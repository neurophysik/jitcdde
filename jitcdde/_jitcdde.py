#!/usr/bin/python3
# -*- coding: utf-8 -*-

from warnings import warn
from itertools import count
import symengine
import numpy as np

from jitcdde.past import Past, Anchor
import jitcdde._python_core as python_core
from jitcxde_common import jitcxde, checker
from jitcxde_common.helpers import sort_helpers, sympify_helpers, find_dependent_helpers
from jitcxde_common.symbolic import collect_arguments, count_calls, replace_function
from jitcxde_common.transversal import GroupHandler
import chspy

_default_min_step = 1e-10

#sigmoid = lambda x: 1/(1+np.exp(-x))
#sigmoid = lambda x: 1 if x>0 else 0
sigmoid = lambda x: (np.tanh(x)+1)/2

#: the symbol for time for defining the differential equation. You may just as well define the an analogous symbol directly with SymEngine or SymPy, but using this function is the best way to get the most of future versions of JiTCDDE, in particular avoiding incompatibilities. You can import a SymPy variant from the submodule `sympy_symbols` instead (see `SymPy vs. SymEngine`_ for details).
t = symengine.Symbol("t", real=True)

def y(index,time=t):
	"""
	the function representing the DDE’s past and present states used for defining the differential equation. The first integer argument denotes the component. The second, optional argument is a symbolic expression denoting the time. This automatically expands to using `current_y`, `past_y`, and `anchors`; so do not be surprised when you look at the output and it is different than what you entered or expected. You can import a SymPy variant from the submodule `sympy_symbols` instead (see `SymPy vs. SymEngine`_ for details).
	"""
	if time == t:
		return current_y(index)
	else:
		return past_y(time, index, anchors(time))

def dy(index,time):
	"""
	This is the function representing the DDE’s past derivative used for defining the differential equation. The first integer argument denotes the component. The second, argument denotes the time. If you use this, you get a neutral DDE which may make addressing initial discontinuities more difficult. Do not use this to get the current derivative. Instead compute it using your dynamical equations. This will not work with tangential Lyapunov exponents.

	**This feature is experimental.**
	
	This automatically expands to using `current_y`, `past_y`, and `anchors`; so do not be surprised when you look at the output and it is different than what you entered or expected. You can import a SymPy variant from the submodule `sympy_symbols` instead (see `SymPy vs. SymEngine`_ for details).
	"""
	if time == t:
		raise ValueError("Do not use `dy` to compute the current derivative. Use your dynamical equations instead.")
	else:
		return past_dy(time, index, anchors(time))

#: the symbol for the current state for defining the differential equation. It is a function and the integer argument denotes the component. This is only needed for specific optimisations of large DDEs; in all other cases use `y` instead. You can import a SymPy variant from the submodule `sympy_symbols` instead (see `SymPy vs. SymEngine`_ for details).
current_y = symengine.Function("current_y")

#: the symbol for DDE’s past state for defining differential equation. It is a function with the first integer argument denoting the component and the second argument being a pair of past points (as being returned by `anchors`) from which the past state is interpolated (or, in rare cases, extrapolated). This is only needed for specific optimisations of large DDEs; in all other cases use `y` instead. You can import a SymPy variant from the submodule `sympy_symbols` instead (see `SymPy vs. SymEngine`_ for details).
past_y = symengine.Function("past_y")

#: the symbol for DDE’s past derivative for defining differential equation. It is a function with the first integer argument denoting the component and the second argument being a pair of past points (as being returned by `anchors`) from which the past state is interpolated (or, in rare cases, extrapolated). This is only needed for specific optimisations of large DDEs; in all other cases use `dy` instead. You can import a SymPy variant from the submodule `sympy_symbols` instead (see `SymPy vs. SymEngine`_ for details).
past_dy = symengine.Function("past_dy")

#: the symbol representing two anchors for defining the differential equation. It is a function and the float argument denotes the time point to which the anchors pertain. This is only needed for specific optimisations of large DDEs; in all other cases use `y` instead. You can import a SymPy variant from the submodule `sympy_symbols` instead (see `SymPy vs. SymEngine`_ for details).
anchors = symengine.Function("anchors")

def _get_delays(f, helpers=()):
	delay_terms = set().union(*(collect_arguments(entry, anchors) for entry in f()))
	delay_terms.update(*(collect_arguments(helper[1], anchors) for helper in helpers))
	delays = [
			(t-delay_term[0]).simplify()
			for delay_term in delay_terms
		]
	return [0]+delays

def _find_max_delay(delays):
	if all(symengine.sympify(delay).is_Number for delay in delays):
		return float(symengine.sympify(max(delays)).n(real=True))
	else:
		raise ValueError("Delay depends on time, dynamics, or control parameters; cannot determine max_delay automatically. You have to pass it as an argument to jitcdde.")

def _propagate_delays(delays, p, threshold=1e-5):
	result = [0]
	if p!=0:
		for delay in _propagate_delays(delays, p-1, threshold):
			for other_delay in delays:
				new_entry = delay + other_delay
				for old_entry in result:
					if abs(new_entry-old_entry) < threshold:
						break
				else:
					result.append(new_entry)
	return result

def quadrature(integrand,variable,lower,upper,nsteps=20,method="gauss"):
	"""
	If your DDE contains an integral over past points, this utility function helps you to implement it. It returns an estimator of the integral from evaluations of the past at discrete points, employing a numerical quadrature. You probably want to disable automatic simplifications when using this.
	
	Parameters
	----------
	integrand : symbolic expression
	
	variable : symbol
		variable of integration
	
	lower, upper : expressions
		lower and upper limit of integration
	
	nsteps : integer
		number of sampling steps. This should be chosen sufficiently high to capture all relevant aspects of your dynamics.
	
	method : `"midpoint"` or `"gauss"`
		which method to use for numerical integration. So far Gauß–Legendre quadrature (`"gauss"`; needs SymPy) and the midpoint rule (`"midpoint"`) are available.
		Use the midpoint rule if you expect your integrand to exhibit structure on a time scale smaller than (`upper` − `lower`)/`nsteps`.
		Otherwise or when in doubt, use `"gauss"`.
	"""
	sample = lambda pos: integrand.subs(variable,pos)
	
	if method == "midpoint":
		half_step = (symengine.sympify(upper-lower)/symengine.sympify(nsteps)/2).simplify()
		return 2*half_step*sum(
				sample(lower+(1+2*i)*half_step)
				for i in range(nsteps)
			)
	elif method == "gauss":
		from sympy.integrals.quadrature import gauss_legendre
		factor = (symengine.sympify(upper-lower)/2).simplify()
		return factor*sum(
				weight*sample(lower+(1+pos)*factor)
				for pos,weight in zip(*gauss_legendre(nsteps,20))
			)
	else:
		raise NotImplementedError("I know no integration method named %s."%method)

class UnsuccessfulIntegration(Exception):
	"""
		This exception is raised when the integrator cannot meet the accuracy and step-size requirements. If you want to know the exact state of your system before the integration fails or similar, catch this exception.
	"""
	pass

class jitcdde(jitcxde):
	"""
	Parameters
	----------
	f_sym : iterable of symbolic expressions or generator function yielding symbolic expressions or dictionary
		If an iterable or generator function, the `i`-th element is the `i`-th component of the value of the DDE’s derivative :math:`f(t,y)`. If a dictionary, it has to map the dynamical variables to its derivatives and the dynamical variables must be `y(0), y(1), …`.
	
	helpers : list of length-two iterables, each containing a symbol and an expression
		Each helper is a variable that will be calculated before evaluating the derivative and can be used in the latter’s computation. The first component of the tuple is the helper’s symbol as referenced in the derivative or other helpers, the second component describes how to compute it from `t`, `y` and other helpers. This is for example useful to realise a mean-field coupling, where the helper could look like `(mean, sum(y(i) for i an range(100))/100)`. (See `the JiTCODE documentation <http://jitcode.readthedocs.io/#module-SW_of_Roesslers>`_ for an example.)
	
	n : integer
		Length of `f_sym`. While JiTCDDE can easily determine this itself (and will, if necessary), this may take some time if `f_sym` is a generator function and `n` is large. Take care that this value is correct – if it isn’t, you will not get a helpful error message.
	
	delays : iterable of expressions or floats
		The delays of the dynamics. If not given, JiTCDDE will determine these itself if needed. However, this may take some time if `f_sym` is large. Take care that these are correct – if they aren’t, you won’t get a helpful error message.
	
	max_delay : number
		Maximum delay. In case of constant delays and if not given, JiTCDDE will determine this itself. However, this may take some time if `f_sym` is large and `delays` is not given. Take care that this value is not too small – if it is, you will not get a helpful error message. If this value is too large, you may run into memory issues for long integration times and calculating Lyapunov exponents (with `jitcdde_lyap`) may take forever.
	
	control_pars : list of symbols
		Each symbol corresponds to a control parameter that can be used when defining the equations and set after compilation with `set_parameters`. Using this makes sense if you need to do a parameter scan with short integrations for each parameter and you are spending a considerable amount of time compiling.
	
	callback_functions : iterable
		Python functions that should be called at integration time (callback) when evaluating the derivative. Each element of the iterable represents one callback function as a tuple containing (in that order):
		
		*	A SymEngine function object used in `f_sym` to represent the function call. If you want to use any JiTCDDE features that need the derivative, this must have a properly defined `f_diff` method with the derivative being another callback function (or constant).
		*	The Python function to be called. This function will receive the state array (`y`) as the first argument. All further arguments are whatever you use as arguments of the SymEngine function in `f_sym`. These can be any expression that you might use in the definition of the derivative and contain, e.g., dynamical variables (current or delayed), time, control parameters, and helpers. The only restriction is that the arguments are floats (and not vectors, anchors or similar). The return value must also be a float (or something castable to float). It is your responsibility to ensure that this function adheres to these criteria, is deterministic and sufficiently smooth with respect its arguments; expect nasty errors otherwise.
		*	The number of arguments, **excluding** the state array as mandatory first argument. This means if you have a variadic Python function, you cannot just call it with different numbers of arguments in `f_sym`, but you have to define separate callbacks for each of numer of arguments.
		
		See `this example <https://github.com/neurophysik/jitcdde/blob/master/examples/sunflower_callback.py>`_ for how to use this.
	
	verbose : boolean
		Whether JiTCDDE shall give progress reports on the processing steps.
	
	module_location : string
		location of a module file from which functions are to be loaded (see `save_compiled`). If you use this, you need not give `f_sym` as an argument, but then you must give `n` and `max_delay`. If you used `control_pars` or `callback_functions`, you have to provide them again. Also note that the integrator may lack some functionalities, depending on the arguments you provide.
	"""
	
	dynvar = current_y
	
	def __init__(
			self,
			f_sym = (), *,
			helpers = None,
			n = None,
			delays = None,
			max_delay = None,
			control_pars = (),
			callback_functions = (),
			verbose = True,
			module_location = None
		):
		
		super(jitcdde,self).__init__(n,verbose,module_location)
		
		self.f_sym = self._handle_input(f_sym)
		if not hasattr(self,"n_basic"):
			self.n_basic = self.n
		self.helpers = sort_helpers(sympify_helpers(helpers or []))
		self.control_pars = control_pars
		self.callback_functions = callback_functions
		
		self.initiate_past()
		self.integration_parameters_set = False
		self.DDE = None
		self.verbose = verbose
		self.delays = delays
		self.max_delay = max_delay
		
		self.initial_discontinuities_handled = False
	
	def initiate_past(self):
		self.past = Past( n=self.n, n_basic=self.n_basic )
	
	@property
	def delays(self):
		if self._delays is None:
			self.delays = _get_delays(self.f_sym, self.helpers)
		return self._delays
	
	@delays.setter
	def delays(self, new_delays):
		self._delays = new_delays
		if (self._delays is not None) and (0 not in self._delays):
			self._delays = np.append(self._delays,0)
		self._max_delay = None
	
	@property
	def max_delay(self):
		if self._max_delay is None:
			self._max_delay = _find_max_delay(self.delays)
			assert self._max_delay >= 0.0, "Negative maximum delay."
		return self._max_delay
	
	@max_delay.setter
	def max_delay(self, new_max_delay):
		if new_max_delay is not None:
			assert new_max_delay >= 0.0, "Negative maximum delay."
		self._max_delay = new_max_delay
	
	@checker
	def _check_non_empty(self):
		self._check_assert( self.f_sym(), "f_sym is empty." )
	
	@checker
	def _check_valid_arguments(self):
		for i,entry in enumerate(self.f_sym()):
			indizes  = [argument[0] for argument in collect_arguments(entry,current_y)]
			indizes += [argument[1] for argument in collect_arguments(entry,past_y   )]
			indizes += [argument[1] for argument in collect_arguments(entry,past_dy  )]
			for index in indizes:
				self._check_assert(
						index >= 0,
						"y is called with a negative argument (%i) in equation %i." % (index,i),
					)
				self._check_assert(
						index < self.n,
						"y is called with an argument (%i) higher than the system’s dimension (%i) in equation %i." % (index,self.n,i),
					)
	
	@checker
	def _check_valid_symbols(self):
		valid_symbols = [t] + [helper[0] for helper in self.helpers] + list(self.control_pars)
		
		for i,entry in enumerate(self.f_sym()):
			for symbol in entry.atoms(symengine.Symbol):
				self._check_assert(
						symbol in valid_symbols,
						"Invalid symbol (%s) in equation %i."  % (symbol.name,i),
					)
	
	def add_past_point(self, time, state, derivative):
		"""
		adds an anchor from which the initial past of the DDE is interpolated.
		
		Parameters
		----------
		time : float
			the temporal position of the anchor.
		state : iterable of floats
			the state of the anchor. The dimension of the array must match the dimension of the differential equation (`n`).
		derivative : iterable of floats
			the derivative at the anchor. The dimension of the array must match the dimension of the differential equation (`n`).
		"""
		self.reset_integrator(idc_unhandled=True)
		self.past.add((time,state,derivative))

	def add_past_points(self, anchors):
		"""
		adds multiple anchors from which the past of the DDE is interpolated.
		
		Parameters
		----------
		anchors : iterable of tuples
			Each tuple must have components corresponding to the arguments of `add_past_point`.
		"""
		self.reset_integrator(idc_unhandled=True)
		for anchor in anchors:
			self.past.add(anchor)
	
	def constant_past(self,state,time=0):
		"""
		initialises the past with a constant state.
		
		Parameters
		----------
		state : iterable of floats
			The length must match the dimension of the differential equation (`n`).
		time : float
			The time at which the integration starts.
		"""
		
		self.reset_integrator(idc_unhandled=True)
		self.past.constant(state,time)
	
	def past_from_function(self,function,times_of_interest=None,max_anchors=100,tol=5):
		"""
		automatically determines anchors describing the past of the DDE from a given function, i.e., a piecewise cubic Hermite interpolation of the function at automatically selected time points will be the initial past. As this process involves heuristics, it is not perfect. For a better control of the initial conditions, use `add_past_point`.
		
		Parameters
		----------
		function : callable or iterable of symbolic expressions
			If callable, this takes the time as an argument and returns an iterable of floats that is the initial state of the past at that time.
			If an iterable of expressions, each expression represents how the initial past of the respective component depends on `t` (requires SymPy).
			In both cases, the lengths of the iterable must match the dimension of the differential equation (`n`).
			
		times_of_interest : iterable of numbers or `None`
			Initial set of time points considered for the interpolation. The highest value will be the starting point of the integration. Further interpolation anchors may be added in between the given anchors depending on heuristic criteria.
			If `None`, these will be automatically chosen depending on the maximum delay and the integration will start at :math:`t=0`.
		
		max_anchors : positive integer
			The maximum number of anchors that this routine will create (including those for the times_of_interest).
		
		tol : integer
			This is a parameter for the heuristics, more precisely the number of digits considered for precision in several places.
			The higher this value, the more likely it is that the heuristic adds anchors.
		"""
		
		self.reset_integrator(idc_unhandled=True)
		if times_of_interest is None:
			times_of_interest = np.linspace(-self.max_delay,0,10)
		else:
			times_of_interest = sorted(times_of_interest)
		
		self.past.from_function(function,times_of_interest,max_anchors,tol)
	
	def get_state(self):
		"""
		Returns an object that represents all anchors currently used by the integrator, which compeletely define the current state. The object is a `CubicHermiteSpline <https://chspy.readthedocs.io>`_ instance (with a few special extensions for JiTCDDE), which allows you to extract all sorts of information from it if you want.
		
		The format can also be used as an argument for `add_past_points`. An example where this is useful is when you want to switch between plain integration and one that also obtains Lyapunov exponents. You can also use this to implement time-dependent equations, however, you need to be careful to truncate the result properly. Moreover, if your delay changes, you may need to set the `max_delay` accordingly to avoid too much past being discarded before you call this method.
		
		If you reinitialise this integrator after calling this, this past will be used.
		"""
		self._initiate()
		self.DDE.forget(self.max_delay)
		self.past.clear()
		self.past.extend(self.DDE.get_full_state())
		return self.past

	def purge_past(self):
		"""
		Clears the past and resets the integrator. You need to define a new past (using `add_past_point`) after this.
		"""
		
		self.past.clear()
		self.reset_integrator(idc_unhandled=True)
	
	def reset_integrator(self,idc_unhandled=False):
		"""
		Resets the integrator, forgetting all integration progress and forcing re-initiation when it is needed next.
		"""
		if idc_unhandled:
			self.initial_discontinuities_handled = False
		self.DDE = None
	
	def generate_lambdas(self,simplify=None):
		"""
		Explicitly initiates a purely Python-based integrator.
		
		Parameter
		---------
		simplify : boolean
			Whether the derivative should be `simplified <http://docs.sympy.org/dev/modules/simplify/simplify.html>`_ (with `ratio=1.0`). The main reason why you could want to disable this is if your derivative is already optimised and so large that simplifying takes a considerable amount of time. If `None`, this will be automatically disabled for `n>10`.
			
		"""
		
		if simplify is None:
			simplify = self.n<=10
		
		if self.callback_functions:
			raise NotImplementedError("Callbacks do not work with lambdification. You must use the C backend.")
		
		assert len(self.past)>1, "You need to add at least two past points first. Usually this means that you did not set an initial past at all."
		
		self.DDE = python_core.dde_integrator(
				self.f_sym,
				self.past,
				self.helpers,
				self.control_pars,
				simplify=simplify,
			)
		self.compile_attempt = False
	
	def _compile_C(self, *args, **kwargs):
		self.compile_C(*args, **kwargs)
	
	def compile_C(
			self,
			simplify = None,
			do_cse = False,
			chunk_size = 100,
			extra_compile_args = None,
			extra_link_args = None,
			verbose = False,
			modulename = None,
			omp = False,
		):
		"""
		translates the derivative to C code using SymEngine’s `C-code printer <https://github.com/symengine/symengine/pull/1054>`_.
		For detailed information many of the arguments and other ways to tweak the compilation, read `these notes <jitcde-common.readthedocs.io>`_.

		Parameters
		----------
		simplify : boolean
			Whether the derivative should be `simplified <http://docs.sympy.org/dev/modules/simplify/simplify.html>`_ (with `ratio=1.0`) before translating to C code. The main reason why you could want to disable this is if your derivative is already optimised and so large that simplifying takes a considerable amount of time. If `None`, this will be automatically disabled for `n>10`.
		
		do_cse : boolean
			Whether SymPy’s `common-subexpression detection <http://docs.sympy.org/dev/modules/rewriting.html#module-sympy.simplify.cse_main>`_ should be applied before translating to C code.
			This is worthwile if your DDE contains the same delay more than once. Otherwise it is almost always better to let the compiler do this (unless you want to set the compiler optimisation to `-O2` or lower). As this requires all entries of `f` at once, it may void advantages gained from using generator functions as an input. Also, this feature uses SymPy and not SymEngine.
		
		chunk_size : integer
			If the number of instructions in the final C code exceeds this number, it will be split into chunks of this size. See `Handling very large differential equations <http://jitcde-common.readthedocs.io/#handling-very-large-differential-equations>`_ on why this is useful and how to best choose this value.
			If smaller than 1, no chunking will happen.
		
		extra_compile_args : iterable of strings
		extra_link_args : iterable of strings
			Arguments to be handed to the C compiler or linker, respectively.
		
		verbose : boolean
			Whether the compiler commands shall be shown. This is the same as Setuptools’ `verbose` setting.
		
		modulename : string or `None`
			The name used for the compiled module.
		
		omp : pair of iterables of strings or boolean
			What compiler arguments shall be used for multiprocessing (using OpenMP). If `True`, they will be selected automatically. If empty or `False`, no compilation for multiprocessing will happen (unless you supply the relevant compiler arguments otherwise).
		"""
		
		self.compile_attempt = False
		
		f_sym_wc = self.f_sym()
		helpers_wc = list(self.helpers) # list is here for copying
		
		if simplify is None:
			simplify = self.n<=10
		if simplify:
			f_sym_wc = (entry.simplify(ratio=1.0) for entry in f_sym_wc)
		
		if do_cse:
			import sympy
			additional_helper = sympy.Function("additional_helper")
			
			_cse = sympy.cse(
					sympy.Matrix(sympy.sympify(list(f_sym_wc))),
					symbols = (additional_helper(i) for i in count())
				)
			helpers_wc.extend(symengine.sympify(_cse[0]))
			f_sym_wc = symengine.sympify(_cse[1][0])
		
		arguments = [
			("self", "dde_integrator * const"),
			("t", "double const"),
			("y", "double", self.n),
			]
		helper_i = 0
		anchor_i = 0
		converted_helpers = []
		self.past_calls = 0
		self.substitutions = {
				control_par: symengine.Symbol("self->parameter_"+control_par.name)
				for control_par in self.control_pars
			}
		
		def finalise(expression):
			expression = expression.subs(self.substitutions)
			self.past_calls += count_calls(expression,anchors)
			return expression
		
		if helpers_wc:
			get_helper = symengine.Function("get_f_helper")
			set_helper = symengine.Function("set_f_helper")
			
			get_anchor = symengine.Function("get_f_anchor_helper")
			set_anchor = symengine.Function("set_f_anchor_helper")
			
			for helper in helpers_wc:
				if (
						type(helper[1]) == type(anchors(0)) and
						helper[1].get_name() == anchors.name
					):
					converted_helpers.append(set_anchor(anchor_i, finalise(helper[1])))
					self.substitutions[helper[0]] = get_anchor(anchor_i)
					anchor_i += 1
				else:
					converted_helpers.append(set_helper(helper_i, finalise(helper[1])))
					self.substitutions[helper[0]] = get_helper(helper_i)
					helper_i += 1
			
			if helper_i:
				arguments.append(("f_helper","double", helper_i))
			if anchor_i:
				arguments.append(("f_anchor_helper","anchor", anchor_i))
			
			self.render_and_write_code(
				converted_helpers,
				name = "helpers",
				chunk_size = chunk_size,
				arguments = arguments,
				omp = False,
				)
		
		set_dy = symengine.Function("set_dy")
		self.render_and_write_code(
			(set_dy(i,finalise(entry)) for i,entry in enumerate(f_sym_wc)),
			name = "f",
			chunk_size = chunk_size,
			arguments = arguments + [("dY", "double", self.n)]
			)
		
		if not self.past_calls:
			warn("Differential equation does not include a delay term.")
		
		self._process_modulename(modulename)
		
		self._render_template(
				n = self.n,
				number_of_helpers = helper_i,
				number_of_anchor_helpers = anchor_i,
				has_any_helpers = anchor_i or helper_i,
				anchor_mem_length = self.past_calls,
				n_basic = self.n_basic,
				control_pars = [par.name for par in self.control_pars],
				tangent_indices = self.tangent_indices if hasattr(self,"tangent_indices") else [],
				chunk_size = chunk_size, # only for OMP
				callbacks = [(fun.name,n_args) for fun,_,n_args in self.callback_functions],
			)
		
		self._compile_and_load(verbose,extra_compile_args,extra_link_args,omp)
	
	def _initiate(self):
		if self.compile_attempt is None:
			self._attempt_compilation()
		
		if self.DDE is None:
			assert len(self.past)>1, "You need to add at least two past points first. Usually this means that you did not set an initial past at all."
			
			if self.compile_attempt:
				self.DDE = self.jitced.dde_integrator(
						self.past,
						*[callback for _,callback,_ in self.callback_functions],
					)
			else:
				self.generate_lambdas()
		
		self._set_integration_parameters()
	
	def set_parameters(self, *parameters):
		"""
		Set the control parameters defined by the `control_pars` argument of the `jitcdde`. Note that you probably want to use `purge_past` and address initial discontinuities every time after you do this.

		Parameters
		----------
		parameters : floats
			Values of the control parameters.
			You can also use a single iterable containing these.
			Either way, the order must be the same as in the `control_pars` argument of the `jitcdde`.
		"""
		
		self._initiate()
		self.initial_discontinuities_handled = False
		try:
			self.DDE.set_parameters(*parameters[0])
		except TypeError:
			self.DDE.set_parameters(*parameters)
		else:
			if len(parameters)>1:
				raise TypeError("Argument must either be a single sequence or multiple numbers.")
	
	def _set_integration_parameters(self):
		if not self.integration_parameters_set:
			self.report("Using default integration parameters.")
			self.set_integration_parameters()
	
	def set_integration_parameters(self,
			atol = 1e-10,
			rtol = 1e-5,
			first_step = 1.0,
			min_step = _default_min_step,
			max_step = 10.0,
			decrease_threshold = 1.1,
			increase_threshold = 0.5,
			safety_factor = 0.9,
			max_factor = 5.0,
			min_factor = 0.2,
			pws_factor = 3,
			pws_atol = 0.0,
			pws_rtol = 1e-5,
			pws_max_iterations = 10,
			pws_base_increase_chance = 0.1,
			pws_fuzzy_increase = False,
			):
		
		"""
		Sets the parameters for the step-size adaption. Arguments starting with `pws` (past within step) are only relevant if the delay is shorter than the step size.
		
		Parameters
		----------
		atol : float
		rtol : float
			The tolerance of the estimated integration error is determined as :math:`\\texttt{atol} + \\texttt{rtol}·|y|`. The step-size adaption algorithm is the same as for the GSL. For details see `its documentation <http://www.gnu.org/software/gsl/manual/html_node/Adaptive-Step_002dsize-Control.html>`_.
		
		first_step : float
			The step-size adaption starts with this value.
			
		min_step : float
			Should the step-size have to be adapted below this value, the integration is aborted and `UnsuccessfulIntegration` is raised.
		
		max_step : float
			The step size will be capped at this value.
			
		decrease_threshold : float
			If the estimated error divided by the tolerance exceeds this, the step size is decreased.
		
		increase_threshold : float
			If the estimated error divided by the tolerance is smaller than this, the step size is increased.
		
		safety_factor : float
			To avoid frequent adaption, all freshly adapted step sizes are multiplied with this factor.
		
		max_factor : float
		min_factor : float
			The maximum and minimum factor by which the step size can be adapted in one adaption step.
		
		pws_factor : float
			Factor of step-size adaptions due to a delay shorter than the time step. If dividing the step size by `pws_factor` moves the delay out of the time step, it is done. If this is not possible, if the iterative algorithm does not converge within `pws_max_iterations`, or if it converges within fewer iterations than `pws_factor`, the step size is decreased or increased, respectively, by this factor.
		
		pws_atol : float
		pws_rtol : float
			If the difference between two successive iterations is below the tolerance determined with these parameters, the iterations are considered to have converged.
		
		pws_max_iterations : integer
			The maximum number of iterations before the step size is decreased.
		
		pws_base_increase_chance : float
			If the normal step-size adaption calls for an increase and the step size was adapted due to the past lying within the step, there is at least this chance that the step size is increased.
			
		pws_fuzzy_increase : boolean
			Whether the decision to try to increase the step size shall depend on chance. The upside of this is that it is less likely that the step size gets locked at a unnecessarily low value. The downside is that the integration is not deterministic anymore. If False, increase probabilities will be added up until they exceed 0.9, in which case an increase happens.
		"""
		
		if first_step > max_step:
			first_step = max_step
			warn("Decreasing first_step to match max_step")
		if min_step > first_step:
			min_step = first_step
			warn("Decreasing min_step to match first_step")
		
		assert decrease_threshold>=1.0, "decrease_threshold smaller than 1"
		assert increase_threshold<=1.0, "increase_threshold larger than 1"
		assert max_factor>=1.0, "max_factor smaller than 1"
		assert min_factor<=1.0, "min_factor larger than 1"
		assert safety_factor<=1.0, "safety_factor larger than 1"
		assert atol>=0.0, "negative atol"
		assert rtol>=0.0, "negative rtol"
		if atol==0 and rtol==0:
			warn("atol and rtol are both 0. You probably do not want this.")
		assert pws_atol>=0.0, "negative pws_atol"
		assert pws_rtol>=0.0, "negative pws_rtol"
		assert 0<pws_max_iterations, "non-positive pws_max_iterations"
		assert 2<=pws_factor, "pws_factor smaller than 2"
		assert pws_base_increase_chance>=0, "negative pws_base_increase_chance"
		
		self.atol = atol
		self.rtol = rtol
		self.dt = first_step
		self.min_step = min_step
		self.max_step = max_step
		self.decrease_threshold = decrease_threshold
		self.increase_threshold = increase_threshold
		self.safety_factor = safety_factor
		self.max_factor = max_factor
		self.min_factor = min_factor
		self.pws_atol = pws_atol
		self.pws_rtol = pws_rtol
		self.pws_max_iterations = pws_max_iterations
		self.pws_factor = pws_factor
		self.pws_base_increase_chance = pws_base_increase_chance
		
		self.q = 3.
		self.last_pws = False
		self.count = 0
		
		if pws_fuzzy_increase:
			self.do_increase = lambda p: np.random.random() < p
		else:
			self.increase_credit = 0.0
			def do_increase(p):
				self.increase_credit += p
				if self.increase_credit >= 0.9:
					self.increase_credit = 0.0
					return True
				else:
					return False
			self.do_increase = do_increase
		
		self.integration_parameters_set = True
	
	def _control_for_min_step(self):
		if self.dt < self.min_step:
			message = (["",
					"Could not integrate with the given tolerance parameters:\n",
					"atol: %e" % self.atol,
					"rtol: %e" % self.rtol,
					"min_step: %e\n" % self.min_step,
					"The most likely reasons for this are:",
				])
			
			if not self.initial_discontinuities_handled:
				message.append("• You did not sufficiently address initial discontinuities. See https://jitcdde.rtfd.io/#discontinuities for details.")
			
			message.append("• The DDE is ill-posed or stiff.")

			if self.initial_discontinuities_handled:
				message.append("• You used `integrate_blindly` and did not adjust the maximum step or you used `adjust_diff` and did not pay attention to the extrema.")
			
			if self.atol==0:
				message.append("• You did not allow for an absolute error tolerance (atol) though your DDE calls for it. Even a very small absolute tolerance (1e-16) may sometimes help.")
			raise UnsuccessfulIntegration("\n".join(message))
	
	def _increase_chance(self, new_dt):
		q = new_dt/self.last_pws
		is_explicit = sigmoid(1-q)
		far_from_explicit = sigmoid(q-self.pws_factor)
		few_iterations = sigmoid(1-self.count/self.pws_factor)
		profile = is_explicit+far_from_explicit*few_iterations
		return profile + self.pws_base_increase_chance*(1-profile)
	
	def _adjust_step_size(self):
		p = self.DDE.get_p(self.atol, self.rtol)
		if p > self.decrease_threshold:
			self.dt *= max(self.safety_factor*p**(-1/self.q), self.min_factor)
			self._control_for_min_step()
			return False
		else:
			if p <= self.increase_threshold:
				factor = self.safety_factor*p**(-1/(self.q+1)) if p else self.max_factor
				
				new_dt = min(
					self.dt*min(factor, self.max_factor),
					self.max_step
					)
				
				if (not self.last_pws) or self.do_increase(self._increase_chance(new_dt)):
					self.dt = new_dt
					self.count = 0
					self.last_pws = False
			return True
	
	@property
	def t(self):
		"""
		Returns
		-------
		time : float
		The current time of the integrator.
		"""
		self._initiate()
		return self.DDE.get_t()
	
	def integrate(self, target_time):
		"""
		Tries to evolve the dynamics.
		
		Parameters
		----------
		
		target_time : float
			time until which the dynamics is evolved
		
		Returns
		-------
		state : NumPy array
			the computed state of the system at `target_time`.
		"""
		self._initiate()
		
		if self.DDE.get_t() > target_time:
			warn("The target time is smaller than the current time. No integration step will happen. The returned state will be extrapolated from the interpolating Hermite polynomial for the last integration step. You may see this because you try to integrate backwards in time, in which case you did something wrong. You may see this just because your sampling step is small, in which case there is no need to worry.")
		
		if not self.initial_discontinuities_handled:
			warn("You did not explicitly handle initial discontinuities. Proceed only if you know what you are doing. This is only fine if you somehow chose your initial past such that the derivative of the last anchor complies with the DDE. In this case, you can set the attribute `initial_discontinuities_handled` to `True` to suppress this warning. See https://jitcdde.rtfd.io/#discontinuities for details.")
		
		while self.DDE.get_t() < target_time:
			if self.try_single_step(self.dt):
				self.DDE.accept_step()
		
		result = self.DDE.get_recent_state(target_time)
		self.DDE.forget(self.max_delay)
		return result
	
	def try_single_step(self,length):
		self.DDE.get_next_step(length)
		
		if self.DDE.past_within_step:
			self.last_pws = self.DDE.past_within_step
			
			# If possible, adjust step size to make integration explicit:
			if self.dt > self.pws_factor*self.DDE.past_within_step:
				self.dt /= self.pws_factor
				self._control_for_min_step()
				return False
			
			# Try to come within an acceptable error within pws_max_iterations iterations; otherwise adjust step size:
			for self.count in range(1,self.pws_max_iterations+1):
				self.DDE.get_next_step(self.dt)
				if self.DDE.check_new_y_diff(self.pws_atol, self.pws_rtol):
					break
			else:
				self.dt /= self.pws_factor
				self._control_for_min_step()
				return False
		
		return self._adjust_step_size()
	
	def adjust_diff(self,shift_ratio=1e-4):
		"""
		Performs a zero-amplitude (backwards) `jump` whose `width` is `shift_ratio` times the distance to the previous anchor into the past. See the documentation of `jump` for the caveats of this and see `discontinuities` for more information on why you almost certainly need to use this or an alternative way to address initial discontinuities.
			
		Returns
		-------
		minima : NumPy array of floats
		maxima : NumPy array of floats
			The minima or maxima, respectively, of each component during the jump interval. See the documentation of `jump` on why you may want these.
		"""
		self._initiate()
		past = self.get_state()
		width = shift_ratio*(past[-1][0]-past[-2][0])
		time = past[-1][0]
		self.initial_discontinuities_handled = True
		return self.jump(0,time,width,forward=False)
	
	def _prepare_blind_int(self, target_time, step):
		self._initiate()
		
		step = step or self.max_step
		assert step>0, "step must be positive"
		
		total_integration_time = target_time-self.DDE.get_t()
		
		if total_integration_time>0:
			if total_integration_time < step:
				step = total_integration_time
			number = int(round(total_integration_time/step))
			dt = total_integration_time/number
		elif total_integration_time==0:
			number = 0
			dt = 0
		else:
			raise ValueError("Can’t integrate backwards in time.")
		
		assert abs(number*dt-total_integration_time)<1e-10
		return dt, number, total_integration_time
	
	def integrate_blindly(self, target_time, step=None):
		"""
		Evolves the dynamics with a fixed step size ignoring any accuracy concerns. If a delay is smaller than the time step, the state is extrapolated from the previous step. See `discontinuities` for more information on why you almost certainly need to use this or an alternative way to address initial discontinuities.
		
		If the target time equals the current time, `adjust_diff` is called automatically.
		
		Parameters
		----------
		target_time : float
			time until which the dynamics is evolved. In most cases, this should be larger than the maximum delay.
		
		step : float
			aspired step size. The actual step size may be slightly adapted to make it divide the integration time. If `None`, `0`, or otherwise falsy, the maximum step size as set with `max_step` of `set_integration_parameters` is used.
		
		Returns
		-------
		state : NumPy array
			the computed state of the system at `target_time`.
		"""
		
		self.initial_discontinuities_handled = True
		dt,number,_ = self._prepare_blind_int(target_time, step)
		
		if number == 0:
			self.adjust_diff()
		else:
			for _ in range(number):
				self.DDE.get_next_step(dt)
				self.DDE.accept_step()
				self.DDE.forget(self.max_delay)
		
		return self.DDE.get_current_state()
	
	def step_on_discontinuities(
			self, *,
			propagations = 1,
			min_distance = 1e-5,
			max_step = None,
		):
		"""
		Assumes that the derivative is discontinuous at the start of the integration and chooses steps such that propagations of this point via the delays always fall on integration steps (or very close). Between these, adaptive steps are taken if necessary to keep the error within the bounds set by `set_integration_parameters`. If the discontinuity was propagated sufficiently often, it is considered to be smoothed and the integration is stopped. See `discontinuities` for more information on why you almost certainly need to use this or an alternative way to address initial discontinuities.
		
		This only makes sense if you just defined the past (via `add_past_point`) and start integrating, just reset the integrator, or changed control parameters.
		
		In case of an ODE, `adjust_diff` is used automatically.
		
		Parameters
		----------
		propagations : integer
			how often the discontinuity has to propagate to before it’s considered smoothed
		
		min_distance : float
			If two required steps are closer than this, they will be treated as one.
		
		max_step : float
			Retired parameter. Steps are now automatically adapted.
		
		Returns
		-------
		state : NumPy array
			the computed state of the system after integration
		"""
		
		self._initiate()
		self.initial_discontinuities_handled = True
		
		assert min_distance > 0, "min_distance must be positive."
		assert isinstance(propagations,int), "Non-integer number of propagations."
		
		if max_step is not None:
			warn("The max_step parameter is retired and does not need to be used anymore. Instead, step_on_discontinuities now adapts the step size. This will raise an error in the future.")
		
		if not all(symengine.sympify(delay).is_number for delay in self.delays):
			raise ValueError("At least one delay depends on time, dynamics, or control parameters; cannot automatically determine steps.")
		self.delays = [
				# This conversion is due to SymEngine.py issue #227
				float(symengine.sympify(delay).n(real=True))
				for delay in self.delays
			]
		steps = _propagate_delays(self.delays, propagations, min_distance)
		steps.sort()
		steps.remove(0)
		
		if steps:
			for step in steps:
				target_time = self.t + step
				while True:
					current_time = self.DDE.get_t()
					if current_time >= target_time:
						break
					if self.try_single_step( min(self.dt,target_time-current_time) ):
						self.DDE.accept_step()
			
			result = self.DDE.get_recent_state(target_time)
			self.DDE.forget(self.max_delay)
			return result
		else:
			self.adjust_diff()
			return self.DDE.get_current_state()[:self.n_basic]
	
	def jump( self, amplitude, time, width=1e-5, forward=True ):
		"""
		Applies a jump to the state. Since this naturally introduces a discontinuity to the state, it can only be approximated. This is done by adding two anchors in a short temporal distance `width`. With other words, the jump is not instantaneous but just a strong change of the state over a small interval of length `width`. The slope after the jump is computed using the derivative `f`.
		
		A potential source of numerical issues is that the Hermite interpolant becomes extreme during the jump (due to Runge’s phenomenon). Whether this is a problem primarily depends on how these extrema influence delayed dependencies of your derivative. To allow you to estimate the magnitude of this, this function returns the extrema during the jump interval. There are two ways to address this:

		* Integrate with steps such that past values from the jump interval are avoided. You can use `integrate_blindly` to do this.
		* Increase the `width`.
		
		Note that due to the adapted derivative, there are no initial discontinuities after this.
		
		Parameters
		----------
		amplitude : NumPy array of floats
			The amplitude of the jump.
		
		time : float
			The time at which the jump shall happen. Usually this would be the last time to which you integrated.
		
		width : float
			The size of the jump interval (see above). The smaller you choose this, the sharper the jump, but the more likely are numerical problems in its wake. This should be smaller than all delays in the system.
		
		forward : boolean
			Whether the jump interval should begin after `time`. Otherwise it will end at `time` (equivalent to a forward jump starting at `time`−`width`).
		
		Returns
		-------
		minima : NumPy array of floats
		maxima : NumPy array of floats
			The minima or maxima, respectively, of each component during the jump interval. See above on why you may want these.
		"""
		self._initiate()
		self.initial_discontinuities_handled = True
		assert width>=0
		if np.ndim(amplitude)==0:
			amplitude = np.full(self.n,amplitude,dtype=float)
		amplitude = np.atleast_1d(np.array(amplitude,dtype=float))
		assert amplitude.shape == (self.n,)
		
		if forward:
			return self.DDE.apply_jump( amplitude, time      , width )
		else:
			return self.DDE.apply_jump( amplitude, time-width, width )

def _jac(f, helpers, delay, n):
	dependent_helpers = [
			find_dependent_helpers(helpers,y(i,t-delay))
			for i in range(n)
		]
	
	def line(f_entry):
		for j in range(n):
			entry = f_entry.diff(y(j,t-delay))
			for helper in dependent_helpers[j]:
				entry += f_entry.diff(helper[0]) * helper[1]
			yield entry
	
	for f_entry in f():
		yield line(f_entry)


def tangent_vector_f(f, helpers, n, n_lyap, delays, zero_padding=0, simplify=True):
	if f:
		def f_lyap():
			yield from f()
			
			for i in range(n_lyap):
				jacs = [_jac(f, helpers, delay, n) for delay in delays]
				
				for _ in range(n):
					expression = 0
					for delay,jac in zip(delays,jacs):
						for k,entry in enumerate(next(jac)):
							expression += entry * y(k+(i+1)*n,t-delay)
					
					if simplify:
						expression = expression.simplify(ratio=1.0)
					yield expression
			
			for _ in range(zero_padding):
				yield symengine.sympify(0)
	
	else:
		return []
	
	return f_lyap

class jitcdde_lyap(jitcdde):
	"""Calculates the Lyapunov exponents of the dynamics (see the documentation for more details). The handling is the same as that for `jitcdde` except for:
	
	Parameters
	----------
	n_lyap : integer
		Number of Lyapunov exponents to calculate.
	
	simplify : boolean
		Whether the differential equations for the separation function shall be `simplified <http://docs.sympy.org/dev/modules/simplify/simplify.html>`_ (with `ratio=1.0`). Doing so may speed up the time evolution but may slow down the generation of the code (considerably for large differential equations). If `None`, this will be automatically disabled for `n>10`.
		"""
	
	def __init__( self, f_sym=(), n_lyap=1, simplify=None, **kwargs ):
		self.n_basic = kwargs.pop("n",None)
		
		kwargs.setdefault("helpers",())
		kwargs["helpers"] = sort_helpers(sympify_helpers(kwargs["helpers"] or []))

		f_basic = self._handle_input(f_sym,n_basic=True)
		
		if simplify is None:
			simplify = self.n_basic<=10
		
		if "delays" not in kwargs.keys() or not kwargs["delays"]:
			kwargs["delays"] = _get_delays(f_basic,kwargs["helpers"])
		
		assert n_lyap>=0, "n_lyap negative"
		self._n_lyap = n_lyap
		
		f_lyap = tangent_vector_f(
				f = f_basic,
				helpers = kwargs["helpers"],
				n = self.n_basic,
				n_lyap = self._n_lyap,
				delays = kwargs["delays"],
				zero_padding = 0,
				simplify = simplify
			)
		
		super(jitcdde_lyap, self).__init__(
				f_lyap,
				n = self.n_basic*(self._n_lyap+1),
				**kwargs
			)
		
		assert self.max_delay>0, "Maximum delay must be positive for calculating Lyapunov exponents."
	
	def integrate(self, target_time):
		"""
		Like `jitcdde`’s `integrate`, except for orthonormalising the separation functions and:
		
		Returns
		-------
		y : one-dimensional NumPy array
			The state of the system. Same as the output of `jitcdde`’s `integrate`.
		
		lyaps : one-dimensional NumPy array
			The “local” Lyapunov exponents as estimated from the growth or shrinking of the separation function during the integration time of this very `integrate` command.
			
		integration time : float
			The actual integration time during to which the local Lyapunov exponents apply. Note that this is not necessarily the difference between `target_time` and the previous `target_time` as JiTCDDE usually integrates a bit ahead and estimates the output via interpolation. When averaging the Lyapunov exponents, you almost always want to weigh them with the integration time.
			
			If the size of the advance by `integrate` (the sampling step) is smaller than the actual integration step, it may also happen that `integrate` does not integrate at all and the integration time is zero. In this case, the local Lyapunov exponents are returned as `0`, which is as nonsensical as any other result (except perhaps `nan`) but should not matter with a proper weighted averaging.
			
		It is essential that you choose `target_time` properly such that orthonormalisation does not happen too rarely. If you want to control the maximum step size, use the parameter `max_step` of `set_integration_parameters` instead.
		"""
		
		self._initiate()
		old_t = self.DDE.get_t()
		result = super(jitcdde_lyap, self).integrate(target_time)[:self.n_basic]
		delta_t = self.DDE.get_t()-old_t
		
		if delta_t!=0:
			norms = self.DDE.orthonormalise(self._n_lyap, self.max_delay)
			lyaps = np.log(norms) / delta_t
		else:
			lyaps = np.zeros(self._n_lyap)
		
		return result, lyaps, delta_t
	
	def set_integration_parameters(self, **kwargs):
		if self._n_lyap/self.n_basic > 2:
			required_max_step = self.max_delay/(np.ceil(self._n_lyap/self.n_basic/2)-1)
			if "max_step" in kwargs.keys():
				if kwargs["max_step"] > required_max_step:
					kwargs["max_step"] = required_max_step
					warn("Decreased max_step to %f to ensure sufficient dimensionality for Lyapunov exponents." % required_max_step)
			else:
				kwargs["max_step"] = required_max_step
			
			kwargs.setdefault("min_step",_default_min_step)
			
			if kwargs["min_step"] > required_max_step:
				warn("Given the number of desired Lyapunov exponents and the maximum delay in the system, the highest possible step size is lower than the default min_step or the min_step set by you. This is almost certainly a very bad thing. Nonetheless I will lower min_step accordingly.")
			
		super(jitcdde_lyap, self).set_integration_parameters(**kwargs)
	
	def integrate_blindly(self, target_time, step=None):
		"""
		Like `jitcdde`’s `integrate_blindly`, except for orthonormalising the separation functions after each step and the output being analogous to `jitcdde_lyap`’s `integrate`.
		"""
		
		dt,number,total_integration_time = self._prepare_blind_int(target_time, step)
		
		instantaneous_lyaps = []
		
		for _ in range(number):
			self.DDE.get_next_step(dt)
			self.DDE.accept_step()
			self.DDE.forget(self.max_delay)
			norms = self.DDE.orthonormalise(self._n_lyap, self.max_delay)
			instantaneous_lyaps.append(np.log(norms)/dt)
		
		lyaps = np.average(instantaneous_lyaps, axis=0)
		
		return self.DDE.get_current_state()[:self.n_basic], lyaps, total_integration_time

class jitcdde_restricted_lyap(jitcdde):
	"""
	Calculates the largest Lyapunov exponent in orthogonal direction to a predefined plane, i.e. the projection of the separation function onto that plane vanishes. See `this test <https://github.com/neurophysik/jitcdde/blob/master/tests/test_restricted_lyap.py>`_ for an example of usage. Note that coordinate planes (i.e., planes orthogonal to vectors with only one non-zero component) are handled considerably faster. Consider transforming your differential equation to achieve this.

	The handling is the same as that for `jitcdde_lyap` except for:
	
	Parameters
	----------
	vectors : iterable of pairs of NumPy arrays
		A basis of the plane, whose projection shall be removed. The first vector in each pair is the component coresponding to the the state, the second vector corresponds to the derivative.
		
	"""
	
	def __init__(self, f_sym=(), vectors=(), **kwargs):
		self.n_basic = kwargs.pop("n",None)
		
		kwargs.setdefault("helpers",())
		kwargs["helpers"] = sort_helpers(sympify_helpers(kwargs["helpers"] or []))
		
		f_basic = self._handle_input(f_sym,n_basic=True)
		
		if kwargs.pop("simplify",None) is None:
			simplify = self.n_basic<=10
		
		if "delays" not in kwargs.keys() or not kwargs["delays"]:
			kwargs["delays"] = _get_delays(f_basic,kwargs["helpers"])
		
		self.vectors = []
		self.state_components = []
		self.diff_components = []
		
		for vector in vectors:
			assert len(vector[0]) == self.n_basic
			assert len(vector[1]) == self.n_basic
			
			if np.count_nonzero(vector[0])+np.count_nonzero(vector[1]) > 1:
				state = np.array(vector[0], dtype=float, copy=True)
				diff  = np.array(vector[1], dtype=float, copy=True)
				self.vectors.append((state,diff))
			elif np.count_nonzero(vector[0])==1 and np.count_nonzero(vector[1])==0:
				self.state_components.append(vector[0].nonzero()[0][0])
			elif np.count_nonzero(vector[1])==1 and np.count_nonzero(vector[0])==0:
				self.diff_components.append(vector[1].nonzero()[0][0])
			else:
				raise ValueError("One vector contains only zeros.")
		
		f_lyap = tangent_vector_f(
				f = f_basic,
				helpers = kwargs["helpers"],
				n = self.n_basic,
				n_lyap = 1,
				delays = kwargs["delays"],
				zero_padding = 2*self.n_basic*len(self.vectors),
				simplify = simplify
			)
		
		super(jitcdde_restricted_lyap, self).__init__(
				f_lyap,
				n = self.n_basic*(2+2*len(self.vectors)),
				**kwargs
			)
		
		assert self.max_delay>0, "Maximum delay must be positive for calculating Lyapunov exponents."
	
	def remove_projections(self):
		for state_component in self.state_components:
			self.DDE.remove_state_component(state_component)
		for diff_component in self.diff_components:
			self.DDE.remove_diff_component(diff_component)
		norm = self.DDE.remove_projections(self.max_delay, self.vectors)
		return norm
	
	def set_integration_parameters(self, *args, **kwargs):
		super(jitcdde_restricted_lyap, self).set_integration_parameters(*args, **kwargs)
		if (self.state_components or self.diff_components) and not self.atol:
			warn("At least one of your vectors has only one component while your absolute error (atol) is 0. This may cause problems due to spuriously high relative errors. Consider setting atol to some small, non-zero value (e.g., 1e-10) to avoid this.")
	
	def integrate(self, target_time):
		"""
		Like `jitcdde`’s `integrate`, except for normalising and aligning the separation function and:
		
		Returns
		-------
		y : one-dimensional NumPy array
			The state of the system. Same as the output of `jitcdde`’s `integrate`.
		
		lyap : float
			The “local” largest transversal Lyapunov exponent as estimated from the growth or shrinking of the separation function during the integration time of this very `integrate` command.
			
		integration time : float
			The actual integration time during to which the local Lyapunov exponents apply. Note that this is not necessarily difference between `target_time` and the previous `target_time`, as JiTCDDE usually integrates a bit ahead and estimates the output via interpolation. When averaging the Lyapunov exponents, you almost always want to weigh them with the integration time.
			
			If the size of the advance by `integrate` (the sampling step) is smaller than the actual integration step, it may also happen that `integrate` does not integrate at all and the integration time is zero. In this case, the local Lyapunov exponents are returned as `0`, which is as nonsensical as any other result (except perhaps `nan`) but should not matter with a proper weighted averaging.
		
		It is essential that you choose `target_time` properly such that orthonormalisation does not happen too rarely. If you want to control the maximum step size, use the parameter `max_step` of `set_integration_parameters` instead.
		"""
		
		self._initiate()
		old_t = self.DDE.get_t()
		result = super(jitcdde_restricted_lyap, self).integrate(target_time)[:self.n_basic]
		delta_t = self.DDE.get_t()-old_t
		
		if delta_t==0:
			warn("No actual integration happened in this call of integrate. This happens because the sampling step became smaller than the actual integration step. While this is not a problem per se, I cannot return a meaningful local Lyapunov exponent; therefore I return 0 instead.")
			lyap = 0
		else:
			norm = self.remove_projections()
			lyap = np.log(norm) / delta_t
		return result, lyap, delta_t
	
	def integrate_blindly(self, target_time, step=None):
		"""
		Like `jitcdde`’s `integrate_blindly`, except for normalising and aligning the separation function after each step and the output being analogous to `jitcdde_restricted_lyap`’s `integrate`.
		"""
		
		self._initiate()
		self.initial_discontinuities_handled = True
		
		dt,number,total_integration_time = self._prepare_blind_int(target_time, step)
		
		instantaneous_lyaps = []
		
		for _ in range(number):
			self.DDE.get_next_step(dt)
			self.DDE.accept_step()
			self.DDE.forget(self.max_delay)
			norm = self.remove_projections()
			instantaneous_lyaps.append(np.log(norm)/dt)
		
		lyap = np.average(instantaneous_lyaps)
		state = self.DDE.get_current_state()[:self.n_basic]
		
		return state, lyap, total_integration_time


class jitcdde_transversal_lyap(jitcdde,GroupHandler):
	"""
	Calculates the largest Lyapunov exponent in orthogonal direction to a predefined synchronisation manifold, i.e. the projection of the tangent vector onto that manifold vanishes. In contrast to `jitcdde_restricted_lyap`, this performs some transformations tailored to this specific application that may strongly reduce the number of differential equations and ensure a dynamics on the synchronisation manifold.

	Note that all functions for defining the past differ from their analoga from `jitcdde` by having the dimensions of the arguments reduced to the number of groups. This means that only one initial value (of the state or derivative) per group of synchronised components has to be provided (in the same order as the `groups` argument of the constructor).
	
	The handling is the same as that for `jitcdde` except for:
	
	Parameters
	----------
	groups : iterable of iterables of integers
		each group is an iterable of indices that identify dynamical variables that are synchronised on the synchronisation manifold.
	
	simplify : boolean
		Whether the transformed differential equations shall be subjected to SymEngine’s `simplify`. Doing so may speed up the time evolution but may slow down the generation of the code (considerably for large differential equations). If `None`, this will be automatically disabled for `n>10`.
	"""
	
	def __init__( self, f_sym=(), groups=(), simplify=None, **kwargs ):
		GroupHandler.__init__(self,groups)
		self.n = kwargs.pop("n",None)
		
		f_basic,extracted = self.extract_main(self._handle_input(f_sym))
		if simplify is None:
			simplify = self.n<=10
		helpers = sort_helpers(sympify_helpers( kwargs.pop("helpers",[]) ))
		delays = kwargs.pop("delays",()) or _get_delays(f_basic,helpers)
		
		past_z = symengine.Function("past_z")
		current_z = symengine.Function("current_z")
		def z(index,time=t):
			if time == t:
				return current_z(index)
			else:
				return past_z(time, index, anchors(time))
		
		tangent_vectors = {}
		for d in delays:
			z_vector = [z(i,(t-d)) for i in range(self.n)]
			tangent_vectors[d] = self.back_transform(z_vector)
		
		def tangent_vector_f():
			jacs = [
					_jac( f_basic, helpers, delay, self.n )
					for delay in delays
				]
			
			for _ in range(self.n):
				expression = 0
				for delay,jac in zip(delays,jacs):
					try:
						for k,entry in enumerate(next(jac)):
							expression += entry * tangent_vectors[delay][k]
					except StopIteration:
						raise AssertionError("Something got horribly wrong")
				yield expression
		
		current_z_conflate = lambda i: current_z(self.map_to_main(i))
		past_z_conflate = lambda t,i,a: past_z(t,self.map_to_main(i),a)
		
		def finalise(entry):
			entry = replace_function(entry,current_y,current_z_conflate)
			entry = replace_function(entry,past_y,past_z_conflate)
			if simplify:
				entry = entry.simplify(ratio=1)
			entry = replace_function(entry,current_z,current_y)
			entry = replace_function(entry,past_z,past_y)
			return entry
		
		def f_lyap():
			for entry in self.iterate(tangent_vector_f()):
				if type(entry)==int:
					yield finalise(extracted[self.main_indices[entry]])
				else:
					yield finalise(entry[0]-entry[1])
		
		helpers = ((helper[0],finalise(helper[1])) for helper in helpers)
		
		super(jitcdde_transversal_lyap, self).__init__(
				f_lyap,
				n = self.n,
				delays = delays,
				helpers = helpers,
				**kwargs
			)
	
	def initiate_past(self):
		self.past = Past(
				n=self.n,
				n_basic=self.n_basic,
				tangent_indices = self.tangent_indices
			)
	
	def norm(self):
		tangent_vector = self._y[self.tangent_indices]
		norm = np.linalg.norm(tangent_vector)
		tangent_vector /= norm
		if not np.isfinite(norm):
			warn("Norm of perturbation vector for Lyapunov exponent out of numerical bounds. You probably waited too long before renormalising and should call integrate with smaller intervals between steps (as renormalisations happen once with every call of integrate).")
		self._y[self.tangent_indices] = tangent_vector
		return norm
	
	def integrate(self, target_time):
		"""
		Like `jitcdde`’s `integrate`, except for normalising and aligning the separation function and:
		
		Returns
		-------
		y : one-dimensional NumPy array
			The state of the system. Only one initial value per group of synchronised components is returned (in the same order as the `groups` argument of the constructor).
		
		lyap : float
			The “local” largest transversal Lyapunov exponent as estimated from the growth or shrinking of the separation function during the integration time of this very `integrate` command.
			
		integration time : float
			The actual integration time during to which the local Lyapunov exponents apply. Note that this is not necessarily difference between `target_time` and the previous `target_time`, as JiTCDDE usually integrates a bit ahead and estimates the output via interpolation. When averaging the Lyapunov exponents, you almost always want to weigh them with the integration time.
			
			If the size of the advance by `integrate` (the sampling step) is smaller than the actual integration step, it may also happen that `integrate` does not integrate at all and the integration time is zero. In this case, the local Lyapunov exponents are returned as `0`, which is as nonsensical as any other result (except perhaps `nan`) but should not matter with a proper weighted averaging.
		
		It is essential that you choose `target_time` properly such that orthonormalisation does not happen too rarely. If you want to control the maximum step size, use the parameter `max_step` of `set_integration_parameters` instead.
		"""
		
		self._initiate()
		old_t = self.DDE.get_t()
		result = super(jitcdde_transversal_lyap, self).integrate(target_time)[self.main_indices]
		delta_t = self.DDE.get_t()-old_t
		
		if delta_t==0:
			warn("No actual integration happened in this call of integrate. This happens because the sampling step became smaller than the actual integration step. While this is not a problem per se, I cannot return a meaningful local Lyapunov exponent; therefore I return 0 instead.")
			lyap = 0
		else:
			norm = self.DDE.normalise_indices(self.max_delay)
			lyap = np.log(norm) / delta_t
		return result, lyap, delta_t
	
	def integrate_blindly(self, target_time, step=None):
		"""
		Like `jitcdde`’s `integrate_blindly`, except for normalising and aligning the separation function after each step and the output being analogous to `jitcdde_transversal_lyap`’s `integrate`.
		"""
		
		self.initial_discontinuities_handled = True
		dt,number,total_integration_time = self._prepare_blind_int(target_time, step)
		
		instantaneous_lyaps = []
		
		for _ in range(number):
			self.DDE.get_next_step(dt)
			self.DDE.accept_step()
			self.DDE.forget(self.max_delay)
			norm = self.DDE.normalise_indices(self.max_delay)
			instantaneous_lyaps.append(np.log(norm)/dt)
		
		lyap = np.average(instantaneous_lyaps)
		state = self.DDE.get_current_state()[self.main_indices]
		
		return state, lyap, total_integration_time

input_shift = symengine.Symbol("external_input",real=True,negative=False)
input_base_n = symengine.Symbol("input_base_n",integer=True,negative=False)
def input(index,time=t):
	"""
	Function representing an external input (for `jitcdde_input`). The first integer argument denotes the component. The second, optional argument is a symbolic expression denoting the time. This automatically expands to using `current_y`, `past_y`, `anchors`, `input_base_n`, and `input_shift`; so do not be surprised when you look at the output and it is different than what you entered or expected. You can import a SymPy variant from the submodule `sympy_symbols` instead (see `SymPy vs. SymEngine`_ for details).
	"""
	if time!=t:
		warn("Do not use delayed inputs unless you also have undelayed inputs (otherwise you can just shift your time frame). If you use delayed and undelayed inputs, you have to rely on automatic delay detection or explicitly add `input[-1].time+delay` to the delays (and consider it as a max_delay) for each delayed input.")
	return y(index+input_base_n,time-input_shift)

class jitcdde_input(jitcdde):
	"""
	Allows to integrate the DDE (or an ODE) with input. Under the hood, this is handled by adding dummy dynamical variables, in whose past state the input is stored.
	In contrast to other variants of JiTCDDE, the integration must start at :math:`t=0`.
	
	All parameters and methods are as for `jitcdde`, except:
	
	Parameters
	----------
	input : CHSPy CubicHermiteSpline
		The input.
		This has to be a CubicHermiteSpline (specifying the values and derivatives at certain anchor points). Be sure to plot the input to check whether it conforms with your expectations.
	
	"""
	
	def __init__( self, f_sym=(), input=None, **kwargs ):
		if input is None:
			raise ValueError("You must define an input; otherwise just use plain jitcdde.")
		if input[0].time > 0:
			warn(f"Your input past does not begin at t=0 but at t={input[0].time}. Values before the beginning of the past will be extrapolated. You very likely do not want this.")
		
		self.n = kwargs.pop("n",None)
		f_basic = self._handle_input(f_sym)
		self.input_base_n = self.n
		self.n += input.n
		
		self.input_duration = input[-1].time
		substitutions = {
				input_base_n: self.input_base_n,
				input_shift: self.input_duration,
			}
		
		self.regular_past = chspy.CubicHermiteSpline(n=self.input_base_n)
		self.input_past = chspy.CubicHermiteSpline(n=input.n)
		for anchor in input:
			self.input_past.add((
					anchor.time-self.input_duration,
					anchor.state,
					anchor.diff,
				))
		
		def f_full():
			for expression in f_basic():
				yield expression.subs(substitutions)
			# Dummy dynamical variables having constant diff of the last input point to avoid adjustment and to provide the best extrapolation, if required.
			yield from input[-1].diff
		
		if kwargs.setdefault("delays",None) is not None:
			kwargs["delays"] = [ *kwargs["delays"], self.input_duration ]
		
		if kwargs.setdefault("max_delay",None) is not None:
			kwargs["max_delay"] = max( kwargs["max_delay"], self.input_duration )
		
		super().__init__( f_full, n=self.n, **kwargs )
		
		self._past = None
	
	def initiate_past(self):
		pass
	
	@property
	def past(self):
		if self._past is None:
			assert len(self.regular_past)>1, "You need to add at least two past points first. Usually this means that you did not set an initial past at all."
			if self.regular_past[-1].time != 0:
				raise ValueError("For jitcdde_input, the initial past must end at t=0.")
			self._past = chspy.join(self.regular_past,self.input_past)
		return self._past
	
	@past.setter
	def past(self,value):
		raise AssertionError("For jitcdde_input, the past attribute should not be directly set.")
	
	def add_past_point(self, time, state, derivative):
		self._past = None
		self.reset_integrator(idc_unhandled=True)
		self.regular_past.add((time,state,derivative))

	def add_past_points(self, anchors):
		self._past = None
		self.reset_integrator(idc_unhandled=True)
		for anchor in anchors:
			self.regular_past.add(anchor)
	
	def constant_past(self,state,time=0):
		self._past = None
		self.regular_past.constant(state,time)
	
	def past_from_function(self,function,times_of_interest=None,max_anchors=100,tol=5):
		self._past = None
		if times_of_interest is None:
			times_of_interest = np.linspace(-self.max_delay,0,10)
		else:
			times_of_interest = sorted(times_of_interest)
		
		self.regular_past.from_function(function,times_of_interest,max_anchors,tol)
	
	def get_state(self):
		self._initiate()
		self.DDE.forget(self.max_delay)
		self._past = None
		self.regular_past.clear()
		for anchor in self.DDE.get_full_state():
			self.regular_past.append((
					anchor[0],
					anchor[1][:self.input_base_n],
					anchor[2][:self.input_base_n],
				))
		return self.regular_past
	
	def purge_past(self):
		self._past = None
		self.regular_past.clear()
		self.reset_integrator(idc_unhandled=True)
	
	def integrate(self,target_time):
		if target_time>self.input_duration:
			warn("Integrating beyond duration of input. From now on, the input will be extrapolated. You very likely do not want this. Instead define a longer input or stop integrating.")
		return super().integrate(target_time)[:self.input_base_n]
	
	def integrate_blindly(self,*args,**kwargs):
		return super().integrate_blindly(*args,**kwargs)[:self.input_base_n]
	
	def step_on_discontinuities(self,*args,**kwargs):
		raise NotImplementedError("Stepping on discontinuities is not implemented for input yet. Use integrate_blindly or adjust_diff instead.")
	
	def jump(self,amplitude,*args,**kwargs):
		if np.ndim(amplitude)==0:
			amplitude = np.full(self.input_base_n,amplitude,dtype=float)
		assert amplitude.shape == (self.input_base_n,)
		extended_amplitude = np.hstack((amplitude,np.zeros(self.input_past.n)))
		minima,maxima = super().jump(extended_amplitude,*args,**kwargs)
		return minima[:self.input_base_n], maxima[:self.input_base_n]

def test(omp=True,sympy=True):
	"""
		Runs a quick simulation to test whether:
		
		* a compiler is available and can be interfaced by Setuptools,
		* OMP libraries are available and can be assessed,
		* SymPy is available.
		
		The latter two tests can be deactivated with the respective argument. This is not a full software test but rather a quick sanity check of your installation. If successful, this function just finishes without any message.
	"""
	if sympy:
		import sympy
	DDE = jitcdde( [y(1,t-1),-y(0,t-2)], verbose=False )
	DDE.compile_C(chunk_size=1,omp=omp)
	DDE.constant_past([1,2])
	DDE.step_on_discontinuities()
	DDE.integrate(DDE.t+1)

