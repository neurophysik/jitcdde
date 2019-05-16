#!/usr/bin/python3
# -*- coding: utf-8 -*-

from warnings import warn
from itertools import count
import symengine
import numpy as np

import jitcdde._python_core as python_core
from jitcxde_common import jitcxde, checker
from jitcxde_common.helpers import sort_helpers, sympify_helpers, find_dependent_helpers
from jitcxde_common.symbolic import collect_arguments, count_calls, replace_function
from jitcxde_common.numerical import random_direction, rel_dist
from jitcxde_common.transversal import GroupHandler

_default_min_step = 1e-10

#sigmoid = lambda x: 1/(1+np.exp(-x))
#sigmoid = lambda x: 1 if x>0 else 0
sigmoid = lambda x: (np.tanh(x)+1)/2

#: the symbol for time for defining the differential equation. You may just as well define the an analogous symbol directly with SymEngine or SymPy, but using this function is the best way to get the most of future versions of JiTCDDE, in particular avoiding incompatibilities.
t = symengine.Symbol("t", real=True)

def y(index,time=t):
	"""
	the function representing the DDE’s past and present states used for defining the differential equation. The first integer argument denotes the component. The second, optional argument is a symbolic expression denoting the time. This automatically expands to using `current_y`, `past_y`, and `anchors`; so do not be surprised when you look at the output and it is different than what you entered or expected.
	"""
	if time == t:
		return current_y(index)
	else:
		return past_y(time, index, anchors(time))

#: the symbol for the current state for defining the differential equation. It is a function and the integer argument denotes the component. This is only needed for specific optimisations of large DDEs; in all other cases use `y` instead.
current_y = symengine.Function("current_y")

#: the symbol for DDE’s past state for defining differential equation. It is a function with the first integer argument denoting the component and the second argument being a pair of past points (as being returned by `anchors`) from which the past state is interpolated (or, in rare cases, extrapolated). This is only needed for specific optimisations of large DDEs; in all other cases use `y` instead.
past_y = symengine.Function("past_y")

#: the symbol representing two anchors for defining the differential equation. It is a function and the float argument denotes the time point to which the anchors pertain. This is only needed for specific optimisations of large DDEs; in all other cases use `y` instead.
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
		raise ValueError("Delay depends on time or dynamics; cannot determine max_delay automatically. You have to pass it as an argument to jitcdde.")

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
	
	verbose : boolean
		Whether JiTCDDE shall give progress reports on the processing steps.
	
	module_location : string
		location of a module file from which functions are to be loaded (see `save_compiled`). If you use this, you need not give `f_sym` as an argument, but in this case you must give `n` and `max_delay`. Also note that the integrator may lack some functionalities, depending on the arguments you provide.
	"""
	
	dynvar = current_y
	
	def __init__(
			self,
			f_sym = (),
			helpers = None,
			n = None,
			delays = None,
			max_delay = None,
			control_pars = (),
			verbose = True,
			module_location = None
		):
		
		super(jitcdde,self).__init__(n,verbose,module_location)
		
		self.f_sym = self._handle_input(f_sym)
		if not hasattr(self,"n_basic"):
			self.n_basic = self.n
		self.helpers = sort_helpers(sympify_helpers(helpers or []))
		self.control_pars = control_pars
		self.past = []
		self.integration_parameters_set = False
		self.DDE = None
		self.verbose = verbose
		self.delays = delays
		self.max_delay = max_delay
	
	@property
	def delays(self):
		if self._delays is None:
			self.delays = _get_delays(self.f_sym, self.helpers)
		return self._delays
	
	@delays.setter
	def delays(self, new_delays):
		self._delays = new_delays
		if (self._delays is not None) and (0 not in self._delays):
			self._delays.append(0)
	
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
		self.add_past_points([(time,state,derivative)])

	def add_past_points(self, anchors):
		"""
		adds multiple anchors from which the past of the DDE is interpolated.
		
		Parameters
		----------
		anchors : iterable of tuples
			Each tuple must have components corresponding to the arguments of `add_past_point`.
		"""
		self.reset_integrator()
		
		for time, state, derivative in anchors:
			state = np.array(state, copy=True, dtype=float)
			derivative = np.array(derivative, copy=True, dtype=float)
			assert state.shape == (self.n,), "State has wrong shape."
			assert derivative.shape == (self.n,), "Derivative has wrong shape."
			
			if time in [anchor[0] for anchor in self.past]:
				raise ValueError("There already is an anchor with that time.")
			
			self.past.append((time, state, derivative))
		
		self.past.sort(key = lambda anchor: anchor[0])
	
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
		
		if self.past:
			warn("You already added past points in some manner. This routine will ignore them, but not remove them. Be sure that you really want this.")
		
		self.add_past_points([
				( time-1., state, np.zeros_like(state) ),
				( time   , state, np.zeros_like(state) ),
			])
	
	def past_from_function(self,function,times_of_interest=None,max_anchors=100,tol=5):
		"""
		automatically determines anchors describing the past of the DDE from a given function, i.e., a piecewise cubic Hermite interpolation of the function at automatically selected time points will be the initial past. As this process involves heuristics, it is not perfect. For a better control of the initial conditions, use `add_past_point`.
		
		Parameters
		----------
		function : callable or iterable of symbolic expressions
			If callable, this takes the time as an argument and returns an iterable of floats that is the initial state of the past at that time.
			If an iterable of expressions, each expression represents how initial past of the respective component depends on `t`.
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
		
		assert tol>=0, "tol must be non-negative."
		assert max_anchors>0, "Maximum number of anchors must be positive."
		
		if self.past:
			warn("You already added past points in some manner. This routine will ignore them, but not remove them. Be sure that you really want this.")
		
		if times_of_interest is None:
			times_of_interest = np.linspace(-self.max_delay,0,10)
		else:
			times_of_interest = sorted(times_of_interest)
		
		if callable(function):
			array_function = lambda time: np.asarray(function(time))
			def get_anchor(time):
				value = array_function(time)
				eps = time*10**-tol or 10**-tol
				derivative = (array_function(time+eps)-value)/eps
				return [time,value,derivative,None]
		else:
			import sympy
			def get_anchor(time):
				evaluate = lambda expr: sympy.sympify(expr).evalf(tol,subs={t:time})
				value = np.fromiter(
					(evaluate(comp) for comp in function),
					dtype = float,
					)
				derivative = np.fromiter(
					(evaluate(comp.diff(t)) for comp in function),
					dtype = float,
					)
				return [time,value,derivative,None]
		
		anchors = [get_anchor(time) for time in times_of_interest]
		anchors[0][3] = anchors[-1][3] = True
		
		while not all(anchor[3] for anchor in anchors) and len(anchors)<=max_anchors:
			for i in range(len(anchors)-2,-1,-1):
				# Check whether anchors are already sufficiently interpolated by their neighbours or temporally close.
				if not anchors[i][3]:
					guess = python_core.interpolate_vec(anchors[i][0],(anchors[i-1],anchors[i+1]))
					anchors[i][3] = any((
						rel_dist(guess,anchors[i][1]) < 10**-tol,
						rel_dist(anchors[i+1][0],anchors[i-1][0]) < 10**-tol
						))
				
				# Add new anchors, if needed
				if not (anchors[i][3] and anchors[i+1][3]):
					time = np.mean((anchors[i][0],anchors[i+1][0]))
					anchors.insert(i+1,get_anchor(time))
				
				if len(anchors)>max_anchors:
					break
		
		self.add_past_points(anchor[:3] for anchor in anchors)
	
	def get_state(self):
		"""
		obtains a list of all anchors currently used by the integrator, which compeletely define the current state. The format is such that it can be used as an argument for `add_past_points`. An example where this is useful is when you want to switch between plain integration and one that also obtains Lyapunov exponents.

		The states and derivatives are just NumPy wrappers around the C arrays used by the integrator. Therefore changing their content affects the integrator and should not be done unless you do not want to continue using this integrator or know exactly what you’re doing.
		"""
		self.DDE.forget(self.max_delay)
		return self.DDE.get_full_state()

	def purge_past(self):
		"""
		Cleans the past and resets the integrator. You need to define a new past (using `add_past_point`) after this.
		"""
		
		self.past = []
		self.reset_integrator()
	
	def reset_integrator(self):
		"""
		Resets the integrator, forgetting all integration progress and forcing re-initiation when it is needed next.
		"""
		self.DDE = None
	
	def generate_lambdas(self):
		"""
		Explicitly initiates a purely Python-based integrator.
		"""
		
		assert len(self.past)>1, "You need to add at least two past points first. Usually this means that you did not set an initial past at all."
		
		self.DDE = python_core.dde_integrator(
			self.f_sym,
			self.past,
			self.helpers,
			self.control_pars,
			self.n_basic,
			self.tangent_indices if isinstance(self,jitcdde_transversal_lyap) else None
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
				if helper[1].__class__ == anchors:
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
			)
		
		self._compile_and_load(verbose,extra_compile_args,extra_link_args,omp)
	
	def _initiate(self):
		if self.compile_attempt is None:
			self._attempt_compilation()
		
		if self.DDE is None:
			assert len(self.past)>1, "You need to add at least two past points first. Usually this means that you did not set an initial past at all."
			
			if self.compile_attempt:
				self.DDE = self.jitced.dde_integrator(self.past)
			else:
				self.generate_lambdas()
		
		self._set_integration_parameters()
	
	def set_parameters(self, *parameters):
		"""
		Set the control parameters defined by the `control_pars` argument of the `jitcdde`. Note that you probably want to use `purge_past` and address initial discontinuities every time after you do this.

		Parameters
		----------
		parameters : floats
			Values of the control parameters. The order must be the same as in the `control_pars` argument of the `jitcdde`.
		"""
		
		self._initiate()
		self.DDE.set_parameters(*parameters)
	
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
			message = "\n".join(["",
					"Could not integrate with the given tolerance parameters:\n",
					"atol: %e" % self.atol,
					"rtol: %e" % self.rtol,
					"min_step: %e\n" % self.min_step,
					"The most likely reasons for this are:",
					"• You did not sufficiently address initial discontinuities. (If your dynamics is fast, did you adjust the maximum step?)",
					"• The DDE is ill-posed or stiff.",
				])
				
			if self.atol==0:
				message += "\n• You did not allow for an absolute error tolerance (atol) though your DDE calls for it. Even a very small absolute tolerance (1e-16) may sometimes help."
			raise UnsuccessfulIntegration(message)
	
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
		
		while self.DDE.get_t() < target_time:
			self.DDE.get_next_step(self.dt)
			
			if self.DDE.past_within_step:
				self.last_pws = self.DDE.past_within_step
				
				# If possible, adjust step size to make integration explicit:
				if self.dt > self.pws_factor*self.DDE.past_within_step:
					self.dt /= self.pws_factor
					self._control_for_min_step()
					continue
				
				# Try to come within an acceptable error within pws_max_iterations iterations; otherwise adjust step size:
				for self.count in range(1,self.pws_max_iterations+1):
					self.DDE.get_next_step(self.dt)
					if self.DDE.check_new_y_diff(self.pws_atol, self.pws_rtol):
						break
				else:
					self.dt /= self.pws_factor
					self._control_for_min_step()
					continue
			
			if self._adjust_step_size():
				self.DDE.accept_step()
		
		result = self.DDE.get_recent_state(target_time)
		self.DDE.forget(self.max_delay)
		return result
	
	def adjust_diff(self,shift_ratio=1e-4):
		"""
			Moves the last anchor by `shift_ratio` times the distance to the previous anchor into the past. Adds a new anchor in its place that has the same state and time but a slope computed using the derivative `f`.
			
			This may help with addressing initial discontinuities, but it usually doesn’t suffice – unless you have an ODE.
		"""
		self.DDE.adjust_diff(shift_ratio)
	
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
		Evolves the dynamics with a fixed step size ignoring any accuracy concerns. See `discontinuities` as to why you may want to use this. If a delay is smaller than the time step, the state is extrapolated from the previous step.
		
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
			self,
			propagations = 1,
			max_step = None,
			min_distance = 1e-5,
		):
		"""
		Assumes that the derivative is discontinuous at the start of the integration and chooses steps such that propagations of this point via the delays always fall on integration steps (or very close). If the discontinuity was propagated sufficiently often, it is considered to be smoothed and the integration is stopped. See `discontinuities` as to why you may want to use this.
		
		This only makes sense if you just defined the past (via `add_past_point`) and start integrating, just reset the integrator, or changed control parameters.
		
		In case of an ODE, `adjust_diff` is used automatically.
		
		Parameters
		----------
		propagations : integer
			how often the discontinuity has to propagate to before it’s considered smoothed
		
		max_step : float
			maximum step size. If `None`, `0`, or otherwise falsy, the `max_step` as set with `set_integration_parameters` is used.
		
		min_distance : float
			If two required steps are closer than this, they will be treated as one.
		
		Returns
		-------
		state : NumPy array
			the computed state of the system after integration
		"""
		
		assert min_distance > 0, "min_distance must be positive."
		assert isinstance(propagations,int), "Non-integer number of propagations."
		
		if not all(symengine.sympify(delay).is_number for delay in self.delays):
			raise ValueError("At least one delay depends on time or dynamics; cannot automatically determine steps.")
		self.delays = [
				# This conversion is due to SymEngine.py issue #227
				float(symengine.sympify(delay).n(real=True))
				for delay in self.delays
			]
		steps = _propagate_delays(self.delays, propagations, min_distance)
		steps.remove(0)
		steps.sort()
		
		if steps:
			start_time = self.t
			for step in steps:
				result = self.integrate_blindly(start_time+step, max_step)
			return result
		else:
			self._initiate()
			self.adjust_diff()
			return self.DDE.get_current_state()[:self.n_basic]

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
		
		if "helpers" not in kwargs.keys():
			kwargs["helpers"] = ()
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
	
	def add_past_points(self,anchors):
		def new_anchors():
			for time,state,derivative in anchors:
				new_state = [state]
				new_derivative = [derivative]
				for _ in range(self._n_lyap):
					new_state.append(random_direction(self.n_basic))
					new_derivative.append(random_direction(self.n_basic))
				yield time, np.hstack(new_state), np.hstack(new_derivative)
		
		super(jitcdde_lyap,self).add_past_points(new_anchors())
	
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
			
			if not "min_step" in kwargs.keys():
				kwargs["min_step"] = _default_min_step
			
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
	"""Calculates the largest Lyapunov exponent in orthogonal direction to a predefined plane, i.e. the projection of the separation function onto that plane vanishes.
	See `this test <https://github.com/neurophysik/jitcdde/blob/master/tests/test_restricted_lyap.py>`_ for an example of usage.
	Note that coordinate planes (i.e., planes orthogonal to vectors with only one non-zero component) are handled considerably faster. Consider transforming your differential equation to achieve this.

	The handling is the same as that for `jitcdde_lyap` except for:
	
	Parameters
	----------
	vectors : iterable of pairs of NumPy arrays
		A basis of the plane, whose projection shall be removed. The first vector in each pair is the component coresponding to the the state, the second vector corresponds to the derivative.
		
	"""
	
	def __init__(self, f_sym=(), vectors=(), **kwargs):
		self.n_basic = kwargs.pop("n",None)
		
		if "helpers" not in kwargs.keys():
			kwargs["helpers"] = ()
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
	
	def add_past_points(self, anchors):
		padding = np.empty(len(self.vectors)*2*self.n_basic)
		def new_anchors():
			for time,state,derivative in anchors:
				new_state = np.hstack([
						state,
						random_direction(self.n_basic),
						padding
					])
				new_derivative = np.hstack([
						derivative,
						random_direction(self.n_basic),
						padding
					])
				yield time,new_state,new_derivative
		
		super(jitcdde_restricted_lyap,self).add_past_points(new_anchors())
	
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
	
	def add_past_points(self, anchors):
		def new_anchors():
			for time,state,derivative in anchors:
				assert len(state)==len(derivative)==len(self.groups), "State and derivative too long or non matching. Provide only one value per synchronisation group"
				
				new_state = np.empty(self.n)
				new_state[self.main_indices] = state
				new_state[self.tangent_indices] = random_direction(len(self.tangent_indices))
				
				new_derivative = np.empty(self.n)
				new_derivative[self.main_indices] = derivative
				new_derivative[self.tangent_indices] = random_direction(len(self.tangent_indices))
				
				yield time,new_state,new_derivative
		
		super(jitcdde_transversal_lyap,self).add_past_points(new_anchors())
	
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

