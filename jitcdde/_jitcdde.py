#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division

from inspect import isgeneratorfunction
from warnings import warn
from itertools import chain, count
import jitcdde._python_core as python_core
import sympy
import numpy as np
from os import path as path
from setuptools import setup, Extension
from sys import version_info, modules
from tempfile import mkdtemp
from jinja2 import Environment, FileSystemLoader
from jitcdde._helpers import (
	ensure_suffix, count_up,
	get_module_path, modulename_from_path, find_and_load_module, module_from_path,
	render_and_write_code,
	render_template,
	collect_arguments,
	random_direction
	)
from numbers import Number

_default_min_step = 1e-10

#sigmoid = lambda x: 1/(1+np.exp(-x))
#sigmoid = lambda x: 1 if x>0 else 0
sigmoid = lambda x: (np.tanh(x)+1)/2

def provide_basic_symbols():
	"""
	provides the basic symbols that must be used to define the differential equation.
	
	Returns
	-------
	t : SymPy symbol
		represents time
	y : SymPy function
		represents the DDE’s state, with the first integer argument denoting the component. The second, optional argument is a Sympy expression denoting the time. This automatically expands, so do not be surprised when you are looking at the output for some reason and it looks different than what you entered or expected (see `provide_advanced_symbols` for more details).
	"""
	
	return provide_advanced_symbols()[:2]

def provide_advanced_symbols():
	"""
	provides all symbols that you may want to use to to define the differential equation.
	
	You may just as well define the respective symbols and functions directly with SymPy, but using this function is the best way to get the most of future versions of JiTCODE, in particular avoiding incompatibilities. If you wish to use other symbols for the dynamical variables, you can use `convert_to_required_symbols` for conversion.
	
	Returns
	-------
	t : SymPy symbol
		represents time
	y : SymPy function
		same as for `provide_basic_symbols`.
	current_y : SymPy function
		represents the DDE’s current state, with the integer argument denoting the component
	past_y : SymPy function
		represents the DDE’s past state, with the integer argument denoting the component and the second argument being a pair of past points (anchors) from which the past state is interpolated (or, in rare cases, extrapolated).
	anchors : SymPy function
		represents the pair of anchors pertaining to a specific time point with the symbolic argument denoting that time point.
	"""
	
	t = sympy.Symbol("t", real=True)
	current_y = sympy.Function("current_y")
	anchors = sympy.Function("anchors")
	past_y = sympy.Function("past_y")
	
	class y(sympy.Function):
		@classmethod
		def eval(cls, index, time=t):
			if time == t:
				return current_y(index)
			else:
				return past_y(time, index, anchors(time))
	
	return t, y, current_y, past_y, anchors

def _handle_input(f_sym,n):
	if isgeneratorfunction(f_sym):
		n = n or sum(1 for _ in f_sym())
		return ( f_sym, n )
	else:
		len_f = len(f_sym)
		if (n is not None) and(len_f != n):
			raise ValueError("len(f_sym) and n do not match.")
		return (lambda: (entry.doit() for entry in f_sym), len_f)

def _depends_on_any(helper, other_helpers):
	for other_helper in other_helpers:
		if helper[1].has(other_helper[0]):
			return True
	return False

def _sort_helpers(helpers):
	if len(helpers)>1:
		for j,helper in enumerate(helpers):
			if not _depends_on_any(helper, helpers):
				helpers.insert(0,helpers.pop(j))
				break
		else:
			raise ValueError("Helpers have cyclic dependencies.")
		
		helpers[1:] = _sort_helpers(helpers[1:])
	
	return helpers

def _sympify_helpers(helpers):
	return [(helper[0], sympy.sympify(helper[1]).doit()) for helper in helpers]

def _delays(f, helpers=[]):
	t, _, _, _, anchors = provide_advanced_symbols()
	delay_terms = set().union(*(collect_arguments(entry, anchors) for entry in f()))
	delay_terms.update(*(collect_arguments(helper[1], anchors) for helper in helpers))
	
	return [0]+list(map(lambda delay_term: t-delay_term[0], delay_terms))

def _find_max_delay(delays):
	if all(sympy.sympify(delay).is_Number for delay in delays):
		return float(max(delays))
	else:
		raise ValueError("Delay depends on time or dynamics; cannot determine max_delay automatically. You have to pass it as an argument to jitcdde.")

class UnsuccessfulIntegration(Exception):
	"""
		This exception is raised when the integrator cannot meet the accuracy and step-size requirements and the argument `raise_exception` of `set_integration_parameters` is set.
	"""
	
	pass

#: A list with the default extra compile arguments. Use and modify these to get the most of future versions of JiTCODE. Note that without `-Ofast`, `-ffast-math`, or `-funsafe-math-optimizations` (if supported by your compiler), you may experience a considerable speed loss since SymPy uses the `pow` function for small integer powers (`SymPy Issue 8997`_).
DEFAULT_COMPILE_ARGS = [
			"-std=c11",
			"-Ofast","-g0",
			#"-O0","-g","-UNDEBUG",
			"-march=native",
			"-mtune=native",
			"-Wno-unknown-pragmas",
			]

class jitcdde(object):
	"""
	Parameters
	----------
	f_sym : iterable of SymPy expressions or generator function yielding SymPy expressions
		The `i`-th element is the `i`-th component of the value of the DDE’s derivative :math:`f(t,y)`.
	
	helpers : list of length-two iterables, each containing a SymPy symbol and a SymPy expression
		Each helper is a variable that will be calculated before evaluating the derivative and can be used in the latter’s computation. The first component of the tuple is the helper’s symbol as referenced in the derivative or other helpers, the second component describes how to compute it from `t`, `y` and other helpers. This is for example useful to realise a mean-field coupling, where the helper could look like `(mean, sympy.Sum(y(i),(i,0,99))/100)`. (See `example_2` for an example.)
	
	n : integer
		Length of `f_sym`. While JiTCDDE can easily determine this itself (and will, if necessary), this may take some time if `f_sym` is a generator function and `n` is large. Take care that this value is correct – if it isn’t, you will not get a helpful error message.
	
	max_delay : number
		Maximum delay. In case of constant delays and if not given, JiTCDDE will determine this itself. However, this may take some time if `f_sym` is large. Take care that this value is correct – if it isn’t, you will not get a helpful error message.
	"""
	
	def __init__(self, f_sym, helpers=None, n=None, max_delay=None):
		self.f_sym, self.n = _handle_input(f_sym,n)
		self.n_basic = self.n
		self.helpers = _sort_helpers(_sympify_helpers(helpers or []))
		self._tmpdir = None
		self._modulename = "jitced"
		self.past = []
		self.max_delay = max_delay or _find_max_delay(_delays(self.f_sym, self.helpers))
		assert self.max_delay >= 0.0, "Negative maximum delay."
	
	def _tmpfile(self, filename=None):
		if self._tmpdir is None:
			self._tmpdir = mkdtemp()
		
		if filename is None:
			return self._tmpdir
		else:
			return path.join(self._tmpdir, filename)
	
	def add_past_point(self, time, state, derivative):
		"""
		adds an anchor point from which the past of the DDE is interpolated.
		
		Parameters
		----------
		time : float
			the time of the anchor point. Must be later than the time of all previously added points.
		state : NumPy array of floats
			the position of the anchor point. The dimension of the array must match the dimension of the differential equation.
		derivative : NumPy array of floats
			the derivative at the anchor point. The dimension of the array must match the dimension of the differential equation.
		"""
		
		assert state.shape == (self.n,), "State has wrong shape."
		assert derivative.shape == (self.n,), "Derivative has wrong shape."
		
		self.past.append((time, state, derivative))
	
	def generate_f_lambda(self):
		"""
			Prepares a purely Python-based integrator.
		"""
		
		self.DDE = python_core.dde_integrator(self.f_sym, self.past, self.helpers)
	
	def generate_f_c(self, *args, **kwargs):
		raise DeprecationWarning("You are very likely seeing this message because you ignored a warning. You should not do this. Warnings exist for a reason. Well, now it’s an exception. Use generate_f_C instead of generate_f_c.")
	
	def generate_f_C(
		self,
		simplify=True,
		do_cse=True,
		chunk_size=100,
		modulename=None,
		verbose=False,
		extra_compile_args=DEFAULT_COMPILE_ARGS
		):
		"""
		translates the derivative to C code using SymPy’s `C-code printer <http://docs.sympy.org/dev/modules/printing.html#module-sympy.printing.ccode>`_.
		
		Parameters
		----------
		simplify : boolean
			Whether the derivative should be `simplified <http://docs.sympy.org/dev/modules/simplify/simplify.html>`_ (with `ratio=1.0`) before translating to C code. The main reason why you could want to disable this is if your derivative is already  optimised and so large that simplifying takes a considerable amount of time.
		
		do_cse : boolean
			Whether SymPy’s `common-subexpression detection <http://docs.sympy.org/dev/modules/rewriting.html#module-sympy.simplify.cse_main>`_ should be applied before translating to C code.
			This is worthwile if your DDE contains the same delay more than once. Otherwise it is almost always better to let the compiler do this (unless you want to set the compiler optimisation to `-O2` or lower). As this requires all entries of `f` at once, it may void advantages gained from using generator functions as an input.
		
		chunk_size : integer
			If the number of instructions in the final C code exceeds this number, it will be split into chunks of this size. After the generation of each chunk, SymPy’s cache is cleared. See `large_systems` on why this is useful.
			
			If there is an obvious grouping of your :math:`f`, the group size suggests itself for `chunk_size`. For example, if you want to simulate the dynamics of three-dimensional oscillators coupled onto a 40×40 lattice and if the differential equations are grouped first by oscillator and then by lattice row, a chunk size of 120 suggests itself.
			
			If smaller than 1, no chunking will happen.
		
		extra_compile_args : list of strings
			Arguments to be handed to the C compiler on top of what Setuptools chooses. In most situations, it’s best not to write your own list, but modify `DEFAULT_COMPILE_ARGS`, e.g., like this: `compile_C(extra_compile_args = DEFAULT_COMPILE_ARGS + ["--my-flag"])`. However, if your compiler cannot handle one of the DEFAULT_COMPILE_ARGS, you best write your own arguments.

		verbose : boolean
			Whether the compiler commands shall be shown. This is the same as Setuptools’ `verbose` setting.

		modulename : string or `None`
			The name used for the compiled module. If `None` or empty, the filename will be chosen by JiTCDDE based on previously used filenames or default to `jitced.so`. The only reason why you may want to change this is if you want to save the module file for later use (with`save_compiled`). It is not possible to re-use a modulename for a given instance of Python (due to the limitations of Python’s import machinery).
		"""
		
		assert len(self.past)>1, "You need to add the past first."
		
		t, y, current_y, past_y, anchors = provide_advanced_symbols()
		
		f_sym_wc = self.f_sym()
		helpers_wc = self.helpers
		
		if simplify:
			f_sym_wc = (entry.simplify(ratio=1.0) for entry in f_sym_wc)
		
		if do_cse:
			additional_helper = sympy.Function("additional_helper")
			
			_cse = sympy.cse(
					sympy.Matrix(list(f_sym_wc)),
					symbols = (additional_helper(i) for i in count())
				)
			helpers_wc.extend(_cse[0])
			f_sym_wc = _cse[1][0]
		
		if modulename:
			warn("Setting the module name works, but saving and loading are not implemented yet. Your file will be located in %s." % self._tmpfile())
		
		arguments = [
			("self", "dde_integrator * const"),
			("t", "double const"),
			("y", "double", self.n),
			]
		functions = ["current_y","past_y","anchors"]
		helper_i = 0
		anchor_i = 0
		self.substitutions = []
		converted_helpers = []
		self.past_calls = 0
		
		def finalise(expression):
			expression = expression.subs(self.substitutions)
			self.past_calls += expression.count(anchors)
			return expression
		
		if helpers_wc:
			get_helper = sympy.Function("get_f_helper")
			set_helper = sympy.Function("set_f_helper")
			
			get_anchor = sympy.Function("get_f_anchor_helper")
			set_anchor = sympy.Function("set_f_anchor_helper")
			
			for helper in helpers_wc:
				if helper[1].__class__ == anchors:
					converted_helpers.append(set_anchor(anchor_i, finalise(helper[1])))
					self.substitutions.append((helper[0], get_anchor(anchor_i)))
					anchor_i += 1
				else:
					converted_helpers.append(set_helper(helper_i, finalise(helper[1])))
					self.substitutions.append((helper[0], get_helper(helper_i)))
					helper_i += 1
			
			if helper_i:
				arguments.append(("f_helper","double", helper_i))
				functions.extend(["get_f_helper", "set_f_helper"])
			if anchor_i:
				arguments.append(("f_anchor_helper","anchor", anchor_i))
				functions.extend(["get_f_anchor_helper", "set_f_anchor_helper"])
			
			render_and_write_code(
				converted_helpers,
				self._tmpfile,
				"helpers",
				functions,
				chunk_size = chunk_size,
				arguments = arguments
				)
		
		set_dy = sympy.Function("set_dy")
		render_and_write_code(
			(set_dy(i,finalise(entry)) for i,entry in enumerate(f_sym_wc)),
			self._tmpfile,
			"f",
			functions = functions+["set_dy"],
			chunk_size = chunk_size,
			arguments = arguments + [("dY", "double", self.n)]
			)
		
		if modulename:
			if modulename in modules.keys():
				raise NameError("Module name has already been used in this instance of Python.")
			self._modulename = modulename
		else:
			while self._modulename in modules.keys():
				self._modulename = count_up(self._modulename)
		
		sourcefile = self._tmpfile(self._modulename + ".c")
		modulefile = self._tmpfile(self._modulename + ".so")
		
		if path.isfile(modulefile):
			raise OSError("Module file already exists.")
		
		if not self.past_calls:
			warn("Differential equation does not inclued a delay term.")
		
		render_template(
			"jitced_template.c",
			sourcefile,
			n = self.n,
			module_name = self._modulename,
			Python_version = version_info[0],
			number_of_helpers = helper_i,
			number_of_anchor_helpers = anchor_i,
			has_any_helpers = anchor_i or helper_i,
			anchor_mem_length = self.past_calls,
			n_basic = self.n_basic
			)
		
		setup(
			name = self._modulename,
			ext_modules = [Extension(
				self._modulename,
				sources = [sourcefile],
				extra_compile_args = ["-lm", "-I" + np.get_include()] + extra_compile_args
				)],
			script_args = [
				"build_ext",
				"--build-lib", self._tmpfile(),
				"--build-temp", self._tmpfile(),
				"--force",
				#"clean" #, "--all"
				],
			verbose = verbose
			)
		
		jitced = find_and_load_module(self._modulename,self._tmpfile())
		self.DDE = jitced.dde_integrator(self.past)
	
	def set_integration_parameters(self,
			atol = 0.0,
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
			raise_exception = True,
			):
		
		"""
		Sets the parameters for the step-size adaption. Arguments starting with `pws` (past within step) are only relevant if the delay is shorter than the step size.
		
		Parameters
		----------
		atol : float
		rtol : float
			The tolerance of the estimated integration error is determined as :math:`\texttt{atol} + \texttt{rtol}·|y|`. The step-size adaption algorithm is the same as for the GSL. For details see its documentation (TODO: link).
		
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
			Factor of step-size adaptions due to a delay shorter than the time step. If dividing the step size by `pws_factor` moves the delay out of the time step, it is done. If this is not possible and the iterative algorithm does not converge within `pws_max_iterations` or converges within fewer iterations than `pws_factor`, the step size is decreased or increased, respectively, by this factor
		
		pws_atol : float
		pws_rtol : float
			If the difference between two successive iterations is below the tolerance determined with these factors, the iterations are considered to have converged.
		
		pws_max_iterations : integer
			The maximum number of iterations before the step size is decreased.
		
		pws_base_increase_chance : float
			If the normal step-size adaption calls for an increase and the step size was adapted due to the past lying within the step, there is at least this chance that the step size is increased.
			
		pws_fuzzy_increase : boolean
			Whether the decision to try to increase the step size shall depend on chance. The upside of this is that it is less likely that the step size gets locked at a unnecessarily low value. The downside is that the integration is not deterministic anymore. If False, increase probabilities will be added up until they exceed 1, in which case an increase happens.
		
		raise_exception : boolean,
			Whether (`UnsuccessfulIntegration`) shall be raised if the integration fails. You can deal with this by catching this exception. If `False`, there is only a warning and `self.successful` is set to `False`.
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
		self.do_raise_exception = raise_exception
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
				if self.increase_credit >= 0.98:
					self.increase_credit = 0.0
					return True
				else:
					return False
			self.do_increase = do_increase
	
	def _control_for_min_step(self):
		if self.dt < self.min_step:
			raise UnsuccessfulIntegration("Step size under min_step (%e)." % self.min_step)
	
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
		else:
			self.successful = True
			self.DDE.accept_step()
			if p < self.increase_threshold:
				factor = self.safety_factor*p**(-1/(self.q+1)) if p else self.max_factor
				
				new_dt = min(
					self.dt*min(factor, self.max_factor),
					self.max_step
					)
				
				if (not self.last_pws) or self.do_increase(self._increase_chance(new_dt)):
					self.dt = new_dt
					self.count = 0
					self.last_pws = False
	
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
			the computed state of the system at `target_time`. If the integration fails and `raise_exception` is `True`, an array of NaNs is returned.
		"""
		
		try:
			while self.DDE.get_t() < target_time:
				self.successful = False
				while not self.successful:
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
					
					self._adjust_step_size()
		
		except UnsuccessfulIntegration as error:
			self.successful = False
			if self.do_raise_exception:
				raise error
			else:
				warn(str(error))
				return np.nan*np.ones(self.n)
		
		else:
			result = self.DDE.get_recent_state(target_time)
			self.DDE.forget(self.max_delay)
			return result

	def integrate_blindly(self, target_time, step=0.1):
		"""
		Evolves the dynamics with a fixed step size ignoring any accuracy concerns. If a delay is smaller than the time step, the state is extrapolated from the previous step.
		
		For many systems, arbitrary initial conditions inevitably lead to high integration errors due to the disagreement of state and derivative. Evolving the system for a while should solves this issue.
		
		Parameters
		----------
		target_time : float
			time until which the dynamics is evolved
		
		step : float
			aspired step size. The actual step size may be slightly adapted to make it divide the integration time.
		
		Returns
		-------
		state : NumPy array
			the computed state of the system at `target_time`.
		"""
		
		total_integration_time = target_time-self.DDE.get_t()
		number = int(round(total_integration_time/step))
		dt = total_integration_time/number
		
		assert(number*dt == total_integration_time)
		for _ in range(number):
			self.DDE.get_next_step(dt)
			self.DDE.accept_step()
			self.DDE.forget(self.max_delay)
		
		return self.DDE.get_current_state()


def _jac(f, helpers, delay, n):
	t,y = provide_basic_symbols()
	
	dependent_helpers = [[] for i in range(n)]
	for i in range(n):
		for helper in helpers:
			derivative = sympy.diff(helper[1], y(i,t-delay))
			for other_helper in dependent_helpers[i]:
				derivative += sympy.diff(helper[1],other_helper[0]) * other_helper[1]
			if derivative:
				dependent_helpers[i].append( (helper[0], derivative) )
	
	def line(f_entry):
		for j in range(n):
			entry = sympy.diff( f_entry, y(j,t-delay) )
			for helper in dependent_helpers[j]:
				entry += sympy.diff(f_entry,helper[0]) * helper[1]
			yield entry
	
	for f_entry in f():
		yield line(f_entry)


class jitcdde_lyap(jitcdde):
	"""the handling is the same as that for `jitcdde` except for:
	
	Parameters
	----------
	n_lyap : integer
		Number of Lyapunov exponents to calculate.
		
	delays : iterable of SymPy expressions
		The delays of the dynamics. If not given, JiTCDDE will determine these itself. However, this may take some time if `f_sym` is large. Take care that these are correct – if they aren’t, you won’t get a helpful error message.
	"""
	
	def __init__(self, f_sym, helpers=[], n=None, max_delay=None, n_lyap=1, delays=None):
		f_basic, n = _handle_input(f_sym,n)
		
		if delays:
			act_delays = delays + ([] if (0 in delays) else [0])
		else:
			act_delays = _delays(f_basic, helpers)
		max_delay = max_delay or _find_max_delay(act_delays)
		
		assert n_lyap>=0, "n_lyap negative"
		self._n_lyap = n_lyap
		
		helpers = _sort_helpers(_sympify_helpers(helpers or []))
		
		t,y = provide_basic_symbols()
		
		def f_lyap():
			#Replace with yield from, once Python 2 is dead:
			for entry in f_basic():
				yield entry
			
			for i in range(self._n_lyap):
				jacs = [_jac(f_basic, helpers, delay, n) for delay in act_delays]
				
				for _ in range(n):
					expression = 0
					for delay,jac in zip(act_delays,jacs):
						for k,entry in enumerate(next(jac)):
							expression += entry * y(k+(i+1)*n, t-delay)
					
					yield sympy.simplify(expression, ratio=1.0)
		
		super(jitcdde_lyap, self).__init__(
			f_lyap,
			helpers = helpers,
			n = n*(self._n_lyap+1),
			max_delay = max_delay
			)
		
		self.n_basic = n
	
	def add_past_point(self, time, state, derivative):
		new_state = [state]
		new_derivative = [derivative]
		for _ in range(self._n_lyap):
			new_state.append(random_direction(self.n_basic))
			new_derivative.append(random_direction(self.n_basic))
		
		super(jitcdde_lyap, self).add_past_point(time, np.hstack(new_state), np.hstack(new_derivative))
	
	
	def integrate(self, target_time):
		"""
		Like `jitcdde`’s `integrate`, except for orthonormalising the separation functions and:
		
		Returns
		-------
		y : one-dimensional NumPy array
			The first `len(f_sym)` entries are the state of the system.
			The remaining entries are the “local” Lyapunov exponents as estimated from the growth or shrinking of the tangent vectors during the integration time of this very `integrate` command.
		"""
		# TODO formula and citation like for JiTCODE?
		
		old_t = self.DDE.get_t()
		result = super(jitcdde_lyap, self).integrate(target_time)[:self.n_basic]
		delta_t = self.DDE.get_t()-old_t
		
		norms = self.DDE.orthonormalise(self._n_lyap, self.max_delay)
		
		lyaps = np.log(norms) / delta_t
		
		return np.hstack((result, lyaps))
	
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
	
	def integrate_blindly(self, target_time, step=0.1):
		"""
		Like `jitcdde`’s `integrate_blindly`, except for orthonormalising the separation functions after each step and an output is analogous to `jitcdde_lyap`’s `integrate`.
		"""
		
		total_integration_time = target_time-self.DDE.get_t()
		number = int(round(total_integration_time/step))
		dt = total_integration_time/number
		assert(number*dt == total_integration_time)
		
		instantaneous_lyaps = []
		
		for _ in range(number):
			self.DDE.get_next_step(dt)
			self.DDE.accept_step()
			self.DDE.forget(self.max_delay)
			norms = self.DDE.orthonormalise(self._n_lyap, self.max_delay)
			instantaneous_lyaps.append(np.log(norms)/dt)
		
		lyaps = np.average(instantaneous_lyaps, axis=0)
		
		return np.hstack((self.DDE.get_current_state()[:self.n_basic], lyaps))
