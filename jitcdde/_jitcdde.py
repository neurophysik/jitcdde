#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division

from inspect import isgeneratorfunction
from warnings import warn
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
	collect_arguments
	)

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

def _delays(delay_terms):
	t, _, _, _, _ = provide_advanced_symbols()
	for delay_term in delay_terms:
		delay = t - delay_term[0]
		if not delay.is_Number:
			raise ValueError("Delay depends on time or dynamics; cannot determine max_delay automatically. You have to pass it as an argument to jitcdde.")
		yield float(delay)

def _find_max_delay(f, helpers=[]):
	_, _, _, _, anchors = provide_advanced_symbols()
	delay_terms = []
	for entry in f():
		delay_terms.extend(collect_arguments(entry, anchors))
	for helper in helpers:
		delay_terms.extend(collect_arguments(helper[1], anchors))
	return max(_delays(delay_terms))

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
		self.helpers = _sort_helpers(_sympify_helpers(helpers or []))
		self._y = []
		self._tmpdir = None
		self._modulename = "jitced"
		self.past = []
		self.max_delay = max_delay or _find_max_delay(self.f_sym, self.helpers)
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
	
	def generate_f_c(
		self,
		chunk_size=100,
		verbose=False,
		modulename=None,
		extra_compile_args=DEFAULT_COMPILE_ARGS
		):
		"""
			Generates the source for the C-based integrator, compiles, and loads it.
		"""
		
		t, y, current_y, past_y, anchors = provide_advanced_symbols()
		
		if self.helpers:
			raise NotImplementedError("Helpers for C are not implemented yet, but they will be soon.")
		
		set_dy = sympy.Function("set_dy")
		render_and_write_code(
			(set_dy(i,entry) for i,entry in enumerate(self.f_sym())),
			self._tmpfile,
			"f",
			["set_dy","current_y","past_y","anchors"],
			chunk_size = chunk_size,
			arguments = [
				("self", "dde_integrator * const"),
				("t", "double const"),
				("y", "double", self.n),
				("dY", "double", self.n)
				]
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
		
		render_template(
			"jitced_template.c",
			sourcefile,
			n = self.n,
			module_name = self._modulename,
			Python_version = version_info[0],
			#has_helpers = bool(helpers),
			)
		
		setup(
			name = self._modulename,
			ext_modules = [Extension(
				self._modulename,
				sources = [sourcefile],
				extra_compile_args = extra_compile_args
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
		
		past_calls = sum(entry.count(anchors) for entry in self.f_sym())
		assert past_calls>0, "No DDE."
		
		jitced = find_and_load_module(self._modulename,self._tmpfile())
		self.DDE = jitced.dde_integrator(self.past, past_calls)
	
	def set_integration_parameters(self,
			atol = 0.0,
			rtol = 1e-5,
			first_step = 1.0,
			min_step = 1e-10,
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
		
		assert min_step <= first_step <= max_step, "Bogus step parameters."
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
				new_dt = min(
					self.dt*min(self.safety_factor*p**(-1/(self.q+1)), self.max_factor),
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
		"""
		
		total_integration_time = target_time-self.DDE.get_t()
		number = int(round(total_integration_time/step))
		dt = total_integration_time/number
		
		assert(number*dt == total_integration_time)
		for _ in range(number):
			self.DDE.get_next_step(dt)
			self.DDE.accept_step()
			self.DDE.forget(self.max_delay)
