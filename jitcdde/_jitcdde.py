#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division

from inspect import isgeneratorfunction
from warnings import warn
import jitcdde._python_core as python_core
import sympy
import numpy as np

#sigmoid = lambda x: 1/(1+np.exp(-x))
#sigmoid = lambda x: 1 if x>0 else 0
sigmoid = lambda x: (np.tanh(x)+1)/2

def provide_advanced_symbols():
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

def provide_basic_symbols():
	return provide_advanced_symbols()[:2]

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

class UnsuccessfulIntegration(Exception):
	pass

class jitcdde():
	def __init__(self, f_sym, helpers=None, n=None):
		self.f_sym, self.n = _handle_input(f_sym,n)
		self.f = None
		self.helpers = _sort_helpers(_sympify_helpers(helpers or []))
		self._y = []
		self._tmpdir = None
		self._modulename = "jitced"
		self.past = []
		
	def add_past_point(self, time, state, diff):
		self.past.append((time, state, diff))
	
	def generate_f_lambda(self):
		self.DDE = python_core.dde_integrator(self.f_sym(), self.past, self.helpers)
	
	def set_integration_parameters(self,
			max_delay,
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
			pws_atol = 0.0,
			pws_rtol = 1e-5,
			pws_max_iterations = 10,
			pws_factor = 3,
			pws_base_increase_chance = 0.1,
			pws_fuzzy_increase = False,
			raise_exception = False,
			):
		
		"""
		TODO: component-wise cool shit
		"""
		
		assert max_delay >= 0.0, "Negative maximum delay."
		assert min_step <= first_step <= max_step, "Bogus step parameters."
		assert decrease_threshold>=1.0, "decrease_threshold smaller than 1"
		assert increase_threshold<=1.0, "increase_threshold larger than 1"
		assert max_factor>=1.0, "max_factor smaller than 1"
		assert min_factor<=1.0, "min_factor larger than 1"
		assert safety_factor<=1.0, "safety_factor larger than 1"
		assert np.all(atol>=0.0), "negative atol"
		assert np.all(rtol>=0.0), "negative rtol"
		if atol==0 and rtol==0:
			warn("atol and rtol are both 0. You probably do not want this.")
		assert np.all(pws_atol>=0.0), "negative pws_atol"
		assert np.all(pws_rtol>=0.0), "negative pws_rtol"
		assert 0<pws_max_iterations, "non-positive pws_max_iterations"
		assert 2<=pws_factor, "pws_factor smaller than 2"
		assert pws_base_increase_chance>=0, "negative pws_base_increase_chance"
		
		self.max_delay = max_delay
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
		p = np.max(np.abs(self.DDE.error)/(self.atol + self.rtol*np.abs(self.DDE.past[-1][1])))
		
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
		try:
			while self.DDE.t < target_time:
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
							old_new_y = self.DDE.past[-1][1]
							self.DDE.get_next_step(self.dt)
							new_y = self.DDE.past[-1][1]
							difference = np.abs(new_y-old_new_y)
							tolerance = self.pws_atol + np.abs(self.pws_rtol*new_y)
							if np.all(difference <= tolerance):
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
			result = self.DDE.get_past_state(
				target_time,
				(self.DDE.past[-2], self.DDE.past[-1])
				)
			self.DDE.forget(self.max_delay)
			return result

	def integrate_blindly(self, target_time, step=0.1):
		total_integration_time = target_time-self.DDE.t
		number = int(round(total_integration_time/step))
		dt = total_integration_time/number
		
		assert(number*dt == total_integration_time)
		for _ in range(number):
			self.DDE.get_next_step(dt)
			self.DDE.accept_step()
			self.DDE.forget(self.max_delay)
