#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from jitcdde.past import Past
from chspy import interpolate, interpolate_diff, extrema_from_anchors, CubicHermiteSpline

NORM_THRESHOLD = 1e-30

class dde_integrator(Past):
	"""
	Class for the Shampine–Thompson integrator.
	"""
	def __init__(self,
				f,
				past,
				helpers = (),
				control_pars = (),
				simplify = True,
			):
		assert isinstance(past,CubicHermiteSpline)
		super().__init__(anchors=past)
		self.t, self.y, self.diff = self[-1]
		self.old_new_y = None
		
		self.parameters = []
		
		from jitcdde._jitcdde import t, y, past_y, past_dy, anchors
		from sympy import DeferredVector, sympify, lambdify
		Y = DeferredVector("Y")
		substitutions = list(reversed(helpers)) + [(y(i),Y[i]) for i in range(self.n)]
		
		past_calls = 0
		f_wc = []
		for entry in f():
			new_entry = sympify(entry).subs(substitutions)
			if simplify:
				new_entry.simplify(ratio=1.0)
			past_calls += new_entry.count(anchors)
			f_wc.append(new_entry)
		
		F = lambdify(
				[t, Y] + list(control_pars),
				f_wc,
				[
					{
						anchors.name: self.get_anchors_with_mem,
						past_y .name: interpolate,
						past_dy.name: interpolate_diff,
					},
					"math"
				]
			)
		
		self.f = lambda *args: np.array(F(*args)).flatten()
		
		# storage of search positions (cursors) for lookup of anchors corresponding to a given time (see `get_anchors_with_mem`)
		# Note that this approach is tailored to mimic the C implementation (using linked lists) as good as possible and therefore is rather clunky in Python.
		self.anchor_mem = (len(past)-1)*np.ones(past_calls, dtype=int)
	
	@property
	def t(self):
		return self._t
	
	@t.setter
	def t(self,value):
		self._t = value
	
	def set_parameters(self, *parameters):
		self.parameters = parameters
	
	def get_t(self):
		return self.t
	
	def get_anchors_with_mem(self, t):
		"""
		Find the two anchors neighbouring `t` using the anchor memory.
		If `t` is outside the ranges of times covered by the anchors, return the two nearest anchors.
		"""
		s = self.anchor_mem[self.anchor_mem_index]
		
		if t > self.t:
			s = len(self)-2
			self.past_within_step = max(self.past_within_step,t-self.t)
		else:
			while self[s].time>=t and s>0:
				s -= 1
			while self[s+1].time<t:
				s += 1
		
		self.anchor_mem[self.anchor_mem_index] = s
		self.anchor_mem_index += 1
		
		return (self[s], self[s+1])
	
	def eval_f(self, t, y):
		self.anchor_mem_index = 0
		return self.f(t, y, *self.parameters)
	
	def get_next_step(self, delta_t):
		"""
		performs an integration step.
		"""
		self.past_within_step = False
		
		try:
			k_1 = self.diff
			k_2 = self.eval_f(self.t + 0.5 *delta_t, self.y + 0.5 *delta_t*k_1)
			k_3 = self.eval_f(self.t + 0.75*delta_t, self.y + 0.75*delta_t*k_2)
			new_y = self.y + (delta_t/9.) * (2*k_1 + 3*k_2 + 4*k_3)
			new_t = self.t + delta_t
			k_4 = new_diff = self.eval_f(new_t, new_y)
			self.error = delta_t * (5*k_1 - 6*k_2 - 8*k_3 + 9*k_4) * (1/72.)
		except ValueError:
			# give some tentative result and force a step-size adaption
			new_y = self.y + self.diff*delta_t
			new_diff = self.diff
			new_t = self.t + delta_t
			self.error = np.array(self.n*[np.inf])
		
		if self[-1].time==self.t:
			self.append((new_t, new_y, new_diff))
		else:
			try:
				self.old_new_y = self[-1].state
			except AttributeError:
				pass
			
			self[-1] = (new_t, new_y, new_diff)
	
	def get_p(self, atol, rtol):
		"""
			computes the coefficient that summarises the integration error.
		"""
		with np.errstate(divide='ignore', invalid='ignore'):
			return np.nanmax(
					np.abs(self.error)
					/(atol + rtol*np.abs(self[-1].state))
				)
	
	def check_new_y_diff(self, atol, rtol):
		"""
			For past-within-step iterations: Checks whether the difference between the new and old approximation of the next step is below the given tolerance level.
		"""
		if self.old_new_y is not None:
			difference = np.abs(self[-1].state-self.old_new_y)
			tolerance = atol + np.abs(rtol*self[-1].state)
			return np.all(difference <= tolerance)
		else:
			return False
	
	def accept_step(self):
		self.t, self.y, self.diff = self[-1]
		self.old_new_y = None
	
	def forget(self, delay):
		"""
		Remove past points that are “out of reach” of the delay.
		"""
		super().forget(delay)
		self.anchor_mem = np.minimum(self.anchor_mem,len(self)-1)
	
	def truncate(self,time):
		super().truncate(time)
		self.anchor_mem = np.minimum(self.anchor_mem,len(self)-1)
		self.accept_step()
	
	def extrema_in_last_step(self):
		extrema =  extrema_from_anchors(self[-2:])
		return extrema.minima, extrema.maxima
	
	def apply_jump( self, change, time, width=1e-5 ):
		new_time = time+width
		new_value = self.get_state(new_time) + change
		new_diff = self.eval_f(new_time,new_value)
		
		self.truncate(time)
		self.append((new_time,new_value,new_diff))
		self.accept_step()
		return self.extrema_in_last_step()

