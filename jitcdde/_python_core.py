#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import jitcdde._jitcdde

import sympy
import numpy as np
from itertools import count

MIN_GARBAGE = 100

class dde_integrator(object):
	def __init__(self,
				f,
				past,
				helpers = []
			):
		self.past = past
		self.t, self.y, self.diff = self.past[-1]
		self.n = len(self.y)
		self.step_count = 0 # For benchmarking purposes.
		self.last_garbage = -1
		
		t, y, current_y, past_y, anchors = jitcdde._jitcdde.provide_advanced_symbols()
		Y = sympy.symarray("Y", self.n)
		substitutions = helpers[::-1] + [(y(i),Y[i]) for i in range(self.n)]
		
		past_calls = 0
		f_wc = []
		for entry in f():
			new_entry = entry.subs(substitutions).simplify(ratio=1.0)
			past_calls += new_entry.count(anchors)
			f_wc.append(new_entry)
		
		F = self.F = sympy.lambdify(
			[t]+[Yentry for Yentry in Y],
			f_wc,
			{
				anchors.__name__: self.get_past_anchors,
				past_y .__name__: self.get_past_value
			})
		
		self.f = lambda T,ypsilon: np.array(F(T,*ypsilon)).flatten()
		
		self.anchor_mem = (len(past)-1)*np.ones(past_calls, dtype=int)
	
	def get_t(self):
		return self.t
	
	def get_past_anchors(self, t):
		s = self.anchor_mem[self.anchor_mem_index]
		
		if t > self.t:
			s = len(self.past)-2
			self.past_within_step = max(self.past_within_step,t-self.t)
		else:
			while self.past[s][0]>t and s>0:
				s -= 1
			while self.past[s+1][0]<t:
				s += 1
		
		self.anchor_mem[self.anchor_mem_index] = s
		self.anchor_mem_index += 1
		
		return (self.past[s], self.past[s+1])
	
	def get_past_value(self, t, i, anchors):
		q = (anchors[1][0]-anchors[0][0])
		x = (t-anchors[0][0]) / q
		a = anchors[0][1][i]
		b = anchors[0][2][i] * q
		c = anchors[1][1][i]
		d = anchors[1][2][i] * q
		
		return (1-x) * ( (1-x) * (b*x + (a-c)*(2*x+1)) - d*x**2) + c
	
	def get_recent_state(self, t):
		anchors = self.past[-2], self.past[-1]
		q = (anchors[1][0]-anchors[0][0])
		x = (t-anchors[0][0]) / q
		a = anchors[0][1]
		b = anchors[0][2] * q
		c = anchors[1][1]
		d = anchors[1][2] * q
		
		output = (1-x) * ( (1-x) * (b*x + (a-c)*(2*x+1)) - d*x**2) + c
		assert type(output) == np.ndarray
		return output
	
	def eval_f(self, t, y):
		self.anchor_mem_index = 0
		return self.f(t, y)
	
	def get_next_step(self, delta_t):
		self.step_count += 1
		
		self.past_within_step = False
		
		k_1 = self.diff
		k_2 = self.eval_f(self.t + 0.5 *delta_t, self.y + 0.5 *delta_t*k_1)
		k_3 = self.eval_f(self.t + 0.75*delta_t, self.y + 0.75*delta_t*k_2)
		new_y = self.y + delta_t * (2*k_1 + 3*k_2 + 4*k_3) / 9
		new_t = self.t + delta_t 
		k_4 = new_diff = self.eval_f(new_t, new_y)
		self.error = (5*k_1 - 6*k_2 - 8*k_3 + 9*k_4) / 72
		
		if self.past[-1][0]==self.t:
			self.past.append((new_t, new_y, new_diff))
		else:
			try: self.old_new_y = self.past[-1][1]
			except AttributeError: pass
			
			self.past[-1] = (new_t, new_y, new_diff)
	
	def get_p(self, atol, rtol):
		return np.max(np.abs(self.error)/(atol + rtol*np.abs(self.past[-1][1])))
	
	def check_new_y_diff(self, atol, rtol):
		difference = np.abs(self.past[-1][1]-self.old_new_y)
		tolerance = atol + np.abs(rtol*self.past[-1][1])
		return np.all(difference <= tolerance)
	
	def accept_step(self):
		self.t, self.y, self.diff = self.past[-1]

	def forget(self, delay):
		threshold = self.t - delay
		while self.past[self.last_garbage+2][0] < threshold:
			self.last_garbage += 1
		
		if self.last_garbage >= MIN_GARBAGE:
			self.past = self.past[self.last_garbage+1:]
			self.anchor_mem -= self.last_garbage+1
			self.last_garbage = -1
	
