#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

MAX_GARBAGE = 10
NORM_THRESHOLD = 1e-30

def interpolate(t,i,anchors):
	return interpolate_vec(t,anchors)[i]

def interpolate_vec(t,anchors):
	"""
		Returns the value of a cubic Hermite interpolant of the anchors at time t.
	"""
	q = (anchors[1][0]-anchors[0][0])
	x = (t-anchors[0][0]) / q
	a = anchors[0][1]
	b = anchors[0][2] * q
	c = anchors[1][1]
	d = anchors[1][2] * q
	
	return (1-x) * ( (1-x) * (b*x + (a-c)*(2*x+1)) - d*x**2) + c

sumsq = lambda x: np.sum(x**2)

# The matrix induced by the scalar product of the cubic Hermite interpolants of two anchors, if their distance is normalised to 1.
sp_matrix = np.array([
			[156,  22,  54, -13],
			[ 22,   4,  13,  -3],
			[ 54,  13, 156, -22],
			[-13,  -3, -22,   4],
		])/420

# The matrix induced by the scalar product of the cubic Hermite interpolants of two anchors, if their distance is normalised to 1, but the initial portion z of the interval is not considered for the scalar product.
def partial_sp_matrix(z):
	h_1 = - 120*z**7 - 350*z**6 - 252*z**5
	h_2 = -  60*z**7 - 140*z**6 -  84*z**5
	h_3 = - 120*z**7 - 420*z**6 - 378*z**5
	h_4 = -  70*z**6 - 168*z**5 - 105*z**4
	h_6 =            - 105*z**4 - 140*z**3
	h_7 =            - 210*z**4 - 420*z**3
	h_5 = 2*h_2 + 3*h_4
	h_8 = - h_5 + h_7 - h_6 - 210*z**2
	
	return np.array([
			[  2*h_3   , h_1    , h_7-2*h_3        , h_5              ],
			[    h_1   , h_2    , h_6-h_1          , h_2+h_4          ],
			[ h_7-2*h_3, h_6-h_1, 2*h_3-2*h_7-420*z, h_8              ],
			[   h_5    , h_2+h_4, h_8              , -h_1+h_2+h_5+h_6 ]
		])/420

def norm_sq_interval(anchors, indizes):
	"""
		Returns the norm of the interpolant of `anchors` for the `indizes`.
	"""
	q = (anchors[1][0]-anchors[0][0])
	vector = np.vstack([
			anchors[0][1][indizes]    , # a
			anchors[0][2][indizes] * q, # b
			anchors[1][1][indizes]    , # c
			anchors[1][2][indizes] * q, # d
		])
	
	return np.einsum(
			vector   , [0,2],
			sp_matrix, [0,1],
			vector   , [1,2],
		)*q

def norm_sq_partial(anchors, indizes, start):
	"""
		Returns the norm of the interpolant of `anchors` for the `indizes`, but only taking into account the time after `start`.
	"""
	q = (anchors[1][0]-anchors[0][0])
	z = (start-anchors[1][0]) / q
	vector = np.vstack([
			anchors[0][1][indizes]    , # a
			anchors[0][2][indizes] * q, # b
			anchors[1][1][indizes]    , # c
			anchors[1][2][indizes] * q, # d
		])
	
	return np.einsum(
			vector              , [0,2],
			partial_sp_matrix(z), [0,1],
			vector              , [1,2],
		)*q

def scalar_product_interval(anchors, indizes_1, indizes_2):
	"""
		Returns the scalar product of the interpolants of `anchors` for `indizes_1` (one side of the product) and `indizes_2` (other side).
	"""
	q = (anchors[1][0]-anchors[0][0])
	
	vector_1 = np.vstack([
		anchors[0][1][indizes_1],     # a_1
		anchors[0][2][indizes_1] * q, # b_1
		anchors[1][1][indizes_1],     # c_1
		anchors[1][2][indizes_1] * q, # d_1
	])
	
	vector_2 = np.vstack([
		anchors[0][1][indizes_2],     # a_2
		anchors[0][2][indizes_2] * q, # b_2
		anchors[1][1][indizes_2],     # c_2
		anchors[1][2][indizes_2] * q, # d_2
	])
	
	return np.einsum(
		vector_1, [0,2],
		sp_matrix, [0,1],
		vector_2, [1,2]
		)*q

def scalar_product_partial(anchors, indizes_1, indizes_2, start):
	"""
		Returns the scalar product of the interpolants of `anchors` for `indizes_1` (one side of the product) and `indizes_2` (other side), but only taking into account the time after `start`.
	"""
	q = (anchors[1][0]-anchors[0][0])
	z = (start-anchors[1][0]) / q
	
	vector_1 = np.vstack([
		anchors[0][1][indizes_1],     # a_1
		anchors[0][2][indizes_1] * q, # b_1
		anchors[1][1][indizes_1],     # c_1
		anchors[1][2][indizes_1] * q, # d_1
	])
	
	vector_2 = np.vstack([
		anchors[0][1][indizes_2],     # a_2
		anchors[0][2][indizes_2] * q, # b_2
		anchors[1][1][indizes_2],     # c_2
		anchors[1][2][indizes_2] * q, # d_2
	])
	
	return np.einsum(
		vector_1, [0,2],
		partial_sp_matrix(z), [0,1],
		vector_2, [1,2]
		)*q

class dde_integrator(object):
	"""
	Class for the Shampine–Thompson integrator.
	"""
	def __init__(self,
				f,
				past,
				helpers = (),
				control_pars = (),
				n_basic = None,
				tangent_indices = (),
			):
		self.past = past
		self.t, self.y, self.diff = self.past[-1]
		self.n = len(self.y)
		self.n_basic = n_basic or self.n
		self.tangent_indices = tangent_indices
		self.last_garbage = -1
		self.old_new_y = None
		
		self.parameters = []
		
		from jitcdde._jitcdde import t, y, past_y, anchors
		from sympy import DeferredVector, sympify, lambdify
		Y = DeferredVector("Y")
		substitutions = list(reversed(helpers)) + [(y(i),Y[i]) for i in range(self.n)]
		
		past_calls = 0
		f_wc = []
		for entry in f():
			new_entry = sympify(entry).subs(substitutions).simplify(ratio=1.0)
			past_calls += new_entry.count(anchors)
			f_wc.append(new_entry)
		
		F = lambdify(
				[t, Y] + list(control_pars),
				f_wc,
				[
					{
						anchors.name: self.get_past_anchors,
						past_y .name: interpolate
					},
					"math"
				]
			)
		
		self.f = lambda *args: np.array(F(*args)).flatten()
		
		# storage of search positions (cursors) for lookup of anchors corresponding to a given time (see `get_past_anchors`)
		# Note that this approach is tailored to mimic the C implementation (using linked lists) as good as possible and therefore is rather clunky in Python.
		self.anchor_mem = (len(past)-1)*np.ones(past_calls, dtype=int)
	
	def set_parameters(self, *parameters):
		self.parameters = parameters
	
	def get_t(self):
		return self.t
	
	def get_past_anchors(self, t):
		"""
			Find the two anchors neighbouring `t`.
			If `t` is outside the ranges of times covered by the anchors, return the two nearest anchors.
		"""
		s = self.anchor_mem[self.anchor_mem_index]
		
		if t > self.t:
			s = len(self.past)-2
			self.past_within_step = max(self.past_within_step,t-self.t)
		else:
			while self.past[s][0]>=t and s>0:
				s -= 1
			while self.past[s+1][0]<t:
				s += 1
		
		self.anchor_mem[self.anchor_mem_index] = s
		self.anchor_mem_index += 1
		
		return (self.past[s], self.past[s+1])
	
	def get_recent_state(self, t):
		"""
		Interpolate the state at time `t` from the last two anchors.
		With other words, this assumes that `t` lies within the last integration step.
		"""
		anchors = self.past[-2], self.past[-1]
		output = interpolate_vec(t,anchors)
		assert type(output) == np.ndarray
		return output
	
	def get_current_state(self):
		return self.past[-1][1]
	
	def get_full_state(self):
		return self.past
	
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
			self.error = (5*k_1 - 6*k_2 - 8*k_3 + 9*k_4) * (1/72.)
		except ValueError:
			# give some tentative result and force a step-size adaption
			new_y = self.y + self.diff*delta_t
			new_diff = self.diff
			new_t = self.t + delta_t
			self.error = np.array(self.n*[np.inf])
		
		if self.past[-1][0]==self.t:
			self.past.append((new_t, new_y, new_diff))
		else:
			try:
				self.old_new_y = self.past[-1][1]
			except AttributeError:
				pass
			
			self.past[-1] = (new_t, new_y, new_diff)
	
	def get_p(self, atol, rtol):
		"""
			computes the coefficient that summarises the integration error.
		"""
		with np.errstate(divide='ignore', invalid='ignore'):
			return np.nanmax(
					np.abs(self.error)
					/(atol + rtol*np.abs(self.past[-1][1]))
				)
	
	def check_new_y_diff(self, atol, rtol):
		"""
			For past-within-step iterations: Checks whether the difference between the new and old approximation of the next step is below the given tolerance level.
		"""
		if self.old_new_y is not None:
			difference = np.abs(self.past[-1][1]-self.old_new_y)
			tolerance = atol + np.abs(rtol*self.past[-1][1])
			return np.all(difference <= tolerance)
		else:
			return False
	
	def accept_step(self):
		self.t, self.y, self.diff = self.past[-1]
		self.old_new_y = None
	
	def adjust_diff(self,shift_ratio):
		"""
		adds another anchor with the same time and state as the last one but with the derivative computed with f.
		"""
		self.diff = self.eval_f(self.t,self.y)
		new_anchor = (self.t,self.y.copy(),self.diff)
		
		gap = self.past[-1][0]-self.past[-2][0]
		self.past[-1] = (
				self.past[-1][0] - shift_ratio*gap,
				self.past[-1][1],
				self.past[-1][2],
			)
		assert self.past[-1][0]>self.past[-2][0]
		self.past.append(new_anchor)
		self.t, self.y, self.diff = self.past[-1]
	
	def forget(self, delay):
		"""
		Remove past points that are “out of reach” of the delay.
		"""
		threshold = self.t - delay
		while self.past[self.last_garbage+2][0] < threshold:
			self.last_garbage += 1
		
		if self.last_garbage >= MAX_GARBAGE:
			self.past = self.past[self.last_garbage+1:]
			self.anchor_mem -= self.last_garbage+1
			self.last_garbage = -1
	
	# ------------------------------------
	
	def norm(self, delay, indizes):
		"""
			Computes the norm between the Hermite interpolants of the past for the given indizes taking into account the time between self.t − delay and self.t.
		"""
		threshold = self.t - delay
		
		i = 0
		while self.past[i+1][0] < threshold:
			i += 1
		
		# partial norm of first relevant interval
		anchors = (self.past[i],self.past[i+1])
		norm_sq = norm_sq_partial(anchors, indizes, threshold)
		
		# full norms of all others
		for i in range(i+1, len(self.past)-1):
			anchors = (self.past[i],self.past[i+1])
			norm_sq += norm_sq_interval(anchors, indizes)
		
		return np.sqrt(norm_sq)
	
	def scalar_product(self, delay, indizes_1, indizes_2):
		"""
			Computes the scalar product of the Hermite interpolants of the past between `indizes_1` (one side of the product) and `indizes_2` (other side) taking into account the time between self.t − delay and self.t.
		"""
		threshold = self.t - delay
		
		i = 0
		while self.past[i+1][0] < threshold:
			i += 1
		
		# partial scalar product of first relevant interval
		anchors = (self.past[i],self.past[i+1])
		sp = scalar_product_partial(anchors, indizes_1, indizes_2, threshold)
		
		# full scalar product of all others
		for i in range(i+1, len(self.past)-1):
			anchors = (self.past[i],self.past[i+1])
			sp += scalar_product_interval(anchors, indizes_1, indizes_2)
		
		return sp
	
	def scale_past(self, indizes, factor):
		for anchor in self.past:
			anchor[1][indizes] *= factor
			anchor[2][indizes] *= factor
	
	def subtract_from_past(self, indizes_1, indizes_2, factor):
		for anchor in self.past:
			anchor[1][indizes_1] -= factor*anchor[1][indizes_2]
			anchor[2][indizes_1] -= factor*anchor[2][indizes_2]
	
	def orthonormalise(self, n_lyap, delay):
		"""
		Orthonormalise separation functions (with Gram-Schmidt) and return their norms after orthogonalisation (but before normalisation).
		"""
		
		vectors = np.split(np.arange(self.n, dtype=int), n_lyap+1)[1:]
		
		norms = []
		for i,vector in enumerate(vectors):
			for j in range(i):
				sp = self.scalar_product(delay, vector, vectors[j])
				self.subtract_from_past(vector, vectors[j], sp)
			norm = self.norm(delay, vector)
			if norm > NORM_THRESHOLD:
				self.scale_past(vector, 1./norm)
			norms.append(norm)
		
		return np.array(norms)
	
	def remove_projections(self, delay, vectors):
		"""
		Remove projections of separation function to vectors and return norm after normalisation.
		"""
		
		sep_func = np.arange(self.n_basic, 2*self.n_basic, 1, dtype=int)
		assert np.all(sep_func == np.split(np.arange(self.n, dtype=int), 2+2*len(vectors))[1])
		assert self.n_basic == len(sep_func)
		d = len(vectors)*2
		
		def get_dummy(index):
			return np.arange((index+2)*self.n_basic, (index+3)*self.n_basic)
		
		dummy_num = 0
		len_dummies = 0
		for anchor in self.past:
			for vector in vectors:
				# Setup dummy 
				dummy = get_dummy(dummy_num)
				for other_anchor in self.past:
					other_anchor[1][dummy] = np.zeros(self.n_basic)
					other_anchor[2][dummy] = np.zeros(self.n_basic)
				anchor[1][dummy] = vector[0]
				anchor[2][dummy] = vector[1]
				
				# Orthonormalise dummies
				past_dummies = [get_dummy( (dummy_num-i-1) % d ) for i in range(len_dummies)]
				
				for past_dummy in past_dummies:
					sp = self.scalar_product(delay, dummy, past_dummy)
					self.subtract_from_past(dummy, past_dummy, sp)
				
				norm = self.norm(delay, dummy)
				if norm > NORM_THRESHOLD:
					self.scale_past(dummy, 1./norm)
					
					# remove projection to dummy
					sp = self.scalar_product(delay, sep_func, dummy)
					self.subtract_from_past(sep_func, dummy, sp)
				else:
					self.scale_past(dummy, 0.0)
				
				len_dummies += 1
				dummy_num = (dummy_num+1)%d
			
			if len_dummies > len(vectors):
				len_dummies -= len(vectors)
		
		for anchor in self.past:
			anchor[1][2*self.n_basic:] = 0.0
			anchor[2][2*self.n_basic:] = 0.0
		
		# Normalise separation function
		norm = self.norm(delay, sep_func)
		self.scale_past(sep_func, 1./norm)
		
		return norm
	
	def normalise_indices(self, delay):
		"""
		Normalise the separation function of the tangent indices (with Gram-Schmidt) and return the norms (before normalisation).
		"""
		
		norm = self.norm(delay,self.tangent_indices)
		if norm > NORM_THRESHOLD:
			self.scale_past(self.tangent_indices,1./norm)
		return norm
	
	def remove_state_component(self, index):
		for anchor in self.past:
			anchor[1][self.n_basic+index] = 0.0
	
	def remove_diff_component(self, index):
		for anchor in self.past:
			anchor[2][self.n_basic+index] = 0.0
	
	
