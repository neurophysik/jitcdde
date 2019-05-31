#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

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

def interpolate_diff(t,i,anchors):
	return interpolate_diff_vec(t,anchors)[i]

def interpolate_diff_vec(t,anchors):
	"""
		Returns the derivative of a cubic Hermite interpolant of the anchors at time t.
	"""
	q = (anchors[1][0]-anchors[0][0])
	x = (t-anchors[0][0]) / q
	a = anchors[0][1]
	b = anchors[0][2] * q
	c = anchors[1][1]
	d = anchors[1][2] * q
	
	return ( (1-x)*(b-x*3*(2*(a-c)+b+d)) + d*x ) /q

def extrema(anchors):
	"""
		Returns two arrays containing the minima and maxima of the Hermite interpolant for the anchors (within the interval spanned by them).
	"""
	q = (anchors[1][0]-anchors[0][0])
	retransform = lambda x: q*x+anchors[0][0]
	a = anchors[0][1]
	b = anchors[0][2] * q
	c = anchors[1][1]
	d = anchors[1][2] * q
	
	minima = np.minimum(anchors[0][1],anchors[1][1])
	maxima = np.maximum(anchors[0][1],anchors[1][1])
	
	radicant = b**2 + b*d + d**2 + 3*(a-c)*(3*(a-c) + 2*(b+d))
	A = 1/(2*a + b - 2*c + d)
	B = a + 2*b/3 - c + d/3
	
	n = len(anchors[0][1])
	for i in range(n):
		if radicant[i]>=0:
			times = { retransform((B[i]+sign*np.sqrt(radicant[i])/3)*A[i]) for sign in (-1,1) }
			values = [
					interpolate(time,i,anchors)
					for time in times
					if anchors[0][0] < time < anchors[1][0]
				]
			minima[i] = min(values+[minima[i]])
			maxima[i] = max(values+[maxima[i]])
	
	return minima,maxima

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

def norm_sq_interval(anchors, indices):
	"""
		Returns the norm of the interpolant of `anchors` for the `indices`.
	"""
	q = (anchors[1][0]-anchors[0][0])
	vector = np.vstack([
			anchors[0][1][indices]    , # a
			anchors[0][2][indices] * q, # b
			anchors[1][1][indices]    , # c
			anchors[1][2][indices] * q, # d
		])
	
	return np.einsum(
			vector   , [0,2],
			sp_matrix, [0,1],
			vector   , [1,2],
		)*q

def norm_sq_partial(anchors, indices, start):
	"""
		Returns the norm of the interpolant of `anchors` for the `indices`, but only taking into account the time after `start`.
	"""
	q = (anchors[1][0]-anchors[0][0])
	z = (start-anchors[1][0]) / q
	vector = np.vstack([
			anchors[0][1][indices]    , # a
			anchors[0][2][indices] * q, # b
			anchors[1][1][indices]    , # c
			anchors[1][2][indices] * q, # d
		])
	
	return np.einsum(
			vector              , [0,2],
			partial_sp_matrix(z), [0,1],
			vector              , [1,2],
		)*q

def scalar_product_interval(anchors, indices_1, indices_2):
	"""
		Returns the scalar product of the interpolants of `anchors` for `indices_1` (one side of the product) and `indices_2` (other side).
	"""
	q = (anchors[1][0]-anchors[0][0])
	
	vector_1 = np.vstack([
		anchors[0][1][indices_1],     # a_1
		anchors[0][2][indices_1] * q, # b_1
		anchors[1][1][indices_1],     # c_1
		anchors[1][2][indices_1] * q, # d_1
	])
	
	vector_2 = np.vstack([
		anchors[0][1][indices_2],     # a_2
		anchors[0][2][indices_2] * q, # b_2
		anchors[1][1][indices_2],     # c_2
		anchors[1][2][indices_2] * q, # d_2
	])
	
	return np.einsum(
		vector_1, [0,2],
		sp_matrix, [0,1],
		vector_2, [1,2]
		)*q

def scalar_product_partial(anchors, indices_1, indices_2, start):
	"""
		Returns the scalar product of the interpolants of `anchors` for `indices_1` (one side of the product) and `indices_2` (other side), but only taking into account the time after `start`.
	"""
	q = (anchors[1][0]-anchors[0][0])
	z = (start-anchors[1][0]) / q
	
	vector_1 = np.vstack([
		anchors[0][1][indices_1],     # a_1
		anchors[0][2][indices_1] * q, # b_1
		anchors[1][1][indices_1],     # c_1
		anchors[1][2][indices_1] * q, # d_1
	])
	
	vector_2 = np.vstack([
		anchors[0][1][indices_2],     # a_2
		anchors[0][2][indices_2] * q, # b_2
		anchors[1][1][indices_2],     # c_2
		anchors[1][2][indices_2] * q, # d_2
	])
	
	return np.einsum(
		vector_1, [0,2],
		partial_sp_matrix(z), [0,1],
		vector_2, [1,2]
		)*q

class Past(list):
	def __init__(self,past=None):
		super().__init__(past or [])
	
	def copy(self):
		return Past(super().copy())
	
	@property
	def t(self):
		"""
		The time of the last anchor. This may be overwritten in subclasses.
		"""
		return self[-1][0]
	
	def get_anchors(self, time):
		"""
			Find the two anchors (before `self.t`) neighbouring `t`.
			If `t` is outside the ranges of times covered by the anchors, return the two nearest anchors.
		"""
		
		if time > self.t:
			return (self[-2], self[-1])
		else:
			s = 0
			while self[s][0]>=time and s>0:
				s -= 1
			while self[s+1][0]<time:
				s += 1
			return (self[s], self[s+1])
	
	def get_recent_state(self, t):
		"""
		Interpolate the state at time `t` from the last two anchors.
		With other words, this assumes that `t` lies within the last integration step.
		"""
		anchors = self[-2], self[-1]
		output = interpolate_vec(t,anchors)
		assert type(output) == np.ndarray
		return output
	
	def get_current_state(self):
		return self[-1][1]
	
	def get_full_state(self):
		return self
	
	def forget(self, delay, max_garbage=10):
		"""
		Remove all but `max_garbage` past points that are “out of reach” of the delay with respect to `self.t`.
		"""
		threshold = self.t - delay
		last_garbage = -1
		while self[last_garbage+2][0] < threshold:
			last_garbage += 1
		
		if last_garbage >= max_garbage:
			self.__init__(self[last_garbage+1:])
	
	def norm(self, delay, indices):
		"""
			Computes the norm between the Hermite interpolants of the past for the given indices taking into account the time between `self.t` − `delay` and `self.t`.
		"""
		threshold = self.t - delay
		
		i = 0
		while self[i+1][0] < threshold:
			i += 1
		
		# partial norm of first relevant interval
		anchors = (self[i],self[i+1])
		norm_sq = norm_sq_partial(anchors, indices, threshold)
		
		# full norms of all others
		for i in range(i+1, len(self)-1):
			anchors = (self[i],self[i+1])
			norm_sq += norm_sq_interval(anchors, indices)
		
		return np.sqrt(norm_sq)
	
	def scalar_product(self, delay, indices_1, indices_2):
		"""
			Computes the scalar product of the Hermite interpolants of the past between `indices_1` (one side of the product) and `indices_2` (other side) taking into account the time between `self.t` − `delay` and `self.t`.
		"""
		threshold = self.t - delay
		
		i = 0
		while self[i+1][0] < threshold:
			i += 1
		
		# partial scalar product of first relevant interval
		anchors = (self[i],self[i+1])
		sp = scalar_product_partial(anchors, indices_1, indices_2, threshold)
		
		# full scalar product of all others
		for i in range(i+1, len(self)-1):
			anchors = (self[i],self[i+1])
			sp += scalar_product_interval(anchors, indices_1, indices_2)
		
		return sp
	
	def scale(self, indices, factor):
		"""
			Scales the past and derivative for `indices` by `factor`.
		"""
		for anchor in self:
			anchor[1][indices] *= factor
			anchor[2][indices] *= factor
	
	def subtract(self, indices_1, indices_2, factor):
		"""
			Substract the past and derivative for `indices_2` multiplied by `factor` from `indices_1`.
		"""
		for anchor in self:
			anchor[1][indices_1] -= factor*anchor[1][indices_2]
			anchor[2][indices_1] -= factor*anchor[2][indices_2]
	
	def last_index_before(self,time):
		"""
			Returns the index of the last anchor before `time`.
		"""
		i = len(self)-2
		while self[i][0] >= time:
			i -= 1
		return i
	
	def truncate(self,time):
		"""
		Interpolates an anchor at `time` and removes all later anchors.
		"""
		assert self[0][0]<=time<=self[-1][0], "truncation time must be within current range of anchors"
		i = self.last_index_before(time)
		
		value =     interpolate_vec(time,(self[i],self[i+1]))
		diff = interpolate_diff_vec(time,(self[i],self[i+1]))
		self[i+1] = (time,value,diff)
		
		self.__init__(self[:i+2])
		assert len(self)>=1
		# TODO: self.anchor_mem = np.minimum(self.anchor_mem,len(self.past)-1)
		# TODO: self.accept_step()
	
	def extrema_in_last_step(self):
		return extrema(self[-2:])

