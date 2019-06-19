#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from jitcxde_common.numerical import random_direction

NORM_THRESHOLD = 1e-30


class Anchor(tuple):
	def __new__( cls, time, state, diff ):
		state = np.atleast_1d(np.array(state,dtype=float,copy=True))
		diff  = np.atleast_1d(np.array(diff ,dtype=float,copy=True))
		if len(state.shape) != 1:
			raise ValueError("State must be a number or one-dimensional iterable.")
		if state.shape != diff.shape:
			raise ValueError("State and diff do not match in shape.")
		return super().__new__(cls,(time,state,diff))
	
	def __init__(self, *args):
		self.time  = self[0]
		self.state = self[1]
		self.diff  = self[2]

def interpolate(t,i,anchors):
	return interpolate_vec(t,anchors)[i]

def interpolate_vec(t,anchors):
	"""
		Returns the value of a cubic Hermite interpolant of the anchors at time t.
	"""
	q = (anchors[1].time-anchors[0].time)
	x = (t-anchors[0].time) / q
	a = anchors[0].state
	b = anchors[0].diff * q
	c = anchors[1].state
	d = anchors[1].diff * q
	
	return (1-x) * ( (1-x) * (b*x + (a-c)*(2*x+1)) - d*x**2) + c

def interpolate_diff(t,i,anchors):
	return interpolate_diff_vec(t,anchors)[i]

def interpolate_diff_vec(t,anchors):
	"""
		Returns the derivative of a cubic Hermite interpolant of the anchors at time t.
	"""
	q = (anchors[1].time-anchors[0].time)
	x = (t-anchors[0].time) / q
	a = anchors[0].state
	b = anchors[0].diff * q
	c = anchors[1].state
	d = anchors[1].diff * q
	
	return ( (1-x)*(b-x*3*(2*(a-c)+b+d)) + d*x ) /q

def extrema(anchors):
	"""
		Finds the extrema of the Hermite interpolant for the anchors (within the interval spanned by them).
		Returns two arrays containing the minima and maxima of the Hermite interpolant for the anchors (within the interval spanned by them) as well as two arrays containing the position of these extrema.
		
		Returns
		-------
		minimal, maxima : NumPy arrays
			the values of the extrema for each component
		
		arg_min, arg_max : NumPy arrays
			the positions of the extrema for each component
	"""
	q = (anchors[1].time-anchors[0].time)
	retransform = lambda x: q*x+anchors[0].time
	a = anchors[0].state
	b = anchors[0].diff * q
	c = anchors[1].state
	d = anchors[1].diff * q
	
	condition = anchors[1].state>anchors[0].state
	arg_min = np.where(condition,anchors[0].time,anchors[1].time)
	arg_max = np.where(condition,anchors[1].time,anchors[0].time)
	minima = np.minimum(anchors[0].state,anchors[1].state)
	maxima = np.maximum(anchors[0].state,anchors[1].state)
	
	radicant = b**2 + b*d + d**2 + 3*(a-c)*(3*(a-c) + 2*(b+d))
	A = 1/(2*a + b - 2*c + d)
	B = a + 2*b/3 - c + d/3
	
	n = len(anchors[0].state)
	for i in range(n):
		if radicant[i]>=0:
			for sign in (-1,1):
				x = (B[i]+sign*np.sqrt(radicant[i])/3)*A[i]
				if 0<x<1:
					value = (1-x) * ( (1-x) * (b[i]*x + (a[i]-c[i])*(2*x+1)) - d[i]*x**2) + c[i]
					if value<minima[i]:
						minima[i] = value
						arg_min[i] = retransform(x)
					elif value>maxima[i]:
						maxima[i] = value
						arg_max[i] = retransform(x)
	
	return minima,maxima,arg_min,arg_max

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
	q = (anchors[1].time-anchors[0].time)
	vector = np.vstack([
			anchors[0].state[indices]   , # a
			anchors[0].diff[indices] * q, # b
			anchors[1].state[indices]   , # c
			anchors[1].diff[indices] * q, # d
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
	q = (anchors[1].time-anchors[0].time)
	z = (start-anchors[1].time) / q
	vector = np.vstack([
			anchors[0].state[indices]   , # a
			anchors[0].diff[indices] * q, # b
			anchors[1].state[indices]   , # c
			anchors[1].diff[indices] * q, # d
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
	q = (anchors[1].time-anchors[0].time)
	
	vector_1 = np.vstack([
		anchors[0].state[indices_1],    # a_1
		anchors[0].diff[indices_1] * q, # b_1
		anchors[1].state[indices_1],    # c_1
		anchors[1].diff[indices_1] * q, # d_1
	])
	
	vector_2 = np.vstack([
		anchors[0].state[indices_2],    # a_2
		anchors[0].diff[indices_2] * q, # b_2
		anchors[1].state[indices_2],    # c_2
		anchors[1].diff[indices_2] * q, # d_2
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
	q = (anchors[1].time-anchors[0].time)
	z = (start-anchors[1].time) / q
	
	vector_1 = np.vstack([
		anchors[0].state[indices_1],    # a_1
		anchors[0].diff[indices_1] * q, # b_1
		anchors[1].state[indices_1],    # c_1
		anchors[1].diff[indices_1] * q, # d_1
	])
	
	vector_2 = np.vstack([
		anchors[0].state[indices_2],    # a_2
		anchors[0].diff[indices_2] * q, # b_2
		anchors[1].state[indices_2],    # c_2
		anchors[1].diff[indices_2] * q, # d_2
	])
	
	return np.einsum(
			vector_1, [0,2],
			partial_sp_matrix(z), [0,1],
			vector_2, [1,2]
		)*q

class Past(list):
	def __init__(self,n=None,past=(),n_basic=None,tangent_indices=()):
		if n is None:
			assert isinstance(past,Past)
			Past.__init__( self, past.n, past, n_basic=past.n_basic, tangent_indices=past.tangent_indices )
		else:
			self.n = n
			self.n_basic = n_basic or self.n
			self.tangent_indices = tangent_indices
			
			if self.tangent_indices:
				self.main_indices = [i for i in range(n) if i not in self.tangent_indices]
			
			super().__init__( [self.prepare_anchor(anchor) for anchor in past] )
			self.sort()
	
	def prepare_anchor(self,x):
		x = x if isinstance(x,Anchor) else Anchor(*x)
		
		if self.tangent_indices and len(x.state)<self.n:
			assert len(x.state)==len(self.main_indices)
			
			new_state = np.resize(x.state,self.n)
			new_state[self.main_indices] = x.state
			new_state[self.tangent_indices] = random_direction(len(self.tangent_indices))
			
			new_diff = np.resize(x.diff,self.n)
			new_diff[self.main_indices] = x.diff
			new_diff[self.tangent_indices] = random_direction(len(self.tangent_indices))
			
			x = Anchor(x.time,new_state,new_diff)
		
		if x.state.shape not in [ (self.n,), (self.n_basic,) ]:
			raise ValueError("State has wrong shape.")
		
		if self.n_basic<self.n and x.state.shape==(self.n_basic,):
			assert self.n%self.n_basic == 0
			new_state = np.resize(x.state,self.n)
			new_diff  = np.resize(x.diff ,self.n)
			for i in range(1,self.n//self.n_basic):
				indices = slice(i*self.n_basic,(i+1)*self.n_basic)
				new_state[indices] = random_direction(self.n_basic)
				new_diff [indices] = random_direction(self.n_basic)
			x = Anchor(x.time,new_state,new_diff)
		
		return x
	
	def append(self,anchor):
		anchor = self.prepare_anchor(anchor)
		if self and anchor.time <= self[-1].time:
			raise ValueError("Anchor must follow last one in time. Consider using `add` instead.")
		super().append(anchor)
	
	def extend(self,anchors):
		for anchor in anchors:
			self.append(anchor)
	
	def copy(self):
		return Past(past=self)
	
	def __setitem__(self,key,item):
		anchor = self.prepare_anchor(item)
		if (
					(key!= 0 and key!=-len(self)   and self[key-1].time>=anchor.time)
				or  (key!=-1 and key!= len(self)-1 and self[key+1].time<=anchor.time)
			):
			raise ValueError("Anchor’s time does not fit.")
		super().__setitem__(key,anchor)
	
	def sort(self):
		self.check_for_duplicate_times()
		super().sort( key = lambda anchor: anchor.time )
	
	def check_for_duplicate_times(self):
		if len(set(anchor.time for anchor in self)) != len(self):
			raise ValueError("You cannot have two anchors with the same time.")
	
	def add(self,anchor):
		super().append( self.prepare_anchor(anchor) )
		self.sort()
	
	def clear_from(self,n):
		while len(self)>n:
			self.pop()
	
	def clear(self):
		super().__init__()
	
	@property
	def t(self):
		"""
		The time of the last anchor. This may be overwritten in subclasses.
		"""
		return self[-1].time
	
	def get_anchors(self, time):
		"""
			Find the two anchors (before `self.t`) neighbouring `t`.
			If `t` is outside the ranges of times covered by the anchors, return the two nearest anchors.
		"""
		
		if time > self.t:
			return (self[-2], self[-1])
		else:
			s = 0
			while self[s].time>=time and s>0:
				s -= 1
			while self[s+1].time<time:
				s += 1
			return (self[s], self[s+1])
	
	def get_recent_state(self, t):
		"""
		Interpolate the state at time `t` from the last two anchors.
		With other words, this assumes that `t` lies between the last two anchors.
		"""
		anchors = self[-2], self[-1]
		output = interpolate_vec(t,anchors)
		assert type(output) == np.ndarray
		return output
	
	def get_current_state(self):
		return self[-1].state
	
	def get_full_state(self):
		return self
	
	def forget(self, delay):
		"""
		Remove all past points that are “out of reach” of the delay with respect to `self.t`.
		"""
		threshold = self.t - delay
		while self[1].time<threshold:
			self.pop(0)
	
	def norm(self, delay, indices):
		"""
			Computes the norm between the Hermite interpolants of the past for the given indices taking into account the time between `self.t` − `delay` and `self.t`.
		"""
		threshold = self.t - delay
		
		i = 0
		while self[i+1].time < threshold:
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
		while self[i+1].time < threshold:
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
			anchor.state[indices] *= factor
			anchor.diff [indices] *= factor
	
	def subtract(self, indices_1, indices_2, factor):
		"""
			Substract the past and derivative for `indices_2` multiplied by `factor` from `indices_1`.
		"""
		for anchor in self:
			anchor.state[indices_1] -= factor*anchor.state[indices_2]
			anchor.diff [indices_1] -= factor*anchor.diff [indices_2]
	
	def last_index_before(self,time):
		"""
			Returns the index of the last anchor before `time`.
		"""
		assert len(self)>=2
		assert self[0].time<=time
		
		i = len(self)-2
		while self[i].time >= time:
			i -= 1
		return i
	
	def truncate(self,time):
		"""
		Interpolates an anchor at `time` and removes all later anchors.
		"""
		assert self[0].time<=time<=self[-1].time, f"Truncation time must be within current range of anchors."
		i = self.last_index_before(time)
		
		value =     interpolate_vec(time,(self[i],self[i+1]))
		diff = interpolate_diff_vec(time,(self[i],self[i+1]))
		self[i+1] = Anchor(time,value,diff)
		
		self.clear_from(i+2)
		assert len(self)>=1
	
	def orthonormalise(self, n_lyap, delay):
		"""
		Orthonormalise separation functions (with Gram-Schmidt) and return their norms after orthogonalisation (but before normalisation).
		"""
		
		vectors = np.split(np.arange(self.n, dtype=int), n_lyap+1)[1:]
		
		norms = []
		for i,vector in enumerate(vectors):
			for j in range(i):
				sp = self.scalar_product(delay, vector, vectors[j])
				self.subtract(vector, vectors[j], sp)
			norm = self.norm(delay, vector)
			if norm > NORM_THRESHOLD:
				self.scale(vector, 1./norm)
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
		for anchor in self:
			for vector in vectors:
				# Setup dummy
				dummy = get_dummy(dummy_num)
				for other_anchor in self:
					other_anchor.state[dummy] = np.zeros(self.n_basic)
					other_anchor.diff [dummy] = np.zeros(self.n_basic)
				anchor.state[dummy] = vector[0]
				anchor.diff [dummy] = vector[1]
				
				# Orthonormalise dummies
				past_dummies = [get_dummy( (dummy_num-i-1) % d ) for i in range(len_dummies)]
				
				for past_dummy in past_dummies:
					sp = self.scalar_product(delay, dummy, past_dummy)
					self.subtract(dummy, past_dummy, sp)
				
				norm = self.norm(delay, dummy)
				if norm > NORM_THRESHOLD:
					self.scale(dummy, 1./norm)
					
					# remove projection to dummy
					sp = self.scalar_product(delay, sep_func, dummy)
					self.subtract(sep_func, dummy, sp)
				else:
					self.scale(dummy, 0.0)
				
				len_dummies += 1
				dummy_num = (dummy_num+1)%d
			
			if len_dummies > len(vectors):
				len_dummies -= len(vectors)
		
		for anchor in self:
			anchor.state[2*self.n_basic:] = 0.0
			anchor.diff[2*self.n_basic:] = 0.0
		
		# Normalise separation function
		norm = self.norm(delay, sep_func)
		self.scale(sep_func, 1./norm)
		
		return norm
	
	def normalise_indices(self, delay):
		"""
		Normalise the separation function of the tangent indices and return the norm (before normalisation).
		"""
		
		norm = self.norm(delay,self.tangent_indices)
		if norm > NORM_THRESHOLD:
			self.scale(self.tangent_indices,1./norm)
		return norm
	
	def remove_state_component(self, index):
		for anchor in self:
			anchor.state[self.n_basic+index] = 0.0
	
	def remove_diff_component(self, index):
		for anchor in self:
			anchor.diff[self.n_basic+index] = 0.0

