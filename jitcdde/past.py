#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from jitcxde_common.numerical import random_direction
from chspy import CubicHermiteSpline, Anchor

NORM_THRESHOLD = 1e-30

class Past(CubicHermiteSpline):
	def __init__(self,n=None,anchors=(),n_basic=None,tangent_indices=()):
		if n is None:
			assert isinstance(anchors,Past)
			Past.__init__( self, anchors.n, anchors, n_basic=anchors.n_basic, tangent_indices=anchors.tangent_indices )
		else:
			self.n_basic = n_basic or n
			self.tangent_indices = tangent_indices
			if self.tangent_indices:
				self.main_indices = [i for i in range(n) if i not in self.tangent_indices]
			super().__init__( n, anchors )
	
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
	
	def get_full_state(self):
		return self
	
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

