import unittest

import numpy as np
from numpy.testing import assert_allclose

from jitcdde.past import Past


class normalisation_test(unittest.TestCase):
	def test_orthonormalisation(self):
		rng = np.random.default_rng(seed=42)
		m = 4
		past = Past(n=m,anchors=[
				(time, rng.normal(0,1,m), rng.normal(0,1,m))
				for time in sorted(rng.uniform(-3,0,5))
			])
		delay = rng.uniform(0.0,2.0)
		past.orthonormalise(m-1, delay)
		
		for j in range(1,m):
			self.assertAlmostEqual(past.norm(delay, j), 1.0)
			
			for k in range(j,m):
				control = 1.0 if k==j else 0.0
				sp = past.scalar_product(delay, j, k)
				self.assertAlmostEqual(sp, control)

class remove_projection_test(unittest.TestCase):
	def setUp(self):
		rng = np.random.default_rng(seed=42)
		self.n_basic = 3
		self.n = 6*self.n_basic
		
		self.original_past = Past(n=self.n,n_basic=self.n_basic)
		for i in range(rng.integers(3,10)):
			if i==0:
				time = rng.uniform(-10,10)
			else:
				time = self.original_past[-1][0] + 0.1 + rng.random()
			state = rng.random(self.n)
			diff  = rng.random(self.n)
			self.original_past.append((time, state, diff))
		
		self.past = self.original_past.copy()
	
	def test_remove_first_component(self):
		rng = np.random.default_rng(seed=42)
		empty = lambda: np.zeros(self.n_basic)
		vectors = [
				(empty(), empty()),
				(empty(), empty())
			]
		component = rng.integers(0,self.n_basic)
		vectors[0][0][component] = 1
		vectors[1][1][component] = 1
		delay = self.original_past[-1][0]-self.original_past[0][0]
		
		self.past.remove_projections(delay, vectors)
		for anchor in self.past:
			self.assertAlmostEqual(anchor[1][self.n_basic+component], 0.0)
			self.assertAlmostEqual(anchor[2][self.n_basic+component], 0.0)
	
	def test_double_removal(self):
		rng = np.random.default_rng(seed=42)
		random_vector = lambda: rng.random(self.n_basic)
		vectors = [
			(random_vector(), random_vector()),
			(random_vector(), random_vector())
			]
		delay = (self.original_past[-1][0]-self.original_past[0][0])*rng.uniform(0.5,1.5)
		
		self.past.remove_projections(delay, vectors)
		past_copy = [(anchor[0], np.copy(anchor[1]), np.copy(anchor[2])) for anchor in self.past]
		for anchor_A, anchor_B in zip(past_copy, self.past, strict=True):
			assert_allclose(anchor_A[1], anchor_B[1])
			assert_allclose(anchor_A[2], anchor_B[2])
		
		norm = self.past.remove_projections(delay, vectors)
		self.assertAlmostEqual(norm, 1.0)
		
		for anchor_A, anchor_B in zip(past_copy, self.past, strict=True):
			assert_allclose(anchor_A[1], anchor_B[1])
			assert_allclose(anchor_A[2], anchor_B[2])

if __name__ == "__main__":
	unittest.main(buffer=True)

