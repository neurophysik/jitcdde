#!/usr/bin/python3
# -*- coding: utf-8 -*-

from jitcdde.past import Past, scalar_product_interval, scalar_product_partial, norm_sq_interval, norm_sq_partial, interpolate, interpolate_diff, extrema

import symengine
import numpy as np
from numpy.testing import assert_allclose
from itertools import chain
import unittest

m = 4

coeff = np.random.random((m,4))
poly = [lambda x, j=j: sum(coeff[j]*(x**np.arange(4))) for j in range(m)]
diff = [lambda x, j=j: sum(np.arange(1,4)*coeff[j,1:]*(x**np.arange(3))) for j in range(m)]

assert poly[0](1.0) != poly[1](1.0)
assert diff[0](1.0) != diff[1](1.0)

past = Past([
		(
			0.0,
			np.array([ poly[j](0.0) for j in range(m) ]),
			np.array([ diff[j](0.0) for j in range(m) ]),
		),
		(
			0.5,
			np.array([ poly[j](0.5) for j in range(m) ]),
			np.array([ diff[j](0.5) for j in range(m) ]),
		),
		(
			2.0,
			np.random.random(m),
			np.random.random(m)
		),
	])

class interpolation_test(unittest.TestCase):
	def test_anchors(self):
		for s in range(len(past)-1):
			t = past[s][0]
			anchors = (past[s],past[s+1])
			for j in range(m):
				self.assertAlmostEqual(
						past[s][1][j],
						interpolate(t, j, anchors),
					)
				self.assertAlmostEqual(
						past[s][2][j],
						interpolate_diff(t, j, anchors),
					)
	
	def test_interpolation(self):
		anchors = (past[0], past[1])
		for t in np.linspace(past[0][0],past[1][0],100):
			for j in range(m):
				self.assertAlmostEqual(
						poly[j](t),
						interpolate(t, j, anchors),
					)
				self.assertAlmostEqual(
						diff[j](t),
						interpolate_diff(t, j, anchors),
					)

class get_anchors_test(unittest.TestCase):
	def test_get_anchors(self):
		for s in range(len(past)-1):
			r = np.random.random()
			t = r*past[s][0] + (1-r)*past[s+1][0]
			anchors = past.get_anchors(t)
			self.assertEqual(anchors[0], past[s])
			self.assertEqual(anchors[1], past[s+1])
	
	def test_too_early(self):
		t = past[0][0] - 1.0
		anchors = past.get_anchors(t)
		self.assertEqual(anchors[0], past[0])
		self.assertEqual(anchors[1], past[1])

	def test_too_late(self):
		t = past[-1][0] + 1.0
		anchors = past.get_anchors(t)
		self.assertEqual(anchors[0], past[-2])
		self.assertEqual(anchors[1], past[-1])

class metrics_test(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.past = past.copy()
	
	def test_compare_norm_with_brute_force(self):
		delay = np.random.uniform(0.0,2.0)
		end = past[-1][0]
		start = end - delay
		
		# Very blunt numerical integration
		N = 100000
		factor = (end-start)/N
		bf_norm_sq = 0
		for t in np.linspace(start,end,N):
			anchors = self.past.get_anchors(t)
			for j in range(m):
				bf_norm_sq += interpolate(t, j, anchors)**2*factor
		
		norm = self.past.norm(delay, np.array(range(m)))
		
		self.assertAlmostEqual(norm, np.sqrt(bf_norm_sq),4)
		
	def test_compare_sp_with_brute_force(self):
		delay = np.random.uniform(0.0,2.0)
		end = past[-1][0]
		start = end - delay
		
		# Very blunt numerical quadrature
		N = 100000
		factor = (end-start)/N
		bf_sp_sq = 0
		for t in np.linspace(start,end,N):
			anchors = self.past.get_anchors(t)
			bf_sp_sq += (
				  interpolate(t, 0, anchors)
				* interpolate(t, 2, anchors)
				* factor)
			bf_sp_sq += (
				  interpolate(t, 1, anchors)
				* interpolate(t, 3, anchors)
				* factor)
		
		sp = self.past.scalar_product(delay, [0,1], [2,3])
		
		self.assertAlmostEqual(sp, bf_sp_sq, 4)
	
	def test_untrue_partials_norms(self):
		for i in range(len(self.past)-1):
			anchors = (self.past[i], self.past[i+1])
			start = np.random.randint(0,m-1)
			length = np.random.randint(1,m-start)
			indizes = list(range(start, start+length))
			norm = norm_sq_interval(anchors, indizes)
			partial_norm = norm_sq_partial(anchors, indizes, anchors[0][0])
			self.assertAlmostEqual(norm, partial_norm)
	
	def test_untrue_partials_sp(self):
		for i in range(len(self.past)-1):
			anchors = (self.past[i], self.past[i+1])
			start_1 = np.random.randint(0,m-1)
			start_2 = np.random.randint(0,m-1)
			length = np.random.randint(1,m-max(start_1, start_2))
			indizes_1 = list(range(start_1, start_1+length))
			indizes_2 = list(range(start_2, start_2+length))
			sp = scalar_product_interval(anchors, indizes_1, indizes_2)
			psp = scalar_product_partial(anchors, indizes_1, indizes_2, anchors[0][0])
			self.assertAlmostEqual(sp, psp)

class truncation_test(unittest.TestCase):
	def test_truncation(self):
		truncation_time = np.random.uniform(past[-2][0],past[-1][0])
		truncated_past = past.copy()
		truncated_past.truncate(truncation_time)
		
		assert truncated_past[-1][0] == truncation_time
		
		anchors = (past[-2], past[-1])
		anchors_trunc = (truncated_past[-2],truncated_past[-1])
		for t in np.linspace(past[-2][0],truncation_time,30):
			for j in range(m):
				self.assertAlmostEqual(
						interpolate( t, j, anchors       ),
						interpolate( t, j, anchors_trunc ),
					)
				self.assertAlmostEqual(
						interpolate_diff( t, j, anchors       ),
						interpolate_diff( t, j, anchors_trunc ),
					)

class extrema_test(unittest.TestCase):
	def test_arg_extreme_given_extrema(self):
		n = 3
		positions = np.random.random(2)
		state = np.random.random(n)
		past = [
				( positions[0], state                       , np.zeros(n) ),
				( positions[1], state+np.random.uniform(0,5), np.zeros(n) ),
			]
		minima,maxima = extrema(past)
		assert_allclose(minima,past[0][1])
		assert_allclose(maxima,past[1][1])
	
	def test_arg_extreme_simple_polynomial(self):
		T = symengine.Symbol("T")
		poly = 2*T**3 - 3*T**2 - 36*T + 17
		arg_extremes = [-2,3]
		arrify = lambda expr,t: np.atleast_1d(float(expr.subs({T:t})))
		past = [
				( t, arrify(poly,t), arrify(poly.diff(T),t) )
				for t in arg_extremes
			]
		minimum,maximum = extrema(past)
		assert_allclose(minimum,arrify(poly,arg_extremes[1]))
		assert_allclose(maximum,arrify(poly,arg_extremes[0]))
	
	def test_extrema_in_last_step(self):
		n = 10
		past = Past([
				(time,np.random.random(n),0.1*np.random.random(n))
				for time in sorted(np.random.uniform(-10,10,3))
			])
		values = np.vstack(
				past.get_recent_state(time)
				for time in np.linspace(past[-2][0],past[-1][0],10000)
			)
		minima,maxima = past.extrema_in_last_step()
		assert_allclose( minima, np.min(values,axis=0), atol=1e-3 )
		assert_allclose( maxima, np.max(values,axis=0), atol=1e-3 )

if __name__ == "__main__":
	unittest.main(buffer=True)

