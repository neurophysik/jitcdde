#!/usr/bin/python3
# -*- coding: utf-8 -*-

from jitcdde.hermite_spline import CubicHermiteSpline, scalar_product_interval, scalar_product_partial, norm_sq_interval, norm_sq_partial, interpolate, interpolate_diff, extrema_from_anchors

import symengine
import numpy as np
from numpy.testing import assert_allclose
import unittest

m = 4

coeff = np.random.random((m,4))
poly = [lambda x, j=j: sum(coeff[j]*(x**np.arange(4))) for j in range(m)]
diff = [lambda x, j=j: sum(np.arange(1,4)*coeff[j,1:]*(x**np.arange(3))) for j in range(m)]

assert poly[0](1.0) != poly[1](1.0)
assert diff[0](1.0) != diff[1](1.0)

spline = CubicHermiteSpline(m, [
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
		for s in range(len(spline)-1):
			t = spline[s][0]
			anchors = (spline[s],spline[s+1])
			for j in range(m):
				self.assertAlmostEqual(
						spline[s][1][j],
						interpolate(t, j, anchors),
					)
				self.assertAlmostEqual(
						spline[s][2][j],
						interpolate_diff(t, j, anchors),
					)
	
	def test_interpolation(self):
		anchors = (spline[0], spline[1])
		for t in np.linspace(spline[0][0],spline[1][0],100):
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
		for s in range(len(spline)-1):
			r = np.random.random()
			t = r*spline[s][0] + (1-r)*spline[s+1][0]
			anchors = spline.get_anchors(t)
			self.assertEqual(anchors[0], spline[s])
			self.assertEqual(anchors[1], spline[s+1])
	
	def test_too_early(self):
		t = spline[0][0] - 1.0
		anchors = spline.get_anchors(t)
		self.assertEqual(anchors[0], spline[0])
		self.assertEqual(anchors[1], spline[1])

	def test_too_late(self):
		t = spline[-1][0] + 1.0
		anchors = spline.get_anchors(t)
		self.assertEqual(anchors[0], spline[-2])
		self.assertEqual(anchors[1], spline[-1])

class metrics_test(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.spline = spline.copy()
	
	def test_compare_norm_with_brute_force(self):
		delay = np.random.uniform(0.0,2.0)
		end = spline[-1][0]
		start = end - delay
		
		# Very blunt numerical integration
		N = 100000
		factor = (end-start)/N
		bf_norm_sq = 0
		for t in np.linspace(start,end,N):
			anchors = self.spline.get_anchors(t)
			for j in range(m):
				bf_norm_sq += interpolate(t, j, anchors)**2*factor
		
		norm = self.spline.norm(delay, np.array(range(m)))
		
		self.assertAlmostEqual(norm, np.sqrt(bf_norm_sq),4)
		
	def test_compare_sp_with_brute_force(self):
		delay = np.random.uniform(0.0,2.0)
		end = spline[-1][0]
		start = end - delay
		
		# Very blunt numerical quadrature
		N = 100000
		factor = (end-start)/N
		bf_sp_sq = 0
		for t in np.linspace(start,end,N):
			anchors = self.spline.get_anchors(t)
			bf_sp_sq += (
				  interpolate(t, 0, anchors)
				* interpolate(t, 2, anchors)
				* factor)
			bf_sp_sq += (
				  interpolate(t, 1, anchors)
				* interpolate(t, 3, anchors)
				* factor)
		
		sp = self.spline.scalar_product(delay, [0,1], [2,3])
		
		self.assertAlmostEqual(sp, bf_sp_sq, 4)
	
	def test_untrue_partials_norms(self):
		for i in range(len(self.spline)-1):
			anchors = (self.spline[i], self.spline[i+1])
			start = np.random.randint(0,m-1)
			length = np.random.randint(1,m-start)
			indizes = list(range(start, start+length))
			norm = norm_sq_interval(anchors, indizes)
			partial_norm = norm_sq_partial(anchors, indizes, anchors[0][0])
			self.assertAlmostEqual(norm, partial_norm)
	
	def test_untrue_partials_sp(self):
		for i in range(len(self.spline)-1):
			anchors = (self.spline[i], self.spline[i+1])
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
		truncation_time = np.random.uniform(spline[-2][0],spline[-1][0])
		truncated_spline = spline.copy()
		truncated_spline.truncate(truncation_time)
		
		assert truncated_spline[-1][0] == truncation_time
		
		anchors = (spline[-2], spline[-1])
		anchors_trunc = (truncated_spline[-2],truncated_spline[-1])
		for t in np.linspace(spline[-2][0],truncation_time,30):
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
	def test_given_extrema(self):
		n = 100
		positions = sorted(np.random.random(2))
		state = np.random.random(n)
		spline = CubicHermiteSpline(n, [
				( positions[0], state                       , np.zeros(n) ),
				( positions[1], state+np.random.uniform(0,5), np.zeros(n) ),
			])
		result = extrema_from_anchors(spline)
		assert_allclose(result.arg_min,spline[0].time)
		assert_allclose(result.arg_max,spline[1].time)
		assert_allclose(result.minima,spline[0][1])
		assert_allclose(result.maxima,spline[1][1])
	
	def test_simple_polynomial(self):
		T = symengine.Symbol("T")
		poly = 2*T**3 - 3*T**2 - 36*T + 17
		arg_extremes = [-2,3]
		arrify = lambda expr,t: np.atleast_1d(float(expr.subs({T:t})))
		spline = CubicHermiteSpline(1, [
				( t, arrify(poly,t), arrify(poly.diff(T),t) )
				for t in arg_extremes
			])
		result = extrema_from_anchors(spline)
		assert_allclose(result.minima,arrify(poly,arg_extremes[1]))
		assert_allclose(result.maxima,arrify(poly,arg_extremes[0]))
		assert_allclose(result.arg_min,arg_extremes[1])
		assert_allclose(result.arg_max,arg_extremes[0])
	
	def test_arbitrary_anchors(self):
		n = 100
		spline = CubicHermiteSpline(n, [
				(time,np.random.normal(0,1,n),np.random.normal(0,0.1,n))
				for time in sorted(np.random.uniform(-10,10,2))
			])
		
		times = np.linspace(spline[0].time,spline[1].time,10000)
		values = np.vstack( spline.get_recent_state(time) for time in times )
		
		result = extrema_from_anchors(spline[-2:])
		assert_allclose( result.minima, np.min(values,axis=0), atol=1e-3 )
		assert_allclose( result.maxima, np.max(values,axis=0), atol=1e-3 )
		assert_allclose( result.arg_min, times[np.argmin(values,axis=0)], atol=1e-3)
		assert_allclose( result.arg_max, times[np.argmax(values,axis=0)], atol=1e-3)
	
	def test_multiple_anchors(self):
		for _ in range(10):
			n = 100
			spline = CubicHermiteSpline(n, [
					(time,np.random.normal(0,1,n),np.random.normal(0,0.1,n))
					for time in sorted(np.random.uniform(-10,10,3))
				])
			
			beginning, end = sorted(np.random.uniform(spline[0].time,spline[-1].time,2))
			times = np.linspace(beginning,end,10000)
			values = np.vstack( spline.get_state(time) for time in times )
			
			result = spline.extrema(beginning,end)
			assert_allclose( result.minima, np.min(values,axis=0), atol=1e-3 )
			assert_allclose( result.maxima, np.max(values,axis=0), atol=1e-3 )
			assert_allclose( result.arg_min, times[np.argmin(values,axis=0)], atol=1e-3 )
			assert_allclose( result.arg_max, times[np.argmax(values,axis=0)], atol=1e-3 )

class TestErrors(unittest.TestCase):
	def test_wrong_shape(self):
		with self.assertRaises(ValueError):
			CubicHermiteSpline( 1, [ (1,[2,3],[4,5]) ] )
	
	def test_wrong_diff_shape(self):
		with self.assertRaises(ValueError):
			CubicHermiteSpline( 1, [ (1,[2],[3,4]) ] )
	
	def test_wrong_time(self):
		with self.assertRaises(ValueError):
			CubicHermiteSpline( 1, [ (0,[0],[0]), (0,[1],[2]) ] )
	
	def test_replace_with_same(self):
		spline = CubicHermiteSpline( 1, [(0,[0],[0])])
		spline[-1] = (0,[1],[2])


if __name__ == "__main__":
	unittest.main(buffer=True)

