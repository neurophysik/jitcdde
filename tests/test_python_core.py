#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from jitcdde._python_core import dde_integrator
from jitcdde._jitcdde import provide_advanced_symbols

import sympy
import numpy as np
#from numpy.testing import assert_allclose
from itertools import chain
import unittest

m = 4

coeff = np.random.random((m,4))
poly = [lambda x, j=j: sum(coeff[j]*(x**np.arange(4))) for j in range(m)]
diff = [lambda x, j=j: sum(np.arange(1,4)*coeff[j,1:]*(x**np.arange(3))) for j in range(m)]

assert(poly[0](1.0) != poly[1](1.0))
assert(diff[0](1.0) != diff[1](1.0))

past = [
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
	]

class interpolation_test(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.DDE = dde_integrator(lambda: [], past)
	
	def test_anchors(self):
		for s in range(len(past)-1):
			t = past[s][0]
			anchors = (past[s],past[s+1])
			for j in range(m):
				value = self.DDE.get_past_value(t, j, anchors)
				self.assertAlmostEqual(past[s][1][j], value)
	
	def test_interpolation(self):
		anchors = (past[0], past[1])
		for t in np.linspace(past[0][0],past[1][0],100):
			for j in range(m):
				value = self.DDE.get_past_value(t, j, anchors)
				self.assertAlmostEqual(poly[j](t), value)

class get_anchors_test(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.DDE = dde_integrator(lambda: [], past)
		self.DDE.anchor_mem = np.ones(1000, dtype=int)
	
	def setUp(self):
		self.DDE.anchor_mem_index = 0
		self.DDE.past_within_step = False
	
	def test_get_anchors(self):
		for s in range(len(past)-1):
			r = np.random.random()
			t = r*past[s][0] + (1-r)*past[s+1][0]
			anchors = self.DDE.get_past_anchors(t)
			self.assertEqual(anchors[0], past[s])
			self.assertEqual(anchors[1], past[s+1])
			self.assertFalse(self.DDE.past_within_step)
	
	def test_too_early(self):
		t = past[0][0] - 1.0
		anchors = self.DDE.get_past_anchors(t)
		self.assertEqual(anchors[0], past[0])
		self.assertEqual(anchors[1], past[1])
		self.assertFalse(self.DDE.past_within_step)

	def test_too_late(self):
		t = past[-1][0] + 1.0
		anchors = self.DDE.get_past_anchors(t)
		self.assertEqual(anchors[0], past[-2])
		self.assertEqual(anchors[1], past[-1])
		self.assertTrue(self.DDE.past_within_step)

class metrics_test(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.DDE = dde_integrator(lambda: [], past[:])
		self.DDE.anchor_mem = np.ones(1000, dtype=int)
		
	def test_compare_norm_with_brute_force(self):
		delay = np.random.uniform(0.0,2.0)
		end = past[-1][0]
		start = end - delay
		
		# Very blunt numerical integration
		N = 100000
		factor = (end-start)/N
		bf_norm_sq = 0
		for t in np.linspace(start,end,N):
			self.DDE.anchor_mem_index = 0
			anchors = self.DDE.get_past_anchors(t)
			for j in range(m):
				bf_norm_sq += self.DDE.get_past_value(t, j, anchors)**2*factor
		
		norm = self.DDE.norm(delay, range(m))
		
		self.assertAlmostEqual(norm, np.sqrt(bf_norm_sq),4)
		
	def test_compare_sp_with_brute_force(self):
		delay = np.random.uniform(0.0,2.0)
		end = past[-1][0]
		start = end - delay
		
		# Very blunt numerical integration
		N = 100000
		factor = (end-start)/N
		bf_sp_sq = 0
		for t in np.linspace(start,end,N):
			self.DDE.anchor_mem_index = 0
			anchors = self.DDE.get_past_anchors(t)
			bf_sp_sq += (
				  self.DDE.get_past_value(t, 0, anchors)
				* self.DDE.get_past_value(t, 2, anchors)
				* factor)
			bf_sp_sq += (
				  self.DDE.get_past_value(t, 1, anchors)
				* self.DDE.get_past_value(t, 3, anchors)
				* factor)
		
		sp = self.DDE.scalar_product(delay, [0,1], [2,3])
		
		self.assertAlmostEqual(sp, bf_sp_sq, 4)


tau = 15
p = 10

t, y, current_y, past_y, anchors = provide_advanced_symbols()
def f():
	yield 0.25 * y(0,t-tau) / (1.0 + y(0,t-tau)**p) - 0.1*y(0,t)

y0 = 0.8
dy0 = -0.0794952762375263
past_generator = lambda: [
		( -1.0, np.array([y0-dy0]), np.array([dy0])),
		(  0.0, np.array([y0    ]), np.array([dy0]))
	]

expected_y = 0.724447497727209
expected_error = -1.34096023725590e-5

class integration_test(unittest.TestCase):
	def setUp(self):
		self.DDE = dde_integrator(f, past_generator())
	
	def test_integration(self):
		self.assertEqual(self.DDE.y[0], y0)
		self.assertEqual(self.DDE.diff[0], dy0)
		self.assertEqual(self.DDE.t, 0.0)
		
		self.DDE.get_next_step(1.0)
		
		self.assertEqual(self.DDE.y[0], y0)
		self.assertEqual(self.DDE.diff[0], dy0)
		self.assertEqual(self.DDE.t, 0.0)
		self.assertAlmostEqual(self.DDE.error[0], expected_error)
		
		self.DDE.accept_step()
		
		self.assertAlmostEqual(self.DDE.y[0], expected_y)
		self.assertEqual(self.DDE.t, 1.0)

delayed_y = sympy.Symbol("delayed_y")
f_alt_helpers = [(delayed_y, y(0,t-tau))]
def f_alt():
	yield 0.25 * delayed_y / (1.0 + delayed_y**p) - 0.1*y(0,t)

class integration_test_with_helpers(integration_test):
	def setUp(self):
		self.DDE = dde_integrator(f_alt, past_generator(), f_alt_helpers)

class double_integration_test(unittest.TestCase):
	def test_integration(self):
		past = past_generator()
		double_past = []
		for entry in past:
			double_past += [(
				entry[0],
				np.hstack((entry[1],entry[1])),
				np.hstack((entry[2],entry[2]))
				)]
		
		self.DDE = dde_integrator(lambda: chain(f(),f()), double_past)
		
		self.assertEqual(self.DDE.y[0], y0)
		self.assertEqual(self.DDE.diff[0], dy0)
		self.assertEqual(self.DDE.y[1], y0)
		self.assertEqual(self.DDE.diff[1], dy0)
		self.assertEqual(self.DDE.t, 0.0)
		
		self.DDE.get_next_step(1.0)
		
		self.assertEqual(self.DDE.y[0], y0)
		self.assertEqual(self.DDE.diff[0], dy0)
		self.assertEqual(self.DDE.y[1], y0)
		self.assertEqual(self.DDE.diff[1], dy0)
		self.assertEqual(self.DDE.t, 0.0)
		self.assertAlmostEqual(self.DDE.error[1], expected_error)
		
		self.DDE.accept_step()
		
		self.assertAlmostEqual(self.DDE.y[0], expected_y)
		self.assertAlmostEqual(self.DDE.y[1], expected_y)
		self.assertEqual(self.DDE.t, 1.0)



unittest.main(buffer=True)

