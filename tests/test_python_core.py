#!/usr/bin/python3
# -*- coding: utf-8 -*-

from jitcdde._python_core import dde_integrator
from jitcdde.past import Past
from jitcdde._jitcdde import t, y, current_y, past_y, anchors

import symengine
import numpy as np
from numpy.testing import assert_allclose
from itertools import chain
import unittest


tau = 15
p = 10

def f():
	yield 0.25 * y(0,t-tau) / (1.0 + y(0,t-tau)**p) - 0.1*y(0,t)

y0 = 0.8
dy0 = -0.0794952762375263
past = Past(1,[
		( -1.0, np.array([y0-dy0]), np.array([dy0])),
		(  0.0, np.array([y0    ]), np.array([dy0])),
	])

expected_y = 0.724447497727209
expected_error = -1.34096023725590e-5

class integration_test(unittest.TestCase):
	def setUp(self):
		self.DDE = dde_integrator(f, past)
	
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

delayed_y = symengine.Symbol("delayed_y")
f_alt_helpers = [(delayed_y, y(0,t-tau))]
def f_alt():
	yield 0.25 * delayed_y / (1.0 + delayed_y**p) - 0.1*y(0,t)

class integration_test_with_helpers(integration_test):
	def setUp(self):
		self.DDE = dde_integrator(f_alt, past, f_alt_helpers)

class double_integration_test(unittest.TestCase):
	def test_integration(self):
		double_past = Past(2*past.n)
		for entry in past:
			double_past.append((
				entry[0],
				np.hstack((entry[1],entry[1])),
				np.hstack((entry[2],entry[2]))
				))
		
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


class jump_test(unittest.TestCase):
	def test_jump(self):
		n = 10
		τ = 10
		width = 10**np.random.uniform(-7,0)*τ
		factor = np.random.random(n)
		
		times = sorted(np.random.uniform(0,τ,5))
		past = Past(n, [
				(time,np.random.random(n),0.1*np.random.random(n))
				for time in times
			])
		jump_time = np.random.uniform(past[-2][0],past[-1][0])
		jump_size = np.random.random(n)
		
		# Use a derivative for which it is clear how a jump affects it:
		f = lambda: [ factor[i]*y(i) + y(i,t-τ) for i in range(n) ]
		self.DDE = dde_integrator(f,past)
		
		state = self.DDE.get_recent_state(jump_time+width)
		derivative = self.DDE.eval_f(jump_time+width,state)
		
		self.DDE.apply_jump(jump_size,jump_time,width)
		
		assert self.DDE[-1][0] == jump_time+width
		assert_allclose( self.DDE[-1][1], state+jump_size )
		assert_allclose( self.DDE[-1][2], derivative+factor*jump_size )

if __name__ == "__main__":
	unittest.main(buffer=True)

