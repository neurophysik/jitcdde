#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from jitcdde import (
	provide_advanced_symbols,
	jitcdde,
	UnsuccessfulIntegration,
	_find_max_delay
	)

import sympy
import numpy as np
from numpy.testing import assert_allclose
import unittest

t, y, current_y, past_y, anchors = provide_advanced_symbols()

omega = np.array([0.88167179, 0.87768425])
k = 0.25
delay = 4.5

f = [
	omega[0] * (-y(1) - y(2)),
	omega[0] * (y(0) + 0.165 * y(1)),
	omega[0] * (0.2 + y(2) * (y(0) - 10.0)),
	omega[1] * (-y(4) - y(5)) + k * (y(0,t-delay) - y(3)),
	omega[1] * (y(3) + 0.165 * y(4)),
	omega[1] * (0.2 + y(5) * (y(3) - 10.0))
	]

def get_past_points():
	data = np.loadtxt("two_Roessler_past.dat")
	for point in data:
		yield (point[0], np.array(point[1:7]), np.array(point[7:13]))

y_10_ref = np.loadtxt("two_Roessler_y10.dat")
T = 10

test_parameters = {
	"raise_exception": True,
	"rtol": 1e-7,
	"atol": 1e-7,
	"pws_rtol": 1e-3,
	"pws_atol": 1e-3,
	"first_step": 30,
	"max_step": 100,
	"min_step": 1e-30,
	}

class TestIntegration(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.DDE = jitcdde(f)
		self.DDE.set_integration_parameters(**test_parameters)
	
	def setUp(self):
		for point in get_past_points():
			self.DDE.add_past_point(*point)
		self.DDE.generate_f_lambda()
		self.y_10 = None
	
	def assert_consistency_with_previous(self, value):
		if self.y_10 is None:
			self.y_10 = value
		else:
			self.assertEqual(value, self.y_10)
	
	def test_integration(self):
		for t in np.linspace(0, T, 10, endpoint=True):
			value = self.DDE.integrate(t)
		assert_allclose(value, y_10_ref)
		self.assert_consistency_with_previous(value)
		
	def test_integration_one_big_step(self):
		value = self.DDE.integrate(T)
		assert_allclose(value, y_10_ref)
		self.assert_consistency_with_previous(value)
	
	def test_tiny_steps(self):
		for t in np.linspace(0.0, T, 10000, endpoint=True):
			value = self.DDE.integrate(t)
		assert_allclose(value, y_10_ref)
		self.assert_consistency_with_previous(value)
		
	def tearDown(self):
		self.DDE.past = []

tiny_delay = 1e-30
f_with_tiny_delay = [
	omega[0] * (-y(1) - y(2)),
	omega[0] * (y(0) + 0.165 * y(1,t-tiny_delay)),
	omega[0] * (0.2 + y(2) * (y(0) - 10.0)),
	omega[1] * (-y(4) - y(5)) + k * (y(0,t-delay) - y(3,t-tiny_delay)),
	omega[1] * (y(3) + 0.165 * y(4)),
	omega[1] * (0.2 + y(5) * (y(3) - 10.0))
	]

class TestPastWithinStep(TestIntegration):
	@classmethod
	def setUpClass(self):
		self.DDE = jitcdde(f_with_tiny_delay)
		self.DDE.set_integration_parameters(pws_fuzzy_increase=False, **test_parameters)

class TestPastWithinStepFuzzy(TestIntegration):
	@classmethod
	def setUpClass(self):
		self.DDE = jitcdde(f)
		self.DDE.set_integration_parameters(pws_fuzzy_increase=True, **test_parameters)

def f_generator():
	yield omega[0] * (-y(1) - y(2))
	yield omega[0] * (y(0) + 0.165 * y(1,t-tiny_delay))
	yield omega[0] * (0.2 + y(2) * (y(0) - 10.0))
	yield omega[1] * (-y(4) - y(5)) + k * (y(0,t-delay) - y(3))
	yield omega[1] * (y(3) + 0.165 * y(4))
	yield omega[1] * (0.2 + y(5) * (y(3) - 10.0))

class TestGenerator(TestIntegration):
	@classmethod
	def setUpClass(self):
		self.DDE = jitcdde(f_generator)
		self.DDE.set_integration_parameters(**test_parameters)

delayed_y, y3m10, coupling_term = sympy.symbols("delayed_y y3m10 coupling_term")
f_alt_helpers = [
	(delayed_y, y(0,t-delay)),
	(coupling_term, k * (delayed_y - y(3))),
	(y3m10, y(3)-10)
	]

f_alt = [
	omega[0] * (-y(1) - y(2)),
	omega[0] * (y(0) + 0.165 * y(1)),
	omega[0] * (0.2 + y(2) * (y(0) - 10.0)),
	omega[1] * (-y(4) - y(5)) + coupling_term,
	omega[1] * (y3m10 + 10 + 0.165 * y(4)),
	omega[1] * (0.2 + y(5) * y3m10)
	]

class TestHelpers(TestIntegration):
	@classmethod
	def setUpClass(self):
		self.DDE = jitcdde(f_alt, f_alt_helpers)
		self.DDE.set_integration_parameters(**test_parameters)

class TestIntegrationParameters(unittest.TestCase):
	def setUp(self):
		self.DDE = jitcdde(f)
		for point in get_past_points():
			self.DDE.add_past_point(*point)
		self.DDE.generate_f_lambda()
		
	def test_min_step_warning(self):
		self.DDE.set_integration_parameters(min_step=1.0, raise_exception=False)
		self.DDE.integrate(1000)
		self.assertFalse(self.DDE.successful)
	
	def test_min_step_error(self):
		self.DDE.set_integration_parameters(min_step=1.0, raise_exception=True)
		with self.assertRaises(UnsuccessfulIntegration):
			self.DDE.integrate(1000)
		self.assertFalse(self.DDE.successful)
	
	def test_rtol_warning(self):
		self.DDE.set_integration_parameters(min_step=1e-3, rtol=1e-10, atol=0, raise_exception=False)
		self.DDE.integrate(1000)
		self.assertFalse(self.DDE.successful)
	
	def test_atol_warning(self):
		self.DDE.set_integration_parameters(min_step=1e-3, rtol=0, atol=1e-10, raise_exception=False)
		self.DDE.integrate(1000)
		self.assertFalse(self.DDE.successful)

class TestFindMaxDelay(unittest.TestCase):
	def test_default(self):
		self.assertEqual(_find_max_delay(f_generator), delay)
	
	def test_helpers(self):
		self.assertEqual(_find_max_delay(lambda:[], f_alt_helpers), delay)
	
	def test_time_dependent_delay(self):
		g = lambda: [y(0,2*t)]
		with self.assertRaises(ValueError):
			_find_max_delay(g)
	
	def test_dynamic_dependent_delay(self):
		g = lambda: [y(0,t-y(0))]
		with self.assertRaises(ValueError):
			_find_max_delay(g)

unittest.main(buffer=True)

