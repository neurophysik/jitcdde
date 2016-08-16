#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from jitcdde import provide_advanced_symbols, jitcdde, UnsuccessfulIntegration

import sympy
import numpy as np
#from numpy.testing import assert_allclose
import unittest


tau = 15
p = 10
t, y, current_y, past_y, anchors = provide_advanced_symbols()
f = [0.25 * y(0,t-tau) / (1.0 + y(0,t-tau)**p) - 0.1*y(0,t)]

def get_past_points():
	data = np.loadtxt("mackey-glass_past.dat")
	for point in data:
		yield (point[0], np.array([point[1]]), np.array([point[2]]))

y_99_ref = 1.2538230733065612

test_parameters = {
	"max_delay": tau,
	"raise_exception": True,
	"rtol": 1e-7,
	"pws_rtol": 1e-7,
	"first_step": 20,
	"max_step": 100,
	"pws_rtol": 1e-7,
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
		self.y_99 = None
	
	def assert_consistency_with_previous(self, value):
		if self.y_99 is None:
			self.y_99 = value
		else:
			self.assertEqual(value, self.y_99)
	
	def test_integration(self):
		for t in range(100):
			value = self.DDE.integrate(t)
		self.assertAlmostEqual(float(value), y_99_ref)
		self.assert_consistency_with_previous(value)
		
	def test_integration_one_big_step(self):
		value = self.DDE.integrate(99.0)
		self.assertAlmostEqual(float(value), y_99_ref)
		self.assert_consistency_with_previous(value)
	
	def test_tiny_steps(self):
		for t in np.linspace(0.0, 99.0, 10000, endpoint=True):
			value = self.DDE.integrate(t)
		self.assertAlmostEqual(float(value), y_99_ref)
		self.assert_consistency_with_previous(value)
		
	def tearDown(self):
		self.DDE.past = []

tiny_delay = 1e-30
f_with_tiny_delay = [0.25 * y(0,t-tau) / (1.0 + y(0,t-tau)**p) - 0.1*y(0,t-tiny_delay)]

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
	yield 0.25 * y(0,t-tau) / (1.0 + y(0,t-tau)**p) - 0.1*y(0,t)

class TestGenerator(TestIntegration):
	@classmethod
	def setUpClass(self):
		self.DDE = jitcdde(f_generator)
		self.DDE.set_integration_parameters(**test_parameters)

delayed_y, denominator, undelayed_term = sympy.symbols("delayed_y denominator undelayed_term")
f_alt_helpers = [
	(denominator, (1.0 + delayed_y**p)),
	(delayed_y, y(0,t-tau)),
	(undelayed_term, - 0.1*y(0,t))
	]
f_alt = [0.25 * delayed_y / denominator + undelayed_term]

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
		self.DDE.set_integration_parameters(max_delay=tau,min_step=1.0)
		self.DDE.integrate(1000)
		self.assertFalse(self.DDE.successful)
	
	def test_min_step_error(self):
		self.DDE.set_integration_parameters(max_delay=tau,min_step=1.0, raise_exception=True)
		with self.assertRaises(UnsuccessfulIntegration):
			self.DDE.integrate(1000)
		self.assertFalse(self.DDE.successful)
	
	def test_rtol_warning(self):
		self.DDE.set_integration_parameters(max_delay=tau,min_step=1e-3, rtol=1e-10, atol=0)
		self.DDE.integrate(1000)
		self.assertFalse(self.DDE.successful)
	
	def test_atol_warning(self):
		self.DDE.set_integration_parameters(max_delay=tau,min_step=1e-3, rtol=0, atol=1e-10)
		self.DDE.integrate(1000)
		self.assertFalse(self.DDE.successful)

class TestPWSParameters(TestIntegrationParameters):
	def setUp(self):
		self.DDE = jitcdde(f_with_tiny_delay)
		for point in get_past_points():
			self.DDE.add_past_point(*point)
		self.DDE.generate_f_lambda()


unittest.main(buffer=True)

