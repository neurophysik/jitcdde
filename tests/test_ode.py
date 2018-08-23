#!/usr/bin/python3
# -*- coding: utf-8 -*-

import platform
import unittest
from jitcdde import t, y, jitcdde
import numpy as np
from numpy.testing import assert_allclose

if platform.system() == "Windows":
	compile_args = None
else:
	from jitcxde_common import DEFAULT_COMPILE_ARGS
	compile_args = DEFAULT_COMPILE_ARGS+["-g","-UNDEBUG"]

# control values:

# some state on the attractor
y0 = np.array([ -0.00338158, -0.00223185, 0.01524253, -0.00613449 ])

y1 = np.array([
	 0.0011789485114731,
	-0.0021947158873226,
	 0.0195744683782066,
	-0.0057801623466600,
	])

f_of_y0 = np.array([
	0.0045396904008868,
	0.00002265673,
	0.0043665702488807,
	0.000328463955
	])

a  = -0.025794
b1 =  0.0065
b2 =  0.0135
c  =  0.02
k  =  0.128

f = [
	y(0) * ( a-y(0) ) * ( y(0)-1.0 ) - y(1) + k * (y(2) - y(0)),
	b1*y(0) - c*y(1),
	y(2) * ( a-y(2) ) * ( y(2)-1.0 ) - y(3) + k * (y(0) - y(2)),
	b2*y(2) - c*y(3)
	]

class basic_test(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.ODE = jitcdde(f)
	
	def setUp(self):
		self.ODE.purge_past()
		self.ODE.add_past_point(-1.0, y0, f_of_y0)
		self.ODE.add_past_point( 0.0, y0, f_of_y0)
	
	def test_C(self):
		self.ODE.compile_C(extra_compile_args=compile_args)
	
	def test_Python(self):
		self.ODE.generate_lambdas()
	
	def integrate(self):
		pass
	
	def tearDown(self):
		self.integrate()
		assert_allclose( self.ODE.integrate(1.0), y1, rtol=1e-5 )

class blind_integration(basic_test):
	def integrate(self):
		self.ODE.integrate_blindly(0.98,0.01)

class no_derivative(basic_test):
	def setUp(self):
		self.ODE.purge_past()
		self.ODE.add_past_point(-1.0, y0, np.random.random(4))
		self.ODE.add_past_point( 0.0, y0, np.random.random(4))
	
	def integrate(self):
		self.ODE.step_on_discontinuities()

class no_derivative_blind(basic_test):
	def integrate(self):
		self.ODE.integrate_blindly(0.0)

tiny_delay = 1e-15
f_with_tiny_delay = [
	y(0,t-tiny_delay) * ( a-y(0) ) * ( y(0)-1.0 ) - y(1) + k * (y(2) - y(0)),
	b1*y(0) - c*y(1),
	y(2) * ( a-y(2) ) * ( y(2)-1.0 ) - y(3) + k * (y(0) - y(2)),
	b2*y(2) - c*y(3)
	]

class TestTinyDelay(basic_test):
	@classmethod
	def setUpClass(self):
		self.ODE = jitcdde(f_with_tiny_delay)

class blind_integration_and_tiny_delay(TestTinyDelay):
	def integrate(self):
		self.ODE.integrate_blindly(0.98,0.01)



if __name__ == "__main__":
	unittest.main(buffer=True)
