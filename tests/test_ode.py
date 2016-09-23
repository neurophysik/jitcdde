#!/usr/bin/python
# -*- coding: utf-8 -*-

from jitcdde import (
	provide_basic_symbols,
	jitcdde,
	UnsuccessfulIntegration,
	_find_max_delay
	)
import numpy as np
from numpy.testing import assert_allclose
import unittest
from sympy import symbols

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

t, y = provide_basic_symbols()

f = [
	y(0) * ( a-y(0) ) * ( y(0)-1.0 ) - y(1) + k * (y(2) - y(0)),
	b1*y(0) - c*y(1),
	y(2) * ( a-y(2) ) * ( y(2)-1.0 ) - y(3) + k * (y(0) - y(2)),
	b2*y(2) - c*y(3)
	]

class basic_test(unittest.TestCase):
	def setUp(self):
		self.ODE = jitcdde(f, max_delay=10.0)
		self.ODE.add_past_point(-1.0, y0, f_of_y0)
		self.ODE.add_past_point( 0.0, y0, f_of_y0)
		self.ODE.set_integration_parameters()
	
	def test_C(self):
		self.ODE.generate_f_C()
	
	def test_Python(self):
		self.ODE.generate_f_lambda()
	
	def tearDown(self):
		assert_allclose( self.ODE.integrate(1.0), y1, rtol=1e-5 )

unittest.main(buffer=True)
