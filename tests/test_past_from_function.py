#!/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest
import symengine
import numpy as np
from numpy.testing import assert_allclose
from jitcdde import jitcdde, t, y

class TestIntegration(unittest.TestCase):
	def setUp(self):
		double_mackey_glass = [
				.25*y(0,t-15) / (1+y(0,t-15)**10) - 0.1*y(0),
				.25*y(1,t-15) / (1+y(1,t-15)**10) - 0.1*y(1)
				]
		self.DDE = jitcdde(double_mackey_glass)
	
	def test_exp(self):
		f_1 = lambda time: [np.exp(time),0.5*np.exp(2*time)]
		f_2 = [symengine.exp(t),symengine.exp(2*t)/2]
		
		results = []
		for function in (f_1,f_2):
			self.DDE.purge_past()
			self.DDE.past_from_function(function)
			self.DDE.step_on_discontinuities()
			times = np.arange( self.DDE.t, self.DDE.t+1000, 10 )
			result = np.vstack( self.DDE.integrate(time) for time in times )
			results.append(result)
		
		assert_allclose(results[0],results[1],atol=0.01,rtol=1e-5)
	
	def test_polynomial(self):
		C = np.random.uniform(-1,1,6)
		f_1 = lambda time: [C[0]*time**2+C[1]*time+C[2],C[3]*time**2+C[4]*time+C[5]]
		f_1_diff = lambda time: [2*C[0]*time+C[1],2*C[3]*time+C[4]]
		f_2 = [ C[0]*t**2+C[1]*t+C[2], C[3]*t**2+C[4]*t+C[5] ]
		
		results = []
		for function in (f_1,f_2,None):
			self.DDE.purge_past()
			if function is None:
				for time in (np.random.uniform(-20,-1), 0.0):
					self.DDE.add_past_point(time,f_1(time),f_1_diff(time))
			else:
				self.DDE.past_from_function(function)
			self.DDE.step_on_discontinuities()
			times = np.arange( self.DDE.t, self.DDE.t+1000, 10 )
			result = np.vstack( self.DDE.integrate(time) for time in times )
			results.append(result)
		
		assert_allclose(results[0],results[1],atol=0.01,rtol=1e-5)
		assert_allclose(results[0],results[2],atol=0.01,rtol=1e-5)
	
	

if __name__ == "__main__":
	unittest.main(buffer=True)

