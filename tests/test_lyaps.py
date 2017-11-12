#!/usr/bin/python3
# -*- coding: utf-8 -*-


import platform
import unittest
import numpy as np
from scipy.stats import sem
from jitcdde import t, y, jitcdde_lyap

if platform.system() == "Windows":
	compile_args = None
else:
	from jitcxde_common import DEFAULT_COMPILE_ARGS
	compile_args = DEFAULT_COMPILE_ARGS+["-g","-UNDEBUG"]

omega = np.array([0.88167179, 0.87768425])
delay = 4.5

f = [
	omega[0] * (-y(1) - y(2)),
	omega[0] * (y(0) + 0.165 * y(1)),
	omega[0] * (0.2 + y(2) * (y(0) - 10.0)),
	omega[1] * (-y(4) - y(5)) + 0.25 * (y(0,t-delay) - y(3)),
	omega[1] * (y(3) + 0.165 * y(4)),
	omega[1] * (0.2 + y(5) * (y(3) - 10.0))
	]

test_parameters = {
	"rtol": 1e-7,
	"atol": 1e-7,
	"pws_rtol": 1e-3,
	"pws_atol": 1e-3,
	"first_step": 30,
	"max_step": 100,
	"min_step": 1e-30,
	}

lyap_controls = [0.0806, 0, -0.0368, -0.1184]

class TestIntegration(unittest.TestCase):
	def setUp(self):
		self.DDE = jitcdde_lyap(f, n_lyap=len(lyap_controls))
		self.DDE.add_past_point(-delay, np.random.random(6), np.random.random(6))
		self.DDE.add_past_point(0.0, np.random.random(6), np.random.random(6))
		self.DDE.set_integration_parameters(**test_parameters)
	
	def test_integrate_blindly(self):
		self.DDE.integrate_blindly(100.0, 0.1)
	
	def test_step_on_discontinuities(self):
		self.DDE.step_on_discontinuities(max_step=0.1)
	
	def tearDown(self):
		lyaps = []
		weights = []
		for T in np.arange(self.DDE.t, self.DDE.t+1000, 10):
			_, lyap, weight = self.DDE.integrate(T)
			lyaps.append(lyap)
			weights.append(weight)
		lyaps = np.vstack(lyaps)
		
		lyap_start = 40
		for i,lyap_control in enumerate(lyap_controls):
			lyap = np.average(lyaps[lyap_start:,i], weights=weights[lyap_start:])
			stderr = sem(lyaps[lyap_start:,i])
			print(lyap,stderr)
			self.assertAlmostEqual(lyap_control, lyap, delta=3*stderr)

class TestSaveAndLoad(TestIntegration):
	def setUp(self):
		DDE_orig = jitcdde_lyap(f, n_lyap=len(lyap_controls))
		filename = DDE_orig.save_compiled(overwrite=True)
		print(filename)
		self.DDE = jitcdde_lyap(
			n=6,
			module_location=filename,
			delays=[delay],
			n_lyap=len(lyap_controls)
			)
		self.DDE.add_past_point(-delay, np.random.random(6), np.random.random(6))
		self.DDE.add_past_point(0.0,    np.random.random(6), np.random.random(6))
		self.DDE.set_integration_parameters(**test_parameters)

class TestOMP(TestIntegration):
	def setUp(self):
		self.DDE = jitcdde_lyap(f, n_lyap=len(lyap_controls))
		self.DDE.add_past_point(-delay, np.random.random(6), np.random.random(6))
		self.DDE.add_past_point(0.0, np.random.random(6), np.random.random(6))
		self.DDE.set_integration_parameters(**test_parameters)
		self.DDE.compile_C(omp=True,chunk_size=15)
	
if __name__ == "__main__":
	unittest.main(buffer=True)

