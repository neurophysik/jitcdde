#!/usr/bin/python3
# -*- coding: utf-8 -*-

from jitcdde import (
		jitcdde,
		t, y,
		UnsuccessfulIntegration,
		_find_max_delay, _get_delays,
		quadrature,
		test,
		input, jitcdde_input,
	)
import platform
import symengine
import numpy as np
from numpy.testing import assert_allclose
import unittest
from itertools import combinations

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
n = len(f)

def get_past_points():
	data = np.loadtxt("two_Roessler_past.dat")
	for point in data:
		yield (point[0], np.array(point[1:7]), np.array(point[7:13]))

y_10_ref = np.loadtxt("two_Roessler_y10.dat")
T = 10

test_parameters = {
		"rtol": 1e-4,
		"atol": 1e-7,
		"pws_rtol": 1e-3,
		"pws_atol": 1e-3,
		"first_step": 30,
		"max_step": 100,
		"min_step": 1e-16,
	}

tolerance = {
		"rtol": 10*test_parameters["rtol"],
		"atol": 10*test_parameters["atol"],
	}

class TestIntegration(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.DDE = jitcdde(f)
		self.DDE.set_integration_parameters(**test_parameters)
	
	def generator(self):
		self.DDE.compile_C(extra_compile_args=compile_args)
	
	def setUp(self):
		self.DDE.add_past_points(get_past_points())
		self.y_10 = None
		if self.DDE.f_sym():
			self.DDE.check()
		self.generator()
		self.DDE.initial_discontinuities_handled = True
	
	def assert_consistency_with_previous(self, value):
		if self.y_10 is None:
			self.y_10 = value
		else:
			self.assertEqual(value, self.y_10)
	
	def test_integration(self):
		for time in np.linspace(0,T,10,endpoint=True):
			value = self.DDE.integrate(time)
		np.savetxt("two_Roessler_y10_new.dat",value,fmt="%.20f")
		assert_allclose(value, y_10_ref, **tolerance)
		self.assert_consistency_with_previous(value)
		
	def test_integration_one_big_step(self):
		value = self.DDE.integrate(T)
		assert_allclose(value, y_10_ref, **tolerance)
		self.assert_consistency_with_previous(value)
	
	def test_integration_with_adjust_diff(self):
		self.DDE.adjust_diff()
		value = self.DDE.integrate(T)
		assert_allclose(value, y_10_ref, **tolerance)
		self.assert_consistency_with_previous(value)
	
	def test_integration_with_zero_jump(self):
		self.DDE.integrate(T/2)
		self.DDE.jump(np.zeros(n),T/2)
		value = self.DDE.integrate(T)
		assert_allclose(value, y_10_ref, **tolerance)
		self.assert_consistency_with_previous(value)
	
	def test_integration_with_annihilating_jumps(self):
		self.DDE.integrate(T/2)
		change = np.random.normal(0,10,n)
		self.DDE.jump(change,T/2)
		self.DDE.jump(-change,T/2)
		value = self.DDE.integrate(T)
		assert_allclose(value, y_10_ref, **tolerance)
		self.assert_consistency_with_previous(value)
	
	def test_tiny_steps(self):
		for time in np.linspace(0.0, T, 10000, endpoint=True):
			value = self.DDE.integrate(time)
		assert_allclose(value, y_10_ref, **tolerance)
		self.assert_consistency_with_previous(value)
		
	def tearDown(self):
		self.DDE.purge_past()

class TestIntegrationLambda(TestIntegration):
	def generator(self):
		self.DDE.generate_lambdas()

class TestIntegrationSaveAndLoad(TestIntegration):
	@classmethod
	def setUpClass(self):
		DDE_orig = jitcdde(f)
		filename = DDE_orig.save_compiled(overwrite=True)
		self.DDE = jitcdde(n=6, module_location=filename, delays=[delay])
		self.DDE.set_integration_parameters(**test_parameters)
	
	def generator(self):
		pass

class TestIntegrationChunks(TestIntegration):
	def generator(self):
		self.DDE.compile_C(chunk_size=1, extra_compile_args=compile_args, simplify=False)

class TestIntegrationOMP(TestIntegration):
	def generator(self):
		self.DDE.compile_C(chunk_size=1, extra_compile_args=compile_args, omp=True, simplify=False)

class TestIntegrationCSE(TestIntegration):
	def generator(self):
		self.DDE.compile_C(do_cse=True, extra_compile_args=compile_args, simplify=False)

tiny_delay = 1e-30
f_with_tiny_delay = [
		omega[0] * (-y(1) - y(2)),
		omega[0] * (y(0) + 0.165 * y(1,t-tiny_delay)),
		omega[0] * (0.2 + y(2) * (y(0) - 10.0)),
		omega[1] * (-y(4) - y(5)) + 0.25 * (y(0,t-delay) - y(3,t-tiny_delay)),
		omega[1] * (y(3) + 0.165 * y(4)),
		omega[1] * (0.2 + y(5) * (y(3) - 10.0))
	]

class TestPastWithinStep(TestIntegration):
	@classmethod
	def setUpClass(self):
		self.DDE = jitcdde(f_with_tiny_delay)
		self.DDE.set_integration_parameters(pws_fuzzy_increase=False, **test_parameters)
	
	# Because the minimum delay must be larger than the jump width:
	def test_integration_with_adjust_diff(self): pass
	def test_integration_with_zero_jump(self): pass
	def test_integration_with_annihilating_jumps(self): pass

class TestPastWithinStepFuzzy(TestIntegration):
	@classmethod
	def setUpClass(self):
		self.DDE = jitcdde(f)
		self.DDE.set_integration_parameters(pws_fuzzy_increase=True, **test_parameters)
	
	# Because the minimum delay must be larger than the jump width:
	def test_integration_with_adjust_diff(self): pass
	def test_integration_with_zero_jump(self): pass
	def test_integration_with_annihilating_jumps(self): pass

class TestPastWithinStepLambda(TestPastWithinStep):
	def generator(self):
		self.DDE.generate_lambdas(simplify=False)

class TestPastWithinStepFuzzyLambda(TestPastWithinStepFuzzy):
	def generator(self):
		self.DDE.generate_lambdas(simplify=False)


def f_generator():
	yield omega[0] * (-y(1) - y(2))
	yield omega[0] * (y(0) + 0.165 * y(1))
	yield omega[0] * (0.2 + y(2) * (y(0) - 10.0))
	yield omega[1] * (-y(4) - y(5)) + 0.25  * (y(0,t-delay) - y(3))
	yield omega[1] * (y(3) + 0.165 * y(4))
	yield omega[1] * (0.2 + y(5) * (y(3) - 10.0))

class TestGenerator(TestIntegration):
	@classmethod
	def setUpClass(self):
		self.DDE = jitcdde(f_generator)
		self.DDE.set_integration_parameters(**test_parameters)

class TestGeneratorPython(TestGenerator):
	def generator(self):
		self.DDE.generate_lambdas(simplify=False)

class TestGeneratorChunking(TestGenerator):
	def generator(self):
		self.DDE.compile_C(chunk_size=1, extra_compile_args=compile_args, simplify=False)


f_dict = { y(i):entry for i,entry in enumerate(f) }

class TestDictionary(TestIntegration):
	@classmethod
	def setUpClass(self):
		self.DDE = jitcdde(f_dict)
		self.DDE.set_integration_parameters(**test_parameters)


delayed_y, y3m10, coupling_term = symengine.symbols("delayed_y y3m10 coupling_term")
f_alt_helpers = [
		(delayed_y, y(0,t-delay)),
		(coupling_term, 0.25 * (delayed_y - y(3))),
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
		self.DDE = jitcdde(f_alt, helpers=f_alt_helpers)
		self.DDE.set_integration_parameters(**test_parameters)

class TestHelpersPython(TestHelpers):
	def generator(self):
		self.DDE.generate_lambdas(simplify=False)

class TestHelpersChunking(TestHelpers):
	def generator(self):
		self.DDE.compile_C(chunk_size=1, extra_compile_args=compile_args, simplify=False)


first_component = lambda y: omega[0] * (-y[1] - y[2])
def fourth_component(y,y0_delayed):
	return omega[1] * (-y[4] - y[5]) + 0.25 * (y0_delayed - y[3])
minus_ten = lambda y,x: x-10.0
call_first_component = symengine.Function("call_first")
call_fourth_component = symengine.Function("call_fourth")
call_minus_ten = symengine.Function("call_minus_ten")

callbacks = [
	( call_first_component, first_component, 0),
	( call_minus_ten, minus_ten, 1 ),
	( call_fourth_component, fourth_component, 1),
]

f_callback = [
		omega[0] * (-y(1) - y(2)),
		omega[0] * (y(0) + 0.165 * y(1)),
		omega[0] * (0.2 + y(2) * call_minus_ten(y(0))),
		call_fourth_component(y(0,t-delay)),
		omega[1] * (y(3) + 0.165 * y(4)),
		omega[1] * (0.2 + y(5) * call_minus_ten(y(3))),
	]

class TestCallback(TestIntegration):
	@classmethod
	def setUpClass(self):
		self.DDE = jitcdde(f_callback,callback_functions=callbacks)
		self.DDE.set_integration_parameters(**test_parameters)

a,b,c,k,tau = symengine.symbols("a b c k tau")
parameters = [0.165, 0.2, 10.0, 0.25, 4.5]
f_params = [
		omega[0] * (-y(1) - y(2)),
		omega[0] * (y(0) + a * y(1)),
		omega[0] * (b + y(2) * (y(0) - c)),
		omega[1] * (-y(4) - y(5)) + k * (y(0,t-tau) - y(3)),
		omega[1] * (y(3) + a * y(4)),
		omega[1] * (b + y(5) * (y(3) - c))
	]

class TestParameters(TestIntegration):
	@classmethod
	def setUpClass(self):
		self.DDE = jitcdde(f_params, control_pars=[a,b,c,k,tau],max_delay=parameters[-1])
		self.DDE.set_integration_parameters(**test_parameters)
	
	def generator(self):
		self.DDE.compile_C(chunk_size=1, extra_compile_args=compile_args, simplify=False)
		self.DDE.set_parameters(*parameters)

class TestParametersPython(TestParameters):
	def generator(self):
		self.DDE.generate_lambdas(simplify=False)
		self.DDE.set_parameters(*parameters)

class TestParametersList(TestParameters):
	def generator(self):
		self.DDE.compile_C(chunk_size=1, extra_compile_args=compile_args, simplify=False)
		self.DDE.set_parameters(parameters)



class TestIntegrationParameters(unittest.TestCase):
	def setUp(self):
		self.DDE = jitcdde(f)
		self.DDE.add_past_points(get_past_points())
		self.DDE.compile_C(extra_compile_args=compile_args, simplify=False)
		
	def test_min_step_error(self):
		self.DDE.set_integration_parameters(min_step=1.0)
		with self.assertRaises(UnsuccessfulIntegration):
			self.DDE.integrate(1000)
	
	def test_rtol_error(self):
		self.DDE.set_integration_parameters(min_step=1e-3, rtol=1e-10, atol=0)
		with self.assertRaises(UnsuccessfulIntegration):
			self.DDE.integrate(1000)
	
	def test_atol_error(self):
		self.DDE.set_integration_parameters(min_step=1e-3, rtol=0, atol=1e-10)
		with self.assertRaises(UnsuccessfulIntegration):
			self.DDE.integrate(1000)

class TestJump(unittest.TestCase):
	def test_jump(self):
		DDE = jitcdde(f)
		DDE.set_integration_parameters(**test_parameters)
		DDE.add_past_points(get_past_points())
		DDE.compile_C(extra_compile_args=compile_args, simplify=False)
		old_state = DDE.integrate(T)
		
		width = 1e-5
		change = np.random.normal(0,5,n)
		DDE.jump(change,T,width)
		past = DDE.get_state()
		
		assert past[-2][0] == T
		assert past[-1][0] == T+width
		assert_allclose( past[-2][1], old_state )
		assert_allclose( past[-1][1], old_state+change, rtol=1e-3 )

class TestFindMaxDelay(unittest.TestCase):
	def test_default(self):
		self.assertEqual(_find_max_delay(_get_delays(f_generator)), delay)
	
	def test_helpers(self):
		self.assertEqual(_find_max_delay(_get_delays(lambda:[], f_alt_helpers)), delay)
	
	def test_time_dependent_delay(self):
		g = lambda: [y(0,2*t)]
		with self.assertRaises(ValueError):
			_find_max_delay(_get_delays(g))
	
	def test_dynamic_dependent_delay(self):
		g = lambda: [y(0,t-y(0))]
		with self.assertRaises(ValueError):
			_find_max_delay(_get_delays(g))

class TestCheck(unittest.TestCase):
	def test_check_index_negative(self):
		DDE = jitcdde([y(-1)])
		with self.assertRaises(ValueError):
			DDE.check()
	
	def test_check_index_too_high(self):
		DDE = jitcdde([y(1)])
		with self.assertRaises(ValueError):
			DDE.check()
	
	def test_check_undefined_variable(self):
		x = symengine.symbols("x")
		DDE = jitcdde([x])
		with self.assertRaises(ValueError):
			DDE.check()

class TestQuadrature(unittest.TestCase):
	def test_midpoint(self):
		f = symengine.Function("f")
		t = symengine.Symbol("t")
		control = (f(1)+f(3)+f(5)+f(7)+f(9))*2
		result = quadrature(f(t),t,0,10,nsteps=5,method="midpoint")
		self.assertEqual(control,result)
	
	def test_numeric(self):
		t = symengine.Symbol("t")
		for method in ["gauss","midpoint"]:
			with self.subTest(method=method):
				result = quadrature(t**2,t,0,1,nsteps=100,method=method)
				self.assertAlmostEqual( float(result.n(real=True)), 1/3, places=3 )
	
	def test_no_method(self):
		t = symengine.Symbol("t")
		with self.assertRaises(NotImplementedError):
			quadrature(t**2,t,0,1,method="tai")

class TestInput(unittest.TestCase):
	def setUp(self):
		DDE = jitcdde(f)
		DDE.set_integration_parameters(**test_parameters)
		DDE.compile_C(extra_compile_args=compile_args)
		DDE.add_past_points(get_past_points())
		DDE.integrate(T)
		self.result = DDE.get_state()
	
	def test_input(self):
		combos = [
			combo
			for l in range(1,n)
			for combo in combinations(range(n),l)
		]
		
		for combo in np.random.choice(combos,3,replace=False):
			combo = [3,5]
			substitutions = { y(i):input(i) for i in combo }
			f_input = [expression.subs(substitutions) for expression in f]
			DDE = jitcdde_input(f,self.result)
			DDE.set_integration_parameters(**test_parameters)
			DDE.compile_C(extra_compile_args=compile_args, simplify=False)
			DDE.add_past_points(get_past_points())
			value = DDE.integrate(T)
			assert_allclose(value, y_10_ref, **tolerance)

class TestTest(unittest.TestCase):
	def test_test(self):
		for sympy in [False,True]:
			for omp in [True,False]:
				test(omp=omp,sympy=sympy)

if __name__ == "__main__":
	unittest.main(buffer=True)

