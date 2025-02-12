"""
Creates instances of the Python and C core for the same DDE and subjects them to a series of random commands (within a reasonable margin). As both cores should behave identically, the results should not differ – except for details of the numerical implementation, which may cause the occasional deviation due to the chaoticity of the Rössler oscillators used for testing.

The argument is the number of runs.
"""

import platform
import random
from sys import argv, stdout

import numpy as np
from numpy.testing import assert_allclose

from jitcdde import jitcdde, t, y
from jitcdde._python_core import dde_integrator as py_dde_integrator
from jitcdde.past import Past


if platform.system() == "Windows":
	compile_args = None
else:
	from jitcxde_common import DEFAULT_COMPILE_ARGS
	compile_args = [*DEFAULT_COMPILE_ARGS,"-g","-UNDEBUG","-O1"]

class FailedComparison(Exception):
	pass

def compare(x,y):
	try:
		assert_allclose(x,y,rtol=1e-3,atol=1e-5)
	except AssertionError as error:
		print("\n")
		print (x,y)
		raise FailedComparison(error.args[0]) from error

number_of_runs = int(argv[1])

omega = np.array([0.88167179, 0.87768425])
k = 0.25
delay = 4.5

def f():
	yield omega[0] * (-y(1) - y(2))
	yield omega[0] * (y(0) + 0.165 * y(1))
	yield omega[0] * (0.2 + y(2) * (y(0) - 10.0))
	yield omega[1] * (-y(4) - y(5)) + k * (y(0,t-delay) - y(3))
	yield omega[1] * (y(3) + 0.165 * y(4))
	yield omega[1] * (0.2 + y(5) * (y(3) - 10.0))
	yield 0
	yield 0

n = 8

errors = 0
rng = np.random.default_rng(seed=42)
pyrng = random.Random(int(rng.integers(1_000_000)))

for i,realisation in enumerate(range(number_of_runs)):
	print(".", end="")
	stdout.flush()
	
	tangent_indices = pyrng.sample(range(n),rng.integers(1,n-1))
	past_seed = rng.integers(1000000)
	
	def past(past_seed=past_seed, tangent_indices=tangent_indices):
		lrng = np.random.default_rng(past_seed)
		result = Past(n=n,n_basic=2,tangent_indices=tangent_indices)
		for time in lrng.uniform(-10,0,lrng.integers(2,10)):
			result.add(( time, lrng.random(n), 0.1*lrng.random(n) ))
		return result
	
	P = py_dde_integrator(f,past())
	
	DDE = jitcdde(f)
	DDE.n_basic = 2
	DDE.tangent_indices = tangent_indices
	DDE.compile_C(chunk_size=rng.integers(0,7),extra_compile_args=compile_args)
	C = DDE.jitced.dde_integrator(past())
	
	def get_next_step(P=P, C=C):
		r = rng.uniform(1e-5,1e-3)
		P.get_next_step(r)
		C.get_next_step(r)
	
	def get_t(P=P, C=C):
		compare(P.get_t(), C.get_t())
	
	def get_recent_state(P=P, C=C):
		time = P.get_t()+rng.uniform(-0.1, 0.1)
		compare(P.get_recent_state(time), C.get_recent_state(time))
	
	def get_current_state(P=P, C=C):
		compare(P.get_current_state(), C.get_current_state())

	def get_full_state(P=P, C=C):
		A = P.get_full_state()
		B = C.get_full_state()
		for a,b in zip(A,B, strict=True):
			compare(a[0], b[0])
			compare(a[1], b[1])
			compare(a[2], b[2])

	def get_p(P=P, C=C):
		r = 10**rng.uniform(-10,-5)
		q = 10**rng.uniform(-10,-5)
		compare(P.get_p(r,q), C.get_p(r,q))
	
	def accept_step(P=P, C=C):
		P.accept_step()
		C.accept_step()
	
	def forget(P=P, C=C):
		P.forget(delay)
		C.forget(delay)
	
	def check_new_y_diff(P=P, C=C):
		r = 10**rng.uniform(-10,-5)
		q = 10**rng.uniform(-10,-5)
		compare(P.check_new_y_diff(r,q), C.check_new_y_diff(r,q))
	
	def past_within_step(P=P, C=C):
		compare(P.past_within_step, C.past_within_step)
	
	def orthonormalise(P=P, C=C):
		d = rng.uniform(0.1*delay, delay)
		compare(P.orthonormalise(3, d), C.orthonormalise(3, d))
	
	def remove_projections(P=P, C=C):
		if rng.random()>0.1:
			
			d = rng.uniform(0.1*delay, delay)
			if rng.integers(0,2):
				vector = tuple(rng.uniform(-1,1,(2,2)))
			else:
				vector = tuple(rng.integers(-1,2,(2,2)).astype(float))
			
			A = P.remove_projections(d,[vector])
			B = C.remove_projections(d,[vector])
			compare(A , B)
		
		else:
			i = rng.integers(0,2)
			if rng.integers(0,2):
				P.remove_state_component(i)
				C.remove_state_component(i)
			else:
				P.remove_diff_component(i)
				C.remove_diff_component(i)
	
	def normalise_indices(P=P, C=C):
		d = rng.uniform(0.1*delay, delay)
		compare(P.normalise_indices(d), C.normalise_indices(d))
	
	def reduced_interval(P=P, C=C):
		interval   = ( C.get_full_state()[0][0], C.get_full_state()[-1][0] )
		interval_2 = ( P.get_full_state()[0][0], P.get_full_state()[-1][0] )
		assert_allclose(interval,interval_2)
		return (
				0.9*interval[0]+0.1*interval[1],
				0.1*interval[0]+0.9*interval[1],
			)
	
	def truncate_past(P=P, C=C):
		accept_step()
		time = rng.uniform(*reduced_interval())
		P.truncate(time)
		C.truncate(time)

	def apply_jump(P=P, C=C):
		accept_step()
		time = rng.uniform(*reduced_interval())
		width = 0.1
		change = rng.normal(0,0.1,n)
		compare(
				P.apply_jump(change,time,width),
				C.apply_jump(change,time,width),
			)
	
	get_next_step()
	get_next_step()
	
	actions = [
			get_next_step,
			get_t,
			get_recent_state,
			get_current_state,
			get_full_state,
			get_p,
			accept_step,
			forget,
			check_new_y_diff,
			past_within_step,
			orthonormalise,
			remove_projections,
			normalise_indices,
			truncate_past,
			apply_jump,
		]
	
	for i in range(10):
		action = pyrng.sample(actions,1)[0]
		try:
			action()
		except FailedComparison:
			print("\n--------------------")
			print(f"Results did not match in realisation {realisation} in action {i}:")
			print(action.__name__)
			print("--------------------")
			
			errors += 1
			break

print(f"Runs with errors: {errors} / {number_of_runs}")
raise SystemExit(errors)
