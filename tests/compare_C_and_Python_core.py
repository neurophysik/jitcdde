#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Creates instances of the Python and C core for the same DDE and subjects them to a series of random commands (within a reasonable margin). As both cores should behave identically, the results should not differ – except for details of the numerical implementation, which may cause the occasional deviation due to the chaoticity of the Rössler oscillators used for testing.

The argument is the number of runs.
"""

from jitcdde._python_core import dde_integrator as py_dde_integrator
from jitcdde.past import Past
from jitcdde import t, y, jitcdde

import numpy as np
from numpy.testing import assert_allclose
import platform
import random
from sys import argv, stdout

if platform.system() == "Windows":
	compile_args = None
else:
	from jitcxde_common import DEFAULT_COMPILE_ARGS
	compile_args = DEFAULT_COMPILE_ARGS+["-g","-UNDEBUG","-O0"]

class FailedComparison(Exception):
	pass

def compare(x,y):
	try:
		assert_allclose(x,y,rtol=1e-3,atol=1e-5)
	except AssertionError as error:
		print("\n")
		print (x,y)
		raise FailedComparison(error.args[0])

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

for i,realisation in enumerate(range(number_of_runs)):
	print(".", end="")
	stdout.flush()
	
	RNG = np.random.RandomState()
	py_RNG = random.Random(RNG.randint(1000000))
	
	tangent_indices = py_RNG.sample(range(n),RNG.randint(1,n-1))
	past_seed = RNG.randint(1000000)
	
	def past():
		RNG = np.random.RandomState(past_seed)
		result = Past(n=n,n_basic=2,tangent_indices=tangent_indices)
		for time in RNG.uniform(-10,0,RNG.randint(2,10)):
			result.add(( time, RNG.rand(n), 0.1*RNG.rand(n) ))
		return result
	
	P = py_dde_integrator(f,past())
	
	DDE = jitcdde(f)
	DDE.n_basic = 2
	DDE.tangent_indices = tangent_indices
	DDE.compile_C(chunk_size=RNG.randint(0,7),extra_compile_args=compile_args)
	C = DDE.jitced.dde_integrator(past())
	
	def get_next_step():
		r = RNG.uniform(1e-5,1e-3)
		P.get_next_step(r)
		C.get_next_step(r)
	
	def get_t():
		compare(P.get_t(), C.get_t())
	
	def get_recent_state():
		time = P.get_t()+RNG.uniform(-0.1, 0.1)
		compare(P.get_recent_state(time), C.get_recent_state(time))
	
	def get_current_state():
		compare(P.get_current_state(), C.get_current_state())

	def get_full_state():
		A = P.get_full_state()
		B = C.get_full_state()
		for a,b in zip(A,B):
			compare(a[0], b[0])
			compare(a[1], b[1])
			compare(a[2], b[2])

	def get_p():
		r = 10**RNG.uniform(-10,-5)
		q = 10**RNG.uniform(-10,-5)
		compare(P.get_p(r,q), C.get_p(r,q))
	
	def accept_step():
		P.accept_step()
		C.accept_step()
	
	def forget():
		P.forget(delay)
		C.forget(delay)
	
	def check_new_y_diff():
		r = 10**RNG.uniform(-10,-5)
		q = 10**RNG.uniform(-10,-5)
		compare(P.check_new_y_diff(r,q), C.check_new_y_diff(r,q))
	
	def past_within_step():
		compare(P.past_within_step, C.past_within_step)
	
	def orthonormalise():
		d = RNG.uniform(0.1*delay, delay)
		compare(P.orthonormalise(3, d), C.orthonormalise(3, d))
	
	def remove_projections():
		if RNG.rand()>0.1:
			
			d = RNG.uniform(0.1*delay, delay)
			if RNG.randint(0,2):
				vector = tuple(RNG.uniform(-1,1,(2,2)))
			else:
				vector = tuple(RNG.randint(-1,2,(2,2)).astype(float))
			
			A = P.remove_projections(d,[vector])
			B = C.remove_projections(d,[vector])
			compare(A , B)
		
		else:
			i = RNG.randint(0,2)
			if RNG.randint(0,2):
				P.remove_state_component(i)
				C.remove_state_component(i)
			else:
				P.remove_diff_component(i)
				C.remove_diff_component(i)
	
	def normalise_indices():
		d = RNG.uniform(0.1*delay, delay)
		compare(P.normalise_indices(d), C.normalise_indices(d))
	
	def reduced_interval():
		interval   = ( C.get_full_state()[0][0], C.get_full_state()[-1][0] )
		interval_2 = ( P.get_full_state()[0][0], P.get_full_state()[-1][0] )
		assert_allclose(interval,interval_2)
		return (
				0.9*interval[0]+0.1*interval[1],
				0.1*interval[0]+0.9*interval[1],
			)
	
	def truncate_past():
		accept_step()
		time = RNG.uniform(*reduced_interval())
		P.truncate(time)
		C.truncate(time)

	def apply_jump():
		accept_step()
		time = RNG.uniform(*reduced_interval())
		width = 0.1
		change = RNG.normal(0,0.1,n)
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
		action = py_RNG.sample(actions,1)[0]
		try:
			action()
		except FailedComparison as error:
			print("\n--------------------")
			print("Results did not match in realisation %i in action %i:" % (realisation, i))
			print(action.__name__)
			print("--------------------")
			
			errors += 1
			break

print("Runs with errors: %i / %i" % (errors, number_of_runs))

