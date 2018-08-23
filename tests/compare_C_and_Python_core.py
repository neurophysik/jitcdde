#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Creates instances of the Python and C core for the same DDE and subjects them to a series of random commands (within a reasonable margin). As both cores should behave identically, the results should not differ – except for details of the numerical implementation, which may cause the occasional deviation due to the chaoticity of the Rössler oscillators used for testing.

The argument is the number of runs.
"""

from jitcdde._python_core import dde_integrator as py_dde_integrator
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
		assert_allclose(x,y,rtol=1e-5,atol=1e-5)
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

seed = np.random.randint(0,100000)
def past():
	np.random.seed(seed)
	return [
	(np.random.uniform(-10, -5), np.random.random(n), np.random.random(n)),
	(0.0                       , np.random.random(n), np.random.random(n))
	]


errors = 0

for realisation in range(number_of_runs):
	print(".", end="")
	stdout.flush()
	
	tangent_indices = random.sample(range(n),random.randint(1,n-1))
	
	P = py_dde_integrator(
			f,
			past(),
			n_basic = 2,
			tangent_indices = tangent_indices
		)
	
	DDE = jitcdde(f)
	DDE.n_basic = 2
	DDE.tangent_indices = tangent_indices
	DDE.compile_C(chunk_size=random.randint(0,7))
	C = DDE.jitced.dde_integrator(past())
	
	def get_next_step():
		r = random.uniform(1e-5,1e-3)
		P.get_next_step(r)
		C.get_next_step(r)
	
	def get_t():
		compare(P.get_t(), C.get_t())
	
	def get_recent_state():
		time = P.get_t()+random.uniform(-0.1, 0.1)
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
		r = 10**random.uniform(-10,-5)
		q = 10**random.uniform(-10,-5)
		compare(P.get_p(r,q), C.get_p(r,q))
	
	def accept_step():
		P.accept_step()
		C.accept_step()
	
	def forget():
		P.forget(delay)
		C.forget(delay)
	
	def check_new_y_diff():
		r = 10**random.uniform(-10,-5)
		q = 10**random.uniform(-10,-5)
		compare(P.check_new_y_diff(r,q), C.check_new_y_diff(r,q))
	
	def past_within_step():
		compare(P.past_within_step, C.past_within_step)
	
	def orthonormalise():
		d = np.random.uniform(0.1*delay, delay)
		compare(P.orthonormalise(3, d), C.orthonormalise(3, d))
	
	def remove_projections():
		if np.random.random()>0.1:
			
			d = np.random.uniform(0.1*delay, delay)
			if np.random.randint(0,2):
				vector = tuple(np.random.uniform(-1,1,(2,2)))
			else:
				vector = tuple(np.random.randint(-1,2,(2,2)).astype(float))
			
			A = P.remove_projections(d,[vector])
			B = C.remove_projections(d,[vector])
			compare(A , B)
		
		else:
			i = np.random.randint(0,2)
			if np.random.randint(0,2):
				P.remove_state_component(i)
				C.remove_state_component(i)
			else:
				P.remove_diff_component(i)
				C.remove_diff_component(i)
	
	def normalise_indices():
		d = np.random.uniform(0.1*delay, delay)
		compare(P.normalise_indices(d), C.normalise_indices(d))
	
	def adjust_diff():
		accept_step()
		shift_ratio = np.random.uniform(0,1)
		P.adjust_diff(shift_ratio)
		C.adjust_diff(shift_ratio)
	
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
			adjust_diff
		]
	
	for i in range(30):
		action = random.sample(actions,1)[0]
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

