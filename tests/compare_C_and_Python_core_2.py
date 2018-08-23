#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Creates instances of the Python and C core for the same DDE and subjects them to a series of random commands (within a reasonable margin). As both cores should behave identically, the results should not differ – except for details of the numerical implementation, which may cause the occasional deviation.

The argument is the number of runs.
"""

from jitcdde._python_core import dde_integrator as py_dde_integrator
from jitcdde import jitcdde, t, y

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

compare = lambda x,y: assert_allclose(x,y,rtol=1e-7,atol=1e-7)

number_of_runs = int(argv[1])

past = [
	( 0.0, np.array([random.random()]), np.array([random.random()]) ),
	( 0.5, np.array([random.random()]), np.array([random.random()]) ),
	( 2.0, np.array([random.random()]), np.array([random.random()]) )
	]

tau = 15
p = 10

tiny_delay = 1e-30
def f():
	yield 0.25 * y(0,t-tau) / (1.0 + y(0,t-tau)**p) - 0.1*y(0,t-tiny_delay)
past_calls = 3

def past_points():
	return [anchor for anchor in past]

errors = 0

for realisation in range(number_of_runs):
	print(".", end="")
	stdout.flush()
	
	P = py_dde_integrator(f, past_points())
	
	DDE = jitcdde(f)
	DDE.compile_C()
	C = DDE.jitced.dde_integrator(past_points())
	
	def get_next_step():
		r = random.uniform(1e-5,1)
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
		P.forget(tau)
		C.forget(tau)
	
	def check_new_y_diff():
		r = 10**random.uniform(-10,-5)
		q = 10**random.uniform(-10,-5)
		compare(P.check_new_y_diff(r,q), C.check_new_y_diff(r,q))
	
	def past_within_step():
		compare(P.past_within_step, C.past_within_step)
	
	def adjust_diff():
		accept_step()
		shift_ratio = np.random.uniform(0,1)
		P.adjust_diff(shift_ratio)
		C.adjust_diff(shift_ratio)
	
	get_next_step()
	get_next_step()
	
	actions = [get_next_step, get_t, get_recent_state, get_current_state, get_full_state, get_p, accept_step, forget, check_new_y_diff, past_within_step, adjust_diff]
	
	for i in range(30):
		action = random.sample(actions,1)[0]
		try:
			action()
		except AssertionError as error:
			print("--------------------")
			print("Results did not match in realisation %i in action %i:" % (realisation, i))
			print(action.__name__)
			print("--------------------")
			
			errors += 1
			break

print("Runs with errors: %i / %i" % (errors, number_of_runs))
