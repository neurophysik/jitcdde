#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Creates instances of the Python and C core for the same DDE and subjects them to a series of random commands (within a reasonable margin). As both cores should behave identically, the results should not differ – except for details of the numerical implementation, which may cause the occasional deviation.

The argument is the number of runs.
"""

from __future__ import print_function
from jitcdde._python_core import dde_integrator as py_dde_integrator
from jitcdde._jitcdde import t, y, current_y, past_y, anchors
from jitcxde_common import (
	ensure_suffix, count_up,
	get_module_path, modulename_from_path, find_and_load_module, module_from_path,
	render_and_write_code,
	render_template,
	)

import sympy
import numpy as np
from numpy.testing import assert_allclose
import random
from setuptools import setup, Extension
from sys import version_info, modules, argv, stdout
from os import path as path
from tempfile import mkdtemp
from jinja2 import Environment, FileSystemLoader

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


tmpdir = None
def tmpfile(filename=None):
	global tmpdir
	tmpdir = tmpdir or mkdtemp()
	
	if filename is None:
		return tmpdir
	else:
		return path.join(tmpdir, filename)

modulename = "jitced"

errors = 0

for realisation in range(number_of_runs):
	print(".", end="")
	stdout.flush()
	
	P = py_dde_integrator(f, past_points())

	set_dy = sympy.Function("set_dy")
	render_and_write_code(
		(set_dy(i,entry) for i,entry in enumerate(f())),
		tmpfile,
		"f",
		["set_dy","current_y","past_y","anchors"],
		chunk_size = 100,
		)
		
	modulename = count_up(modulename)

	render_template(
		"jitced_template.c",
		tmpfile(modulename + ".c"),
		folder = path.join(path.dirname(__file__),"..","jitcdde"),
		n = 1,
		module_name = modulename,
		Python_version = version_info[0],
		has_any_helpers = False,
		number_of_helpers = 0,
		number_of_anchor_helpers = 0,
		anchor_mem_length = past_calls,
		n_basic = 1
		)

	setup(
		name = modulename,
		ext_modules = [Extension(
			modulename,
			sources = [tmpfile(modulename + ".c")],
			extra_compile_args = ["-g", "-UNDEBUG", "-Wno-unknown-pragmas", "-O2", "-std=c11"]
			)],
		script_args = ["build_ext","--build-lib",tmpfile(),"--build-temp",tmpfile(),"--force",],
		verbose = False
		)
	
	jitced = find_and_load_module(modulename,tmpfile())
	C = jitced.dde_integrator(past_points())
	
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
	
	get_next_step()
	get_next_step()
	
	actions = [get_next_step, get_t, get_recent_state, get_current_state, get_full_state, get_p, accept_step, forget, check_new_y_diff, past_within_step]
	
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
