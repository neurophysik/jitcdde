#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Creates instances of the Python and C core and subjects for the same DDE them to a series of random commands (within a reasonable margin). As both cores should behave identically, the results should not differ – except for details of the numerical implementation, which may cause the occasional deviation due to the chaoticity of the Rössler oscillators used for testing.

The argument is the number of runs.
"""

from __future__ import print_function
from jitcdde._python_core import dde_integrator as py_dde_integrator
from jitcdde._jitcdde import provide_advanced_symbols
from jitcdde._helpers import (
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

compare = lambda x,y: assert_allclose(x,y,rtol=1e-5,atol=1e-5)

number_of_runs = int(argv[1])

t, y, current_y, past_y, anchors = provide_advanced_symbols()

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

def past_points():
	data = np.loadtxt("two_Roessler_past.dat")
	return [(point[0], np.array(point[1:7]), np.array(point[7:13])) for point in data]

n = 6

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
		chunk_size = 3,
		arguments = [
			("self", "dde_integrator * const"),
			("t", "double const"),
			("y", "double", n),
			("dY", "double", n)
			]
		)
		
	modulename = count_up(modulename)

	render_template(
		"jitced_template.c",
		tmpfile(modulename + ".c"),
		n = n,
		module_name = modulename,
		Python_version = version_info[0],
		has_any_helpers = False,
		number_of_helpers = 0,
		number_of_anchor_helpers = 0,
		anchor_mem_length = 1,
		)

	setup(
		name = modulename,
		ext_modules = [Extension(
			modulename,
			sources = [tmpfile(modulename + ".c")],
			extra_compile_args = ["-g", "-UNDEBUG", "-O2", "-Wall", "-pedantic", "-Wno-unknown-pragmas", "-std=c11"]
			)],
		script_args = ["build_ext","--build-lib",tmpfile(),"--build-temp",tmpfile(),"--force",],
		verbose = False
		)
	
	jitced = find_and_load_module(modulename,tmpfile())
	C = jitced.dde_integrator(past_points())
	
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
	
	get_next_step()
	get_next_step()
	
	actions = [get_next_step, get_t, get_recent_state, get_current_state, get_p, accept_step, forget, check_new_y_diff, past_within_step]
	
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
