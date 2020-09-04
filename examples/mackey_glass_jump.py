#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
This is a very simple example implementing the Mackey–Glass system with an additional jump.
See https://github.com/neurophysik/jitcdde/issues/23 for details.
"""

from jitcdde import jitcdde, y, t
import numpy

τ = 15
n = 10
β = 0.25
γ = 0.1

f = [ β * y(0,t-τ) / (1 + y(0,t-τ)**n) - γ*y(0) ]
DDE = jitcdde(f,verbose=False)

DDE.constant_past([1.0])

DDE.step_on_discontinuities()

for time in numpy.arange(DDE.t, DDE.t+1000, 10):
	print( time, *DDE.integrate(time) )

DDE.jump(1,DDE.t)

for time in numpy.arange(DDE.t, DDE.t+1000, 10):
	print( time, *DDE.integrate(time) )
