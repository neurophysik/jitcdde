#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
This is a very simple example for a state-dependent delay.
See https://github.com/neurophysik/jitcdde/issues/7 for details.
"""


from jitcdde import jitcdde, y, t
import numpy

f = [ -y(0,t-2-y(0)**2) + 5 ]
DDE = jitcdde(f,max_delay=1e10)

eps = 0.1
DDE.add_past_point(-2.0    , [ 4.5], [0.0])
DDE.add_past_point(-1.0-eps, [ 4.5], [0.0])
DDE.add_past_point(-1.0+eps, [-0.5], [0.0])
DDE.add_past_point( 0.0    , [-0.5], [0.0])

DDE.integrate_blindly(0.01)

data = []
for time in [ DDE.t, *numpy.linspace(0.9,2,30) ]:
	data.append([time, DDE.integrate(time)])
numpy.savetxt("timeseries.dat", data)

