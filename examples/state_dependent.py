#!/usr/bin/python3
# -*- coding: utf-8 -*-

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
for time in numpy.arange(0,2.0,0.001)+DDE.t:
	data.append([time, DDE.integrate(time)])
numpy.savetxt("timeseries.dat", data)

