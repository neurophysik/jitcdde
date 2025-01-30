#!/usr/bin/python3

"""
This is a very simple example for a state-dependent delay.
See https://github.com/neurophysik/jitcdde/issues/7 for details.
"""


import numpy as np

from jitcdde import jitcdde, t, y


f = [ -y(0,t-2-y(0)**2) + 5 ]
DDE = jitcdde(f,max_delay=1e10)

eps = 0.1
DDE.add_past_point(-2.0    , [ 4.5], [0.0])
DDE.add_past_point(-1.0-eps, [ 4.5], [0.0])
DDE.add_past_point(-1.0+eps, [-0.5], [0.0])
DDE.add_past_point( 0.0    , [-0.5], [0.0])

DDE.integrate_blindly(0.01)

steps = np.linspace(0.9,2,30)
data = np.empty((steps.size + 1, 2))
for n, time in enumerate([ DDE.t, *steps ]):
	data[n] = [time, DDE.integrate(time)[0]]
np.savetxt("timeseries.dat", data)

