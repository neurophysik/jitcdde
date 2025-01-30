"""
This is an example implementing laminar chaos as discovered by: https://doi.org/10.1103/PhysRevLett.120.084102
"""

import numpy as np
from symengine import sin

from jitcdde import jitcdde, t, y


T = 200
A = 0.9/2/np.pi
τ_0 = 1.50
f = lambda z: 4*z*(1-z)
τ = τ_0 + A*sin(2*np.pi*t)

model = [ T*( -y(0) + f(y(0,t-τ)) ) ]

DDE = jitcdde(model,max_delay=τ_0+A,verbose=False)
DDE.past_from_function([0.4+0.2*sin(t)])
DDE.set_integration_parameters(max_step=0.01,first_step=0.01)
DDE.integrate_blindly(τ_0+A,0.01)

for time in DDE.t + 100 + np.arange(0,10,0.01):
	print(DDE.integrate(time)[0])
