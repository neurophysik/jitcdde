"""
	Experimental implementation of a neutral DDE. See https://github.com/neurophysik/jitcdde/issues/24 for details. `neutral.py` contains a considerably more efficient implementation.
"""

from jitcdde import jitcdde, y, dy, t, UnsuccessfulIntegration
from symengine import tanh, sqrt, exp
import numpy as np

sech = lambda x: 2/(exp(x)+exp(-x))
eps = 1e-5
abs = lambda x: sqrt(x**2+eps**2)

ε = [ 0.03966, 0.03184, 0.02847 ]
ν = [ 1, 2.033, 3.066 ]
μ = [ 0.16115668456085775, 0.14093420256851111, 0.11465065353644151 ]
ybar_0 = 0
τ = 1.7735
ζ = [ 0.017940997406325931, 0.015689701773967984, 0.012763648066925721 ]

ydot = [ y(i) for i in range(3,6) ]
y_tot     = lambda time: sum(  y(i,time) for i in range(  3) )
ydot_tot  = lambda time: sum(  y(i,time) for i in range(3,6) )
yddot_tot = lambda time: sum( dy(i,time) for i in range(3,6) )

f = { y(i):ydot[i] for i in range(3) }
f.update( { ydot[i]:
		μ[i] * sech(ybar_0 - y_tot(t-τ))**2 *
		(yddot_tot(t-τ) + 2*ydot_tot(t-τ)**2*tanh(ybar_0 - y_tot(t-τ)))
		- 2*ζ[i] * abs(y_tot(t)) * ydot_tot(t)
		- ε[i] * ν[i] * ydot[i]
		- ν[i]**2 * y(i)
		for i in range(3)
	} )

DDE = jitcdde(f,verbose=False)

np.random.seed(23)
DDE.constant_past(np.random.normal(0,1,6))
DDE.adjust_diff()

for time in DDE.t+np.arange(0.1,100,0.1):
	print(time,*DDE.integrate(time))

