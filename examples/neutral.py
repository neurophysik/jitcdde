"""
	Experimental implementation of a neutral DDE. See https://github.com/neurophysik/jitcdde/issues/24 for details. `simple_neutral.py` contains a clearer, but less efficient implementation.
"""

from jitcdde import jitcdde, y, current_y, past_dy, past_y, t, anchors
from symengine import tanh, sqrt, exp, Symbol
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

anchors_past = Symbol("anchors_past")
difference = Symbol("difference")
factor_μ = Symbol("factor_μ")
factor_ζ = Symbol("factor_ζ")

ydot = [ y(i) for i in range(3,6) ]
y_tot      = sum( current_y(i)                for i in range(  3) )
ydot_tot   = sum( current_y(i)                for i in range(3,6) )
y_past     = sum( past_y (t-τ,i,anchors_past) for i in range(  3) )
ydot_past  = sum( past_y (t-τ,i,anchors_past) for i in range(3,6) )
yddot_past = sum( past_dy(t-τ,i,anchors_past) for i in range(3,6) )

helpers = {
		( anchors_past, anchors(t-τ) ),
		( difference, ybar_0-y_past ),
		( factor_μ, sech(difference)**2 * (yddot_past + 2*ydot_past**2*tanh(difference)) ),
		( factor_ζ, 2*abs(y_tot)*ydot_tot ),
	}

f = { y(i):ydot[i] for i in range(3) }
f.update( {
		ydot[i]:
		μ[i]*factor_μ - ζ[i]*factor_ζ - ε[i]*ν[i]*ydot[i] - ν[i]**2*y(i)
		for i in range(3)
	} )

DDE = jitcdde(f,helpers=helpers,verbose=False)

np.random.seed(23)
DDE.constant_past(np.random.normal(0,1,6))
DDE.adjust_diff()

for time in DDE.t+np.arange(0.1,100,0.1):
	print(time,*DDE.integrate(time))

