import platform
from jitcdde import jitcdde, y, current_y, past_dy, past_y, t, anchors
from symengine import tanh, sqrt, exp, Symbol
import numpy as np
import unittest

if platform.system() == "Windows":
	compile_args = None
else:
	from jitcxde_common import DEFAULT_COMPILE_ARGS
	compile_args = DEFAULT_COMPILE_ARGS+["-g","-UNDEBUG"]

initial = [ 0.66698806, 0.02581308, -0.77761941, 0.94863382, 0.70167179, -1.05108156 ]
control = [-0.72689205, 0.09982525, 0.41167269, 0.86527044, -0.50028942, 0.7227548 ]
T = 5

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

class TestNeutral(unittest.TestCase):
	def setUp(self):
		self.DDE = jitcdde(f,helpers=helpers,verbose=False)
		self.DDE.set_integration_parameters(rtol=1e-5)
		self.DDE.constant_past(initial)
	
	def test_compiled(self):
		self.DDE.compile_C(extra_compile_args=compile_args)
	
	def test_Python_core(self):
		self.DDE.generate_lambdas(simplify=False)
	
	def tearDown(self):
		self.DDE.adjust_diff()
		np.testing.assert_allclose(self.DDE.integrate(T),control,rtol=1e-4)

if __name__ == "__main__":
	unittest.main(buffer=True)

