#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Suppose we want to integrate the Mackey–Glass delay differential equation: :math:`\\dot{y} = f(y)` with :math:`y∈ℝ`, 

.. math::

	f(y) = β \\frac{y(t-τ)}{1+y(t-τ)^n} - γ·y(t),
	
and with :math:`β = 0.25`, :math:`γ = 0.1`, and :math:`n = 10`.
First we do some importing and define the constants:

.. literalinclude:: ../examples/mackey_glass.py
	:dedent: 1
	:lines: 60-66

Amongst our imports were the symbols for the state (`y`) and time (`t`), which have to be used to write down the differential equation such that JiTCDDE can process it.
Using them, we can write down the right-hand side of the differential equation as a list of expressions.
As it is one-dimensional, the list contains only one element. 
We can then initiate JiTCDDE:

.. literalinclude:: ../examples/mackey_glass.py
	:dedent: 1
	:lines: 68-69

We want the initial condition and past to be :math:`y(t<0) = 1`.
Hence we can use `constant_past`.
This automatically results in the integration starting at :math:`t=0`.

.. literalinclude:: ../examples/mackey_glass.py
	:dedent: 1
	:lines: 71

If we calculate the derivative from our initial conditions, we obtain :math:`f(t=0) = 0.025`, which does not agree with the :math:`\dot{y}(t=0) = 0` as explicitly defined in the initial conditions. Practically, this would result in an error if we started integrating without further precautions.
`step_on_discontinuities` makes some tailored integration steps to avoid this problem and to allow for the discontinuity to be smoothed out by temporal evolution.
(See `discontinuities` for alternatives and more details on this).

.. literalinclude:: ../examples/mackey_glass.py
	:dedent: 1
	:lines: 73

Finally, we can perform the actual integration.
In our case, we integrate for 10000 time units with a sampling rate of 10 time units. We query the current time of the integrator (`DDE.t`) to start wherever `step_on_discontinuities` ended. `integrate` returns the state after integration, which we collect in the list `data`.
Finally, we use `numpy.savetxt` to store this to the file `timeseries.dat`.

.. literalinclude:: ../examples/mackey_glass.py
	:dedent: 1
	:lines: 75-78

Taking everything together, our code is:

.. literalinclude:: ../examples/mackey_glass.py
	:dedent: 1
	:lines: 60-78
"""


if __name__ == "__main__":
	from jitcdde import jitcdde, y, t
	import numpy
	
	τ = 15
	n = 10
	β = 0.25
	γ = 0.1
	
	f = [ β * y(0,t-τ) / (1 + y(0,t-τ)**n) - γ*y(0) ]
	DDE = jitcdde(f)
	
	DDE.constant_past([1.0])
	
	DDE.step_on_discontinuities()
	
	data = []
	for time in numpy.arange(DDE.t, DDE.t+10000, 10):
		data.append( DDE.integrate(time) )
	numpy.savetxt("timeseries.dat", data)
