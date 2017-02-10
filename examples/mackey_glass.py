#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Suppose we want to integrate the Mackey–Glass delay differential equation: :math:`\\dot{y} = f(y)` with :math:`y∈ℝ`, 

.. math::

	f(y) = β \\frac{y(t-τ)}{1+y(t-τ)^n} - γ·y(t),
	
and with :math:`β = 0.25`, :math:`γ = 0.1`, and :math:`n = 10`.
First we do some importing and define the control parameters:

.. literalinclude:: ../examples/mackey_glass.py
	:linenos:
	:dedent: 1
	:lines: TODO

JiTCDDE needs the differential equation being written down using specific symbols for the state (`y`) and time (`t`). These are provided to us by `provide_basic_symbols`:

.. literalinclude:: ../examples/mackey_glass.py
	:linenos:
	:dedent: 1
	:lines: TODO

Now, we can write down the right-hand side of the differential equation as a list of expressions. As it is one-dimensional, the list contains only one element. We can then initiate JiTCDDE:

.. literalinclude:: ../examples/mackey_glass.py
	:linenos:
	:dedent: 1
	:lines: TODO

We want the initial condition and past to be :math:`y(t<0) = 1`. To achieve this, we use two anchors with a state of :math:`1` and a derivative of :math:`0`. This automatically results in the integration starting at :math:`t=0` – the time of the youngest anchor.

.. literalinclude:: ../examples/mackey_glass.py
	:linenos:
	:dedent: 1
	:lines: TODO

If we calculate the derivative from our initial conditions, we obtain :math:`f(t=0) = 0.025`, which does not agree with the :math:`\dot{y}(t=0) = 0` as explicitly defined in the initial conditions. Practically, this results in a discontinuity of the derivative at :math:`t=0`, which in turn TODO
"""



if __name__ == "__main__":
	# example-start
	from jitcdde import provide_basic_symbols, jitcdde
	import numpy as np
	
	τ = 15
	n = 10
	β = 0.25
	γ = 0.1
	
	t, y = provide_basic_symbols()
	
	f = [ β * y(0,t-τ) / (1 + y(0,t-τ)**n) - γ*y(0) ]
	DDE = jitcdde(f)
	
	DDE.add_past_point(-1.0, [1.0], [0.0])
	DDE.add_past_point( 0.0, [1.0], [0.0])
	
	DDE.step_on_discontinuities()
	
	data = []
	for T in np.arange(DDE.t, DDE.t+10000, 10):
		data.append( DDE.integrate(T) )
	data = np.vstack(data)
	
	np.savetxt("timeseries.dat", data)
