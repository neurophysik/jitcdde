#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Suppose we want to integrate the Mackey–Glass delay differential equation: :math:`\dot{y} = f(y)` with :math:`y∈ℝ`, 

.. math::

	f(y) = β\frac{y(t-τ)}{1+y(t-τ)^n} - γ·y(t),
	
and with :math:`β = 0.25`, :math:`γ = 0.1`, and :math:`n = 10`.
The following code integrates the above for 10000 time units, with :math:`y(t<0) = 1`, and writes the results to :code:`timeseries.dat`:
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
	
	DDE.add_past_point( 0.0, [1.0], [0.0])
	DDE.add_past_point(-1.0, [1.0], [0.0])
	
	DDE.step_on_discontinuities()
	
	data = []
	for T in np.arange(DDE.t, DDE.t+10000, 10):
		data.append( DDE.integrate(T) )
	data = np.vstack(data)
	
	np.savetxt("timeseries.dat", data)
