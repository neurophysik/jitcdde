#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Suppose, we want to implement a network of :math:`n=100` delay-coupled Kuramoto oscillators, governed by the following differential equations:

.. math::

	\\dot{y}_i = ω + \\frac{c}{n-1} \\sum_{j=0}^{n-1} A_{ji} \\sin\\Big(y_j(t-τ_{ij})-y_i(t)\\Big),

where :math:`ω=1`, :math:`c=42`, :math:`τ_{ij} \\sim \\mathcal{U}([\\frac{π}{5},2π])`, and :math:`A` is the adjacency matrix of a random unweighted directed network with self-connections where each node exists with a probability :math:`q=0.05`.

Without further ado, here is the example code; highlighted lines will be commented upon below:

.. literalinclude:: ../examples/kuramoto_network.py
	:linenos:
	:start-after: example-st\u0061rt
	:dedent: 1
	:emphasize-lines: 9, 10, 15, 21, 24

Explanation of selected features and choices:

* Line 9 is just a quick way to generate the network described above. For moore complex network, you will either have to write more complex function or use dedicated modules. (In fact this example was chosen such that the network creation is very simple.)

* The values of :math:`τ` are initialised globally (line 10). We shoould not just define a function here, because if we were trying to calculate Lyapunov exponents or the Jacobian, the generator function would be called multiple times, and thus the value of the parameter would not be consistent (which would be desastrous).

* In line 15, we use `symengine.sin` – in contrast to `math.sin` or `numpy.sin`.

* In line 21, we explicitly use absolute instead of relative errors, as the latter make no sense for Kuramoto oscillators.

* In line 24, we integrate blindly with a maximum time step of 0.1 up to the maximal delay to ensure that initial discontinuities have smoothened out.
"""

if __name__ == "__main__":
	# example-start
	from jitcdde import jitcdde, y, t
	from numpy import pi, arange, random, max
	from symengine import sin
	
	n = 100
	ω = 1
	c = 42
	q = 0.05
	A = random.choice( [1,0], size=(n,n), p=[q,1-q] )
	τ = random.uniform( pi/5, 2*pi, size=(n,n) )
	
	def kuramotos():
		for i in range(n):
			yield ω + c/(n-1)*sum(
						sin(y(j,t-τ[i,j])-y(i))
						for j in range(n)
						if A[j,i]
					)
	
	I = jitcdde(kuramotos,n=n,verbose=False)
	I.set_integration_parameters(rtol=0,atol=1e-5)
	
	I.constant_past( random.uniform(0,2*pi,n), time=0.0 )
	I.integrate_blindly( max(τ) , 0.1 )
	
	for time in I.t + arange(0,400,0.2):
		print(*I.integrate(time) % (2*pi))
