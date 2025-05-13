#!/usr/bin/python3

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
	:emphasize-lines: 11, 12, 17, 22, 23, 26

Explanation of selected features and choices:

* Line 11 is just a quick way to generate the network described above. For more complex networks, you will either have to write more complex function or use dedicated modules. (In fact this example was chosen such that the network creation is very simple.)

* The values of :math:`τ` are initialised globally (line 12). We should not just define a function here, because if we were trying to calculate Lyapunov exponents or the Jacobian, the generator function would be called multiple times, and thus the value of the parameter would not be consistent (which would be disastrous).

* In line 17, we use `symengine.sin` – in contrast to `math.sin` or `numpy.sin`.

* In line 22, we explicitly specify the delays to speed things up a little.

* In line 23, we explicitly use absolute instead of relative errors, as the latter make no sense for Kuramoto oscillators.

* In line 26, we integrate blindly with a maximum time step of 0.1 up to the maximal delay to ensure that initial discontinuities have smoothened out.
"""

if __name__ == "__main__":
	# example-start
	import numpy as np
	from symengine import sin

	from jitcdde import jitcdde, t, y
	
	rng = np.random.default_rng(seed=42)
	n = 100
	ω = 1
	c = 42
	q = 0.05
	A = rng.choice( [1,0], size=(n,n), p=[q,1-q] )
	τ = rng.uniform( np.pi/5, 2*np.pi, size=(n,n) )
	
	def kuramotos():
		for i in range(n):
			yield ω + c/(n-1)*sum(
						sin(y(j,t-τ[i,j])-y(i))
						for j in range(n)
						if A[j,i]
					)
	
	solver = jitcdde(kuramotos,n=n,verbose=False,delays=τ.flatten())
	solver.set_integration_parameters(rtol=0,atol=1e-5)
	
	solver.constant_past( rng.uniform(0,2*np.pi,n), time=0.0 )
	solver.integrate_blindly( np.max(τ) , 0.1 )
	
	for time in solver.t + np.arange(0,400,0.2):
		print(*solver.integrate(time) % (2*np.pi))
