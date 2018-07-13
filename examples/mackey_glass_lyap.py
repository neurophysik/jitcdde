#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
For instance, we can calculate and print the Lyapunov exponents for the Mackey–Glass system as follows (central changes to the example from `example` highlighted):

.. literalinclude:: ../examples/mackey_glass_lyap.py
	:dedent: 1
	:lines: 17-49
	:emphasize-lines: 11-12, 19-20, 22, 28, 30-33
	:linenos:

Note that `integrate` does not only return local Lyapunov exponents but also the length of the time interval to which they apply (which differs from the time spanned by the `integrate` command and may even be zero). This length should be used to weigh the local Lyapunov exponents for statistical processing, like in line 31.
"""

if __name__ == "__main__":
	from jitcdde import jitcdde_lyap, y, t
	import numpy
	from scipy.stats import sem
	
	τ = 15
	n = 10
	β = 0.25
	γ = 0.1
	
	f = [ β * y(0,t-τ) / (1 + y(0,t-τ)**n) - γ*y(0) ]
	n_lyap = 4
	DDE = jitcdde_lyap(f, n_lyap=n_lyap)
	
	DDE.constant_past([1.0])
	
	DDE.step_on_discontinuities(max_step=1.0)
	
	data = []
	lyaps = []
	weights = []
	for time in numpy.arange(DDE.t, DDE.t+10000, 10):
		state, lyap, weight = DDE.integrate(time)
		data.append(state)
		lyaps.append(lyap)
		weights.append(weight)
	
	numpy.savetxt("timeseries.dat", data)
	lyaps = numpy.vstack(lyaps)
	
	for i in range(n_lyap):
		Lyap = numpy.average(lyaps[400:,i], weights=weights[400:])
		stderr = sem(lyaps[400:,i]) # Note that this only an estimate
		print("%i. Lyapunov exponent: % .4f +/- %.4f" % (i+1,Lyap,stderr))
