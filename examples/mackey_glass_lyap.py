#!/usr/bin/python3

"""
For instance, we can calculate and print the Lyapunov exponents for the Mackey–Glass system as follows (central changes to the example from `example` highlighted):

.. literalinclude:: ../examples/mackey_glass_lyap.py
	:dedent: 1
	:lines: 19-49
	:emphasize-lines: 9-10, 17-18, 20, 26, 28-31
	:linenos:

Note that `integrate` does not only return local Lyapunov exponents but also the length of the time interval to which they apply (which differs from the time spanned by the `integrate` command and may even be zero). This length should be used to weigh the local Lyapunov exponents for statistical processing, like in line 29.
"""

if __name__ == "__main__":
	import numpy as np
	from scipy.stats import sem

	from jitcdde import jitcdde_lyap, t, y
	
	τ = 15
	n = 10
	β = 0.25
	γ = 0.1
	
	f = [ β * y(0,t-τ) / (1 + y(0,t-τ)**n) - γ*y(0) ]
	n_lyap = 4
	DDE = jitcdde_lyap(f, n_lyap=n_lyap)
	
	DDE.constant_past([1.0])
	
	DDE.step_on_discontinuities()
	
	data = []
	lyaps = []
	weights = []
	for time in np.arange(DDE.t, DDE.t+10000, 10):
		state, lyap, weight = DDE.integrate(time)
		data.append(state)
		lyaps.append(lyap)
		weights.append(weight)
	
	np.savetxt("timeseries.dat", data)
	lyaps = np.vstack(lyaps)
	
	for i in range(n_lyap):
		Lyap = np.average(lyaps[400:,i], weights=weights[400:])
		stderr = sem(lyaps[400:,i]) # Note that this only an estimate
		print(f"{i+1}. Lyapunov exponent: {Lyap:.4f} +/- {stderr:.4f}")
