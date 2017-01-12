from __future__ import print_function
from jitcdde import provide_advanced_symbols, jitcdde_lyap
import numpy as np
from scipy.stats import sem

tau = 15
p = 10

t, y, current_y, past_y, anchors = provide_advanced_symbols()

f = [0.25 * y(0,t-tau) / (1 + y(0,t-tau)**p) - 0.1*y(0)]

n_lyap = 4
DDE = jitcdde_lyap(f, n_lyap=n_lyap)

y0 = 0.8
dy = -0.0794952762375263
DDE.add_past_point(-1.0, [y0-dy], [dy])
DDE.add_past_point( 0.0, [y0   ], [dy])

#DDE.integrate_blindly(DDE.t()+15.0,1.0)
DDE.step_on_discontinuities(1,1.0)


dt = 10.0
data = []
for T in np.arange(DDE.t+dt,10000,dt):
	data.append( DDE.integrate(T) )
data = np.vstack(data)

np.savetxt("timeseries.dat", data)

n = len(f)
for i in range(n_lyap):
	lyap = np.average(data[400:,n+i], weights=data[400:,-1])
	stderr = sem(data[400:,n+i]) # Note that this only an estimate
	print("%i. Lyapunov exponent: % .4f +/- %.4f" % (i+1,lyap,stderr))

