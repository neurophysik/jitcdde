from __future__ import print_function
from jitcdde import provide_advanced_symbols, jitcdde
import numpy as np

tau = 15
p = 10

t, y, current_y, past_y, anchors = provide_advanced_symbols()

f = [0.25 * y(0,t-tau) / (1.0 + y(0,t-tau)**p) - 0.1*y(0)]

DDE = jitcdde(f)

y0 = 0.3
dy = -0.4
DDE.add_past_point( 0.0, np.array([y0   ]), np.array([dy]))
DDE.add_past_point(-1.0, np.array([y0-dy]), np.array([dy]))

dt = 10.0
DDE.step_on_discontinuities()

data = []
for T in np.arange(DDE.t()+dt,10000,dt):
	data.append( DDE.integrate(T) )
data = np.vstack(data)

np.savetxt("timeseries.dat", data)