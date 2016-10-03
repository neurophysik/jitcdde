from __future__ import print_function
from jitcdde import provide_advanced_symbols, jitcdde
import numpy as np

tau = 15
p = 10

t, y, current_y, past_y, anchors = provide_advanced_symbols()

f = [0.25 * y(0,t-tau) / (1.0 + y(0,t-tau)**p) - 0.1*y(0)]

DDE = jitcdde(f)

y0 = 0.8
dy = -0.0794952762375263
DDE.add_past_point(-1.0, np.array([y0-dy]), np.array([dy]))
DDE.add_past_point( 0.0, np.array([y0   ]), np.array([dy]))

DDE.generate_f_C()
DDE.set_integration_parameters()

dt = 1.0

values = np.vstack(DDE.integrate(t) for t in np.arange(dt,1000,dt))

delay = tau/dt
np.savetxt(
	"timeseries.dat",
	np.hstack((values[delay:], values[:-delay]))
	)

