from jitcdde import t, y, jitcdde
import numpy as np

ω = np.random.normal(0.89, 0.0089, 2)
k = 0.25
delay = 4.5

f = [
	ω[0] * (-y(1) - y(2)),
	ω[0] * (y(0) + 0.165 * y(1)),
	ω[0] * (0.2 + y(2) * (y(0) - 10.0)),
	ω[1] * (-y(4) - y(5)) + k * (y(0,t-delay) - y(3)),
	ω[1] * (y(3) + 0.165 * y(4)),
	ω[1] * (0.2 + y(5) * (y(3) - 10.0))
	]

DDE = jitcdde(f)

start_state = np.random.uniform(-0.5,0.5,6)

DDE.add_past_point(-delay, start_state, np.zeros(6))
DDE.add_past_point(0.0   , start_state, np.zeros(6))

DDE.step_on_discontinuities()

times = np.arange(DDE.t,DDE.t+1000,0.1)
data = np.vstack(DDE.integrate(T) for T in times)
np.savetxt("two_roesslers.dat", data)

