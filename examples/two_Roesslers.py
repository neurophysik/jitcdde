from __future__ import print_function
from jitcdde import provide_advanced_symbols, jitcdde
import numpy as np

t, y, current_y, past_y, anchors = provide_advanced_symbols()

omega = np.random.normal(0.89, 0.0089, 2)
k = 0.25
delay = 4.5

f = [
	omega[0] * (-y(1) - y(2)),
	omega[0] * (y(0) + 0.165 * y(1)),
	omega[0] * (0.2 + y(2) * (y(0) - 10.0)),
	omega[1] * (-y(4) - y(5)) + k * (y(0,t-delay) - y(3)),
	omega[1] * (y(3) + 0.165 * y(4)),
	omega[1] * (0.2 + y(5) * (y(3) - 10.0))
	]

DDE = jitcdde(f, max_delay=delay)

start_state = np.random.uniform(-0.5,0.5,6)

DDE.add_past_point(-delay, start_state, np.zeros(6))
DDE.add_past_point(0.0   , start_state, np.zeros(6))

DDE.generate_f_C()
DDE.set_integration_parameters(rtol=1e-5)

pre_T = 100
DDE.integrate_blindly(pre_T)

dt = 0.1
with open("two_roesslers.dat", "w") as ausgabe:
	for T in np.arange(pre_T+dt,pre_T+1000,dt):
		print(T)
		state = DDE.integrate(T)
		ausgabe.write((6*"%f\t"+"\n")%tuple(state))
