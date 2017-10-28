#!/usr/bin/python3
# -*- coding: utf-8 -*-

from jitcdde import jitcdde_restricted_lyap, y, t
import numpy as np
from scipy.stats import sem
from symengine import Symbol

a = -0.025794
b =  0.01
c =  0.02
k = Symbol("k")

f = [
		y(0)*(a-y(0))*(y(0)-1.0) - y(1) + k*(y(2)-y(0)),
		b*y(0) - c*y(1,t-1),
		y(2)*(a-y(2))*(y(2)-1.0) - y(3) + k*(y(0)-y(2)),
		b*y(2) - c*y(3,t-1)
	]

vectors = [
		( np.array([1,0,1,0]), np.array([0,0,0,0]) ),
		( np.array([0,0,0,0]), np.array([1,0,1,0]) ),
		( np.array([0,1,0,1]), np.array([0,0,0,0]) ),
		( np.array([0,0,0,0]), np.array([0,1,0,1]) )
	]

DDE = jitcdde_restricted_lyap( f, vectors=vectors, verbose=False, control_pars=[k] )
DDE.compile_C(simplify=False)

scenarios = [
		{"k": 0    , "sign": 0},
		{"k":-0.128, "sign": 1},
		{"k":-0.2  , "sign": 1},
		{"k": 0.128, "sign":-1},
		{"k": 0.2  , "sign":-1},
	]

for scenario in scenarios:
	DDE.purge_past()
	
	if scenario["sign"]<0:
		initial_state = np.random.random(4)
	else:
		single = np.random.random(2)
		initial_state = np.hstack([single,single])
	DDE.constant_past(initial_state,0.0)
	assert DDE.t==0
	
	DDE.set_parameters(scenario["k"])
	
	for time in np.arange(1,1000):
		DDE.integrate_blindly(time)
	
	lyaps = []
	weights = []
	states = []
	times = DDE.t + np.arange(100,100000,100)
	for i,time in enumerate(times):
		state,lyap,weight = DDE.integrate(time)
		lyaps.append(lyap)
		weights.append(weight)
		states.append(state)
		assert not np.isnan(lyap)
	
	# Check that we are still on the synchronisation manifold:
	assert state[0]==state[2]
	assert state[1]==state[3]
	
	Lyap = np.average(lyaps,weights=weights)
	margin = sem(lyaps)
	sign = np.sign(Lyap) if abs(Lyap)>margin else 0
	assert sign==scenario["sign"], "Test failed in scenario %s. (Lyapunov exponent: %f±%f)" % (scenario,Lyap,margin)
	print(".",end="",flush=True)

print("")

