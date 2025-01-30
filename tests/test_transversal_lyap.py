"""
Integration test of jitcdde_restricted_lyap and jitcdde_transversal_lyap by comparing their results to each other for a synchronised scenario.
"""

import platform
from itertools import combinations
from warnings import warn

import numpy as np
from scipy.stats import sem
from symengine import Symbol

from jitcdde import jitcdde_restricted_lyap, jitcdde_transversal_lyap, t, y


if platform.system() == "Windows":
	compile_args = None
else:
	from jitcxde_common import DEFAULT_COMPILE_ARGS
	compile_args = [*DEFAULT_COMPILE_ARGS,"-g","-UNDEBUG"]

rng = np.random.default_rng(seed=42)

a = -0.025794
b =  0.01
c =  0.02
τ =  1.0
k = Symbol("k")

scenarios = [
	{
	"name":
		"grouped by variables",
	"f":
		[
			y(0)*(a-y(0))*(y(0)-1.0) - y(3) + k*(y(1)-y(0)),
			y(1)*(a-y(1))*(y(1)-1.0) - y(4) + k*(y(2)-y(1)),
			y(2)*(a-y(2))*(y(2)-1.0) - y(5) + k*(y(0)-y(2)),
			b*y(0) - c*y(3,t-τ),
			b*y(1) - c*y(4,t-τ),
			b*y(2) - c*y(5,t-τ),
		],
	"vectors":
		[
			( [1.,1.,1.,0.,0.,0.], [0.,0.,0.,0.,0.,0.] ) ,
			( [0.,0.,0.,0.,0.,0.], [1.,1.,1.,0.,0.,0.] ) ,
			( [0.,0.,0.,1.,1.,1.], [0.,0.,0.,0.,0.,0.] ) ,
			( [0.,0.,0.,0.,0.,0.], [0.,0.,0.,1.,1.,1.] ) ,
		],
	"groups":
		( [0,1,2], [3,4,5] ),
	},
	{
	"name":
		"grouped by oscillators",
	"f":
		[
			y(0)*(a-y(0))*(y(0)-1.0) - y(1) + k*(y(2)-y(0)),
			b*y(0) - c*y(1,t-τ),
			y(2)*(a-y(2))*(y(2)-1.0) - y(3) + k*(y(4)-y(2)),
			b*y(2) - c*y(3,t-τ),
			y(4)*(a-y(4))*(y(4)-1.0) - y(5) + k*(y(0)-y(4)),
			b*y(4) - c*y(5,t-τ),
		],
	"vectors":
		[
			( [1.,0.,1.,0.,1.,0.], [0.,0.,0.,0.,0.,0.] ) ,
			( [0.,0.,0.,0.,0.,0.], [1.,0.,1.,0.,1.,0.] ) ,
			( [0.,1.,0.,1.,0.,1.], [0.,0.,0.,0.,0.,0.] ) ,
			( [0.,0.,0.,0.,0.,0.], [0.,1.,0.,1.,0.,1.] ) ,
		],
	"groups":
		( [0,2,4], [1,3,5] ),
	},
]

couplings = [
		{"k":-0.1, "sign": 1},
		{"k":-0.2, "sign": 1},
		{"k": 0.1, "sign":-1},
		{"k": 0.2, "sign":-1},
		{"k": 0  , "sign": 0},
	]

class TestPrerequisiteBroken(AssertionError):
	pass

for scenario in scenarios:
	n = len(scenario["f"])
	
	DDE1 = jitcdde_restricted_lyap(
			scenario["f"],
			vectors = scenario["vectors"],
			verbose = False,
			control_pars = [k],
		)
	# Simplification would lead to trajectories diverging from the synchronisation manifold due to numerical noise.
	DDE1.compile_C(simplify=False,extra_compile_args=compile_args)
	
	DDE2 = jitcdde_transversal_lyap(
			scenario["f"],
			groups = scenario["groups"],
			verbose = False,
			control_pars = [k],
		)
	DDE2.compile_C(extra_compile_args=compile_args)
	
	def check_manifold(k, DDE1=DDE1, scenario=scenario):
		message = f"The dynamics left the synchronisation manifold when {scenario['name']} with coupling {k}. If this fails, this is a problem with the test and not with what is tested or any software involved.\n\nSpecifically, this test only works when the backend (Symengine plus compiler) implements certain computations completely symmetrically. This needs not and cannot be reasonably controlled (and no, turning off compiler optimisation doesn’t necessarily help as it often restores symmetries broken by Symengine). It’s only something exploited by this test to make it work in the first place."
		for anchor in DDE1.get_state():
			for group in scenario["groups"]:
				for i,j in combinations(group,2):
					if anchor[1][i]!=anchor[1][j] or anchor[2][i]!=anchor[2][j]:
						raise TestPrerequisiteBroken(message)
	
	for coupling in couplings:
		DDE1.purge_past()
		DDE2.purge_past()
		
		single = rng.random(2)
		initial_state = np.empty(n)
		for j,group in enumerate(scenario["groups"]):
			for i in group:
				initial_state[i] = single[j]
		DDE1.constant_past(initial_state,0.0)
		DDE2.constant_past(single,0.0)
		
		DDE1.set_parameters(coupling["k"])
		DDE2.set_parameters(coupling["k"])
		
		for time in range(1,100):
			DDE1.integrate_blindly(time)
			DDE2.integrate_blindly(time)
		
		lyaps1 = []
		lyaps2 = []
		weights1 = []
		weights2 = []
		assert DDE1.t==DDE2.t
		times = DDE1.t + np.arange(100,10000,100)
		try:
			for time in times:
				check_manifold(coupling["k"])
				_,lyap1,weight1 = DDE1.integrate(time)
				_,lyap2,weight2 = DDE2.integrate(time)
				lyaps1.append(lyap1)
				lyaps2.append(lyap2)
				weights1.append(weight1)
				weights2.append(weight2)
		except TestPrerequisiteBroken as err:
			warn(str(err),stacklevel=2)
			print( "X", end="", flush=True )
			continue
		
		Lyap1 = np.average(lyaps1,weights=weights1)
		Lyap2 = np.average(lyaps2,weights=weights2)
		margin1 = sem(lyaps1)
		margin2 = sem(lyaps2)
		sign1 = np.sign(Lyap1) if abs(Lyap1)>margin1 else 0
		sign2 = np.sign(Lyap2) if abs(Lyap2)>margin2 else 0
		
		assert margin1/Lyap1 < 0.1 or not sign1
		assert margin2/Lyap2 < 0.1 or not sign2
		assert sign1==coupling["sign"]
		assert sign2==coupling["sign"]
		assert abs(Lyap1-Lyap2)<max(margin1,margin2), f"{Lyap1}±{margin1} \t {Lyap2}±{margin2}"
		print( ".", end="", flush=True )

print("")
