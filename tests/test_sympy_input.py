"""
Tests whether things works independent of where symbols are imported from.
"""

import jitcdde
import jitcdde.sympy_symbols
import sympy
import symengine

sympy_t = sympy.Symbol("t",real=True)
def sympy_y(index,time=sympy_t):
	if time == sympy_t:
		return sympy_current_y(index)
	else:
		return sympy_past_y(time, index, sympy_anchors(time))
sympy_current_y = sympy.Function("current_y",real=True)
sympy_past_y = sympy.Function("past_y",real=True)
sympy_anchors = sympy.Function("anchors",real=True)

symengine_t = symengine.Symbol("t",real=True)
def symengine_y(index,time=symengine_t):
	if time == symengine_t:
		return symengine_current_y(index)
	else:
		return symengine_past_y(time, index, symengine_anchors(time))
symengine_current_y = symengine.Function("current_y",real=True)
symengine_past_y = symengine.Function("past_y",real=True)
symengine_anchors = symengine.Function("anchors",real=True)


symengine_manually = [
		symengine_t,
		symengine_y,
		symengine.cos,
	]

sympy_manually = [
		sympy_t,
		sympy_y,
		sympy.cos,
	]

jitcdde_provisions = [
		jitcdde.t,
		jitcdde.y,
		symengine.cos,
	]

jitcdde_sympy_provisions = [
		jitcdde.sympy_symbols.t,
		jitcdde.sympy_symbols.y,
		symengine.cos,
	]

mixed = [
		jitcdde.sympy_symbols.t,
		jitcdde.y,
		sympy.cos,
	]

results = set()

for t,y,cos in [
			symengine_manually,
			sympy_manually,
			jitcdde_provisions,
			jitcdde_sympy_provisions,
			mixed,
		]:
	DDE = jitcdde.jitcdde( [cos(t)*y(0)-y(0,t-1)], verbose=False )
	DDE.constant_past([1],0.0)
	
	DDE.step_on_discontinuities()
	result = DDE.integrate(10)[0]
	results.add(result)

assert len(results)==1

