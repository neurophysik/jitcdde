import sympy

#: the symbol for time for defining the differential equation. This one is different from the one you can import from jitcode directly by being defined via SymPy and thus being better suited for some symbolic processing techniques that are not available in SymEngine yet.
t = sympy.Symbol("t", real=True)

def y(index,time=t):
	"""
	the function representing the DDE’s past and present states used for defining the differential equation. The first integer argument denotes the component. The second, optional argument is a symbolic expression denoting the time. This automatically expands to using `current_y`, `past_y`, and `anchors`; so do not be surprised when you look at the output and it is different than what you entered or expected. This one is different from the one you can import from jitcode directly by being defined via SymPy and thus being better suited for some symbolic processing techniques that are not available in SymEngine yet.
	"""
	if time == t:
		return current_y(index)
	else:
		return past_y(time, index, anchors(time))

#: the symbol for the current state for defining the differential equation. It is a function and the integer argument denotes the component. This is only needed for specific optimisations of large DDEs; in all other cases use `y` instead. This one is different from the one you can import from jitcode directly by being defined via SymPy and thus being better suited for some symbolic processing techniques that are not available in SymEngine yet.
current_y = sympy.Function("current_y",real=True)

#: the symbol for DDE’s past state for defining differential equation. It is a function with the first integer argument denoting the component and the second argument being a pair of past points (as being returned by `anchors`) from which the past state is interpolated (or, in rare cases, extrapolated). This is only needed for specific optimisations of large DDEs; in all other cases use `y` instead. This one is different from the one you can import from jitcode directly by being defined via SymPy and thus being better suited for some symbolic processing techniques that are not available in SymEngine yet.
past_y = sympy.Function("past_y",real=True)

#: the symbol representing two anchors for defining the differential equation. It is a function and the float argument denotes the time point to which the anchors pertain. This is only needed for specific optimisations of large DDEs; in all other cases use `y` instead. This one is different from the one you can import from jitcode directly by being defined via SymPy and thus being better suited for some symbolic processing techniques that are not available in SymEngine yet.
anchors = sympy.Function("anchors",real=True)
