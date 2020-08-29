# In this example, we implement the sunflower equation. First, we do it regularly, then we repeat the process using callbacks.

from jitcdde import jitcdde, y, t
import numpy as np

a = 4.8
b = 0.186
tau = 40

# Regular implementation
# ----------------------
# To implement the sine function, we use SymEngine’s sine. This is a symbolic function that gets translated to a C implementation of the sine function under the hood.

import symengine
f = [
		y(1),
		-a/tau*y(1) - b/tau*symengine.sin(y(0,t-tau))
	]
DDE_regular = jitcdde(f)

# With Callbacks
# --------------
# Now, let’s assume for example’s sake that we did not have a sine function for straightforward use like above (i.e., a symbolic SymEngine function that gets translated to a C implementation). Instead we have to use the one from the Python’s Math library.

my_sine = symengine.Function("my_sine")
f_with_callback = [
		y(1),
		-a/tau*y(1) - b/tau*my_sine(y(0,t-tau))
	]

# We need to introduce a wrapper to match the required signature for callbacks. We also add a `print` statement to see when the callback was called. Except for the latter, we do not use the first argument, which is a vector containing the entire present state of the system:

import math
def my_sine_callback(y,arg):
	print(f"my_sine called with arguments {y} and {arg}")
	return math.sin(arg)

DDE_callback = jitcdde(
		f_with_callback,
		callback_functions = [(my_sine,my_sine_callback,1)],
	)

# Integration
# -----------
# Initialise and address initial discontinuities for both implementations:
DDE_regular.constant_past( [1.0,0.0], time=0.0 )
DDE_regular.adjust_diff()
DDE_callback.constant_past( [1.0,0.0], time=0.0 )
DDE_callback.adjust_diff()
assert DDE_regular.t == DDE_callback.t

# Integrate side by side and compare:
times = DDE_regular.t + np.arange(10,100,10)
for time in times:
	assert DDE_regular.integrate(time)[0] == DDE_callback.integrate(time)[0]

