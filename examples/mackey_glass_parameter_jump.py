"""
The following are a set of examples that implement the Mackey-Glass system with a jump in the value of a parameter. Here we will change the value of β from β₀ to β₁ at time t₀.

The methods range from the most straightforward to more involved ones.

    1. Integrate two separate models with different β values.
    2. Use β as a control parameter for a system and modify its value midway through the integration.
    3. Use jitcdde.input.
    4. Use jitcxde_common.conditional.
    5. Use callbacks to define your own conditional.

The result of all the methods is basically identical for this example, but not all may be feasible for your specific application – there is a reason that so many variants exist. The pros and cons of each method are outlined in their respective sections.

Note that if the parameter that needs to jump is the delay, you’ll need to manually set max_delay when defining the integrator in cases 2–5.
"""

### Makey–Glass with time-dependent parameter

import numpy as np
from matplotlib.pyplot import subplots

from jitcdde import jitcdde, t, y


# Defining parameters
# --------------

τ = 15
n = 10
β0 = 0.25
β1 = 0.4
γ = 0.1

t0 = 400
dt = 1
tf = 800

# 1. Running separate models with the initial and final parameter value
# -------------
# This method spends the least time on actual integration, but requires two compilations. It becomes infeasible once your parameter changes frequently or even continuously.

# Create a function to easily make a model with different values for β:
def make_model(β):
    return [ β * y(0,t-τ) / (1 + y(0,t-τ)**n) - γ*y(0) ]

# First run a model with β0 and integrate it to t0.
DDE0 = jitcdde(make_model(β0))
DDE0.constant_past(1)
DDE0.adjust_diff()
times0 = np.arange(0, t0, dt)
data = [DDE0.integrate(t) for t in times0]

# Create a new model using β1 and set its past to the last state of the other model.
# We use adjust_diff to handle the discontinuities that arise from the change in β:
DDE1 = jitcdde(make_model(β1))
DDE1.add_past_points(DDE0.get_state())
DDE1.adjust_diff()
times1 = np.arange(t0, tf, dt)
data = np.vstack( data + [DDE1.integrate(t) for t in times1] )

# Plot
fig, ax = subplots()
ax.plot(np.hstack((times0, times1)), data)
ax.set_title("two separate models")


# 2. Using a control parameter we modify mid-run
# --------------
# This method requires only one compilation and spends a bit more on each integration. It also becomes infeasible once your parameter changes frequently or even continuously.

from symengine import Symbol


# Create the model with β being a symbol instead of having a value, and integrate it up to t0:
β = Symbol("beta")
f = [ β * y(0,t-τ) / (1 + y(0,t-τ)**n) - γ*y(0) ]

# Run the model with β as its control parameter:
DDE = jitcdde(f, control_pars=[β])
DDE.constant_past(1)
DDE.set_parameters(β0)
DDE.adjust_diff()

times0 = np.arange(0, t0, dt)
data = [DDE.integrate(t) for t in times0]

# Set the parameter to a new value, adjust_diff and integrate the rest of the way:
times1 = np.arange(t0, tf, dt)
DDE.set_parameters(β1)
DDE.adjust_diff()
data = np.vstack( data + [DDE.integrate(t) for t in times1] )

# Plot
fig, ax = subplots()
ax.plot(np.hstack((times0, times1)), data)
ax.set_title("control parameter")


# 3. Using input to modify the parameter mid-run
# --------------
# This method requires more preparation and integration time, but can handle frequent and continuous changes of the input.

from chspy import CubicHermiteSpline

from jitcdde import input, jitcdde_input  # noqa: A004


# Define how the parameter changes over time and put that information into a spline:
input_times = np.arange(0, tf)
parameter_over_time = lambda t: β0 if t<t0 else β1
parameter_spline = CubicHermiteSpline(n=1)
parameter_spline.from_function(parameter_over_time, times_of_interest = input_times)

# Defining Dynamics
β = input(0)
f = [ β * y(0,t-τ) / (1 + y(0,t-τ)**n) - γ*y(0) ]

# Create and integrate the model all the way to the end, input handles the change in the value midway through the integration:
DDE = jitcdde_input(f, parameter_spline)
DDE.constant_past(1)
DDE.adjust_diff()
times = DDE.t + np.arange(0, input_times[-1])
data = np.vstack([DDE.integrate(time) for time in times])

# Plot
fig, ax = subplots()
ax.plot(times, data)
ax.set_title("input")


# 4. Using jitcxde_common.conditional to approximate a jump
# --------------
# This method is very similar to Method 2 in its pros and cons. Its main advantage is that it’s more straightforward to implement and that the conditional can trigger on inputs other than time.

from jitcxde_common import conditional


# Define how the parameter changes over time and define the system dynamics:
β = conditional(t, t0, β0, β1)
f = [ β * y(0,t-τ) / (1 + y(0,t-τ)**n) - γ*y(0) ]

# Create and integrate the model all the  way to the end, conditional handles the change in the value midway through the integration:
DDE = jitcdde(f)
DDE.constant_past(1)
DDE.adjust_diff()
times = DDE.t + np.arange(0, tf)
data = np.vstack([DDE.integrate(time) for time in times])

# Plot
fig, ax = subplots()
ax.plot(times, data)
ax.set_title("conditional")

# 5. Writing the jump ourselves using a callback
# --------------
# This method can not only handle frequent and continuous changes of the input, but also allows to change the input based on the current state of the system. Its efficiency primarily depends on the efficiency of the callback function.

from symengine import Function


# Define the the ~ython function that will handle the jump and the Symengine symbol corresponding to the parameter, which this time is a Function
β = Function("param_jump")
def param_jump_callback(y, t):
    return β0 if t<t0 else β1

# Define the dynamics of the system, β is now an explicit function of time
f = [ β(t) * y(0,t-τ) / (1 + y(0,t-τ)**n) - γ*y(0) ]

# Create and integrate the model all the way to the end, the callback handles the change in the value midway through the integration:
DDE = jitcdde(f, callback_functions=[ (β, param_jump_callback, 1) ])
DDE.constant_past(1)
DDE.adjust_diff()
times = DDE.t + np.arange(0, tf)
data = np.vstack([DDE.integrate(time) for time in times])

# Plot
fig, ax = subplots()
ax.plot(times, data)
ax.set_title("callback")
