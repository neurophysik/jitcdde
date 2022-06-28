"""
The following are a set of examples that implement the Mackey-Glass system 
with a jump in the value of a parameter. Here we will change the value of β 
from β0 to β1 at time t0. 

The result of all the methods is basically identical and they range from the 
most straightforward to more involved ones. Choose whichever one suits your 
usecase. 
    1. Integrating two separate models with diferent β values
    2. Use β as a control parameter for a system amd modify its value midway 
    through the integration
    3. Use jitcdde.input
    4. Use jitcxde_common.conditional
    5. Use callbacks to define our own conditional
    
Note that if the parameter that needs to jump is the delay, you'll need to 
manually set max_delay when defining the integrator in cases 2-5.
"""

### Makey-Glass with time-dependent parameter

import numpy as np
from jitcdde import y, t, jitcdde
from matplotlib.pyplot import subplots

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
# --------------

# Create a function to easily make a model with different values for β
def make_model(β):
    return [ β * y(0,t-τ) / (1 + y(0,t-τ)**n) - γ*y(0) ]

# First run a model with β0 and integrate it to t0.
DDE0 = jitcdde(make_model(β0))
DDE0.constant_past(1)
DDE0.adjust_diff()
times0 = np.arange(0, t0, dt)
data = [DDE0.integrate(t) for t in times0]

# Create a new model using β1 and set its past to the last state of the other model
# We use adjust_diff to handle the discontinuities that arise from the change in β
DDE1 = jitcdde(make_model(β1))
DDE1.add_past_points(DDE0.get_state())
DDE1.adjust_diff()
times1 = np.arange(t0, tf, dt)
data = np.vstack( data + [DDE1.integrate(t) for t in times1])

# Plot
fig, ax = subplots()
ax.plot(np.hstack((times0, times1)), data)
ax.set_title('two separate models')


# 2. Using a control parameter we modify mid-run
# --------------

from symengine import Symbol

# Create the model with β being a symbol instead of having a value, and integrate it up to t0
β = Symbol('b')
f = [ β * y(0,t-τ) / (1 + y(0,t-τ)**n) - γ*y(0) ]

# Run the model with βas its contorl parameter
DDE = jitcdde(f, control_pars=[β])
DDE.constant_past(1)
DDE.set_parameters(β0)
DDE.adjust_diff()

times0 = np.arange(0, t0, dt)
data = [DDE.integrate(t) for t in times0]

# Set the parameter to a new value, adjust_diff and integrate the rest of the way
times1 = np.arange(t0, tf, dt)
DDE.set_parameters(β1)
DDE.adjust_diff()
data = np.vstack( data + [DDE.integrate(t) for t in times1])

# Plot
fig, ax = subplots()
ax.plot(np.hstack((times0, times1)), data)
ax.set_title('contorl parameter')


# 3. Using input to modify the parameter mid-run
# --------------

from jitcdde import jitcdde_input, input
from chspy import CubicHermiteSpline

# Define how the parameter changes over time and put that information into a spline
input_times = np.arange(0, tf)
parameter_over_time = lambda t: β0 if t<t0 else β1
parameter_spline = CubicHermiteSpline(n=1)
parameter_spline.from_function(parameter_over_time, times_of_interest = input_times)

# Defining Dynamics 
β = input(0)
f = [ β * y(0,t-τ) / (1 + y(0,t-τ)**n) - γ*y(0) ]

# Create and integrate the model all the  way to the end, input handles the 
# change in the value midway through the integration
DDE = jitcdde_input(f, parameter_spline)
DDE.constant_past(1)
DDE.adjust_diff()
times = DDE.t + np.arange(0, input_times[-1])
data = np.vstack([DDE.integrate(time) for time in times])

# Plot
fig, ax = subplots()
ax.plot(np.hstack((times0, times1)), data)
ax.set_title('input')


# 4. Using jitcxde_common.conditional to approjimate a jump
# --------------

from jitcxde_common import conditional

# Define how the parameter changes over time and define the system dynamics
β = conditional(t, t0, β0, β1)
f = [ β * y(0,t-τ) / (1 + y(0,t-τ)**n) - γ*y(0) ]

# Create and integrate the model all the  way to the end, conditional handles 
# the change in the value midway through the integration
DDE = jitcdde_input(f, parameter_spline)
DDE.constant_past(1)
DDE.adjust_diff()
times = DDE.t + np.arange(0, tf)
data = np.vstack([DDE.integrate(time) for time in times])

# Plot
fig, ax = subplots()
ax.plot(times, data)
ax.set_title('conditional')

# 5. Writing the jump ourselves using a callback
# --------------

from symengine import Function

# Define the the python function that will handle the jump and the symengine 
# symbol corresponding to the parameter, which this time is a Function
β = Function('param_jump')
def param_jump_callback(y, t):
    return β0 if t<t0 else β1

# Define the dynamics of the system, β is now an explicit function of time
f = [ β(t) * y(0,t-τ) / (1 + y(0,t-τ)**n) - γ*y(0) ]

# Create and integrate the model all the way to the end, the callback handles 
# the change in the value midway through the integration
DDE = jitcdde(f, callback_functions=[ (β, param_jump_callback, 1) ])
DDE.constant_past(1)
DDE.adjust_diff()
times = DDE.t + np.arange(0, tf)
data = np.vstack([DDE.integrate(time) for time in times])

# Plot
fig, ax = subplots()
ax.plot(times, data)
ax.set_title('callback')
