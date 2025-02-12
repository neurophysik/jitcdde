import numpy as np
from matplotlib import pyplot as plt

from chspy import CubicHermiteSpline

from jitcdde import input, jitcdde_input, t, y  # noqa: A004


rng = np.random.default_rng(seed=42)

# Defining Input
# --------------

# input_data and input_times would usually be real data. We choose them the way we do to keep the example simple.

input_times = np.linspace(0, 70, 20)
max_time = input_times[-1]

input_data = rng.normal(size=(20,2))
input_spline = CubicHermiteSpline.from_data(input_times,input_data)

fig,axes = plt.subplots()
input_spline.plot(axes)
axes.set_title("input")
plt.show()

# Defining Dynamics
# -----------------

τ = 10
f = [
	-y(0) + y(1,t-τ) + input(0),
	-y(1) + y(0,t-τ) + input(1),
]

# Actual Integration
# ------------------

DDE = jitcdde_input(f,input_spline)
DDE.constant_past(np.zeros(len(f)))
DDE.adjust_diff()
times = np.linspace(DDE.t, max_time, 100)
result = np.vstack([DDE.integrate(time) for time in times])

# Plotting
# --------

fig,axes = plt.subplots()
axes.plot(times, result)
axes.set_title("result")
plt.show()

