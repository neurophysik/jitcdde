JiTCDDE stands for just-in-time compilation for delay differential equations (DDEs). It makes use of `the method described by Thompson and Shampine <http://dx.doi.org/10.1016/S0168-9274(00)00055-6>`_ which is based on the Bogacki–Shampine Runge–Kutta method. Briefly, past states are obtained from a cubic Hermite interpolation with the result of past integration steps being the anchors.

It can also calculate Lyapunov exponents with a method similar to `the one described by Farmer <http://dx.doi.org/10.1016/0167-2789(82)90042-2>`_, which in turn is essentially `the method described by Benettin et al. <http://dx.doi.org/10.1007/BF02128236>`_. However, there is a crucial change to adapt those to the Shampine–Thompson approach, namely that for purposes of obtaining the norms of tangent vectors and applying the Gram–Schmidt orthonormalisation, a function scalar product is used.

JiTCDDE is designed in analogy to `JiTCODE <http://github.com/neurophysik/jitcode>`_:
It takes an iterable (or generator function) of `SymPy <http://www.sympy.org/>`_ expressions, translates them to C code, compiles them (and an integrator wrapped around them) on the fly, and allows you to operate this integrator from Python.

This is work in progress, so you are particularly encouraged to report any issues and feature request on the `Issue Tracker <http://github.com/neurophysik/jitcdde/issues>`_ or directly to me.

`Here <http://jitcdde.readthedocs.io>`_ is a documentation automatically generated from docstrings (which will be extended soon).

