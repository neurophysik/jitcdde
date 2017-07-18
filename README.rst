JiTCDDE stands for just-in-time compilation for delay differential equations (DDEs). It makes use of `the method described by Thompson and Shampine <http://dx.doi.org/10.1016/S0168-9274(00)00055-6>`_ which is based on the Bogacki–Shampine Runge–Kutta method.
JiTCDDE is designed in analogy to `JiTCODE <http://github.com/neurophysik/jitcode>`_:
It takes an iterable (or generator function) of `SymPy <http://www.sympy.org/>`_ expressions, translates them to C code, compiles them (and an integrator wrapped around them) on the fly, and allows you to operate this integrator from Python.
If you want to integrate ordinary or stochastic differential equations, check out
`JiTCODE <http://github.com/neurophysik/jitcode>`_, or
`JiTCSDE <http://github.com/neurophysik/jitcsde>`_, respectively.

* `Documentation <http://jitcdde.readthedocs.io>`_

* `Issue Tracker <http://github.com/neurophysik/jitcdde/issues>`_

* `Installation instructions <http://jitcde-common.readthedocs.io/#installation>`_ (or just ``pip install jitcdde``).

This work was supported by the Volkswagen Foundation (Grant No. 88463).

