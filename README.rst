JiTCDDE stands for just-in-time compilation for delay differential equations (DDEs). It makes use of the method described by Thompson and Shampine and is designed in analogy to `JiTCODE <http://github.com/neurophysik/jitcode>`:
It takes an iterable (or generator function) of `SymPy <http://www.sympy.org/>`_ expressions, translates them to C code, compiles them (and an integrator wrapped around them) on the fly, and allows you to operate this integrator from Python. It can also calculate Lyapunov exponents.

This is work in progress, so you are particularly encouraged to report any issues and feature request on the `Issue Tracker <http://github.com/neurophysik/jitcdde/issues>`_ or directly to me.

`Here <http://jitcdde.readthedocs.io>`_ is a documentation automatically generated from docstrings (which will be extended soon).

