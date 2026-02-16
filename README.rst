JiTCDDE stands for just-in-time compilation for delay differential equations (DDEs). It makes use of `the method described by Thompson and Shampine <https://doi.org/10.1016/S0168-9274(00)00055-6>`__ which is based on the Bogacki–Shampine Runge–Kutta method.
JiTCDDE is designed in analogy to `JiTCODE <https://github.com/neurophysik/jitcode>`__:
It takes an iterable (or generator function) of `SymPy <https://www.sympy.org/>`__ expressions, translates them to C code, compiles them (and an integrator wrapped around them) on the fly, and allows you to operate this integrator from Python.
If you want to integrate ordinary or stochastic differential equations, check out
`JiTCODE <https://github.com/neurophysik/jitcode>`__ or
`JiTCSDE <https://github.com/neurophysik/jitcsde>`__, respectively.

* `Documentation <https://jitcdde.readthedocs.io>`__ – Read this to get started and for reference. Don’t miss that some topics are addressed in the `common JiTC*DE documentation <https://jitcde-common.readthedocs.io>`__.

* `Paper <https://doi.org/10.1063/1.5019320>`__ – Read this for the scientific background. Cite this (`BibTeX <https://raw.githubusercontent.com/neurophysik/jitcxde_common/master/citeme.bib>`__) if you wish to give credit or to shift blame.

* `Issue Tracker <https://github.com/neurophysik/jitcdde/issues>`__ – Please report any bugs here. Also feel free to ask for new features.

* `Installation instructions <https://jitcde-common.readthedocs.io/#installation>`__. In most cases, ``pip install jitcdde`` or similar should do the job.

This work was supported by the Volkswagen Foundation (Grant No. 88463).

