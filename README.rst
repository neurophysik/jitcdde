JiTCDDE stands for just-in-time compilation for delay differential equations (DDEs). It makes use of `the method described by Thompson and Shampine <http://dx.doi.org/10.1016/S0168-9274(00)00055-6>`_ which is based on the Bogacki–Shampine Runge–Kutta method.
JiTCDDE is designed in analogy to `JiTCODE <http://github.com/neurophysik/jitcode>`_:
It takes an iterable (or generator function) of `SymPy <http://www.sympy.org/>`_ expressions, translates them to C code, compiles them (and an integrator wrapped around them) on the fly, and allows you to operate this integrator from Python.
If you want to integrate ordinary or stochastic differential equations, check out
`JiTCODE <http://github.com/neurophysik/jitcode>`_, or
`JiTCSDE <http://github.com/neurophysik/jitcsde>`_, respectively.

* `Documentation <http://jitcdde.readthedocs.io>`_ – Read this to get started and for reference. Don’t miss that some topics are addressed in the `common JiTC*DE documentation <http://jitcde-common.readthedocs.io>`_.

* `Paper <https://doi.org/10.1063/1.5019320>`_ – Read this for the scientific background. Cite this (`BibTeX <https://raw.githubusercontent.com/neurophysik/jitcxde_common/master/citeme.bib>`_) if you wish to give credit or to shift blame.

* `Issue Tracker <http://github.com/neurophysik/jitcdde/issues>`_ – Please report any bugs here. Also feel free to ask for new features.

* `Installation instructions <http://jitcde-common.readthedocs.io/#installation>`_. In most cases, `pip3 install jitcode` or similar should do the job.

This work was supported by the Volkswagen Foundation (Grant No. 88463).

