"""Vendored subset of the kernel-xgb library.

Only the two modules that `aiv.weak_kernel` orchestrates are copied here:
`kernel` (LeafKernel) and `weakness` (spectral clustering, cluster error report,
JS weak-region profile). Both depend only on numpy, scipy, scikit-learn and
pandas. See aiv/_vendor/__init__.py for why this is vendored.
"""
