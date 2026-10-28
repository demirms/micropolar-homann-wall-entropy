"""Supercritical example check: for Delta_m=1, Pr=0.7, Ha=1 and alpha=3 the minimum of theta of the selected (Robin) solution
(paper: about -5.4), and for alpha=2.4 the maximum of theta (paper: 2.40).
Solver: independent Chebyshev solver (two resolutions). Produces: these two values quoted in Section 5.
Output: screen only (no file).
Run:  python theta_min_supercritical_K2.py      (seconds; needs independent_chebyshev_K2.py in the same folder)
Reproducible: deterministic.
Requirements: Python 3.12, numpy, scipy 1.18 (see README.md).
"""
import numpy as np
from independent_chebyshev_K2 import Grid, full, P, solve_theta
for N, L in ((120, 16.0), (160, 22.0)):
    g = Grid(N, L); _, f = full(g, P(D=1))
    t3 = solve_theta(g, f['F'], 0.7, 3.0); t24 = solve_theta(g, f['F'], 0.7, 2.4)
    print(N, L, 'min theta(alpha=3) = %.5f' % t3.min(), 'max theta(alpha=2.4) = %.5f' % t24.max())
