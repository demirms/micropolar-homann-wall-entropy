"""Distance between the special points and magnitude of the residual wall irreversibility.

Aim: measure the physical weight of the minimum-wall-entropy result: d (distance between the zero-shear and the zero-couple-
stress nodes), the residual term (Br/Omega) AB/(A+B) d^2 and its ratio to N_s(0)_min and to the frictional part.
Ha=1, Pr=0.7, lambda=0.5, alpha=0.2, lm=sm=0.5, Br/Omega=0.1; Vr = 0, 1, 2. Solver: independent Chebyshev solver.
Produces: the "at most 1.2 % of the wall value for Vr <= 2" quoted in the Conclusions and the node separations in Section 5.
Output: output/point_separation_magnitude_K2.json
Run:  python point_separation_magnitude_K2.py      (seconds; needs independent_chebyshev_K2.py in the same folder)
Reproducible: deterministic.
Requirements: Python 3.12, numpy, scipy 1.18 (see README.md).
"""
import json, os
import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')   # all outputs are written here
os.makedirs(OUT, exist_ok=True)
import numpy as np
from independent_chebyshev_K2 import Grid, full, P

g = Grid(120, 16.0); rows = []
for D in (1, 3, 5):
    p = P(D=D); w, _ = full(g, p)
    A = (1 + D) * w['F2'] ** 2; B = D / (p['lm'] * p['sm']) * w['G1'] ** 2
    for Vr in (0.0, 1.0, 2.0):
        dx = -w['M1'] / w['F2'] + w['N1'] / w['G1']
        dz = Vr * (-w['H1'] / w['F2'] - w['R1'] / w['G1'])
        d2 = dx ** 2 + dz ** 2
        resid = 0.1 * A * B / (A + B) * d2
        nsmin = w['T1'] ** 2 + resid + 0.1 * p['Ha'] * Vr ** 2
        rows.append(dict(D=D, Vr=Vr, d=float(np.sqrt(d2)), resid=resid, Nsmin=nsmin,
                         resid_over_Nsmin_pct=100 * resid / nsmin))
        print(rows[-1])
json.dump(rows, open(os.path.join(OUT, 'point_separation_magnitude_K2.json'), 'w'), indent=1, default=float)
