"""Extra numbers quoted in the paper.

1) Momentum-layer thickness eta_99 (F' = 0.99) at Ha = 1, base values, Delta_m = 0, 1, 3, 5, compared with the micro-length
   sqrt(j~) = sqrt(Delta_m / sigma_m) (sigma_m = 0.5).
2) Separation of the zero-couple and zero-shear points along the plate, zeta_c - zeta_0 (V_r = 1), and the ratio
   zeta_c / zeta_0 for Ha = 0, 1, 3 (from the T12 entry of the independent Chebyshev results).
3) Hermite approach: (4 - alpha_c) / sqrt(Pr) from the 'hermite' entries of the independent Chebyshev results.
Uses the independent Chebyshev solver.
Produces: the slopes zeta_c/zeta_0 and the percentages (30-36 % and 47-54 %) of Section 5 and the ratios (4-alpha_c)/sqrt(Pr).
Input: output/independent_chebyshev_K2.json (run independent_chebyshev_K2.py first).
Output: output/v1_extra_numbers_K2.json
Run:  python v1_extra_numbers_K2.py      (seconds)
Reproducible: deterministic.
Requirements: Python 3.12, numpy, scipy 1.18 (see README.md).
"""
import json, os
import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')   # all outputs are written here
os.makedirs(OUT, exist_ok=True)
import numpy as np
from scipy.interpolate import interp1d
from independent_chebyshev_K2 import Grid, full, P

out = {}
g = Grid(160, 16.0)
rows = []
for D in (0, 1, 3, 5):
    p = P(D=D)
    w, f = full(g, p)
    Fp = g.D1 @ f['F']
    eta = g.eta[::-1]; fp = Fp[::-1]
    k = np.argmax(fp >= 0.99)
    e99 = float(interp1d(fp[k - 1:k + 1], eta[k - 1:k + 1])(0.99))
    rows.append(dict(D=D, eta99=e99, sqrt_jtilde=float(np.sqrt(D / 0.5)) if D else 0.0))
    print(rows[-1])
out['eta99_Ha1'] = rows

ind = json.load(open(os.path.join(OUT, 'independent_chebyshev_K2.json')))
sep = []
for r in ind['T12']:
    if 'zec' not in r:
        continue
    sep.append(dict(Ha=r['Ha'], D=r['D'], dz=r['zec'] - r['ze0'], ratio=r['zec'] / r['ze0']))
out['separation'] = sep
print(sep)
out['hermite_scaled'] = [dict(Pr=pr, ac=ac, gap=4 - ac, gap_over_sqrtPr=(4 - ac) / np.sqrt(pr))
                         for pr, ac, _ in ind['hermite']]
print(out['hermite_scaled'])
json.dump(out, open(os.path.join(OUT, 'v1_extra_numbers_K2.json'), 'w'), indent=1, default=float)
