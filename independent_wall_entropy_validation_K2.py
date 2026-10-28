"""Independent recomputation, supplement (quick pre-submission check).

Aim: (i) wall entropy generation N_s(0) and Be(0) (Table 5, Ha=1, Pr=0.7) from Eq. (Ns0), using the Chebyshev solution;
(ii) find the minimum wall-entropy point by direct numerical minimisation and compare with Eq. (nodem);
(iii) the validation table (Wang-Ng 2013 Newtonian, Wang 1973 H'(0), two Borrelli 2013 rows) with the Chebyshev solver.
The paper's main code is not used.
Produces: the independent confirmation of Table 5 (N_s(0), Be(0)), of the minimum point of Table 3 and of Table 1.
Output: output/independent_wall_entropy_validation_K2.json
Run:  python independent_wall_entropy_validation_K2.py      (seconds; needs independent_chebyshev_K2.py in the same folder)
Reproducible: deterministic (Nelder-Mead with fixed start).
Requirements: Python 3.12, numpy, scipy 1.18 (see README.md).
"""
import json, os
import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')   # all outputs are written here
os.makedirs(OUT, exist_ok=True)
import numpy as np
from scipy.optimize import minimize
from independent_chebyshev_K2 import Grid, full, P

g = Grid(120, 16.0)
out = {'Ns0': [], 'min_check': []}
for D in (0, 1, 3, 5):
    p = P(D=D); w, _ = full(g, p)
    lm, sm, Ha = p['lm'], p['sm'], p['Ha']
    def Ns0(x, Vr=1.0):
        xi, ze = x
        tx = (1 + D) * (xi * w['F2'] + w['M1']); ty = (1 + D) * (ze * w['F2'] + Vr * w['H1'])
        cx = Vr * w['R1'] - ze * w['G1']; cy = xi * w['G1'] + w['N1']
        cpl = D / (lm * sm) * (cx ** 2 + cy ** 2) if D > 0 else 0.0
        return w['T1'] ** 2 + 0.1 * ((tx ** 2 + ty ** 2) / (1 + D) + cpl + Ha * Vr ** 2)
    v = Ns0((1.0, 0.0))
    out['Ns0'].append([D, v, w['T1'] ** 2 / v])
    if D > 0:
        r = minimize(Ns0, (0.0, 0.5), method='Nelder-Mead', options=dict(xatol=1e-12, fatol=1e-15, maxiter=20000))
        A = (1 + D) * w['F2'] ** 2; B = D / (lm * sm) * w['G1'] ** 2
        xim = (A * (-w['M1'] / w['F2']) + B * (-w['N1'] / w['G1'])) / (A + B)
        zem = (A * (-w['H1'] / w['F2']) + B * (w['R1'] / w['G1'])) / (A + B)
        out['min_check'].append([D, r.x[0], r.x[1], xim, zem, float(np.hypot(r.x[0] - xim, r.x[1] - zem))])
print('Ns0', out['Ns0']); print('min', out['min_check'])
# validation
val = {}
for Pr in (0.7, 7.0):
    w, _ = full(g, dict(D=0, lm=0.5, sm=0.0, Ha=0, lam=1.0, Pr=Pr, al=0.0))
    val['newt_Pr%g' % Pr] = [w['F2'], w['M1'], -w['T1'], w['H1']]
for c1, c2, c3 in ((0.1, 1.5, 0.1), (0.5, 3.0, 0.5)):
    D = 2 * c1 / (c2 - 2 * c1); lm = c3 / (1 + D); sm = c1 / (lm * D); s = np.sqrt(1 + D)
    w, _ = full(g, dict(D=D, lm=lm, sm=sm, Ha=0, lam=0.0, Pr=0.7, al=0.0))
    val['borrelli_%g_%g_%g' % (c1, c2, c3)] = [s * w['F2'], -D * w['G1'], w['delta'] / s]
out['validation'] = val
print(json.dumps(val, indent=1))
json.dump(out, open(os.path.join(OUT, 'independent_wall_entropy_validation_K2.json'), 'w'), indent=1, default=float)
