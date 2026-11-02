"""Effective-viscosity mapping of the wall heat flux.

Checks -theta'(0; Dm, Pr) = -theta'_N(0; Ha, (1+Dm) Pr, alpha) / sqrt(1+Dm), which is exact for lm*sm = 0,
against the lm = 0 reference and the full micropolar solution (Pr = 0.7 and 7, Dm = 0, 1, 3, 5, base values).
Paper item: heat-flux paragraph of the Results (equation of the mapping; 0.15 % at Pr = 0.7, 0.93 % at Pr = 7).
Output: output/check_heat_flux_mapping.json
"""
import json
import os
import numpy as np
from micropolar_solver import solve, wall

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')   # all outputs are written here
os.makedirs(OUT, exist_ok=True)

BASE = dict(lm=0.5, sm=0.5, Ha=1.0, lam=0.5, al=0.2)


def heat_flux(p):
    s = solve(p, 'cyl', L=20.0, tol=1e-9, robin=True)
    assert s.status == 0, s.message
    return -wall(s, p)['T1']


out = []
for Pr in (0.7, 7.0):
    for Dm in (0.0, 1.0, 3.0, 5.0):
        full = heat_flux(dict(BASE, D=Dm, Pr=Pr, sm=(0.0 if Dm == 0 else 0.5)))
        lm0 = heat_flux(dict(BASE, D=Dm, Pr=Pr, lm=0.0))
        newt = heat_flux(dict(BASE, D=0.0, sm=0.0, Pr=(1 + Dm) * Pr))
        mapped = newt / np.sqrt(1 + Dm)
        out.append(dict(Pr=Pr, Dm=Dm, full=full, lm0=lm0, newtonian_at_Pr_star=newt, mapped=mapped,
                        mapped_vs_lm0=abs(mapped - lm0), rel_dev_full_pct=100 * (mapped - full) / full))
        print(f"Pr={Pr:4} Dm={Dm:3}  full={full:.6f}  lm0={lm0:.6f}  mapped={mapped:.6f}  "
              f"|mapped-lm0|={abs(mapped - lm0):.1e}  dev={100 * (mapped - full) / full:+.3f}%")
json.dump(out, open(os.path.join(OUT, 'check_heat_flux_mapping.json'), 'w'), indent=1, default=float)
