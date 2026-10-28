"""Parameter realism check.

Aim: recompute from their definitions the dimensional examples of the paper (engine oil Br/Omega and Br, liquid-metal Ha,
internal heat generation alpha, micro-length j). Property values are approximate values from Incropera's tables (input;
source to be verified by the user). Produces: the engine-oil values (Br/Omega about 0.04 and 0.26, Br about 1.2e-3 and
8e-3) and the dimensional estimates quoted in Section 3 and the Discussion.
Output: screen and output/parameter_realism_K2.json
Run:  python parameter_realism_K2.py      (instant; needs only numpy)
Reproducible: deterministic.
Requirements: Python 3.12, numpy (see README.md).
"""
import json, os
import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')   # all outputs are written here
os.makedirs(OUT, exist_ok=True)
import numpy as np

out = {}
# 1) Engine oil: Br/Omega = mu^2 a T_inf / (rho k dT^2); Br = Omega * (Br/Omega)
oil = {  # T [K]: (mu [Pa s], rho [kg/m3], k [W/mK]) approx. Incropera Table A.5 (to be verified)
    300: (0.486, 884.1, 0.145),
    320: (0.141, 871.8, 0.143),
    340: (0.0531, 859.9, 0.139),
    360: (0.0252, 847.8, 0.138),
}
a, dT = 500.0, 10.0
rows = []
for T, (mu, rho, k) in oil.items():
    BrOm = mu ** 2 * a * T / (rho * k * dT ** 2)
    Om = dT / T
    rows.append(dict(T=T, BrOm=BrOm, Omega=Om, Br=BrOm * Om,
                     corr_newt_pct=100 * BrOm * Om * 10.16, corr_D5_pct=100 * BrOm * Om * 92.35))
out['engine_oil'] = rows
# Br at Br/Omega=0.1, dT=10 K, T_inf=300 K
out['Br_base_dT10'] = 0.1 * 10 / 300
# 2) Liquid metal: B0 for Ha = 1, a = 10 1/s: B0 = sqrt(Ha rho a / sigma)
lm = {'lithium': (510.0, 3.3e6), 'sodium': (927.0, 1.0e7), 'GaInSn': (6360.0, 3.3e6), 'mercury': (13530.0, 1.0e6)}
out['B0_Ha1'] = {n: float(np.sqrt(1 * r * 10.0 / s)) for n, (r, s) in lm.items()}
out['rho_needed_for_0.06T_sigma3e6'] = 0.06 ** 2 * 3e6 / 10.0
# 3) Internal heat generation: Q0 = alpha rho c a
for name, rc in (('water', 4.18e6), ('engine_oil', 1.67e6)):
    out['Q0_alpha1_' + name] = {str(av): rc * av for av in (1.0, 10.0, 500.0)}
# 4) Micro-length: j~ = j rho a / mu. sigma_m = kappa/(rho j a) = Delta/j~  ->  j~ = Delta/sigma_m
out['jtilde_base'] = {str(D): D / 0.5 for D in (1, 3, 5)}
out['sqrt_jtilde_base'] = {str(D): float(np.sqrt(D / 0.5)) for D in (1, 3, 5)}
# lambda_m sigma_m = Delta mu^2/(gamma rho a); with gamma=(mu+kappa/2) j: lambda_m = 1/((1+D/2) j~)
out['lambda_m_if_closure_with_jtilde'] = {str(D): 1 / ((1 + D / 2) * (D / 0.5)) for D in (1, 3, 5)}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(OUT, 'parameter_realism_K2.json'), 'w'), indent=1)
