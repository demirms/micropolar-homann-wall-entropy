"""Small-Prandtl-number correction to the Hermite limit alpha_c -> 4.

Independent derivation (sympy, step by step) of
    4 - alpha_c ~ (4/sqrt(pi)) * delta_inf * sqrt(Pr)
by a boundary perturbation of the Hermite eigenvalue problem, two routes:
  (i)  Sturm-Liouville domain perturbation (Hadamard formula),
  (ii) implicit differentiation of the Hermite function condition H_nu(s0) = 0.
Second-order term from the exact shifted Hermite condition (mpmath/scipy).
Comparison with the numerical alpha_c(Pr) from the independent Chebyshev generalised-eigenvalue code
(independent_chebyshev_K2.py, same folder).
Produces: Eq. (hermite) of the paper, 4 - alpha_c = (4/sqrt(pi)) delta_inf sqrt(Pr) + O(Pr), with the coefficient 1.108 and the
ratios (4-alpha_c)/sqrt(Pr) quoted in Section 5.
Output: output/hermite_correction_K2.json
Run:  python hermite_correction_K2.py      (several minutes)
Reproducible: deterministic.
Requirements: Python 3.12, numpy, scipy 1.18, sympy (see README.md).
"""
import json, os, sys
import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, 'output')   # all outputs are written here
os.makedirs(OUT, exist_ok=True)

out = {}

# ---------------------------------------------------------------- 1. symbolic
s, a, s0, nu = sp.symbols('s alpha s0 nu', real=True)
th = sp.Function('theta')
phi = sp.Function('phi')
# Outer equation: F = eta - delta, s = sqrt(Pr)(eta - delta)
#   theta_ss + 2 s theta_s + alpha theta = 0  on s >= s0 = -delta sqrt(Pr)
eq = sp.diff(th(s), s, 2) + 2 * s * sp.diff(th(s), s) + a * th(s)
sub = eq.subs(th(s), sp.exp(-s**2) * phi(s)).doit()
herm_eq = sp.simplify(sp.expand(sub * sp.exp(s**2)))
out['step1_transformed_equation'] = str(herm_eq)  # phi'' - 2 s phi' + (alpha-2) phi
check1 = sp.simplify(herm_eq - (sp.diff(phi(s), s, 2) - 2 * s * sp.diff(phi(s), s) + (a - 2) * phi(s)))
out['step1_check_zero'] = str(check1)
# Unperturbed (s0 = 0): first odd Hermite polynomial H1 = 2s, alpha - 2 = 2 -> alpha = 4
H1 = sp.hermite(1, s)
res = sp.simplify(sp.diff(H1, s, 2) - 2 * s * sp.diff(H1, s) + 2 * H1)
out['step2_H1'] = str(H1); out['step2_residual'] = str(res)
# Sturm-Liouville form: (p phi')' + lam w phi = 0 with p = w = exp(-s^2), lam = alpha - 2
p = sp.exp(-s**2)
sl = sp.simplify((sp.diff(p * sp.diff(phi(s), s), s) + (a - 2) * p * phi(s)) / p - herm_eq)
out['step3_SL_form_check'] = str(sl)
# Route (i): Hadamard formula for a Dirichlet end s0: d lam/d s0 = p(s0) phi'(s0)^2 / int w phi^2
num = (p * sp.diff(H1, s)**2).subs(s, 0)
den = sp.integrate(p * H1**2, (s, 0, sp.oo))
dlam_ds0_i = sp.simplify(num / den)
out['step4_route_i'] = dict(numerator=str(num), denominator=str(den), dalpha_ds0=str(dlam_ds0_i))
# Route (ii): H_nu(0) = 2^nu sqrt(pi)/Gamma((1-nu)/2); condition H_nu(s0)=0 at (s0, nu) = (0, 1).
# Near nu = 1 put e = (1-nu)/2 and use the series 1/Gamma(e) = e + EulerGamma e^2 + ...
e = sp.symbols('e')
rg = sp.series(1 / sp.gamma(e), e, 0, 3).removeO()
Hnu0_ser = 2**nu * sp.sqrt(sp.pi) * rg.subs(e, (1 - nu) / 2)
dH_dnu = sp.simplify(sp.diff(Hnu0_ser, nu).subs(nu, 1))
dH_ds = sp.diff(H1, s).subs(s, 0)          # dH_nu/ds at nu = 1, s = 0
dnu_ds0 = -dH_ds / dH_dnu
dalpha_ds0_ii = sp.simplify(2 * dnu_ds0)   # alpha = 2 + 2 nu
out['step5_route_ii'] = dict(series_1_over_gamma=str(rg), dHnu0_dnu=str(dH_dnu), dH_ds=str(dH_ds),
                             dalpha_ds0=str(dalpha_ds0_ii))
assert sp.simplify(dlam_ds0_i - dalpha_ds0_ii) == 0
c1 = float(dlam_ds0_i)                     # 4/sqrt(pi)
out['c1_4_over_sqrtpi'] = c1
print('c1 =', dlam_ds0_i, '=', c1, flush=True)
# alpha_c ~ 4 + c1 s0 = 4 - c1 delta sqrt(Pr)

# ---------------------------------------- 2. exact shifted Hermite condition
from scipy.special import hyp1f1, rgamma
from scipy.optimize import brentq
def hermite_fn(v, x):
    """Hermite function H_v(x) (non-integer order), Kummer representation."""
    return 2**v * np.sqrt(np.pi) * (rgamma((1 - v) / 2) * hyp1f1(-v / 2, 0.5, x * x)
                                   - 2 * x * rgamma(-v / 2) * hyp1f1((1 - v) / 2, 1.5, x * x))
def alpha_shift(s0v):
    """Smallest alpha with a Gaussian-decaying solution of theta''+2s theta'+alpha theta=0, theta(s0)=0."""
    v = brentq(lambda v: hermite_fn(v, s0v), 0.3, 1.0 + 1e-12 if s0v < 0 else 1.6, xtol=1e-15)
    return 2 + 2 * v
assert abs(alpha_shift(-1e-9) - 4) < 1e-6
hs = 1e-3
am = alpha_shift(-hs)
am2 = alpha_shift(-2 * hs)
# one-sided (s0 < 0 is the physical side): alpha = 4 + c1 s0 + c2 s0^2
c2 = ((am2 - 4) / (-2 * hs) - (am - 4) / (-hs)) / (-hs)
d1 = 2 * (am - 4) / (-hs) - (am2 - 4) / (-2 * hs)
out['shifted_hermite'] = dict(d1_numeric=d1, c2_numeric=c2)
print('shifted Hermite: d1 =', d1, ' c2 =', c2, flush=True)

# ----------------------------------------------- 3. numerical alpha_c(Pr)
from independent_chebyshev_K2 import Grid, full, alpha_c_eig, P
NN, LL = 120, 16.0
g = Grid(NN, LL)
rows = []
for D in (0, 1, 3, 5):
    _, f = full(g, P(D=D))
    Fc = f['F']; delta = LL - Fc[g.iL]
    coef = np.polynomial.chebyshev.chebfit(2 * g.eta / LL - 1, Fc, NN)
    Fint = (lambda coef, delta: (lambda e: np.where(e <= LL, np.polynomial.chebyshev.chebval(
        2 * np.minimum(e, LL) / LL - 1, coef), e - delta)))(coef, delta)
    for Pr in (0.1, 0.03, 0.01, 0.003, 0.001):
        Lt = np.sqrt(40 / Pr) + 2
        a1 = alpha_c_eig(Fint, Pr, Lt, 180); a2 = alpha_c_eig(Fint, Pr, Lt * 1.3, 230)
        sq = np.sqrt(Pr)
        lead = 4 - c1 * delta * sq
        exact_shift = alpha_shift(-delta * sq)
        rows.append(dict(D=D, delta=float(delta), Pr=Pr, alpha_c=float(a1), check=float(abs(a1 - a2)),
                         ratio=float((4 - a1) / sq), coef_pred=float(c1 * delta),
                         alpha_lead=float(lead), alpha_shift_exact=exact_shift,
                         err_lead=float(a1 - lead), err_shift=float(a1 - exact_shift)))
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
out['table'] = rows
# Richardson-type extrapolation of the ratio to sqrt(Pr) -> 0 (linear in sqrt(Pr), last two points)
ext = {}
for D in (0, 1, 3, 5):
    r = [x for x in rows if x['D'] == D]
    x1, x2 = np.sqrt(r[-2]['Pr']), np.sqrt(r[-1]['Pr'])
    y1, y2 = r[-2]['ratio'], r[-1]['ratio']
    ext[D] = dict(extrapolated=float(y2 - (y2 - y1) / (x2 - x1) * x2), predicted=r[-1]['coef_pred'])
out['ratio_extrapolated'] = ext
print(ext)
json.dump(out, open(os.path.join(OUT, 'hermite_correction_K2.json'), 'w'), indent=1, default=float)
print('done')
