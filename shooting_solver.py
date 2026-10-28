"""Independent second solver: shooting method (DOP853 + fsolve) for the prismatic problem (also usable for 'cyl').

Unknowns: F''(0), H'(0), M'(0), G'(0), N'(0), R'(0), theta'(0). Targets at eta=Ls: F'=1, H=M=G=N=R=theta=0.
Produces: the shooting solution behind the 3e-14 agreement with the collocation values quoted in Section 4 (called by
verify_numerics_K2.py, entry 'solvers' of output/numerics_K2.json). Use: imported; not run on its own.
Requirements: Python 3.12, numpy, scipy 1.18, matplotlib (see README.md).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from micropolar_solver import rhs_factory
def shoot(p, geom='prism', Ls=10.0, guess=None):
    f = rhs_factory(p, geom)
    def ode(t, y): return f(t, y.reshape(15,1)).ravel()
    def res(s):
        y0 = np.array([0,0,s[0], 1,s[1], 0,s[2], 0,s[3], 0,s[4], 0,s[5], 1,s[6]], float)
        r = solve_ivp(ode, (0,Ls), y0, method='DOP853', rtol=1e-12, atol=1e-13)
        yL = r.y[:,-1]
        return [yL[1]-1, yL[3], yL[5], yL[7], yL[9], yL[11], yL[13]]
    g = guess if guess is not None else [1.0,-0.7,0.15,0.15,0.01,0.1,-0.3]
    s, info, ier, msg = fsolve(res, g, full_output=True, xtol=1e-13)
    return dict(F2=s[0],H1=s[1],M1=s[2],G1=s[3],N1=s[4],R1=s[5],T1=s[6]), ier, np.max(np.abs(res(s)))
