"""Comparison with published micropolar axisymmetric data: Borrelli, Giantesio & Patria (2013), CAMWA 66:472-489, Table 1
(M^2=0, c=1).

Mapping (Lukaszewicz -> Eringen: kappa=2 rho nu_r, mu=rho(nu-nu_r); their eta scale sqrt(a/(nu+nu_r)) = our eta/s,
s=sqrt(1+Delta_m)):
  c3 = lm(1+Dm),  c2 = 2 lm sm (1+Dm),  c1 = lm sm Dm
  => Dm = 2c1/(c2-2c1),  lm = c3/(1+Dm),  sm = c1/(lm Dm)
  f''(0)_B = s F''(0),  G'(0)_B = -Dm G'(0),  alpha_B = delta_inf/s.
Produces: the micropolar comparison rows of Table 1 (same numbers as in output/tables_K2.json, key 3).
Run:  python validate_borrelli2013.py      (prints to screen; a few seconds)
Reproducible: deterministic.
Requirements: Python 3.12, numpy, scipy 1.18 (see README.md).
"""
import numpy as np
from micropolar_solver import solve, wall
ref=[(0.1,1.5,0.1,1.3013,-0.0558,0.5665),(0.1,1.5,0.5,1.3030,-0.0531,0.5667),(0.1,3.0,0.1,1.3043,-0.0470,0.5670),(0.1,3.0,0.5,1.3051,-0.0458,0.5671),
     (0.5,1.5,0.1,1.2583,-0.2793,0.5568),(0.5,1.5,0.5,1.2669,-0.2657,0.5576),(0.5,3.0,0.1,1.2734,-0.2354,0.5594),(0.5,3.0,0.5,1.2774,-0.2292,0.5598)]
rows=[]
for c1,c2,c3,f2,G1,al in ref:
    D=2*c1/(c2-2*c1); lm=c3/(1+D); sm=c1/(lm*D); s=np.sqrt(1+D)
    p=dict(D=D,lm=lm,sm=sm,Ha=0,lam=0,Pr=0.7,al=0); so=solve(p,'cyl',L=25,tol=1e-10); w=wall(so,p)
    f2p=s*w['F2']; G1p=-D*w['G1']; alp=(25-so.sol(25)[0])/s
    rows.append((c1,c2,c3,f2,f2p,G1,G1p,al,alp))
    print('c=(%.1f,%.1f,%.1f) Δm=%.4f λm=%.4f σm=%.4f | f″: %.4f / %.5f | G′: %.4f / %.5f | α: %.4f / %.5f'%(c1,c2,c3,D,lm,sm,f2,f2p,G1,G1p,al,alp))
