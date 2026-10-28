"""Solver library: micropolar stagnation-point (Homann) flow past a translating vertical plate.

geom='prism' -> prismatic (plane) problem; geom='cyl' -> cylindrical (axisymmetric) problem, which is the
problem of the paper. Independent implementation with scipy solve_bvp (collocation, bvp4c equivalent).
Produces: the wall gradients F''(0), M'(0), H'(0), G'(0), N'(0), R'(0), theta'(0) and the full profiles used by every
table and figure of the paper (Tables 1-6, Figures 1-5 and the numbers quoted in the text).
Use: imported by the other scripts; "python micropolar_solver.py" prints the base-case wall values.
Requirements: Python 3.12, numpy, scipy 1.18, matplotlib (see README.md).
"""
import numpy as np
from scipy.integrate import solve_bvp

def rhs_factory(p, geom):
    D,lm,sm,Ha,lam,Pr,al = (p[k] for k in ('D','lm','sm','Ha','lam','Pr','al'))
    c = 1.0 if geom=='prism' else 2.0
    def rhs(x, y):
        F,F1,F2,H,H1,M,M1,G,G1,N,N1,R,R1,T,T1 = y
        # G'' from the microrotation equation (prismatic / cylindrical form)
        if geom=='prism':
            G2 = -lm*(F*G1-F1*G) + lm*sm*(2*G-F2)
        else:
            G2 = -lm*(2*F*G1-G*F1) + lm*sm*(2*G-F2)
        F3 = -(c*F*F2 - F1**2 + Ha*(1-F1) - D*G1 + 1)/(1+D)
        if geom=='prism':
            H2 = -(F*H1 - Ha*H + D*R1)/(1+D)
        else:
            H2 = -(2*F*H1 - F1*H - Ha*H + D*R1)/(1+D)
        M2 = -(c*F*M1 - F1*M - Ha*M + lam*T - D*N1)/(1+D)
        N2 = -lm*(c*F*N1 - G*M) + lm*sm*(2*N - M1)
        if geom=='prism':
            R2 = -lm*F*R1 + lm*sm*(2*R + H1)
        else:
            R2 = -lm*(2*F*R1 + G*H) + lm*sm*(2*R + H1)
        T2 = -(c*Pr*F*T1 + al*Pr*T)
        return np.vstack([F1,F2,F3,H1,H2,M1,M2,G1,G2,N1,N2,R1,R2,T1,T2])
    return rhs

def make_bc(S=0.0, robin=None):
  """robin=(c,Pr,al): asymptotic Robin condition for theta: theta'(L)+[cPr(L-beta)-(al/c-1)/(L-beta)]theta(L)=0,
  with L-beta = F(L) (F = eta-beta in the outer flow). robin=None -> theta(L)=0."""
  def bc(a, b):
    if robin is None:
        tb = b[13]
    else:
        c,Pr,al = robin; zL=b[0]
        tb = b[14] + (c*Pr*zL - (al/c-1.0)/zL)*b[13]
    return np.array([a[0],a[1]-S,b[1]-1, a[3]-1,b[3], a[5],b[5], a[7],b[7], a[9],b[9], a[11],b[11], a[13]-1,tb])
  return bc

def solve(p, geom='prism', L=15.0, tol=1e-9, n=600, guess=None, robin=False):
    x = np.linspace(0, L, n)
    if guess is None:
        y = np.zeros((15, n)); e = np.exp(-x)
        y[0]=x-1+e; y[1]=1-e; y[2]=e; y[3]=e; y[4]=-e; y[13]=e; y[14]=-e
    else:
        y = guess.sol(x)
    s = solve_bvp(rhs_factory(p, geom), make_bc(p.get('S',0.0), ((1.0 if geom=='prism' else 2.0),p['Pr'],p['al']) if robin else None), x, y, tol=tol, max_nodes=500000)
    return s

def wall(s, p):
    y0 = s.sol(0.0)
    D = p['D']
    return dict(F2=y0[2], H1=y0[4], M1=y0[6], G1=y0[8], N1=y0[10], R1=y0[12], T1=y0[14])

BASE = dict(D=1.0, lm=0.5, sm=0.5, Ha=1.0, lam=0.5, Pr=0.7, al=0.2)
if __name__ == '__main__':
    for g in ('prism','cyl'):
        s = solve(BASE, g); print(g, s.status, {k: round(v,6) for k,v in wall(s,BASE).items()})
