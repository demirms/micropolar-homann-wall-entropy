"""Local entropy generation of the cylindrical (axisymmetric) problem, dimensionless (library module).

Ns = theta'^2 + (Br/Omega){ (2+Delta)[6F'^2 + 1/2 (M'+xi F'')^2 + 1/2 (Vr H'+zeta F'')^2]
     + (Delta/2)[(Vr(H'+2R)+zeta(F''-2G))^2 + (M'+xi(F''-2G)-2N)^2]
     + (Delta/(lm sm))[2(1-b)G^2 + (Vr R'-zeta G')^2 + (N'+xi G')^2]
     + Ha[(M+xi F')^2 + (Vr H+zeta F')^2] },   xi=x/eps, zeta=y/eps, Vr=V0/(a eps), b=beta_v/gamma_v (|b|<=1).
Produces: N_s and its components (Eq. for the dissipation function), the excess entropy S_ex; used for Figures 2, 4, 5
and Tables 3, 5 (through make_results_K2.py, extra_K2.py, reviewer_checks_K2.py). Use: imported by other scripts.
Requirements: Python 3.12, numpy, scipy 1.18, matplotlib (see README.md).
"""
import numpy as np
def Ns_parts(y, p, xi, zeta, Vr, BrOm, b=0.0):
    F,F1,F2,H,H1,M,M1,G,G1,N,N1,R,R1,T,T1 = y
    D,lm,sm,Ha = p['D'],p['lm'],p['sm'],p['Ha']
    heat=T1**2
    visc=(2+D)*(6*F1**2+0.5*(M1+xi*F2)**2+0.5*(Vr*H1+zeta*F2)**2) \
        +0.5*D*((Vr*(H1+2*R)+zeta*(F2-2*G))**2+(M1+xi*(F2-2*G)-2*N)**2)
    couple=(D/(lm*sm))*(2*(1-b)*G**2+(Vr*R1-zeta*G1)**2+(N1+xi*G1)**2) if D>0 else 0*F
    joule=Ha*((M+xi*F1)**2+(Vr*H+zeta*F1)**2)
    return dict(heat=heat,visc=BrOm*visc,couple=BrOm*couple,joule=BrOm*joule)
def Ns(y,p,xi,zeta,Vr,BrOm,b=0.0):
    d=Ns_parts(y,p,xi,zeta,Vr,BrOm,b); t=sum(d.values()); return t, d['heat']/t
def Ns_inf(p,xi,zeta,BrOm):
    return BrOm*(6*(2+p['D'])+p['Ha']*(xi**2+zeta**2))
def S_ex(sol,p,xi,zeta,Vr,BrOm,b=0.0,n=20001):
    from scipy.integrate import simpson
    L=sol.x[-1]; x=np.linspace(0,L,n); d=Ns_parts(sol.sol(x),p,xi,zeta,Vr,BrOm,b)
    return simpson(sum(d.values())-Ns_inf(p,xi,zeta,BrOm),x=x)
