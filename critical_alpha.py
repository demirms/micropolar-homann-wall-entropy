"""Critical heat-generation parameter alpha_c and zero-heat-flux value alpha_0 (library module + demonstration run).

Homogeneous problem theta''+c Pr F theta'+alpha Pr theta=0, theta(0)=0, theta rapidly decaying; alpha_c is its first
eigenvalue. Also the dependence of -theta'(0) on alpha and the alpha_0 at which it changes sign.
Produces: Table 4 (delta_inf, alpha_0, alpha_c) via verify_numerics_K2.py (entry 'T1' of output/numerics_K2.json).
Use: imported; "python critical_alpha.py" prints alpha_c, alpha_0 for the prismatic and cylindrical cases (not used
in the paper). Requirements: Python 3.12, numpy, scipy 1.18, matplotlib (see README.md).
"""
import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import brentq
from micropolar_solver import solve
def flowF(p,geom,L=20):
    q=dict(p); q['al']=0.0; q['lam']=0.0
    s=solve(q,geom,L=L); return s
def theta_wall(al,sF,Pr,c,L=20):
    x=np.linspace(0,L,1500); F=lambda z: sF.sol(z)[0]
    beta=L-sF.sol(L)[0]
    def fun(z,y): return np.vstack([y[1],-(c*Pr*F(z)*y[1]+al*Pr*y[0])])
    def bc(a,b):
        zL=L-beta; pexp=al/c-1
        return np.array([a[0]-1, b[1]+(c*Pr*zL-pexp/zL)*b[0]])
    y0=np.vstack([np.exp(-x),-np.exp(-x)])
    r=solve_bvp(fun,bc,x,y0,tol=1e-9,max_nodes=300000)
    return -r.sol(0)[1]
def eig(sF,Pr,c,L=20,guess=2.0):
    x=np.linspace(0,L,1500); F=lambda z: sF.sol(z)[0]; beta=L-sF.sol(L)[0]
    def fun(z,y,pp): return np.vstack([y[1],-(c*Pr*F(z)*y[1]+pp[0]*Pr*y[0])])
    def bc(a,b,pp):
        zL=L-beta; pexp=pp[0]/c-1
        return np.array([a[0], a[1]-1, b[1]+(c*Pr*zL-pexp/zL)*b[0]])
    y0=np.vstack([x*np.exp(-x**2/4),(1-x**2/2)*np.exp(-x**2/4)])
    r=solve_bvp(fun,bc,x,y0,p=[guess],tol=1e-9,max_nodes=300000)
    return r.p[0], r.status
def first_pole(sF,Pr,c,amax=3.0,da=0.01):
    """Finds the first pole of -theta'(0) by scanning alpha (initial guess for the smallest eigenvalue)."""
    al=np.arange(da,amax,da); q=[theta_wall(a,sF,Pr,c) for a in al]
    for i in range(len(al)-1):
        if q[i]<-2 and q[i+1]>2: return al[i]
    return None
if __name__=='__main__':
    for geom,c in (('prism',1),('cyl',2)):
        for Pr in (0.7,7):
            for D in (0,1,3,5):
                p=dict(D=D,lm=0.5,sm=0.5,Ha=1,lam=0.5,Pr=Pr,al=0.2)
                sF=flowF(p,geom); g=first_pole(sF,Pr,c,amax=4.0 if c==2 else 3.0)
                ac,st=eig(sF,Pr,c,guess=g)
                a0=brentq(lambda a: theta_wall(a,sF,Pr,c), 0.0, ac*0.999)
                beta=20-sF.sol(20)[0]
                print(geom,'Pr=%g Δ=%d β=%.5f α_c=%.5f α_0=%.5f -θ′(0)|α=0.2=%.6f'%(Pr,D,beta,ac,a0,theta_wall(0.2,sF,Pr,c)),flush=True)
