"""Numerical verification of the paper (Section 4) and the data of Tables 1 and 4.

(a) independence of the domain length L and of the tolerance, (b) collocation versus the independent shooting solver,
(c) Proposition 1 (cylindrical), (d) Newtonian axisymmetric validation (Homann; Wang 1973; Wang & Ng 2013, Table 1),
(e) Table 4 (delta_inf, alpha_0, alpha_c), plus the Hermite limit Pr -> 0 and the Robin-versus-Dirichlet and uniqueness
tests (random initial guesses, fixed seed 1).
Output: output/numerics_K2.json (entries indep, solvers, robin, unique, prop1, newt_Pr0.7, newt_Pr7, T1, hermite).
Produces: Table 1 (Newtonian rows), Table 4, and the convergence/agreement numbers quoted in Section 4.
Run:  python verify_numerics_K2.py      (heavy: the alpha_c scans take several minutes)
Reproducible: deterministic (the only random numbers use a fixed seed).
Requirements: Python 3.12, numpy, scipy 1.18 (see README.md).
"""
import numpy as np, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')   # all outputs are written here
os.makedirs(OUT, exist_ok=True)
from scipy.optimize import brentq
from micropolar_solver import solve, wall
from entropy_cyl import S_ex
from shooting_solver import shoot
from critical_alpha import flowF, theta_wall, eig, first_pole
p=dict(D=3,lm=0.5,sm=0.5,Ha=1,lam=0.5,Pr=0.7,al=0.2); xi,ze,Vr,B=1.0,0.0,1.0,0.1
keys=['F2','M1','H1','G1','N1','R1','T1']; out={'indep':[]}
for L in (10,15,20,30):
    for tol in (1e-6,1e-10):
        s=solve(p,'cyl',L=L,tol=tol,robin=True); w=wall(s,p)
        row=[L,tol]+[float(w[k]) for k in keys]+[float(S_ex(s,p,xi,ze,Vr,B))]
        out['indep'].append(row); print(L,tol,' '.join('%.8f'%v for v in row[2:]),s.status)
sb=solve(p,'cyl',L=10,tol=1e-10); wb=wall(sb,p)
ws,ier,res=shoot(p,'cyl',Ls=10)
print('shoot',ier,res,max(abs(wb[k]-ws[k]) for k in keys))
out['solvers']={k:[float(wb[k]),float(ws[k])] for k in keys}; out['shoot_res']=float(res)
# Robin vs Dirichlet, L=8, α=0.6
q=dict(p,al=0.6); ref=wall(solve(q,'cyl',L=30,tol=1e-10,robin=True),q)['T1']
eR=abs(wall(solve(q,'cyl',L=8,tol=1e-10,robin=True),q)['T1']-ref); eD=abs(wall(solve(q,'cyl',L=8,tol=1e-10),q)['T1']-ref)
out['robin']=[eR,eD]; print('robin',eR,eD)
# uniqueness: random guesses
rng=np.random.default_rng(1); base=solve(p,'cyl',L=20,tol=1e-10,robin=True); wb20=wall(base,p); md=0
for i in range(6):
    x=np.linspace(0,20,600); y=base.sol(x)*(1+0.3*rng.standard_normal((15,1)))
    class Gs: pass
    g=Gs(); g.sol=lambda z,y=y,x=x: np.array([np.interp(z,x,yy) for yy in y])
    s2=solve(p,'cyl',L=20,tol=1e-10,guess=g,robin=True); md=max(md,max(abs(wall(s2,p)[k]-wb20[k]) for k in keys))
out['unique']=md; print('unique',md)
# Prop 1
err=0
for D in (1,3,5):
    for lm,sm in ((0,0.5),(0.5,0)):
        a=dict(D=D,lm=lm,sm=sm,Ha=1,lam=0.5,Pr=0.7,al=0.2); w=wall(solve(a,'cyl',L=40),a)
        b=dict(D=0,lm=0,sm=0,Ha=1,lam=0.5,Pr=0.7*(1+D),al=0.2); wn=wall(solve(b,'cyl',L=40),b); s=np.sqrt(1+D)
        err=max(err,max(abs(w[k]-wn[k]/s) for k in ('F2','H1','M1','T1')))
out['prop1']=err; print('prop1',err)
# Newtonian axisymmetric checks
for Pr in (0.7,7):
    a=dict(D=0,lm=0,sm=0,Ha=0,lam=1,Pr=Pr,al=0); w=wall(solve(a,'cyl',L=20,tol=1e-10,robin=True),a)
    out['newt_Pr%g'%Pr]=[w['F2'],w['H1'],w['M1'],-w['T1']]; print('newt',Pr,w['F2'],w['H1'],w['M1'],-w['T1'])
# Table 1
t1=[]
for D in (0,1,3,5):
    row=[D]
    for Pr in (0.7,7):
        a=dict(D=D,lm=0.5,sm=0.5 if D>0 else 0.0,Ha=1,lam=0.5,Pr=Pr,al=0.2); sF=flowF(a,'cyl')
        g=first_pole(sF,Pr,2,amax=4.0,da=0.02); ac,st=eig(sF,Pr,2,guess=g)
        a0=brentq(lambda z: theta_wall(z,sF,Pr,2),0.0,ac*0.999)
        if Pr==0.7: row.append(20-sF.sol(20)[0])
        row+=[a0,ac]
    t1.append(row); print('T1',row,flush=True)
out['T1']=t1
# Hermite limit Pr->0 (Δ=0, Ha=1)
a=dict(D=0,lm=0,sm=0,Ha=1,lam=0,Pr=0.1,al=0); sF=flowF(a,'cyl'); hl=[]
for Pr in (0.1,0.03):
    g=first_pole(sF,Pr,2,amax=4.5,da=0.02); hl.append(eig(sF,Pr,2,guess=g)[0])
out['hermite']=hl; print('hermite',hl)
json.dump(out,open(os.path.join(OUT,'numerics_K2.json'),'w'),indent=1,default=float)
