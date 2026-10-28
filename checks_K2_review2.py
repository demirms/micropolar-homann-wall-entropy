"""Consistency checks behind several numbers of the paper: dissipative-heating correction, alpha_c convergence, residuals.

(1) Consistency of the dissipative-heating (Br) correction of the energy equation (axisymmetric outer flow, c=2):
    theta = t0 + xi t1 + zeta t3 + xi^2 t2 + zeta^2 t4 (the xi*zeta term is not forced), exact linear correction phi in Br;
    the dissipation of the outer-flow reference state D_inf is subtracted:
      t2: phi2''+2Pr F phi2'+(alpha-2F')Pr phi2+(d20-d20_inf)=0 ;  t4: same with d02
      t1: phi1''+2Pr F phi1'+(alpha-F')Pr phi1-2Pr M phi2+d10=0 ;  t3: phi3''+2Pr F phi3'+(alpha-F')Pr phi3-2Pr Vr H phi4+d01=0
      t0: phi0''+2Pr F phi0'+alpha Pr phi0+2 phi2+2 phi4-Pr M phi1-Pr Vr H phi3+(d00-d00_inf)=0
    -> the factor Gamma = 8-10 / 87-92 / 3-37 and the Br limits 1.0e-3 and 1.1e-4 quoted in Section 3 (entry 'corr').
(2) alpha_c, and the L/tolerance convergence of -theta'(0) near alpha_c (Pr=7, Delta_m=5, alpha=1.2 and Pr=0.7, Delta_m=1)
    -> "independent of L and tolerance to seven digits" quoted in Section 4 (entry 'alpha').
(3) Largest relative RMS residual over all solutions of the tables (entry 'resid').
Output: output/checks_K2_review2.json
Run:  python checks_K2_review2.py      (heavy, several minutes)
Reproducible: deterministic.
Requirements: Python 3.12, numpy, scipy 1.18 (see README.md).
"""
import numpy as np, json, math
import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')   # all outputs are written here
os.makedirs(OUT, exist_ok=True)
from scipy.integrate import solve_bvp
from scipy.optimize import brentq
from micropolar_solver import solve, wall
from entropy_cyl import Ns_parts
from critical_alpha import first_pole
def P(D,Ha=1,Pr=0.7,al=0.2): return dict(D=D,lm=0.5,sm=0.5 if D>0 else 0.0,Ha=Ha,lam=0.5,Pr=Pr,al=al)
def dco(y,p,Vr):
    Dx=lambda a,b: (lambda d: d['visc']+d['couple']+d['joule'])(Ns_parts(y,p,a,b,Vr,1.0))
    d00=Dx(0,0); dp,dm=Dx(1,0),Dx(-1,0); ep,em=Dx(0,1),Dx(0,-1)
    return d00,(dp-dm)/2,(ep-em)/2,(dp+dm)/2-d00,(ep+em)/2-d00, Dx(1,1)-d00-(dp-dm)/2-(ep-em)/2-((dp+dm)/2-d00)-((ep+em)/2-d00)
def consistency(p,Vr=1.0,L=20,Br=None):
    s=solve(p,'cyl',L=L,tol=1e-10,robin=True); Pr,al=p['Pr'],p['al']
    di=dco(s.sol(L).reshape(15,1),p,Vr); dinf=[float(np.ravel(v)[0]) for v in di[:5]]
    b=1.0 if Br is None else Br
    def fun(x,z):
        y=s.sol(x); F,F1,H,M=y[0],y[1],y[3],y[5]; d=dco(y,p,Vr)
        f0,g0,f1,g1,f3,g3,f2,g2,f4,g4=z
        dg2=-(2*Pr*F*g2+(al-2*F1)*Pr*f2+b*(d[3]-dinf[3]))
        dg4=-(2*Pr*F*g4+(al-2*F1)*Pr*f4+b*(d[4]-dinf[4]))
        dg1=-(2*Pr*F*g1+(al-F1)*Pr*f1-2*Pr*M*f2+b*(d[1]-dinf[1]))
        dg3=-(2*Pr*F*g3+(al-F1)*Pr*f3-2*Pr*Vr*H*f4+b*(d[2]-dinf[2]))
        dg0=-(2*Pr*F*g0+al*Pr*f0+2*f2+2*f4-Pr*M*f1-Pr*Vr*H*f3+b*(d[0]-dinf[0]))
        if Br is not None: dg0=dg0-(2*Pr*F*0)  # in the full problem t0 = theta^(0) + Br*phi0 is handled separately below
        return np.vstack([g0,dg0,g1,dg1,g3,dg3,g2,dg2,g4,dg4])
    x=np.linspace(0,L,800); z=np.zeros((10,x.size))
    if Br is None:
        bc=lambda a,bb: np.array([a[0],bb[0],a[2],bb[2],a[4],bb[4],a[6],bb[6],a[8],bb[8]])
    else:
        z[0]=s.sol(x)[13]; z[1]=s.sol(x)[14]
        bc=lambda a,bb: np.array([a[0]-1,bb[0],a[2],bb[2],a[4],bb[4],a[6],bb[6],a[8],bb[8]])
    r=solve_bvp(fun,bc,x,z,tol=1e-8,max_nodes=300000); z0=r.sol(0.0); T1=s.sol(0.0)[14]
    return dict(T1=T1,dphi=[z0[1],z0[3],z0[5],z0[7],z0[9]],status=r.status,d11max=float(np.max(np.abs(dco(s.sol(x),p,Vr)[5]))))
def C(res,xi,ze):
    d0,d1,d3,d2,d4=res['dphi']; return (d0+xi*d1+ze*d3+xi**2*d2+ze**2*d4)/res['T1']
out={'corr':[],'check':None,'alpha':[],'nearac':[],'resid':None}
for Pr in (0.7,7):
    for D in (0,1,3,5):
        r=consistency(P(D,Pr=Pr)); c0=C(r,0,0); c1=C(r,1,0); cz=C(r,0,1); cm=max(abs(c0),abs(c1),abs(cz))
        out['corr'].append([Pr,D,c0,c1,cz,0.01/cm,r['status'],r['d11max']])
        print('Pr=%g Δ=%d C(0,0)=%.3f C(1,0)=%.3f C(0,1)=%.3f Br1%%=%.3g st=%d d11=%.1e'%(Pr,D,c0,c1,cz,0.01/cm,r['status'],r['d11max']),flush=True)
r=consistency(P(3)); full=consistency(P(3),Br=0.01)
out['check']=[0.01*C(r,1,0), (full['dphi'][0]+full['dphi'][1]+full['dphi'][3])/r['T1']-1]; print('check',out['check'])
# (2) αc convergence, and −θ'(0) near αc
def flow(p,L,tol): q=dict(p); q['al']=0.0; q['lam']=0.0; return solve(q,'cyl',L=L,tol=tol)
def eig(sF,Pr,L,tol,g):
    x=np.linspace(0,L,1500); F=lambda z: sF.sol(z)[0]; dl=L-sF.sol(L)[0]
    fun=lambda z,y,pp: np.vstack([y[1],-(2*Pr*F(z)*y[1]+pp[0]*Pr*y[0])])
    bc=lambda a,b,pp: np.array([a[0],a[1]-1,b[1]+(2*Pr*(L-dl)-(pp[0]/2-1)/(L-dl))*b[0]])
    y0=np.vstack([x*np.exp(-x**2/4),(1-x**2/2)*np.exp(-x**2/4)])
    rr=solve_bvp(fun,bc,x,y0,p=[g],tol=tol,max_nodes=300000); return rr.p[0]
for Pr,D,al in ((0.7,1,2.8),(7,5,1.2)):
    p=P(D,Pr=Pr); g=first_pole(flow(p,20,1e-9),Pr,2,amax=4.0,da=0.02)
    for L,tol in ((10,1e-6),(10,1e-9),(15,1e-9),(20,1e-6),(20,1e-9),(20,1e-10),(30,1e-9)):
        ac=eig(flow(p,L,tol),Pr,L,tol,g); q=dict(p,al=al); w=-wall(solve(q,'cyl',L=L,tol=tol,robin=True),q)['T1']
        out['alpha'].append([Pr,D,L,tol,ac,al,w]); print('Pr=%g Δ=%d L=%d tol=%.0e αc=%.8f −θ′(0)|α=%g=%.6f'%(Pr,D,L,tol,ac,al,w),flush=True)
# (3) residuals
mx=0; nn=[]
for Ha in (0,1,3):
    for D in (0,1,3,5):
        for Pr in (0.7,7):
            q=P(D,Ha,Pr); s=solve(q,'cyl',L=20,tol=1e-9,robin=True); mx=max(mx,s.rms_residuals.max()); nn.append(s.x.size)
out['resid']=[mx,min(nn),max(nn)]; print('resid',out['resid'])
json.dump(out,open(os.path.join(OUT,'checks_K2_review2.json'),'w'),indent=1,default=float)
