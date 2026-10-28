"""Numbers quoted in the text of Section 5: deviations from Proposition 1, peak of M, wall-entropy shares, S_ex(Vr) and
zero crossing Vr*, entropy at Pr = 7, location of the local minimum of N_s.

Output: output/extra_K2.json (entries dev_P1, Mpeak, share, Sex_Vr, T9, Nsmin).
Produces: Table 5 (S_ex at Vr=0 and 1, Vr*) together with T8 of results_K2.json, and the percentages quoted in the text.
Run:  python extra_K2.py        (about a minute; writes only inside output/)
Reproducible: deterministic.
Requirements: Python 3.12, numpy, scipy 1.18 (see README.md).
"""
import numpy as np, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')   # all outputs are written here
os.makedirs(OUT, exist_ok=True)
from micropolar_solver import solve, wall
from entropy_cyl import Ns_parts, Ns_inf, S_ex
BASE=dict(D=3,lm=0.5,sm=0.5,Ha=1,lam=0.5,Pr=0.7,al=0.2)
def P(**k):
    q=dict(BASE); q.update(k)
    if q['D']==0: q['sm']=0.0
    return q
S=lambda p: solve(p,'cyl',L=20,tol=1e-9,robin=True); out={}
dev=[]
for D in (1,3,5):
    w=wall(S(P(D=D)),P(D=D)); w0=wall(S(P(D=D,lm=0.0)),P(D=D,lm=0.0))
    dev.append([D,w0['F2'],w0['H1'],-w0['T1'],100*(w['F2']/w0['F2']-1),100*(w['H1']/w0['H1']-1),100*(w['T1']/w0['T1']-1)])
out['dev_P1']=dev
eta=np.linspace(0,10,20001); mp=[]
for D in (0,1,3,5):
    s=S(P(D=D)); y=s.sol(eta); i=np.argmax(y[5]); mp.append([D,eta[i],y[5][i]])
    i=np.argmax(y[7]) ; 
out['Mpeak']=mp
sh=[]; sv=[]
for D in (0,1,3,5):
    p=P(D=D); s=S(p); d=Ns_parts(s.sol(0.0),p,1,0,1,0.1); t=sum(d.values())
    sh.append([D]+[d[k]/t for k in ('heat','visc','couple','joule')])
    a=S_ex(s,p,1,0,0,0.1); b=S_ex(s,p,1,0,1,0.1)-a
    sv.append([D,a,b,np.sqrt(-a/b) if a<0<b else np.nan, Ns_inf(p,1,0,0.1)])
    # local minimum of Ns near wall
out['share']=sh; out['Sex_Vr']=sv
t9=[]
for D in (0,1,3,5):
    p=P(D=D,Pr=7); s=S(p); d=Ns_parts(s.sol(0.0),p,1,0,1,0.1); t=sum(d.values())
    t9.append([D,t,d['heat']/t,S_ex(s,p,1,0,1,0.1)])
out['T9']=t9
# Ns local min location, Δ=0,3
lm=[]
for D in (0,3):
    p=P(D=D); s=S(p); x=np.linspace(0,3,3001); d=Ns_parts(s.sol(x),p,1,0,1,0.1); t=sum(d.values())
    i=np.argmin(t[:2000]); lm.append([D,x[i],t[i]])
out['Nsmin']=lm
# Be profile max, ζ dependence: Sex at ζ=ζ0
json.dump(out,open(os.path.join(OUT,'extra_K2.json'),'w'),indent=1,default=float)
for k,v in out.items(): print(k); [print('  ',np.round(r,5)) for r in v]
