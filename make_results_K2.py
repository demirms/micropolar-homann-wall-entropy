"""Main results of the paper: wall gradients, special points and wall entropy, as JSON; plus draft figures.

Solves the cylindrical (Homann) problem with the collocation solver for all parameter sets and writes
output/results_K2.json with the entries
  T4 (wall gradients, Ha=0,1,3, Delta_m=0,1,3,5)  -> Table 2 (Ha=1 rows) and the Ha-trend quoted in the text,
  T5 (lambda_m, sigma_m sweep), T6 (-theta'(0) versus alpha, Pr=0.7 and 7), T7 (Ahmadi closure and the Proposition-1
  deviation) -> numbers quoted in Sections 4-5,
  T8 (skin-friction node, couple-stress node, minimum of N_s(0), N_s(0), Be(0), S_ex)  -> Table 3, Table 5 and the node
  location of Figure 1 (read by make_figures_v1_K2.py),
  check_min (brute-force minimisation of N_s(0)) -> the agreement quoted with Eq. (nodem).
It also draws earlier draft figures into output/figures_draft/ (superseded by make_figures_v1_K2.py; not in the paper).
Run:  python make_results_K2.py          (several minutes; writes only inside output/)
Reproducible: deterministic (no random numbers).
Requirements: Python 3.12, numpy, scipy 1.18, matplotlib (see README.md).
"""
import numpy as np, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')   # all outputs are written here
FIGD = os.path.join(OUT, 'figures_draft')
os.makedirs(FIGD, exist_ok=True)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from micropolar_solver import solve, wall
from entropy_cyl import Ns_parts, Ns_inf, S_ex
BASE=dict(D=3,lm=0.5,sm=0.5,Ha=1,lam=0.5,Pr=0.7,al=0.2); XI,ZE,VR,BO=1.0,0.0,1.0,0.1
def P(**k):
    q=dict(BASE); q.update(k)
    if q['D']==0: q['sm']=0.0
    return q
def S(p,L=20): return solve(p,'cyl',L=L,tol=1e-9,robin=True)
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.linewidth':0.8,'xtick.direction':'in','ytick.direction':'in',
 'xtick.top':True,'ytick.right':True,'legend.frameon':False,'lines.linewidth':1.1,'savefig.dpi':300})
LS=['-','--','-.',':']; MK=['o','s','^','v']
def st(i,me=0.12): return dict(color='k',ls=LS[i%4],marker=MK[i%4],markevery=me,ms=3.5,mfc='white',mew=0.8)
K=('F2','M1','H1','G1','N1','R1','T1'); eta=np.linspace(0,6,601); res={}
res['T4']=[[Ha,D]+[wall(S(P(D=D,Ha=Ha)),P(D=D,Ha=Ha))[k] for k in K] for Ha in (0,1,3) for D in (0,1,3,5)]
res['T5']=[[lm,sm]+[wall(S(P(lm=lm,sm=sm)),P(lm=lm,sm=sm))[k] for k in K] for lm in (0.5,1,2) for sm in (0.5,1,2)]
t6=[]
for Pr in (0.7,7):
    for al in (-0.5,0,0.4,0.8,1.2):
        t6.append([Pr,al]+[-wall(S(P(D=D,Pr=Pr,al=al)),P(D=D,Pr=Pr,al=al))['T1'] for D in (0,1,3,5)])
res['T6']=t6
# Ahmadi closure + P1 deviation
t7=[]
for D in (1,3,5):
    p=P(D=D,lm=1/(1+D/2),sm=float(D)); w=wall(S(p),p); q=P(D=D,lm=0.0); w0=wall(S(q),q)
    t7.append([D]+[w[k] for k in K]+[100*(w['F2']/w0['F2']-1),100*(w['H1']/w0['H1']-1),100*(w['T1']/w0['T1']-1)])
res['T7']=t7
# zero-shear, zero-couple and wall-entropy-minimum points (Vr=1), entropy
def points(w,p,Vr):
    D=p['D']; wt=D/(p['lm']*p['sm']) if D>0 else 0.0; A=(1+D)*w['F2']**2; B=wt*w['G1']**2
    xi0=-w['M1']/w['F2']; ze0=-Vr*w['H1']/w['F2']
    xic=-w['N1']/w['G1'] if D>0 else np.nan; zec=Vr*w['R1']/w['G1'] if D>0 else np.nan
    xim=(A*xi0+(B*xic if D>0 else 0))/(A+B); zem=(A*ze0+(B*zec if D>0 else 0))/(A+B)
    return xi0,ze0,xic,zec,xim,zem
t8=[]
for Ha in (0,1,3):
    for D in (0,1,3,5):
        p=P(D=D,Ha=Ha); s_=S(p); w=wall(s_,p); pts=points(w,p,VR)
        d=Ns_parts(s_.sol(0.0),p,XI,ZE,VR,BO); tot=sum(d.values())
        t8.append([Ha,D]+list(pts)+[tot,d['heat']/tot,S_ex(s_,p,XI,ZE,VR,BO)])
res['T8']=t8
# brute-force check of the wall-entropy minimum (Δm=3, Ha=1, Vr=1)
from scipy.optimize import minimize
p=P(); s_=S(p); y0=s_.sol(0.0); w=wall(s_,p)
fmin=minimize(lambda z: sum(Ns_parts(y0,p,z[0],z[1],VR,BO).values()),[0,0],tol=1e-12).x
res['check_min']=[float(fmin[0]),float(fmin[1])]+list(points(w,p,VR)[4:])
json.dump(res,open(os.path.join(OUT,'results_K2.json'),'w'),indent=1)
sols={D:S(P(D=D)) for D in (0,1,3,5)}
# Fig 2 velocity
fig,ax=plt.subplots(1,3,figsize=(7.0,2.4))
for i,D in enumerate((0,1,3,5)):
    y=sols[D].sol(eta)
    for a,idx in zip(ax,(1,5,3)): a.plot(eta,y[idx],label=r'$\Delta_m=%d$'%D,**st(i))
for a,lab,t,yy in zip(ax,(r"$F'$",r'$M$',r'$H$'),'abc',(0.62,0.92,0.92)):
    a.set_xlabel(r'$\eta$'); a.set_ylabel(lab); a.set_xlim(0,6); a.text(0.92,yy,'(%s)'%t,transform=a.transAxes,ha='right')
ax[0].legend(loc='lower right',fontsize=7.5); fig.tight_layout(); fig.savefig(os.path.join(FIGD,'fig2_velocity.png')); plt.close(fig)
# Fig 3 microrotation
fig,ax=plt.subplots(1,3,figsize=(7.0,2.4))
for i,D in enumerate((1,3,5)):
    y=sols[D].sol(eta)
    for a,idx in zip(ax,(7,9,11)): a.plot(eta,y[idx],label=r'$\Delta_m=%d$'%D,**st(i+1))
for a,lab,t in zip(ax,(r'$G$',r'$N$',r'$R$'),'abc'):
    a.set_xlabel(r'$\eta$'); a.set_ylabel(lab); a.set_xlim(0,6); a.text(0.92,0.92,'(%s)'%t,transform=a.transAxes,ha='right'); a.axhline(0,color='0.6',ls=':',lw=0.6)
ax[0].legend(loc='center right',fontsize=7.5); fig.tight_layout(); fig.savefig(os.path.join(FIGD,'fig3_microrotation.png')); plt.close(fig)
# Fig 4 temperature
fig,ax=plt.subplots(1,2,figsize=(6.0,2.5))
for i,D in enumerate((0,1,3,5)): ax[0].plot(eta,sols[D].sol(eta)[13],label=r'$\Delta_m=%d$'%D,**st(i))
for i,al in enumerate((0,0.8,1.6,2.4)):
    s=S(P(D=1,al=al)); ax[1].plot(eta,s.sol(eta)[13],label=r'$\alpha=%.1f$'%al,**st(i))
for a,t in zip(ax,'ab'): a.set_xlabel(r'$\eta$'); a.set_ylabel(r'$\theta$'); a.set_xlim(0,5); a.text(0.04,0.06,'(%s)'%t,transform=a.transAxes)
ax[0].legend(fontsize=7.5); ax[1].legend(fontsize=7.5); fig.tight_layout(); fig.savefig(os.path.join(FIGD,'fig4_temperature.png')); plt.close(fig)
# Fig 5 entropy components
fig,ax=plt.subplots(1,2,figsize=(6.4,2.6),sharey=True)
for a,D,t in zip(ax,(0,3),'ab'):
    p=P(D=D); d=Ns_parts(sols[D].sol(eta),p,XI,ZE,VR,BO); tot=sum(d.values())
    a.plot(eta,tot,color='k',lw=1.6,label=r'$N_s$')
    for i,(k,lab) in enumerate([('heat',r'$N_h$'),('visc',r'$N_d+N_r$'),('couple',r'$N_g$'),('joule',r'$N_J$')]):
        if D==0 and k=='couple': continue
        a.plot(eta,d[k],label=lab,**st(i))
    a.axhline(Ns_inf(p,XI,ZE,BO),color='0.6',ls=':',lw=0.7); a.set_xlabel(r'$\eta$'); a.set_xlim(0,6)
    a.text(0.04,0.92,r'(%s) $\Delta_m=%d$'%(t,D),transform=a.transAxes)
ax[0].set_ylabel(r'$N_s$ and its components'); ax[1].legend(fontsize=7.5,loc='center right'); fig.tight_layout(); fig.savefig(os.path.join(FIGD,'fig5_entropy_components.png')); plt.close(fig)
# Fig 6 wall entropy map in (ξ,ζ), Δm=3
p=P(); s_=sols[3]; w=wall(s_,p); y0=s_.sol(0.0); xg=np.linspace(-2,2,81); zg=np.linspace(-1.5,4,111)
fig,ax=plt.subplots(1,2,figsize=(6.4,2.9),sharey=True)
for a,Vr,t in zip(ax,(0.0,2.0),'ab'):
    X,Z=np.meshgrid(xg,zg); SS=np.vectorize(lambda A,B: sum(Ns_parts(y0,p,A,B,Vr,BO).values()))(X,Z)
    cs=a.contour(X,Z,SS,levels=[0.2,0.3,0.4,0.6,0.8,1.0,1.3,1.6,2.0],colors='k',linewidths=0.6); a.clabel(cs,fontsize=6,fmt='%.1f')
    xi0,ze0,xic,zec,xim,zem=points(w,p,Vr)
    a.plot(xi0,ze0,'ko',ms=4,label='zero wall shear'); a.plot(xic,zec,'k^',ms=4.5,mfc='white',label='zero couple stress')
    a.plot(xim,zem,'ks',ms=4,mfc='white',label=r'minimum of $N_s(0)$')
    a.set_xlabel(r'$\xi$'); a.text(0.04,0.92,r'(%s) $V_r=%g$'%(t,Vr),transform=a.transAxes,bbox=dict(fc='white',ec='none',pad=1))
ax[0].set_ylabel(r'$\zeta$'); ax[1].legend(fontsize=6.5,loc='lower center'); fig.tight_layout(); fig.savefig(os.path.join(FIGD,'fig7_wall_entropy_map.png')); plt.close(fig)
# Fig 7 displacement of special points vs Vr
Vs=np.linspace(0,3,7); fig,ax=plt.subplots(1,3,figsize=(7.0,2.4),sharey=True)
for i,D in enumerate((0,1,3,5)):
    p=P(D=D); w=wall(sols[D],p); pts=np.array([points(w,p,v) for v in Vs])
    for a,col in zip(ax,(1,3,5)):
        if D==0 and col==3: continue
        a.plot(Vs,pts[:,col],label=r'$\Delta_m=%d$'%D,**st(i,1))
for a,lab,t in zip(ax,(r'zero wall shear $\zeta_0$',r'zero couple stress $\zeta_c$',r'minimum of $N_s(0)$, $\zeta_m$'),'abc'):
    a.set_xlabel(r'$V_r$'); a.text(0.04,0.92,'(%s) %s'%(t,lab),transform=a.transAxes,fontsize=7.5)
ax[0].set_ylabel(r'$\zeta$'); ax[2].legend(fontsize=7,loc='lower right'); fig.tight_layout(); fig.savefig(os.path.join(FIGD,'fig7_point_shift.png')); plt.close(fig)
print('done')
