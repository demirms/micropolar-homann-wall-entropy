"""Decomposition of the excess entropy generation and the data of Table 6 and Figure 5.

(a) S_ex = I_h + (Br/Omega) Q(xi,zeta,Vr); the six coefficients of Q (q0, q_xi, q_xixi, q_zetazeta, q_VV, q_zetaV) and the
    critical dissipation group (Br/Omega)*  -> Table 6;
(b) heat generation 2 < alpha < alpha_c: limit L -> infinity of the truncated problem theta(L)=0 toward the Robin solution
    (the 7e-7 and 4e-12 differences quoted in Section 5);
(c) data of Figure 5: (a) S_ex versus Vr, (b) (Br/Omega)* versus Delta_m for Ha = 0, 1, 3.
Output: output/reviewer_K2.json (entries coef, crit, trunc, SexVr) and output/figures_draft/fig6_excess_entropy.png
(draft; the paper figure is made by make_figures_v1_K2.py from reviewer_K2.json).
Run:  python reviewer_checks_K2.py       (heavy, several minutes)
Reproducible: deterministic.
Requirements: Python 3.12, numpy, scipy 1.18, matplotlib (see README.md).
"""
import numpy as np, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')   # all outputs are written here
FIGD = os.path.join(OUT, 'figures_draft')
os.makedirs(FIGD, exist_ok=True)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.integrate import simpson
from micropolar_solver import solve, wall
from entropy_cyl import Ns_parts, Ns_inf
BASE=dict(D=3,lm=0.5,sm=0.5,Ha=1,lam=0.5,Pr=0.7,al=0.2)
def P(**k):
    q=dict(BASE); q.update(k)
    if q['D']==0: q['sm']=0.0
    return q
S=lambda p: solve(p,'cyl',L=20,tol=1e-9,robin=True)
def parts(s,p,n=20001):
    x=np.linspace(0,s.x[-1],n); y=s.sol(x)
    Ih=simpson(y[14]**2,x=x)
    def Q(xi,ze,Vr):
        d=Ns_parts(y,p,xi,ze,Vr,1.0); return simpson(d['visc']+d['couple']+d['joule']-Ns_inf(p,xi,ze,1.0),x=x)
    q0=Q(0,0,0); qx=(Q(1,0,0)-Q(-1,0,0))/2; qxx=(Q(1,0,0)+Q(-1,0,0))/2-q0
    qzz=Q(0,1,0)-q0; qVV=Q(0,0,1)-q0; qzV=Q(0,1,1)-q0-qzz-qVV
    return Ih,dict(q0=q0,qx=qx,qxx=qxx,qzz=qzz,qVV=qVV,qzV=qzV),Q
out={'coef':[],'crit':[]}
for D in (0,1,3,5):
    p=P(D=D); s=S(p); Ih,c,Q=parts(s,p)
    Qb=Q(1,0,1); chk=c['q0']+c['qx']+c['qxx']+c['qVV']-Qb
    out['coef'].append([D,Ih]+[c[k] for k in ('q0','qx','qxx','qzz','qVV','qzV')]+[-Ih/Qb, float(chk)])
    print(D,'Ih=%.5f'%Ih,{k:round(v,5) for k,v in c.items()},'crit=%.4f'%(-Ih/Qb),'chk=%.1e'%chk,flush=True)
for Ha in (0,1,3):
    row=[Ha]
    for D in (0,0.5,1,2,3,4,5):
        p=P(D=D,Ha=Ha); s=S(p); Ih,c,Q=parts(s,p,8001); row.append(-Ih/Q(1,0,1))
    out['crit'].append(row); print('crit Ha',Ha,np.round(row[1:],4),flush=True)
# (b) truncated-problem limit, Δ=1, α=2.5 (2<α<αc=2.828)
p=P(D=1,al=2.5); ref=wall(S(p),p)['T1']; tr=[]
for L in (6,8,10,12,15):
    w=wall(solve(p,'cyl',L=L,tol=1e-10),p)['T1']; tr.append([L,w]); print('trunc L=%d θ′(0)=%.8f ref=%.8f diff=%.1e'%(L,w,ref,abs(w-ref)))
out['trunc']=[ref,tr]
# Sex vs Vr for figure
Vs=np.linspace(0,6,25); sv=[]
for D in (0,1,3,5):
    p=P(D=D); s=S(p); Ih,c,Q=parts(s,p,8001)
    sv.append([Ih+0.1*(c['q0']+c['qx']+c['qxx']+c['qVV']*v**2) for v in Vs])
out['SexVr']=[list(Vs),sv]
json.dump(out,open(os.path.join(OUT,'reviewer_K2.json'),'w'),indent=1,default=float)
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.linewidth':0.8,'xtick.direction':'in','ytick.direction':'in',
 'xtick.top':True,'ytick.right':True,'legend.frameon':False,'lines.linewidth':1.1,'savefig.dpi':300})
LS=['-','--','-.',':']; MK=['o','s','^','v']
fig,ax=plt.subplots(1,2,figsize=(6.4,2.6))
for i,D in enumerate((0,1,3,5)):
    ax[0].plot(Vs,sv[i],color='k',ls=LS[i],marker=MK[i],markevery=4,ms=3.5,mfc='white',mew=0.8,label=r'$\Delta_m=%d$'%D)
ax[0].axhline(0,color='0.6',ls=':',lw=0.7); ax[0].set_xlabel(r'$V_r$'); ax[0].set_ylabel(r'$S_{ex}$'); ax[0].legend(fontsize=7.5,loc='upper left')
Ds=(0,0.5,1,2,3,4,5)
for i,row in enumerate(out['crit']):
    ax[1].plot(Ds,row[1:],color='k',ls=LS[i],marker=MK[i],ms=3.5,mfc='white',mew=0.8,label=r'Ha$=%d$'%row[0])
ax[1].set_yscale('log'); ax[1].set_xlabel(r'$\Delta_m$'); ax[1].set_ylabel(r'$(\mathrm{Br}/\Omega)^*$'); ax[1].legend(fontsize=7.5)
ax[0].text(0.93,0.06,'(a)',transform=ax[0].transAxes,ha='right'); ax[1].text(0.05,0.06,'(b)',transform=ax[1].transAxes)
fig.tight_layout(); fig.savefig(os.path.join(FIGD,'fig6_excess_entropy.png')); print('done')
