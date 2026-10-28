"""Assembles the numerical tables of the paper (as Markdown) from the JSON results of the other scripts.

Reads output/results_K2.json, output/numerics_K2.json, output/extra_K2.json, output/reviewer_K2.json; computes the
Borrelli et al. (2013) comparison on the fly; writes output/tables_K2.json with keys
  '2' numerics, '3' comparison with published results (paper Table 1), '4' wall gradients (paper Table 2),
  '5','6','7' parameter sweeps quoted in the text, '10' special points (paper Table 3),
  '8' entropy (paper Table 5), '9' coefficients (paper Table 6).
Produces: Tables 1, 2, 3, 5, 6 of the paper (Table 4 comes from numerics_K2.json, entry T1).
Run:  python make_tables_K2.py      (after make_results_K2.py, verify_numerics_K2.py, extra_K2.py, reviewer_checks_K2.py)
Reproducible: deterministic.
Requirements: Python 3.12, numpy, scipy 1.18 (see README.md).
"""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')   # all inputs/outputs live here
from micropolar_solver import solve, wall
r=json.load(open(os.path.join(OUT,'results_K2.json'))); nm=json.load(open(os.path.join(OUT,'numerics_K2.json'))); ex=json.load(open(os.path.join(OUT,'extra_K2.json')))
f=lambda v,n=5: ('%.*f'%(n,v)).replace('-','−')
T={}
# Table 2 of tables_K2.json (numerics)
L=["| $L$ | tol | $F''(0)$ | $M'(0)$ | $H'(0)$ | $G'(0)$ | $N'(0)$ | $R'(0)$ | $-\\theta'(0)$ | $S_{ex}$ |",'|---|---|---|---|---|---|---|---|---|---|']
for row in nm['indep']: L.append('| %d | $10^{%d}$ | '%(row[0],round(np.log10(row[1])))+' | '.join(f(v,8) for v in row[2:8])+' | '+f(-row[8],8)+' | '+f(row[9],8)+' |')
keys=['F2','M1','H1','G1','N1','R1','T1']
for lab,i in (('10 (collocation)',0),('10 (shooting)',1)):
    L.append('| %s | — | '%lab+' | '.join(f(nm['solvers'][k][i],10) for k in keys[:6])+' | '+f(-nm['solvers']['T1'][i],10)+' | — |')
T[2]='\n'.join(L)
# Table 3 of tables_K2.json (comparison with published results)
L=['| Case | Quantity | Published | Present |','|---|---|---|---|']
a,b=nm['newt_Pr0.7'],nm['newt_Pr7']
L+=["| Newtonian (Wang & Ng, 2013, Table 1) | $F''(0)$ | 1.31194 | %s |"%f(a[0],6),
    "| Newtonian, Pr = 0.7 | $M'(0)$ | 0.4955 | %s |"%f(a[2],6),"| | $-\\theta'(0)$ | 0.6654 | %s |"%f(a[3],6),
    "| Newtonian, Pr = 7 | $M'(0)$ | 0.2849 | %s |"%f(b[2],6),"| | $-\\theta'(0)$ | 1.5458 | %s |"%f(b[3],6),
    "| Translating plate (Wang, 1973) | $H'(0)$ | −0.93873 | %s |"%f(a[1],6)]
ref=[(0.1,1.5,0.1,1.3013,-0.0558,0.5665),(0.1,1.5,0.5,1.3030,-0.0531,0.5667),(0.1,3.0,0.1,1.3043,-0.0470,0.5670),(0.1,3.0,0.5,1.3051,-0.0458,0.5671),
     (0.5,1.5,0.1,1.2583,-0.2793,0.5568),(0.5,1.5,0.5,1.2669,-0.2657,0.5576),(0.5,3.0,0.1,1.2734,-0.2354,0.5594),(0.5,3.0,0.5,1.2774,-0.2292,0.5598)]
L2=["| $c_1$ | $c_2$ | $c_3$ | $f''(0)$ [B] | present | $g'(0)$ [B] | present | $\\alpha$ [B] | present |",'|---|---|---|---|---|---|---|---|---|']
for c1,c2,c3,f2,G1,al in ref:
    D=2*c1/(c2-2*c1); lm=c3/(1+D); sm=c1/(lm*D); s=np.sqrt(1+D)
    p=dict(D=D,lm=lm,sm=sm,Ha=0,lam=0,Pr=0.7,al=0); so=solve(p,'cyl',L=25,tol=1e-10); w=wall(so,p)
    L2.append('| %g | %g | %g | %s | %s | %s | %s | %s | %s |'%(c1,c2,c3,f(f2,4),f(s*w['F2'],5),f(G1,4),f(-D*w['G1'],5),f(al,4),f((25-so.sol(25)[0])/s,5)))
T[3]='\n'.join(L)+'\n\n*Micropolar fluid: Borrelli et al. (2013), Table 1 ([B]); $c_1$, $c_2$, $c_3$ are their micropolar parameters.*\n\n'+'\n'.join(L2)
# Tables 4, 5 of tables_K2.json
hdr="| %s | %s | $F''(0)$ | $M'(0)$ | $H'(0)$ | $G'(0)$ | $N'(0)$ | $R'(0)$ | $-\\theta'(0)$ |"
L=[hdr%('Ha','$\\Delta_m$'),'|---|---|---|---|---|---|---|---|---|']
for row in r['T4']: L.append('| %d | %d | '%(row[0],row[1])+' | '.join(f(v) for v in row[2:8])+' | '+f(-row[8])+' |')
T[4]='\n'.join(L)
L=[hdr%('$\\lambda_m$','$\\sigma_m$'),'|---|---|---|---|---|---|---|---|---|']
for row in r['T5']: L.append('| %g | %g | '%(row[0],row[1])+' | '.join(f(v) for v in row[2:8])+' | '+f(-row[8])+' |')
T[5]='\n'.join(L)
L=['| Pr | $\\alpha$ | $\\Delta_m=0$ | $\\Delta_m=1$ | $\\Delta_m=3$ | $\\Delta_m=5$ |','|---|---|---|---|---|---|']
for row in r['T6']: L.append('| %g | %s | '%(row[0],f(row[1],1))+' | '.join(f(v) for v in row[2:])+' |')
T[6]='\n'.join(L)
L=["| $\\Delta_m$ | $F''(0)$ | $H'(0)$ | $G'(0)$ | $R'(0)$ | $-\\theta'(0)$ | dev. $F''(0)$ | dev. $H'(0)$ | dev. $\\theta'(0)$ |",'|---|---|---|---|---|---|---|---|---|']
for x in r['T7']: L.append('| %d | %s | %s | %s | %s | %s | %s %% | %s %% | +%s %% |'%(x[0],f(x[1]),f(x[3]),f(x[4]),f(x[6]),f(-x[7]),f(x[8],1),f(x[9],1),f(x[10],2)))
T[7]='\n'.join(L)
L=['| Ha | $\\Delta_m$ | $\\xi_0$ | $\\zeta_0$ | $\\xi_c$ | $\\zeta_c$ | $\\xi_m$ | $\\zeta_m$ | $A$ | $B$ |','|---|---|---|---|---|---|---|---|---|---|']
g=lambda v: '—' if v!=v else f(v,4)
t4={(x[0],x[1]):x for x in r['T4']}
for x in r['T8']:
    w=t4[(x[0],x[1])]; D=x[1]; A=(1+D)*w[2]**2; B=(D/0.25)*w[5]**2 if D>0 else float('nan')
    L.append('| %d | %d | '%(x[0],x[1])+' | '.join(g(v) for v in x[2:8])+' | %s | %s |'%(f(A,3),'—' if B!=B else f(B,3)))
T[10]='\n'.join(L)
L=['| $\\Delta_m$ | $N_s(0)$ | Be(0) | $S_{ex}$ ($V_r=0$) | $S_{ex}$ ($V_r=1$) | $V_r^{*}$ | $N_s(0)$, Pr = 7 | Be(0), Pr = 7 | $S_{ex}$, Pr = 7 |','|---|---|---|---|---|---|---|---|---|']
t8={(x[0],x[1]):x for x in r['T8']}
for sv,t9 in zip(ex['Sex_Vr'],ex['T9']):
    D=sv[0]; x=t8[(1,D)]
    L.append('| %d | %s | %s | %s | %s | %s | %s | %s | %s |'%(D,f(x[8],4),f(x[9],4),f(sv[1],4),f(sv[1]+sv[2],4),f(sv[3],3),f(t9[1],4),f(t9[2],4),f(t9[3],4)))
T[8]='\n'.join(L)
rv=json.load(open(os.path.join(OUT,'reviewer_K2.json')))
L=['| $\\Delta_m$ | $I_h$ | $q_0$ | $q_\\xi$ | $q_2$ | $q_V$ | $q_{\\zeta V}$ | $(\\mathrm{Br}/\\Omega)^*$ |','|---|---|---|---|---|---|---|---|']
for c in rv['coef']: L.append('| %d | '%c[0]+' | '.join(f(v,4) for v in (c[1],c[2],c[3],c[4],c[6],c[7]))+' | %s |'%f(c[8],4))
T[9]='\n'.join(L)
json.dump({str(k):v for k,v in T.items()},open(os.path.join(OUT,'tables_K2.json'),'w'),ensure_ascii=False,indent=0)
print(T[9]); print(T[10])
