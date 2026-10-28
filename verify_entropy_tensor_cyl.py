"""Independent check of the entropy-generation expression (cylindrical problem) against its tensor definition.

Builds the three-dimensional fields from the similarity solution, takes the gradients by numerical differentiation,
computes Phi = t_kl a_kl + m_kl nu_l,k (including alpha_v and beta_v) and the Joule term directly from the tensor
definition, and compares with entropy_cyl.Ns_parts (with b = beta_v/gamma_v = 0.37).
Produces: the agreement "5e-11" with the tensor form quoted in Section 4 (printed relative differences, three points).
Run:  python verify_entropy_tensor_cyl.py      (prints to screen; seconds)
Reproducible: deterministic.
Requirements: Python 3.12, numpy, scipy 1.18 (see README.md).
"""
import numpy as np
from micropolar_solver import solve
from entropy_cyl import Ns_parts
p=dict(D=1.5,lm=0.7,sm=0.8,Ha=1.2,lam=0.5,Pr=0.7,al=0.2); s=solve(p,'cyl',L=20)
mu,rho,a,V0=1.3,0.9,2.0,0.7; eps=np.sqrt(mu/(rho*a)); kap=p['D']*mu; j=kap/(rho*a*p['sm']); gam=mu*j/p['lm']
bet=0.37*gam; alv=0.5*gam; sigB2=p['Ha']*rho*a
def fields(X):
    x,y,z=X; F,F1,F2,H,H1,M,M1,G,G1,N,N1,R,R1,T,T1=s.sol(z/eps)
    f=eps*a*F; fp=a*F1; m=eps*a*M; h=V0*H; r=V0*R/eps; g=a*G/eps; n=a*N
    return np.array([m+x*fp, h+y*fp, -2*f]), np.array([r-y*g, n+x*g, 0.0])
def grad(fun,X,dx=1e-5):
    return np.array([(fun(X+e)-fun(X-e))/(2*dx) for e in np.eye(3)*dx])
e3=np.zeros((3,3,3)); e3[0,1,2]=e3[1,2,0]=e3[2,0,1]=1; e3[0,2,1]=e3[2,1,0]=e3[1,0,2]=-1
for (x,y,z) in [(0.3,-0.5,0.2),(0.8,0.4,0.5),(1.1,1.3,0.05)]:
    X=np.array([x,y,z]); Gv=grad(lambda X: fields(X)[0],X); Gn=grad(lambda X: fields(X)[1],X); v,nu=fields(X)
    akl=Gv-np.einsum('klm,m->kl',e3,nu); t=mu*(Gv+Gv.T)+kap*akl
    mkl=alv*np.trace(Gn)*np.eye(3)+bet*Gn.T+gam*Gn   # m_kl = alpha nu_r,r delta_kl + beta nu_k,l + gamma nu_l,k ; Gn[k,l]=nu_l,k
    Phi=np.sum(t*akl)+np.sum(mkl*Gn)
    Joule=sigB2*(v[0]**2+v[1]**2)
    d=Ns_parts(s.sol(z/eps),p,x/eps,y/eps,V0/(a*eps),1.0,b=bet/gam)
    num=(Phi+Joule)/(mu*a**2); ana=d['visc']+d['couple']+d['joule']
    print('x=%.2f y=%.2f z=%.2f tensor=%.10f expression=%.10f relative difference=%.1e'%(x,y,z,num,ana,abs(num-ana)/ana))
