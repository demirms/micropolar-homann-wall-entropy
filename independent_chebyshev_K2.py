"""Independent recomputation of the key numbers of the paper (independent check, call 2/3).

Aim: reproduce the key numbers by a method independent of the paper's main code (micropolar_solver.py, solve_bvp).
Method: Chebyshev spectral collocation; Newton iteration for the nonlinear (F, G) sub-problem; linear solves for theta,
(M, N) and (H, R) (hierarchical structure); alpha_c from a generalised eigenvalue problem (Dirichlet theta(L)=0, large L).
The equations are written directly from the paper's equations (flow, microrotation, temperature, entropy expressions);
the main code is not imported.
Produces (output/independent_chebyshev_K2.json): Table 2 (entry T6_Ha1, with grid check), Proposition 1 and the effective-
viscosity mapping (P1), the Ahmadi-closure deviations (T9), Tables 5 and 6 and V_r* (T11, Vr_star_newt_xi012, crit_Ha),
Table 3 (T12), Table 4 and the Hermite limit (T1, hermite). Internal table labels (T6, T9, T11, T12, T1) follow the numbering of an earlier working draft; the paper
tables are named in each 'Produces' line.
Run:  python independent_chebyshev_K2.py [output.json]      (default output: output/independent_chebyshev_K2.json)
Reproducible: deterministic (no random numbers).
Requirements: Python 3.12, numpy, scipy 1.18 (see README.md).
"""
import json, sys, os
import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')   # all outputs are written here
os.makedirs(OUT, exist_ok=True)
import numpy as np
from scipy.linalg import eig, solve as lsolve


def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2, np.ones(N - 1), 2]) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, x


def cc_weights(N):
    # Clenshaw-Curtis weights on [-1,1] for x_j = cos(pi j/N)
    theta = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    v = np.ones(N - 1)
    if N % 2 == 0:
        w[0] = w[N] = 1.0 / (N ** 2 - 1)
        for k in range(1, N // 2):
            v -= 2 * np.cos(2 * k * theta[1:-1]) / (4 * k ** 2 - 1)
        v -= np.cos(N * theta[1:-1]) / (N ** 2 - 1)
    else:
        w[0] = w[N] = 1.0 / N ** 2
        for k in range(1, (N - 1) // 2 + 1):
            v -= 2 * np.cos(2 * k * theta[1:-1]) / (4 * k ** 2 - 1)
    w[1:-1] = 2 * v / N
    return w


class Grid:
    def __init__(self, N, L):
        D, x = cheb(N)
        self.N, self.L = N, L
        self.eta = L * (x + 1) / 2          # index 0 -> eta=L, index N -> eta=0
        self.D1 = D * 2 / L
        self.D2 = self.D1 @ self.D1
        self.D3 = self.D2 @ self.D1
        self.w = cc_weights(N) * L / 2
        self.I = np.eye(N + 1)
        self.iL, self.i0 = 0, N


def solve_flow(g, D, lm, sm, Ha, F0=None, G0=None, tol=1e-13):
    n = g.N + 1
    eta = g.eta
    F = eta - 1 + np.exp(-eta) if F0 is None else F0.copy()
    G = np.zeros(n) if G0 is None else G0.copy()
    for it in range(60):
        F1, F2, F3 = g.D1 @ F, g.D2 @ F, g.D3 @ F
        G1, G2 = g.D1 @ G, g.D2 @ G
        R1 = (1 + D) * F3 + 2 * F * F2 - F1 ** 2 + Ha * (1 - F1) - D * G1 + 1
        R2 = G2 + lm * (2 * F * G1 - F1 * G) - lm * sm * (2 * G - F2)
        J11 = (1 + D) * g.D3 + np.diag(2 * F2) + np.diag(2 * F) @ g.D2 - np.diag(2 * F1) @ g.D1 - Ha * g.D1
        J12 = -D * g.D1
        J21 = lm * (np.diag(2 * G1) - np.diag(G) @ g.D1) + lm * sm * g.D2
        J22 = g.D2 + lm * (np.diag(2 * F) @ g.D1 - np.diag(F1)) - 2 * lm * sm * g.I
        # boundary rows for F: F(0)=0 (row N), F'(0)=0 (row N-1), F'(L)=1 (row 0)
        for J, r in ((J11, None), (J12, None)):
            pass
        J11, J12 = J11.copy(), J12.copy()
        J11[g.i0] = g.I[g.i0]; J12[g.i0] = 0; R1[g.i0] = F[g.i0]
        J11[g.i0 - 1] = g.D1[g.i0]; J12[g.i0 - 1] = 0; R1[g.i0 - 1] = F1[g.i0]
        J11[g.iL] = g.D1[g.iL]; J12[g.iL] = 0; R1[g.iL] = F1[g.iL] - 1
        J21, J22 = J21.copy(), J22.copy()
        for i in (g.i0, g.iL):
            J21[i] = 0; J22[i] = g.I[i]; R2[i] = G[i]
        J = np.block([[J11, J12], [J21, J22]])
        d = lsolve(J, -np.hstack([R1, R2]))
        F += d[:n]; G += d[n:]
        if np.max(np.abs(d)) < tol:
            break
    return F, G


def solve_theta(g, F, Pr, al, robin=True):
    A = g.D2 + np.diag(2 * Pr * F) @ g.D1 + al * Pr * g.I
    b = np.zeros(g.N + 1)
    A[g.i0] = g.I[g.i0]; b[g.i0] = 1
    if robin:
        zL = F[g.iL]
        A[g.iL] = g.D1[g.iL] + (2 * Pr * zL - (al / 2 - 1) / zL) * g.I[g.iL]
    else:
        A[g.iL] = g.I[g.iL]
    return lsolve(A, b)


def solve_MN(g, F, G, th, D, lm, sm, Ha, lam):
    n = g.N + 1
    F1 = g.D1 @ F
    A11 = (1 + D) * g.D2 + np.diag(2 * F) @ g.D1 - np.diag(F1) - Ha * g.I
    A12 = -D * g.D1
    A21 = -lm * np.diag(G) + lm * sm * g.D1
    A22 = g.D2 + 2 * lm * np.diag(F) @ g.D1 - 2 * lm * sm * g.I
    A = np.block([[A11, A12], [A21, A22]]); b = np.hstack([-lam * th, np.zeros(n)])
    for i in (g.i0, g.iL):
        for k in (i, n + i):
            A[k] = 0; A[k, k] = 1; b[k] = 0
    s = lsolve(A, b); return s[:n], s[n:]


def solve_HR(g, F, G, D, lm, sm, Ha):
    n = g.N + 1
    F1 = g.D1 @ F
    A11 = (1 + D) * g.D2 + np.diag(2 * F) @ g.D1 - np.diag(F1) - Ha * g.I
    A12 = D * g.D1
    A21 = lm * np.diag(G) - lm * sm * g.D1
    A22 = g.D2 + 2 * lm * np.diag(F) @ g.D1 - 2 * lm * sm * g.I
    A = np.block([[A11, A12], [A21, A22]]); b = np.zeros(2 * n)
    for i in (g.i0, g.iL):
        for k in (i, n + i):
            A[k] = 0; A[k, k] = 1; b[k] = 0
    b[g.i0] = 1.0
    s = lsolve(A, b); return s[:n], s[n:]


def full(g, p, cont=None):
    D, lm, sm, Ha, lam, Pr, al = (p[k] for k in ('D', 'lm', 'sm', 'Ha', 'lam', 'Pr', 'al'))
    F, G = solve_flow(g, D, lm, sm, Ha, *(cont if cont else (None, None)))
    th = solve_theta(g, F, Pr, al)
    M, N = solve_MN(g, F, G, th, D, lm, sm, Ha, lam)
    H, R = solve_HR(g, F, G, D, lm, sm, Ha)
    d = lambda f: (g.D1 @ f)[g.i0]
    w = dict(F2=(g.D2 @ F)[g.i0], M1=d(M), H1=d(H), G1=d(G), N1=d(N), R1=d(R), T1=d(th),
             delta=g.L - F[g.iL])
    return w, dict(F=F, G=G, th=th, M=M, N=N, H=H, R=R)


def Ns_bracket(g, p, f, xi, ze, Vr, b=0.0):
    """Eq. (27b)-(27e) bracket (per unit Br/Omega) minus free-stream value (28)."""
    D, lm, sm, Ha = p['D'], p['lm'], p['sm'], p['Ha']
    F1, F2 = g.D1 @ f['F'], g.D2 @ f['F']
    M, M1, H, H1 = f['M'], g.D1 @ f['M'], f['H'], g.D1 @ f['H']
    G, G1, N, N1, R, R1 = f['G'], g.D1 @ f['G'], f['N'], g.D1 @ f['N'], f['R'], g.D1 @ f['R']
    nd = (2 + D) * (6 * F1 ** 2 + 0.5 * (M1 + xi * F2) ** 2 + 0.5 * (Vr * H1 + ze * F2) ** 2)
    nr = 0.5 * D * ((Vr * (H1 + 2 * R) + ze * (F2 - 2 * G)) ** 2 + (M1 + xi * (F2 - 2 * G) - 2 * N) ** 2)
    ng = (D / (lm * sm)) * (2 * (1 - b) * G ** 2 + (Vr * R1 - ze * G1) ** 2 + (N1 + xi * G1) ** 2) if D > 0 else 0 * F1
    nj = Ha * ((M + xi * F1) ** 2 + (Vr * H + ze * F1) ** 2)
    ninf = 6 * (2 + D) + Ha * (xi ** 2 + ze ** 2)
    return nd + nr + ng + nj - ninf


def entropy(g, p, f, b=0.0):
    Ih = g.w @ (g.D1 @ f['th']) ** 2
    Q = lambda xi, ze, Vr: g.w @ Ns_bracket(g, p, f, xi, ze, Vr, b)
    q0 = Q(0, 0, 0); qx = (Q(1, 0, 0) - Q(-1, 0, 0)) / 2; q2 = (Q(1, 0, 0) + Q(-1, 0, 0)) / 2 - q0
    q2z = Q(0, 1, 0) - q0; qV = Q(0, 0, 1) - q0; qzV = Q(0, 1, 1) - q0 - q2z - qV
    Qb = Q(1, 0, 1)
    return dict(Ih=Ih, q0=q0, qx=qx, q2=q2, q2z=q2z, qV=qV, qzV=qzV, crit=-Ih / Qb,
                Sex_V0=Ih + 0.1 * Q(1, 0, 0), Sex_V1=Ih + 0.1 * Qb)


def alpha_c_eig(F_of_eta, Pr, L, N):
    """Smallest eigenvalue alpha of theta''+2PrF theta'+alpha Pr theta=0, theta(0)=theta(L)=0."""
    g = Grid(N, L)
    F = F_of_eta(g.eta)
    A = g.D2 + np.diag(2 * Pr * F) @ g.D1
    B = -Pr * g.I.copy()
    for i in (g.i0, g.iL):
        A[i] = g.I[i]; B[i] = 0
    ev = eig(A, B, right=False)
    ev = ev[np.isfinite(ev)]
    ev = ev[np.abs(ev.imag) < 1e-8].real
    ev = ev[ev > 0]
    return np.min(ev)


def P(**k):
    q = dict(D=3, lm=0.5, sm=0.5, Ha=1, lam=0.5, Pr=0.7, al=0.2); q.update(k)
    if q['D'] == 0: q['sm'] = 0.0
    return q


if __name__ == '__main__':
    out = {}
    NN, LL = 120, 16.0
    g = Grid(NN, LL)
    g2 = Grid(160, 22.0)            # resolution / domain check
    # 1) Table 6 (Ha=1) and Table 2 base case, with grid check
    t6 = []
    for D in (0, 1, 3, 5):
        w, f = full(g, P(D=D)); w2, _ = full(g2, P(D=D))
        t6.append(dict(D=D, **w, check_maxdiff=max(abs(w[k] - w2[k]) for k in w)))
        print('T6 Ha=1 D=%d' % D, {k: round(v, 8) for k, v in w.items()}, 'grid diff %.1e' % t6[-1]['check_maxdiff'], flush=True)
    out['T6_Ha1'] = t6
    # 2) Proposition 1 (lm=0) and heat-flux deviation (abstract 0.15 %)
    p1 = []
    for D in (1, 3, 5):
        w0, _ = full(g, P(D=D, lm=0.0)); w, _ = full(g, P(D=D))
        p1.append(dict(D=D, F2_P1=w0['F2'], H1_P1=w0['H1'], mT1_P1=-w0['T1'],
                       devF=100 * (w['F2'] / w0['F2'] - 1), devH=100 * (w['H1'] / w0['H1'] - 1),
                       devT=100 * (w['T1'] / w0['T1'] - 1)))
        # exact mapping check: Newtonian with Pr*=(1+D)Pr
        wN, _ = full(g, P(D=0, Pr=(1 + D) * 0.7))
        s = np.sqrt(1 + D)
        p1[-1]['map_err_T1'] = abs(w0['T1'] - wN['T1'] / s)
        p1[-1]['map_err_F2'] = abs(w0['F2'] - wN['F2'] / s)
        print('P1', p1[-1], flush=True)
    out['P1'] = p1
    # 3) closure lm=1/(1+D/2), sm=D (Table 9)
    t9 = []
    for D in (1, 3, 5):
        w0, _ = full(g, P(D=D, lm=0.0)); w, _ = full(g, P(D=D, lm=1 / (1 + D / 2), sm=float(D)))
        t9.append(dict(D=D, F2=w['F2'], H1=w['H1'], mT1=-w['T1'], devF=100 * (w['F2'] / w0['F2'] - 1),
                       devH=100 * (w['H1'] / w0['H1'] - 1), devT=100 * (w['T1'] / w0['T1'] - 1)))
        print('T9', t9[-1], flush=True)
    out['T9'] = t9
    # 4) entropy coefficients (Table 11), Table 10 Sex, Vr*, beta_v effect
    t11 = []
    for D in (0, 1, 3, 5):
        p = P(D=D); w, f = full(g, p); e = entropy(g, p, f)
        _, f2 = full(g2, p); e2 = entropy(g2, p, f2)
        e['Vr_star'] = np.sqrt(-(e['Sex_V0']) / (0.1 * e['qV'])) if e['Sex_V0'] < 0 else float('nan')
        e['crit_grid_diff'] = abs(e['crit'] - e2['crit'])
        if D > 0:
            em = entropy(g, p, f, b=-1.0); ep = entropy(g, p, f, b=1.0)
            e['dSex_b'] = ep['Sex_V1'] - em['Sex_V1']; e['dSex_b_rel'] = 100 * e['dSex_b'] / abs(e['Sex_V1'])
        t11.append(dict(D=D, **e)); print('T11', {k: (round(v, 6) if isinstance(v, float) else v) for k, v in t11[-1].items()}, flush=True)
    out['T11'] = t11
    # Newtonian Vr* at xi=0,1,2
    p = P(D=0); w, f = full(g, p); e = entropy(g, p, f)
    vs = []
    for xi in (0, 1, 2):
        Q0 = g.w @ Ns_bracket(g, p, f, xi, 0, 0)
        vs.append(float(np.sqrt(-(e['Ih'] + 0.1 * Q0) / (0.1 * e['qV']))))
    out['Vr_star_newt_xi012'] = vs; print('Vr* newt xi=0,1,2', vs, flush=True)
    # (Br/Omega)* vs Ha (trend, Fig 6b)
    cr = {}
    for Ha in (0, 3):
        cr[Ha] = []
        for D in (0, 5):
            pp = P(D=D, Ha=Ha); w, f = full(g, pp); cr[Ha].append(entropy(g, pp, f)['crit'])
    out['crit_Ha'] = cr; print('crit vs Ha', cr, flush=True)
    # 5) special points (Table 12, Ha = 0, 1, 3) with Vr = 1
    t12 = []
    for Ha in (0, 1, 3):
        for D in (0, 1, 3, 5):
            p = P(D=D, Ha=Ha); w, _ = full(g, p)
            xi0 = -w['M1'] / w['F2']; ze0 = -w['H1'] / w['F2']; A = (1 + D) * w['F2'] ** 2
            row = dict(Ha=Ha, D=D, xi0=xi0, ze0=ze0, A=A)
            if D > 0:
                xic = -w['N1'] / w['G1']; zec = w['R1'] / w['G1']; B = D / (p['lm'] * p['sm']) * w['G1'] ** 2
                row.update(xic=xic, zec=zec, B=B, xim=(A * xi0 + B * xic) / (A + B), zem=(A * ze0 + B * zec) / (A + B))
            t12.append(row); print('T12', {k: round(v, 5) for k, v in row.items()}, flush=True)
    out['T12'] = t12
    # 6) alpha_c, alpha_0 (Table 1) and Hermite limit
    from scipy.optimize import brentq
    t1 = []
    for D in (0, 1, 3, 5):
        p = P(D=D); _, f = full(g, p)
        Fc = f['F']; delta = LL - Fc[g.iL]
        # barycentric-free interpolation: polynomial fit through Chebyshev nodes via numpy chebfit
        xc = 2 * g.eta / LL - 1
        coef = np.polynomial.chebyshev.chebfit(xc, Fc, NN)
        Fint = lambda e: np.where(e <= LL, np.polynomial.chebyshev.chebval(2 * np.minimum(e, LL) / LL - 1, coef), e - delta)
        row = dict(D=D, delta=delta)
        for Pr in (0.7, 7.0):
            Lt = 16.0 if Pr == 0.7 else 8.0
            a1 = alpha_c_eig(Fint, Pr, Lt, 100); a2 = alpha_c_eig(Fint, Pr, Lt * 1.4, 140)
            gt = Grid(120, Lt); Ft = Fint(gt.eta)
            a0 = brentq(lambda a: (gt.D1 @ solve_theta(gt, Ft, Pr, a))[gt.i0], 0.01, a1 * 0.98)
            row['ac_%g' % Pr] = a1; row['ac_check_%g' % Pr] = abs(a1 - a2); row['a0_%g' % Pr] = a0
        t1.append(row); print('T1', {k: round(v, 6) for k, v in row.items()}, flush=True)
    out['T1'] = t1
    # Hermite limit: D=0, Ha=1, Pr -> 0
    p = P(D=0); _, f = full(g, p); Fc = f['F']; delta = LL - Fc[g.iL]
    coef = np.polynomial.chebyshev.chebfit(2 * g.eta / LL - 1, Fc, NN)
    Fint = lambda e: np.where(e <= LL, np.polynomial.chebyshev.chebval(2 * np.minimum(e, LL) / LL - 1, coef), e - delta)
    herm = []
    for Pr in (0.1, 0.03, 0.01, 0.003):
        Lt = np.sqrt(40 / Pr) + 2
        a1 = alpha_c_eig(Fint, Pr, Lt, 160); a2 = alpha_c_eig(Fint, Pr, Lt * 1.3, 200)
        herm.append([Pr, a1, abs(a1 - a2)]); print('Hermite', herm[-1], flush=True)
    out['hermite'] = herm
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, 'independent_chebyshev_K2.json'), 'w'), indent=1, default=float)
    print('done')
