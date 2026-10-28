"""Figures 1-5 of the paper at final print size (vector PDF with embedded fonts + 600 dpi PNG).

Figure 1: geometry and skin-friction lines (node position from output/results_K2.json, entry T8);
Figure 2: wall entropy generation N_s(0) over the plate, Vr = 0 and 2 (solved here);
Figure 3: temperature profiles for alpha = 0, 0.8, 1.6, 2.4 (solved here);
Figure 4: local entropy generation and its components, Delta_m = 0 and 3 (solved here);
Figure 5: excess entropy S_ex(Vr) and (Br/Omega)* (data from output/reviewer_K2.json, entries SexVr and crit).
All text is >= 8 pt at the size used in the manuscript (full width = 390 pt = 5.40 in; Fig. 3 single column 3.40 in),
black and white.
Run:  python make_figures_v1_K2.py       (after make_results_K2.py and reviewer_checks_K2.py)
Output: output/figures/Figure_N.pdf and Figure_N.png (N = 1..5)
Reproducible: deterministic (same figures for the same matplotlib version).
Requirements: Python 3.12, numpy, scipy 1.18, matplotlib (see README.md).
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from micropolar_solver import solve, wall
from entropy_cyl import Ns_parts, Ns_inf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'output')            # JSON inputs
OUT = os.path.join(DATA, 'figures')            # figures are written here
os.makedirs(OUT, exist_ok=True)
FW = 390.0 / 72.27          # manuscript text width in inches
plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 11, 'xtick.labelsize': 10,
                     'ytick.labelsize': 10, 'legend.fontsize': 10, 'axes.linewidth': 0.8,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                     'legend.frameon': False, 'lines.linewidth': 1.1, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'mathtext.fontset': 'dejavuserif'})
LS = ['-', '--', '-.', ':']; MK = ['o', 's', '^', 'v']
BASE = dict(D=3, lm=0.5, sm=0.5, Ha=1, lam=0.5, Pr=0.7, al=0.2); XI, ZE, VR, BO = 1.0, 0.0, 1.0, 0.1


def P(**k):
    q = dict(BASE); q.update(k)
    if q['D'] == 0: q['sm'] = 0.0
    return q


def S(p, L=20): return solve(p, 'cyl', L=L, tol=1e-9, robin=True)
def st(i, me=0.12): return dict(color='k', ls=LS[i % 4], marker=MK[i % 4], markevery=me, ms=3.5, mfc='white', mew=0.8)


def save(fig, n):
    fig.savefig(os.path.join(OUT, 'Figure_%d.pdf' % n), bbox_inches='tight', pad_inches=0.03)
    fig.savefig(os.path.join(OUT, 'Figure_%d.png' % n), dpi=600, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig); print('Figure', n, 'written', flush=True)


# ------------------------------------------------------------ Figure 1 geometry
r = json.load(open(os.path.join(DATA, 'results_K2.json'))); row = [x for x in r['T8'] if x[0] == 1 and x[1] == 3][0]; xi0, ze0 = row[2], row[3]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(FW, 3.0), gridspec_kw=dict(width_ratios=[1.05, 1]))
ax.set_xlim(-0.7, 3.3); ax.set_ylim(-2.3, 2.45); ax.axis('off')
ax.plot([0, 0], [-2.1, 2.1], 'k', lw=2)
for yy in np.arange(-2.05, 2.1, 0.2): ax.plot([-0.18, 0], [yy - 0.12, yy], 'k', lw=0.6)
for c in (0.12, 0.45, 1.0, 1.8):
    z = np.linspace(0.03, 3.1, 400)
    for sgn in (1, -1):
        x = sgn * np.sqrt(c / z); m = np.abs(x) < 2.1; ax.plot(z[m], x[m], color='k', lw=0.7)
        i = np.argmin(np.abs(z - 1.9)); ax.annotate('', xy=(z[i - 8], x[i - 8]), xytext=(z[i], x[i]), arrowprops=dict(arrowstyle='->', lw=0.7))
ax.annotate('', xy=(-0.45, 2.0), xytext=(-0.45, 1.2), arrowprops=dict(arrowstyle='->', lw=1)); ax.text(-0.55, 1.6, r'$x$', ha='right', va='center')
ax.annotate('', xy=(3.2, 0), xytext=(2.4, 0), arrowprops=dict(arrowstyle='->', lw=1)); ax.text(3.2, 0.12, r'$z$')
ax.add_patch(plt.Circle((-0.45, 0), 0.12, fill=False, lw=0.9)); ax.plot([-0.45], [0], 'ko', ms=2.5); ax.text(-0.45, -0.38, r'$y,\ V_0$', ha='center')
for yb in (-1.4, 1.4): ax.annotate('', xy=(2.0, yb), xytext=(1.0, yb), arrowprops=dict(arrowstyle='->', lw=1.2))
ax.text(1.5, 1.55, r'$B_0$', ha='center'); ax.text(1.5, -1.25, r'$B_0$', ha='center')
ax.annotate('', xy=(2.8, -2.1), xytext=(2.8, -1.4), arrowprops=dict(arrowstyle='->', lw=1)); ax.text(2.9, -1.8, r'$\mathbf{g}$')
ax.text(0.1, -2.3, r'$T_w$'); ax.text(2.75, 1.0, r'$T_\infty$'); ax.text(1.55, 2.2, r'$u\to ax,\ v\to ay,\ w\to -2az$', fontsize=10, ha='center')
ax.text(-0.65, 2.3, '(a)')
bx.set_aspect('equal'); bx.set_xlim(-2.2, 2.2); bx.set_ylim(-2.0, 2.6)
for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
    t = np.linspace(0.12, 3, 50); X = xi0 + t * np.sin(th); Z = ze0 + t * np.cos(th); m = (np.abs(X) < 2.1) & (Z > -1.9) & (Z < 2.5)
    bx.plot(Z[m], X[m], 'k', lw=0.6); j = min(18, m.sum() - 1)
    if m.sum() > 20: bx.annotate('', xy=(Z[j + 2], X[j + 2]), xytext=(Z[j], X[j]), arrowprops=dict(arrowstyle='->', lw=0.6))
bx.plot(ze0, xi0, 'ko', ms=4); bx.plot(0, 0, 'k+', ms=7, mew=0.9)
bx.text(ze0 + 0.12, xi0 - 0.42, r'$(\xi_0,\zeta_0)$', fontsize=10, bbox=dict(fc='white', ec='none', pad=0.3))
bx.text(-0.12, 0.12, r'$O$', fontsize=10, ha='right', va='bottom', bbox=dict(fc='white', ec='none', pad=0.5))
bx.annotate('', xy=(1.8, -1.75), xytext=(0.6, -1.75), arrowprops=dict(arrowstyle='->', lw=1.2)); bx.text(1.2, -1.6, r'$V_0$', ha='center')
bx.set_xlabel(r'$\zeta=y/\varepsilon$'); bx.set_ylabel(r'$\xi=x/\varepsilon$'); bx.text(-2.1, 2.25, '(b)')
fig.tight_layout(pad=0.3); save(fig, 1)

# ------------------------------------------------------ Figure 2 wall entropy map
p = P(); s_ = S(p); w = wall(s_, p); y0 = s_.sol(0.0)
A = (1 + p['D']) * w['F2'] ** 2; B = p['D'] / (p['lm'] * p['sm']) * w['G1'] ** 2
zg = np.linspace(-1.5, 3.5, 161); xg = np.linspace(-2, 2, 129)
ZL, XL = (-1.5, 3.5), (-2, 2)


def label_points(cs, center, angles, margin=0.3):
    """One label per contour level, placed inside the axes away from the frame."""
    pts = []
    for lev, segs in zip(cs.levels, cs.allsegs):
        V = np.vstack(segs) if len(segs) else np.empty((0, 2))
        if len(V) == 0: continue
        inside = (V[:, 0] > ZL[0] + margin) & (V[:, 0] < ZL[1] - margin) & (V[:, 1] > XL[0] + margin) & (V[:, 1] < XL[1] - margin)
        V = V[inside]
        if len(V) == 0: continue
        ang = np.degrees(np.arctan2(V[:, 1] - center[1], V[:, 0] - center[0]))
        for a0 in angles:
            d = np.abs((ang - a0 + 180) % 360 - 180)
            if d.min() < 20: pts.append(tuple(V[np.argmin(d)])); break
    return pts


fig, ax = plt.subplots(1, 2, figsize=(FW, 2.95), sharey=True)
for k, (a, Vr, t) in enumerate(zip(ax, (0.0, 2.0), 'ab')):
    Z, X = np.meshgrid(zg, xg); SS = np.vectorize(lambda xi, ze: sum(Ns_parts(y0, p, xi, ze, Vr, BO).values()))(X, Z)
    cs = a.contour(Z, X, SS, levels=[0.3, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0], colors='k', linewidths=0.6)
    x0, z0 = -w['M1'] / w['F2'], -Vr * w['H1'] / w['F2']; xc, zc = -w['N1'] / w['G1'], Vr * w['R1'] / w['G1']
    xm, zm = (A * x0 + B * xc) / (A + B), (A * z0 + B * zc) / (A + B)
    angles = (-50, -130, 50, 130) if k == 0 else (50, -50, -130, 130)
    a.clabel(cs, fontsize=10, fmt='%.1f', manual=label_points(cs, (zm, xm), angles), inline_spacing=3)
    a.plot(zm, xm, 'ks', ms=10, mfc='none', mew=1.3, label=r'minimum of $N_s(0)$')
    a.plot(z0, x0, 'ko', ms=4, label='zero wall shear')
    a.plot(zc, xc, marker='D', color='k', ls='none', ms=6, mfc='0.6', mew=0.9, label='zero couple stress')
    a.set_xlim(*ZL); a.set_ylim(*XL); a.set_aspect('equal'); a.set_xlabel(r'$\zeta$')
    a.text(0.03, 0.91, r'(%s) $V_r=%g$' % (t, Vr), transform=a.transAxes, bbox=dict(fc='white', ec='none', pad=1))
ax[0].set_ylabel(r'$\xi$')
h, l = ax[1].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.5)
fig.subplots_adjust(left=0.085, right=0.995, bottom=0.2, top=0.99, wspace=0.06); save(fig, 2)

# ----------------------------------------------------- Figure 3 temperature (Δm=1)
eta = np.linspace(0, 6, 601)
fig, a = plt.subplots(figsize=(3.4, 2.6))
for i, al in enumerate((0, 0.8, 1.6, 2.4)):
    s = S(P(D=1, al=al)); a.plot(eta, s.sol(eta)[13], label=r'$\alpha=%.1f$' % al, **st(i))
a.set_xlabel(r'$\eta$'); a.set_ylabel(r'$\theta$'); a.set_xlim(0, 5); a.legend(loc='upper right')
fig.tight_layout(pad=0.3); save(fig, 3)

# -------------------------------------------------- Figure 4 entropy components
sols = {D: S(P(D=D)) for D in (0, 3)}
fig, ax = plt.subplots(1, 2, figsize=(FW, 2.6), sharey=True)
for a, D, t in zip(ax, (0, 3), 'ab'):
    p = P(D=D); d = Ns_parts(sols[D].sol(eta), p, XI, ZE, VR, BO); tot = sum(d.values())
    a.plot(eta, tot, color='k', lw=1.6, label=r'$N_s$')
    for i, (k, lab) in enumerate([('heat', r'$N_h$'), ('visc', r'$N_d+N_r$'), ('couple', r'$N_g$'), ('joule', r'$N_J$')]):
        if D == 0 and k == 'couple': continue
        a.plot(eta, d[k], label=lab, **st(i))
    a.axhline(Ns_inf(p, XI, ZE, BO), color='0.5', ls=':', lw=0.8); a.set_xlabel(r'$\eta$'); a.set_xlim(0, 6)
    a.text(0.04, 0.9, r'(%s) $\Delta_m=%d$' % (t, D), transform=a.transAxes)
ax[0].set_ylabel(r'$N_s$ and its components'); ax[1].legend(loc='center right')
fig.tight_layout(pad=0.3); save(fig, 4)

# ------------------------------------------------- Figure 5 excess entropy
rv = json.load(open(os.path.join(DATA, 'reviewer_K2.json'))); Vs, sv = rv['SexVr']
fig, ax = plt.subplots(1, 2, figsize=(FW, 2.6))
for i, D in enumerate((0, 1, 3, 5)):
    ax[0].plot(Vs, sv[i], color='k', ls=LS[i], marker=MK[i], markevery=4, ms=3.5, mfc='white', mew=0.8, label=r'$\Delta_m=%d$' % D)
ax[0].axhline(0, color='0.5', ls=':', lw=0.8); ax[0].set_xlabel(r'$V_r$'); ax[0].set_ylabel(r'$S_{ex}$'); ax[0].set_ylim(top=8.0); ax[0].legend(loc='upper left', ncol=2, columnspacing=0.8, handlelength=1.8)
Ds = (0, 0.5, 1, 2, 3, 4, 5)
for i, row in enumerate(rv['crit']):
    ax[1].plot(Ds, row[1:], color='k', ls=LS[i], marker=MK[i], ms=3.5, mfc='white', mew=0.8, label=r'Ha$=%d$' % row[0])
ax[1].set_yscale('log'); ax[1].set_xlabel(r'$\Delta_m$'); ax[1].set_ylabel(r'$(\mathrm{Br}/\Omega)^*$'); ax[1].legend()
ax[0].text(0.95, 0.06, '(a)', transform=ax[0].transAxes, ha='right'); ax[1].text(0.05, 0.06, '(b)', transform=ax[1].transAxes)
fig.tight_layout(pad=0.3); save(fig, 5)
print('done')
