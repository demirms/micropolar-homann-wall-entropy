# Wall-stress nodes and minimum wall entropy generation in micropolar Homann flow over a translating plate

Code accompanying the article of the same title by Mehmet Şirin Demir (Istanbul University-Cerrahpaşa).

This repository contains the solver and all scripts that produce the numbers, tables and figures of the article
(axisymmetric Homann stagnation-point flow of an electrically conducting micropolar fluid towards a vertical plate
translating in its own plane; skin-friction node, couple-stress node and minimum of the wall entropy generation).

## Environment

- Python 3.12, numpy 2.5, scipy 1.18, matplotlib 3.11, sympy 1.14 (only `hermite_correction_K2.py`); mpmath is not needed.
- Tested on Windows 10 (8 GB RAM, one script at a time); no GPU, no network access, no random numbers except one fixed-seed test.
- Windows: set `MPLBACKEND=Agg` and `PYTHONUTF8=1` if the console cannot print Greek letters.

## How to run

Run every script from this folder; each script finds its own location and writes only into `output/`
(the JSON files that were produced by the original run are shipped in `output/` as reference values and are overwritten).
The run order is in `run_all.txt` (one script per line, `after:` marks prerequisites):

```
python make_results_K2.py
python verify_numerics_K2.py
python extra_K2.py
python reviewer_checks_K2.py
python make_tables_K2.py
python make_figures_v1_K2.py
python checks_K2_review2.py
python validate_borrelli2013.py
python verify_entropy_tensor_cyl.py
python independent_chebyshev_K2.py
python hermite_correction_K2.py
python independent_wall_entropy_validation_K2.py
python point_separation_magnitude_K2.py
python parameter_realism_K2.py
python theta_min_supercritical_K2.py
python v1_extra_numbers_K2.py
```

Order matters only for `make_tables_K2.py` (needs the JSON files of steps 1-4), `make_figures_v1_K2.py` (needs
`results_K2.json` and `reviewer_K2.json`) and `v1_extra_numbers_K2.py` (needs `independent_chebyshev_K2.json`).
Run times are given at the end of this file.

## Library modules (not run directly)

| File | Content |
|---|---|
| `micropolar_solver.py` | similarity equations (prismatic and cylindrical), `solve_bvp` collocation solver, wall gradients |
| `entropy_cyl.py` | local entropy generation N_s and its components, free-stream value, excess entropy S_ex |
| `critical_alpha.py` | critical heat-generation parameter alpha_c (eigenvalue problem), alpha_0 |
| `shooting_solver.py` | independent second solver (shooting, DOP853 + fsolve) |

## Mapping: script -> paper item

| Script | Output (in `output/`) | Paper item |
|---|---|---|
| `make_results_K2.py` | `results_K2.json` (+ `figures_draft/`, earlier drafts, not in the paper) | Table 2 (wall gradients, Ha=1 rows), Table 3 (nodes, weights A, B), N_s(0) and Be(0) of Table 5, node position of Figure 1, brute-force check of the minimum (Eq. for the minimum point), trends with Ha, lambda_m, sigma_m, Pr, alpha quoted in the text |
| `verify_numerics_K2.py` | `numerics_K2.json` | Table 1 (Newtonian rows), Table 4 (delta_inf, alpha_0, alpha_c), independence of L and tolerance, collocation versus shooting (3e-14), Robin versus Dirichlet condition, uniqueness test, effective-viscosity mapping, Hermite limit |
| `extra_K2.py` | `extra_K2.json` | Table 5 (S_ex for V_r=0 and 1, V_r*), deviations from the effective-viscosity mapping, wall-entropy shares, peak of M |
| `reviewer_checks_K2.py` | `reviewer_K2.json` (+ draft figure) | Table 6 (coefficients of Q, critical (Br/Omega)*), data of Figure 5, truncated-domain test (7e-7, 4e-12) |
| `make_tables_K2.py` | `tables_K2.json` | Tables 1, 2, 3, 5, 6 formatted as Markdown, including the Borrelli et al. (2013) comparison rows |
| `make_figures_v1_K2.py` | `figures/Figure_1..5.pdf/.png` | Figures 1-5 |
| `checks_K2_review2.py` | `checks_K2_review2.json` | dissipative-heating correction (Gamma = 8-10, 87-92, 3-37; Br limits 1.0e-3 and 1.1e-4), alpha_c convergence, maximum residual |
| `validate_borrelli2013.py` | screen | Table 1, micropolar rows (Borrelli, Giantesio and Patria 2013) |
| `verify_entropy_tensor_cyl.py` | screen | agreement of the dissipation function with its tensor form (5e-11, b = 0.37) |
| `independent_chebyshev_K2.py` | `independent_chebyshev_K2.json` | independent Chebyshev spectral check of Tables 1-6 and of alpha_c, V_r* and the effective-viscosity mapping |
| `hermite_correction_K2.py` | `hermite_correction_K2.json` | Hermite small-Prandtl correction, 4 - alpha_c = (4/sqrt(pi)) delta_inf sqrt(Pr) + O(Pr) |
| `independent_wall_entropy_validation_K2.py` | `independent_wall_entropy_validation_K2.json` | independent N_s(0), Be(0) of Table 5, minimum point of Table 3, validation table |
| `point_separation_magnitude_K2.py` | `point_separation_magnitude_K2.json` | node separation d and the residual wall irreversibility at the minimum (at most 1.2 % for V_r <= 2) |
| `parameter_realism_K2.py` | `parameter_realism_K2.json` | dimensional engine-oil example (Br/Omega, Br), liquid-metal Ha, heat-generation and micro-length estimates |
| `theta_min_supercritical_K2.py` | screen | minimum and maximum of theta for alpha = 3 and 2.4 quoted in the text |
| `v1_extra_numbers_K2.py` | `v1_extra_numbers_K2.json` | eta_99 versus micro-length, node-separation ratios, (4 - alpha_c)/sqrt(Pr) |

## Reproducibility

All computations are deterministic. The only random numbers (uniqueness test in `verify_numerics_K2.py`) use a fixed
seed (`default_rng(1)`). Results agree with the shipped reference outputs to a relative tolerance of 1e-12 or better when
Python, numpy and scipy versions are the ones listed above; other versions may change the last digits.
Figures are regenerated with the same matplotlib version for identical drawings. The `figures_draft/` PNGs are earlier
drafts that are not part of the paper.

Internal labels in the independent code (T1, T6, T9, T11, T12) and the equation numbers in comments follow an earlier working
draft; the table above gives the paper numbering.

## Clean-copy test (verification record)

The archive was run end to end in a clean temporary copy (run order of `run_all.txt`, outputs compared with the shipped reference files): 16/16 scripts finished without error in a total of about
650 s (make_results 127 s, verify_numerics 271 s, checks_K2_review2 162 s, all others below 25 s each).
Of 12 JSON outputs, 9 are identical to 1e-12 relative. The other three differ only by rounding noise:
`numerics_K2.json` (`shoot_res`, the shooting residual 6e-15 versus 8e-15), `reviewer_K2.json` (`coef[0][9]`, an identity
check of size 1e-15) and `checks_K2_review2.json` (two entries of `alpha`: -theta'(0) = -50.9020739 near the critical
alpha_c, where the value is ill-conditioned and the solver tolerance is 1e-6; relative difference 1.3e-9 at most).

## License

MIT License (see `LICENSE`).
