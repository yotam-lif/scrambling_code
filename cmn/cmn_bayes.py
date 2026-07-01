r"""Bayesian inference of the peak-DFE boundary floor — consolidated module.

Single home for the floor/pseudogap computation behind Figure 4 and its
supplement. The figures only READ the JSON products listed below; all the
inference lives here.

Model for the near-zero DFE density (deleterious effects u = -Delta >= 0):

    p(u ; N) = c * N^{-alpha} + d * u^theta
             = p0(N)          + pseudogap

  p0(N) = c N^{-alpha}   boundary floor   (alpha = 0  => persistent floor)
  d, theta               pseudogap coefficient / exponent
  N                      DFE sample size (loci / mutations; "m" for FGM)

Likelihood: extended (unbinned Poisson) on a near-zero window u in [0, U] — the
only form that makes the ABSOLUTE floor height identifiable (a shape-only
likelihood fixes just p0/c). Sampled with a hand-rolled Goodman-Weare ensemble
sampler in conditioning coordinates A = d U^theta, P = c Nref^{-alpha}; reported
back as c, d, theta, alpha.

Known values imposed via `infer(fix_theta=..., fix_d=...)`:
  FGM            theta = n/2 - 1 (chi-squared)   -> free c, d, alpha
  SK (p=2)       theta = 1 [, d = 0.3]           -> free c, [d,] alpha
  NK, p-spin p=3 nothing                          -> free theta, d, c, alpha

Data products written to data/ (read by the figures, never recomputed by them):

  floor_effects_cache.pkl    sorted deleterious effects per (model, param, N);
                             slow to build (p=3 is multi-GB) -- built once.
  floor_alpha_by_param.json  {model: {param: [med, lo, hi]}} floor exponent alpha
                             -> fig4_peak_dfes.py inset
  floor_theta_by_param.json  {model: {param: [med, lo, hi]}} pseudogap exponent theta
                             -> figS6_peak_dfe_bayes.py panel A
  fgm_radius_by_n.json       FGM endpoint radius r ~ m^{-gamma} per n
                             -> figS6_peak_dfe_bayes.py panel B

CLI (run from the repo root):

    python cmn/cmn_bayes.py cache    # (re)build floor_effects_cache.pkl  [slow]
    python cmn/cmn_bayes.py alpha    # -> data/floor_alpha_by_param.json
    python cmn/cmn_bayes.py theta    # -> data/floor_theta_by_param.json
    python cmn/cmn_bayes.py radius   # -> data/fgm_radius_by_n.json
    python cmn/cmn_bayes.py all      # alpha + theta + radius (cache must exist)

──────────────────────────────────────────────────────────────────────────────
FINDINGS  (does the boundary floor p0(N) = c N^{-alpha} vanish with system size?)

  SK (p-spin p=2) — TRUSTWORTHY. Pinning the known theta=1 and letting d float,
    the fit returns d ~ 0.28, reproducing the exact SK pseudogap coefficient 0.3
    (0.3 sits inside the CI). The floor then vanishes: alpha ~ 0.29 [0.20, 0.38].
    Mechanism: marginal stability. (`alpha` pins theta=1 for SK only.)
  p-spin p=3 — vanishes, alpha ~ 0.59 [0.47, 0.72]; theta floats to ~1.
  NK (negative control) — alpha ~ 0 across K: a genuine PERSISTENT floor. This is
    the control the method must discriminate from the vanishing cases, and does.
  FGM — the floor vanishes too, but its exponent is assumption-dependent; see the
    caveat below. The published `alpha` product leaves FGM theta FREE.

FGM caveat (why its alpha is soft, and why it is NOT the chi-squared):
  The FGM near-zero DFE is a FLOOR, not the equilibrium chi^2 u^{n/2-1}. Genotype
  sigma in {+-1}^N, phenotype z = sum_i sigma_i a_i (a_i in R^n, |a_i| ~ s sqrt n,
  s=0.05), fitness -|z|^2/2. At a single-flip optimum the cost of flipping locus i
  is u_i = 2|z^{(i)}.a_i| (z^{(i)} = cavity phenotype). For isotropic a_i that
  projection has a finite, smooth density at 0 -> a floor (leading correction
  ~u^2), matching the measured near-0 slope (theta_eff ~ 1, flat) rather than
  n/2-1. The chi^2 is only the r=|z*|->0 limit (the exact peak), which the
  adaptive walk never reaches. The floor exists ONLY because the endpoint sits
  off-optimum at r>0, and it CLOSES as the endpoint radius shrinks with system
  size: r(N) ~ N^{-0.2..-0.5} (this is exactly what the `radius` command / fig.
  S3 panel B measures, and f(0) rises with r within each n). So alpha_FGM ~ the
  r-decay rate ~ 0.2-0.5, which matches a chi^2-theta-fixed fit (0.15-0.39). The
  free-theta fit gives a larger, rising alpha (~0.3/0.9/1.1/1.5 for n=4/8/16/32)
  and is partly MISSPECIFIED — free theta absorbs the r-driven curvature of the
  floor. Treat the FGM exponent as soft; the robust statement is "vanishes, via
  r(N)->0". (To harden it: a high-rep N-sweep, then regress f(0) on the measured
  r to test f(0) prop r^gamma and pin alpha.)

VALIDATION & CONTROLS
  On synthetic data at the real sample size (M = 10 N) the inference recovers a
  known alpha to +-0.03, and is window-robust (U = 10/20/30% quantile give the
  same alpha). A pure theta=1, no-floor null is NOT misread as a floor (returns
  theta ~ 1, d ~ 0; a real floor carries ~37% of the near-zero window mass vs ~3%
  for the null). NK (alpha ~ 0) is the live negative control.

HISTORY
  Supersedes an order-statistic estimator P_hat(0;k) = k/(n u_(k)) = p0(N) +
  [d/(theta+1)] u_(k)^theta, whose fitted slope conflated the floor with a
  pseudogap-rounding bias; the joint floor+pseudogap likelihood below separates
  the two.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
# When run as a script (`python cmn/cmn_bayes.py`) sys.path[0] is cmn/ itself,
# where cmn.py would shadow the `cmn` package as a top-level module. Drop that
# entry and expose the repo root so `from cmn import ...` finds the package.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != SCRIPT_DIR]
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from cmn import cmn, cmn_pspin

DATA = os.path.join(REPO_DIR, "data")
CACHE = os.path.join(DATA, "floor_effects_cache.pkl")
REPS = 10
RNG = np.random.default_rng(0)

# model -> control params, N-sweep, reference N. FGM/p-spin sweep 100..500; NK
# extends to 2000.
SPECS = {
    "FGM":   {"params": [4, 8, 16, 32], "Ns": [100, 200, 300, 400, 500], "Nref": 300},
    "PSPIN": {"params": [2, 3],         "Ns": [100, 200, 300, 400, 500], "Nref": 300},
    "NK":    {"params": [4, 8, 16, 32], "Ns": [100, 200, 300, 400, 500, 1000, 2000], "Nref": 400},
}


# ── effect-size cache: sorted deleterious effects per (model, param, N) ─────────
def _dele(a):
    a = np.asarray(a, float)
    return np.sort(-a[a < 0])


def fgm_eff(n, N):
    p = f"{DATA}/FGM/fgm_rps{REPS}_n{n}_N{N}_sig0.05.pkl"
    if not os.path.exists(p):
        return None
    return [_dele(r["dfes"][-1]) for r in pickle.load(open(p, "rb"))]


def nk_eff(K, N):
    m = glob.glob(f"{DATA}/NK/N_{N}_K_{K}_repeats_*.pkl")
    if not m:
        return None
    return [_dele(np.asarray(r["dfes"][-1], float) * N) for r in pickle.load(open(m[0], "rb"))]


def pspin_eff(P, N):
    name = (f"N400_P{P}_pure_repeats{REPS}.pkl" if N == 400
            else f"N{N}_P{P}_pure_repeats{REPS}.pkl")
    p = f"{DATA}/PSPIN/{name}"
    if not os.path.exists(p):
        return None
    d = pickle.load(open(p, "rb"))
    out = []
    for e in d:
        sigma = cmn.compute_sigma_from_hist(e["init_sigma"], e["flip_seq"])
        out.append(_dele(cmn_pspin.compute_dfe(sigma, e["J"])))
    del d
    return out


LOADERS = {"FGM": fgm_eff, "NK": nk_eff, "PSPIN": pspin_eff}


def build_cache():
    """(Re)build data/floor_effects_cache.pkl from the raw N-sweep data. Slow:
    the p-spin p=3 files are multi-GB and loaded one at a time."""
    cache = {}
    for model, spec in SPECS.items():
        cache[model] = {}
        for prm in spec["params"]:
            cache[model][prm] = {}
            for N in spec["Ns"]:
                reps = LOADERS[model](prm, N)
                if reps is None:
                    continue
                u = np.sort(np.concatenate(reps))
                cache[model][prm][N] = u.astype(np.float32)
                print(f"  {model} {prm} N={N}: M={u.size}")
    pickle.dump(cache, open(CACHE, "wb"))
    print("wrote", CACHE)


# ── Goodman-Weare ensemble sampler (red-black split, batched per half) ──────────
def gw_sampler(lpv, p0, nsteps, a=2.0):
    nw, nd = p0.shape
    pos = p0.copy()
    lp = lpv(pos)
    idx = np.arange(nw)
    keep = []
    for t in range(nsteps):
        for s in (0, 1):
            act = idx[idx % 2 == s]
            oth = idx[idx % 2 != s]
            m = len(act)
            z = ((a - 1.0) * RNG.random(m) + 1.0) ** 2 / a
            partners = oth[RNG.integers(0, len(oth), m)]
            prop = pos[partners] + z[:, None] * (pos[act] - pos[partners])
            lpp = lpv(prop)
            logr = (nd - 1) * np.log(z) + lpp - lp[act]
            accpt = np.log(RNG.random(m)) < logr
            pos[act[accpt]] = prop[accpt]
            lp[act[accpt]] = lpp[accpt]
        keep.append(pos.copy())
    return np.array(keep)


# ── constrained inference: pin the known physics, let only the floor float ─────
def windows(per, Ns, halve=False, q=0.20):
    sc = 2.0 if halve else 1.0
    eff = {N: np.asarray(per[N], float) / sc for N in Ns}
    U = float(np.quantile(eff[max(Ns)], q))
    W, ds = [], []
    for N in Ns:
        u = eff[N]; x = u[u < U]
        W.append((N, u.size, x, U)); ds.append(x.size / (u.size * U))
    return W, U, float(np.median(ds))


def infer(per, Ns, Nref, halve=False, fix_theta=None, fix_d=None, nsteps=6000):
    """Posterior over (theta, d, c, alpha) for one (model, param) N-sweep.

    `per` maps N -> sorted deleterious effects (from the cache). Returns a dict
    of posterior-sample arrays theta/d/c/alpha/P plus the scalar window U.
    Internally sampled in conditioning coordinates A = d U^theta, P = c Nref^{-alpha}.
    """
    W, U, ds = windows(per, Ns, halve=halve)
    Amax = Pmax = 60 * ds
    free = ([] if fix_theta is not None else ["theta"]) \
        + ([] if fix_d is not None else ["A"]) + ["P", "alpha"]
    ix = {nm: i for i, nm in enumerate(free)}

    def unpack(Q):
        n = len(Q)
        theta = Q[:, ix["theta"]] if "theta" in ix else np.full(n, fix_theta)
        A = Q[:, ix["A"]] if "A" in ix else fix_d * U ** theta
        return theta, A, Q[:, ix["P"]], Q[:, ix["alpha"]]

    def lpv(Q):
        Q = np.atleast_2d(Q); n = len(Q)
        theta, A, P, alpha = unpack(Q)
        out = np.full(n, -np.inf)
        ok = ((theta > 0.02) & (theta < 20) & (A >= 0) & (P >= 0) & (P <= Pmax)
              & (alpha > -1.5) & (alpha < 3))
        if "A" in ix:
            ok &= (A <= Amax)
        if not ok.any():
            return out
        th = theta[ok][:, None]; Ag = A[ok][:, None]; Pg = P[ok]; al = alpha[ok]
        acc = np.zeros(ok.sum())
        for N, M, x, Uw in W:
            p0 = Pg * (Nref / N) ** al
            dens = p0[:, None] + Ag * (x[None, :] / Uw) ** th
            with np.errstate(divide="ignore"):
                acc += np.sum(np.log(dens), axis=1) - M * (p0 * Uw + Ag[:, 0] * Uw / (th[:, 0] + 1))
        out[ok] = acc
        return out

    nd = len(free); nw = 40
    base = {"theta": 1.0, "A": ds, "P": ds, "alpha": 0.3}
    start = np.array([base[nm] for nm in free])
    p0 = start * (1 + 0.3 * RNG.standard_normal((nw, nd)))
    for nm in ("A", "P"):
        if nm in ix:
            p0[:, ix[nm]] = np.clip(p0[:, ix[nm]], 1e-4, Amax)
    if "theta" in ix:
        p0[:, ix["theta"]] = np.clip(p0[:, ix["theta"]], 0.1, 8)
    flat = gw_sampler(lpv, p0, nsteps)[2000:].reshape(-1, nd)

    theta, A, P, alpha = unpack(flat)
    d = A / U ** theta
    c = P * Nref ** alpha
    return {"theta": theta, "d": d, "c": c, "alpha": alpha, "P": P, "U": U}


# ── data products ──────────────────────────────────────────────────────────────
# Credible/bootstrap intervals everywhere are the central 95% (2.5-97.5%).
_CI_PCT = [2.5, 97.5]


def _a_ci(r):
    p = np.percentile(r["alpha"], [50] + _CI_PCT)
    return [float(p[0]), float(p[1]), float(p[2])]


def _t_ci(r):
    p = np.percentile(r["theta"], [50] + _CI_PCT)
    return [float(p[0]), float(p[1]), float(p[2])]


def compute_alpha():
    """Floor exponent alpha per parameter, with 95% credible intervals. The only
    constraint is the known SK value theta=1 (p=2); theta is free everywhere
    else. Writes data/floor_alpha_by_param.json -> {model: {param: [med, lo, hi]}}."""
    global RNG
    RNG = np.random.default_rng(0)
    cache = pickle.load(open(CACHE, "rb"))
    out = {"FGM": {}, "PSPIN": {}, "NK": {}}

    for n in SPECS["FGM"]["params"]:
        per = cache["FGM"][n]; Ns = sorted(per)
        out["FGM"][str(n)] = _a_ci(infer(per, Ns, SPECS["FGM"]["Nref"]))   # theta free
        print("FGM", n, out["FGM"][str(n)])

    per = cache["PSPIN"][2]; Ns = sorted(per)
    out["PSPIN"]["2"] = _a_ci(infer(per, Ns, SPECS["PSPIN"]["Nref"], fix_theta=1.0))  # only SK constraint
    print("SK", out["PSPIN"]["2"])
    per = cache["PSPIN"][3]; Ns = sorted(per)
    out["PSPIN"]["3"] = _a_ci(infer(per, Ns, SPECS["PSPIN"]["Nref"]))
    print("p=3", out["PSPIN"]["3"])

    for K in SPECS["NK"]["params"]:
        per = cache["NK"][K]; Ns = [N for N in sorted(per) if N <= 1000]
        out["NK"][str(K)] = _a_ci(infer(per, Ns, 400))
        print("NK", K, out["NK"][str(K)])

    path = os.path.join(DATA, "floor_alpha_by_param.json")
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


def compute_theta():
    """Pseudogap exponent theta per parameter, theta FREE everywhere (a genuine
    inference of the near-zero rise; the known SK theta=1 and FGM chi^2
    theta=n/2-1 are drawn as guides in the figure, not imposed here). Writes
    data/floor_theta_by_param.json -> {model: {param: [med, lo, hi]}}."""
    global RNG
    RNG = np.random.default_rng(0)
    cache = pickle.load(open(CACHE, "rb"))
    out = {"FGM": {}, "PSPIN": {}, "NK": {}}

    for n in SPECS["FGM"]["params"]:
        per = cache["FGM"][n]; Ns = sorted(per)
        out["FGM"][str(n)] = _t_ci(infer(per, Ns, SPECS["FGM"]["Nref"]))   # theta free
        print("FGM", n, out["FGM"][str(n)])

    for P in SPECS["PSPIN"]["params"]:                                     # p=2 (SK), p=3
        per = cache["PSPIN"][P]; Ns = sorted(per)
        out["PSPIN"][str(P)] = _t_ci(infer(per, Ns, SPECS["PSPIN"]["Nref"]))  # theta free
        print("PSPIN", P, out["PSPIN"][str(P)])

    for K in SPECS["NK"]["params"]:
        per = cache["NK"][K]; Ns = [N for N in sorted(per) if N <= 1000]
        out["NK"][str(K)] = _t_ci(infer(per, Ns, 400))                     # theta free
        print("NK", K, out["NK"][str(K)])

    path = os.path.join(DATA, "floor_theta_by_param.json")
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


# ── FGM endpoint radius r ~ m^{-gamma} (mechanism that closes the FGM floor) ────
_RADIUS_NS = [4, 8, 16, 32]              # phenotype dimension n
_RADIUS_MS = [100, 200, 300, 400, 500]   # number of loci m (= N in the file name)
_RADIUS_NBOOT = 4000


def _load_radii(n, m):
    """Endpoint radii r=|traj[-1]| over reps for FGM (n, m), or None if missing."""
    path = os.path.join(DATA, "FGM", f"fgm_rps10_n{n}_N{m}_sig0.05.pkl")
    if not os.path.exists(path):
        return None
    reps = pickle.load(open(path, "rb"))
    return np.asarray([float(np.linalg.norm(np.asarray(r["traj"], float)[-1])) for r in reps], float)


def _fit_gamma(logm, logr):
    """Least-squares slope of logr vs logm; return -slope (= gamma)."""
    A = np.vstack([logm, np.ones_like(logm)]).T
    slope, _ = np.linalg.lstsq(A, logr, rcond=None)[0]
    return -slope


def compute_radius():
    """FGM endpoint radius r=|z*| vs number of loci m, fit r ~ m^{-gamma} per
    phenotype dimension n with a bootstrap CI over reps. Writes
    data/fgm_radius_by_n.json."""
    global RNG
    RNG = np.random.default_rng(0)
    out = {}
    for n in _RADIUS_NS:
        per_rep = {}
        for m in _RADIUS_MS:
            r = _load_radii(n, m)
            if r is not None:
                per_rep[m] = r
        ms = sorted(per_rep)
        if len(ms) < 2:
            print(f"n={n}: insufficient data, skipping")
            continue

        r_mean = np.array([per_rep[m].mean() for m in ms])
        r_std = np.array([per_rep[m].std(ddof=1) for m in ms])
        logm = np.log(np.asarray(ms, float))

        # Point estimate: fit to the per-N means (log space).
        gamma_hat = _fit_gamma(logm, np.log(r_mean))

        # Bootstrap over reps: resample the REPS replicates at each m, recompute
        # the per-m mean, refit. Central 95% gives the credible interval.
        nrep = min(len(per_rep[m]) for m in ms)
        gammas = np.empty(_RADIUS_NBOOT)
        for b in range(_RADIUS_NBOOT):
            boot_mean = np.empty(len(ms))
            for i, m in enumerate(ms):
                idx = RNG.integers(0, len(per_rep[m]), nrep)
                boot_mean[i] = per_rep[m][idx].mean()
            gammas[b] = _fit_gamma(logm, np.log(boot_mean))
        lo, hi = np.percentile(gammas, _CI_PCT)

        out[str(n)] = {"Ns": ms, "r_mean": r_mean.tolist(), "r_std": r_std.tolist(),
                       "gamma": [float(gamma_hat), float(lo), float(hi)]}
        print(f"n={n}: gamma = {gamma_hat:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"(r {r_mean[0]:.3f} -> {r_mean[-1]:.3f})")

    path = os.path.join(DATA, "fgm_radius_by_n.json")
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


_COMMANDS = {"cache": build_cache, "alpha": compute_alpha,
             "theta": compute_theta, "radius": compute_radius}


def main(argv):
    cmds = argv or ["all"]
    if cmds == ["all"]:
        cmds = ["alpha", "theta", "radius"]
    unknown = [c for c in cmds if c not in _COMMANDS]
    if unknown:
        print(__doc__)
        print(f"unknown command(s): {unknown}")
        return
    for c in cmds:
        print(f"== {c} ==")
        _COMMANDS[c]()


if __name__ == "__main__":
    main(sys.argv[1:])
