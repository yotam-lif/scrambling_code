r"""Bayesian inference of Fisher's Geometric Model DFE parameters from data.

For every empirical DFE (Ascensao et al. and Couce et al.) we infer the joint
posterior over the three parameters of the analytic isotropic FGM DFE derived in
Theory/fgm_dfe.tex (and implemented in cmn/cmn_fgm.py):

    n      phenotypic dimension          (integer)
    sigma  mutation-step standard dev.
    r      current radius (= distance to the fitness optimum)

    Delta = exp(-|r + delta|^2 / 2) - exp(-r^2 / 2),   delta ~ N(0, sigma^2 I_n).

Priors (flat):  n ~ U{1,...,60},  sigma ~ U[0, 10],  r ~ U[0, 100].

Method.  The posterior is computed on a grid (exact marginalisation; n is
discrete, so a grid is the natural tool and gives P(n) directly). The likelihood
is a fine-binned multinomial with CDF-exact bin probabilities (cmn_fgm.
fgm_bin_loglik) -- ~50x cheaper than a per-point likelihood over ~10^4 effects
and statistically equivalent for a smooth DFE with a few hundred bins. A coarse
stage-1 grid (log-spaced sigma, full feasible r, all n) locates the posterior
mass; a refined stage-2 grid resolves it.

Key structural fact: the FGM DFE has hard support [-e^{-r^2/2}, 1-e^{-r^2/2}], so
the data's most-deleterious effect caps r from above and its most-beneficial
effect floors r from below. r is therefore largely pinned by the data range; n
and sigma shape the distribution within it. (This also means r is sensitive to a
single extreme outlier -- reported as the feasible_r interval per dataset.)

Two kinds of fit:
  * per-DFE  -- independent (n, sigma, r) for each Ascensao DFE;
  * shared   -- one (n, sigma) shared across an experiment, r free per timepoint:
                Couce 0K -> 2K -> 15K, and the Ascensao ancestor->offshoot pairs
                R->L and R->S (R = ancestor; L, S = evolved offshoots).
Couce is fit shared-only and is the sole content of the one figure; Ascensao is
fit both ways and exported as text.

Robustness: the deleterious extremes are mostly lethals / essential-gene knockouts
that FGM does not model, so a small lower-tail fraction (TRIM_DEL) is dropped; the
beneficial tail is kept (it carries the distance-to-optimum signal). See the
TRIM_DEL / TRIM_BEN config block.

Outputs (written next to the repo's other products):
    data/fgm_dfe_fit.json               {per_dfe (Ascensao), shared (all)} summaries
    data/fgm_dfe_asencao_params.txt     human-readable Ascensao parameter table
    figs_paper/fig_fgm_fit_shared.pdf   Couce shared-(n,sigma) fit (the only figure)

Run from anywhere:  python code_figs/figSX_experiment_bayes.py
"""
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import logsumexp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from cmn import cmn_fgm

DATA = os.path.join(REPO_DIR, "data")
FIGS = os.path.join(REPO_DIR, "figs_paper")
ASENCAO_DIR = os.path.join(DATA, "asencao_dfe_arrays")
COUCE_DIR = os.path.join(DATA, "alex_code")
JSON_PATH = os.path.join(DATA, "fgm_dfe_fit.json")
ASENCAO_TXT = os.path.join(DATA, "fgm_dfe_asencao_params.txt")

# ── priors ──────────────────────────────────────────────────────────────────
N_MIN, N_MAX = 1, 60          # n ~ U{N_MIN..N_MAX}
SIGMA_MAX = 10.0              # sigma ~ U[0, SIGMA_MAX]
R_MAX = 100.0                 # r ~ U[0, R_MAX]

# ── tail trimming (per data source) ───────────────────────────────────────────
# The FGM DFE has hard support [-e^{-r^2/2}, 1-e^{-r^2/2}], so the data's extreme
# effects pin r. The strongly-DELETERIOUS extremes are largely lethals / essential-
# gene knockouts / noisy large-magnitude estimates that FGM is not meant to model;
# dropping a small lower-tail fraction lets sigma match the bulk. The BENEFICIAL
# tail carries the distance-to-optimum signal (best beneficial effect <-> r via the
# 1-e^{-r^2/2} reach), so it is kept -- except a tiny upper-tail trim in Ascensao to
# remove implausible isolated beneficial outliers (s ~ 0.3-0.5) that otherwise make
# r run away once the deleterious counterweight is trimmed.
# Each entry is (frac_deleterious, frac_beneficial).
TRIM_COUCE = (0.02, 0.0)
TRIM_ASENCAO = (0.005, 0.002)

# ── inference grid resolution ─────────────────────────────────────────────────
NBINS = 250                  # multinomial bins over [min(data), max(data)]
NSIG1, NR1 = 40, 34          # stage-1 (coarse) sigma / r grid
NSIG2, NR2 = 64, 64          # stage-2 (refined) sigma / r grid
SIG1_LO, SIG1_HI = 1e-3, 5.0  # stage-1 sigma search range (log-spaced)
CHUNK = 4000                 # grid points evaluated per likelihood call

plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update({"axes.labelsize": 13, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 10})
_CMR = sns.color_palette("CMRmap", 5)
DATA_FILL = (0.5, 0.5, 0.5, 0.35)
MODEL_COLOR = _CMR[2]


# ══════════════════════════════════════════════════════════════════════════════
# Data loading -- one cleaned array of fitness effects per DFE
# ══════════════════════════════════════════════════════════════════════════════
def _trim(v, trim):
    """Drop ``trim[0]`` off the deleterious (lower) tail and ``trim[1]`` off the
    beneficial (upper) tail (asymmetric robustness to non-FGM outliers)."""
    frac_del, frac_ben = trim
    lo = np.quantile(v, frac_del) if frac_del > 0.0 else -np.inf
    hi = np.quantile(v, 1.0 - frac_ben) if frac_ben > 0.0 else np.inf
    return v[(v >= lo) & (v <= hi)]


def load_couce():
    """Couce et al. marginal DFEs for the three backgrounds (0K / 2K / 15K).

    fitted1 is the inferred selection coefficient per transposon-insertion site;
    keep abn > 1, drop NaN / failed-fit sentinels, de-duplicate identical fits.
    """
    files = [("0K", "Rfitted_fil.txt"),
             ("2K", "2Kfitted_fil.txt"),
             ("15K", "15Kfitted_fil.txt")]
    out = []
    for label, fname in files:
        t = pd.read_csv(os.path.join(COUCE_DIR, fname), sep="\t").dropna(subset=["fitted1"])
        t = t.drop_duplicates(subset=["fitted1"])
        t = t[t["abn"] > 1]
        v = t["fitted1"].to_numpy(float)
        v = v[v > -100.0]                     # drop -107 failed-fit sentinel
        out.append((label, _trim(v, TRIM_COUCE)))
    return out


def load_asencao():
    """Ascensao et al. DFEs: one array per background (L / R / S) per experiment."""
    out = []
    for d in sorted(os.listdir(ASENCAO_DIR)):
        sub = os.path.join(ASENCAO_DIR, d)
        if not os.path.isdir(sub):
            continue
        for arr in ("L", "R", "S"):
            p = os.path.join(sub, f"{arr}.npy")
            if not os.path.exists(p):
                continue
            v = np.load(p).astype(float)
            v = v[np.isfinite(v)]
            out.append((f"Asc {d} {arr}", _trim(v, TRIM_ASENCAO)))
    return out


def groups():
    """Groups that share (n, sigma) but have independent r.

    Couce: the LTEE Ara+2 time series 0K -> 2K -> 15K. Ascensao: R is the ancestor
    and L, S are two independent evolved offshoots, so each experiment gives two
    ancestor->evolved pairs, R->L and R->S, treated separately (ancestor first).
    """
    g = {"Couce": ["0K", "2K", "15K"]}
    for d in sorted(os.listdir(ASENCAO_DIR)):
        sub = os.path.join(ASENCAO_DIR, d)
        if not os.path.isdir(sub):
            continue
        has = {a: os.path.exists(os.path.join(sub, f"{a}.npy")) for a in ("L", "R", "S")}
        if has["R"]:
            for off in ("L", "S"):
                if has[off]:
                    g[f"Asc {d} R->{off}"] = [f"Asc {d} R", f"Asc {d} {off}"]
    return g


# ══════════════════════════════════════════════════════════════════════════════
# Posterior on a (n, sigma, r) grid
# ══════════════════════════════════════════════════════════════════════════════
def feasible_r(dmin, dmax, eps=1e-4):
    """[r_lo, r_hi] for which the data range fits inside the FGM support.

    Need 1 - e^{-r^2/2} >= dmax (beneficial reach) and -e^{-r^2/2} <= dmin
    (deleterious reach), intersected with the prior [0, R_MAX].
    """
    r_lo = np.sqrt(-2.0 * np.log(1.0 - dmax)) if dmax < 1.0 else 0.0
    r_lo = max(r_lo, 0.0)
    if dmin < 0.0 and -dmin < 1.0:
        r_hi = np.sqrt(-2.0 * np.log(-dmin))
    else:
        r_hi = R_MAX
    r_hi = min(r_hi, R_MAX)
    span = r_hi - r_lo
    return r_lo + eps * span, r_hi - eps * span


def _loglik_cube(counts, edges, n_vals, sig_vals, r_vals):
    """Log-likelihood on the full n x sigma x r grid (shape Nn, Nsig, Nr)."""
    N, S, R = np.meshgrid(n_vals.astype(float), sig_vals, r_vals, indexing="ij")
    flat = np.empty(N.size)
    nf, sf, rf = N.ravel(), S.ravel(), R.ravel()
    for i in range(0, nf.size, CHUNK):
        sl = slice(i, i + CHUNK)
        flat[sl] = cmn_fgm.fgm_bin_loglik(counts, edges, nf[sl], sf[sl], rf[sl])
    return flat.reshape(N.shape)


def _cell_widths(x):
    """Integration weight per grid node (np.gradient = central differences)."""
    x = np.asarray(x, float)
    if x.size == 1:
        return np.array([1.0])
    return np.gradient(x)


def _quantiles(vals, cellprob, qs):
    """Quantiles of a discretised distribution given per-cell probabilities."""
    order = np.argsort(vals)
    v = np.asarray(vals, float)[order]
    w = np.asarray(cellprob, float)[order]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    # midpoint convention so a single dominant cell returns its own value
    cdf_mid = cdf - 0.5 * w / w.sum()
    return np.interp(qs, cdf_mid, v)


def _summarise(ll, n_vals, sig_vals, r_vals):
    """Marginals, MAP, mean, median and 95% CI from a log-likelihood cube."""
    wsig, wr = _cell_widths(sig_vals), _cell_widths(r_vals)
    logmass = ll + np.log(wsig)[None, :, None] + np.log(wr)[None, None, :]
    mass = np.exp(logmass - np.nanmax(logmass))
    mass = np.where(np.isfinite(mass), mass, 0.0)
    total = mass.sum()

    Pn = mass.sum(axis=(1, 2)) / total
    psig = mass.sum(axis=(0, 2)) / total
    pr = mass.sum(axis=(0, 1)) / total

    i, j, k = np.unravel_index(np.argmax(ll), ll.shape)
    qs = [0.025, 0.5, 0.975]
    nq = _quantiles(n_vals, Pn, qs)
    sq = _quantiles(sig_vals, psig, qs)
    rq = _quantiles(r_vals, pr, qs)
    return {
        "map": {"n": int(n_vals[i]), "sigma": float(sig_vals[j]), "r": float(r_vals[k])},
        "mean": {"n": float(np.sum(n_vals * Pn)), "sigma": float(np.sum(sig_vals * psig)),
                 "r": float(np.sum(r_vals * pr))},
        "median": {"n": float(nq[1]), "sigma": float(sq[1]), "r": float(rq[1])},
        "ci95": {"n": [float(nq[0]), float(nq[2])], "sigma": [float(sq[0]), float(sq[2])],
                 "r": [float(rq[0]), float(rq[2])]},
        "P_n": {int(nv): float(p) for nv, p in zip(n_vals, Pn) if p > 1e-4},
        "loglik_max": float(np.max(ll)),
        "_Pn": Pn, "_psig": psig, "_pr": pr,
        "_n_vals": n_vals, "_sig_vals": sig_vals, "_r_vals": r_vals,
    }


def _bbox(ll, n_vals, sig_vals, r_vals, thresh=25.0):
    """Axis ranges of the joint high-likelihood region {ll > max - thresh}.

    Using the joint region (rather than per-axis marginal quantiles) is what
    keeps the diagonal n-sigma degeneracy ridge inside the refined grid.
    """
    m = ll > (ll.max() - thresh)
    ni = np.where(m.any(axis=(1, 2)))[0]
    si = np.where(m.any(axis=(0, 2)))[0]
    ri = np.where(m.any(axis=(0, 1)))[0]
    return (int(n_vals[ni.min()]), int(n_vals[ni.max()]),
            float(sig_vals[si.min()]), float(sig_vals[si.max()]),
            float(r_vals[ri.min()]), float(r_vals[ri.max()]))


def _prep(effects):
    """Histogram + feasible-r box for one DFE (the per-dataset likelihood inputs)."""
    dmin, dmax = float(effects.min()), float(effects.max())
    r_lo, r_hi = feasible_r(dmin, dmax)
    edges = np.linspace(dmin, dmax, NBINS + 1)
    counts, _ = np.histogram(effects, bins=edges)
    return {"effects": effects, "edges": edges, "counts": counts,
            "r_lo": r_lo, "r_hi": r_hi, "dmin": dmin, "dmax": dmax}


def infer(effects):
    """Two-stage grid posterior over (n, sigma, r) for one DFE."""
    p = _prep(effects)
    r_lo, r_hi = p["r_lo"], p["r_hi"]

    # stage 1 -- coarse, wide
    n1 = np.arange(N_MIN, N_MAX + 1)
    sig1 = np.geomspace(SIG1_LO, SIG1_HI, NSIG1)
    r1 = np.linspace(r_lo, r_hi, NR1)
    ll1 = _loglik_cube(p["counts"], p["edges"], n1, sig1, r1)

    # stage 2 -- refine over the joint high-posterior bounding box (padded)
    nlo, nhi, slo, shi, rlo2, rhi2 = _bbox(ll1, n1, sig1, r1)
    nlo, nhi = max(N_MIN, nlo - 1), min(N_MAX, nhi + 1)
    slo, shi = max(slo / 1.3, 1e-4), min(shi * 1.3, SIGMA_MAX)
    rpad = 0.05 * (rhi2 - rlo2) + 1e-6
    rlo2, rhi2 = max(rlo2 - rpad, r_lo), min(rhi2 + rpad, r_hi)

    n2 = np.arange(nlo, nhi + 1)
    sig2 = np.linspace(slo, shi, NSIG2)
    r2 = np.linspace(rlo2, rhi2, NR2)
    ll2 = _loglik_cube(p["counts"], p["edges"], n2, sig2, r2)
    res = _summarise(ll2, n2, sig2, r2)
    res["data"] = {"N": int(effects.size), "min": p["dmin"], "max": p["dmax"],
                   "mean": float(effects.mean()), "sd": float(effects.std())}
    res["feasible_r"] = [float(r_lo), float(r_hi)]
    res["_effects"] = effects
    return res


# ══════════════════════════════════════════════════════════════════════════════
# Shared-(n, sigma) inference across timepoints of one experiment (r free per tp)
#
# With (n, sigma) shared and r_k independent, the likelihood factorises:
#   P(n, sigma, r_1..r_K) = prod_k L_k(n, sigma, r_k).
# Integrating each r_k out gives the shared-parameter marginal
#   log P(n, sigma) = sum_k log integral L_k(n, sigma, r_k) dr_k,
# and each timepoint's r posterior is recovered as the (n, sigma)-averaged
# conditional. So we evaluate one log-likelihood cube per timepoint on a COMMON
# (n, sigma) grid (each with its own feasible r axis) and reduce over r.
# ══════════════════════════════════════════════════════════════════════════════
def _logZ_over_r(counts, edges, n_vals, sig_vals, r_vals):
    """log integral of the likelihood over r -> (Nn, Nsig); also returns the cube."""
    ll = _loglik_cube(counts, edges, n_vals, sig_vals, r_vals)
    logwr = np.log(_cell_widths(r_vals))
    return logsumexp(ll + logwr[None, None, :], axis=2), ll


def _shared_logpost(preps, n_vals, sig_vals, nr):
    """Joint shared-(n, sigma) log-posterior summed over timepoints."""
    logP = np.zeros((n_vals.size, sig_vals.size))
    cubes, rgrids, logZs = [], [], []
    for p in preps:
        r_vals = np.linspace(p["r_lo"], p["r_hi"], nr)
        logZ, ll = _logZ_over_r(p["counts"], p["edges"], n_vals, sig_vals, r_vals)
        logP = logP + logZ
        cubes.append(ll); rgrids.append(r_vals); logZs.append(logZ)
    return logP, cubes, rgrids, logZs


def _marginal_stats(vals, cellprob):
    """median + 95% CI + mean for a 1-D discretised marginal."""
    q = _quantiles(vals, cellprob, [0.025, 0.5, 0.975])
    return {"median": float(q[1]), "ci95": [float(q[0]), float(q[2])],
            "mean": float(np.sum(np.asarray(vals, float) * cellprob / cellprob.sum()))}


def infer_shared(group):
    """Joint fit of one experiment: shared (n, sigma), independent r per timepoint.

    ``group`` is a list of (label, effects). Returns shared n/sigma posteriors and
    a per-timepoint r posterior.
    """
    labels = [lab for lab, _ in group]
    preps = [_prep(eff) for _, eff in group]

    # stage 1 -- common coarse (n, sigma) grid
    n1 = np.arange(N_MIN, N_MAX + 1)
    sig1 = np.geomspace(SIG1_LO, SIG1_HI, NSIG1)
    logP1, *_ = _shared_logpost(preps, n1, sig1, NR1)

    # refine (n, sigma) over the joint high-posterior box
    m = logP1 > logP1.max() - 25.0
    ni = np.where(m.any(axis=1))[0]
    si = np.where(m.any(axis=0))[0]
    nlo, nhi = max(N_MIN, int(n1[ni.min()]) - 1), min(N_MAX, int(n1[ni.max()]) + 1)
    slo = max(float(sig1[si.min()]) / 1.3, 1e-4)
    shi = min(float(sig1[si.max()]) * 1.3, SIGMA_MAX)

    n2 = np.arange(nlo, nhi + 1)
    sig2 = np.linspace(slo, shi, NSIG2)
    logP, cubes, rgrids, logZs = _shared_logpost(preps, n2, sig2, NR2)

    # shared-(n, sigma) marginals
    wsig = _cell_widths(sig2)
    mass = np.exp(logP + np.log(wsig)[None, :] - logP.max())
    mass = np.where(np.isfinite(mass), mass, 0.0)
    Pn = mass.sum(axis=1) / mass.sum()
    psig = mass.sum(axis=0) / mass.sum()
    i, j = np.unravel_index(np.argmax(logP), logP.shape)

    out = {
        "timepoints": labels,
        "shared": {
            "map": {"n": int(n2[i]), "sigma": float(sig2[j])},
            "n": {**_marginal_stats(n2, Pn), "P_n": {int(nv): float(pp)
                  for nv, pp in zip(n2, Pn) if pp > 1e-4}},
            "sigma": _marginal_stats(sig2, psig),
        },
        "r": {}, "_r_post": {},
    }
    # per-timepoint r posterior: marginalise (n, sigma) out of the joint, holding
    # this timepoint's full r-dependence and the OTHER timepoints' evidence. We sum
    # the other logZ_j directly (never -inf minus -inf) to stay numerically clean.
    logwsig = np.log(wsig)[None, :]
    for k, (lab, ll, r_vals) in enumerate(zip(labels, cubes, rgrids)):
        logother = sum(lz for j, lz in enumerate(logZs) if j != k) \
            if len(logZs) > 1 else np.zeros_like(logP)
        logw = (logother + logwsig)[:, :, None]       # (Nn, Nsig, 1)
        logjoint = logw + ll + np.log(_cell_widths(r_vals))[None, None, :]
        logr = logsumexp(logjoint, axis=(0, 1))       # (Nr,)
        pr = np.exp(logr - logr.max())
        st = _marginal_stats(r_vals, pr)
        st["map"] = float(r_vals[np.argmax(logr)])
        out["r"][lab] = st
        out["_r_post"][lab] = (r_vals, pr / pr.sum())
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════
def _fmt(med, ci):
    return f"{med:.3g} [{ci[0]:.3g}, {ci[1]:.3g}]"


def plot_shared(sr, data_map, path):
    """Shared-(n, sigma) fit for one experiment: one panel per timepoint, each the
    data DFE with the best-fit FGM model at the shared (n, sigma) and that
    timepoint's r."""
    n, s = sr["shared"]["map"]["n"], sr["shared"]["map"]["sigma"]
    n_ci, s_ci = sr["shared"]["n"]["ci95"], sr["shared"]["sigma"]["ci95"]
    labs = sr["timepoints"]
    fig, axes = plt.subplots(1, len(labs), figsize=(3.9 * len(labs), 3.3))
    axes = np.atleast_1d(axes)
    for ax, lab in zip(axes, labs):
        eff = data_map[lab]
        r, r_ci = sr["r"][lab]["map"], sr["r"][lab]["ci95"]
        ax.hist(eff, bins=70, density=True, color=DATA_FILL, edgecolor="none",
                label="data")
        dlo, dhi = cmn_fgm.fgm_support(r)
        xs = np.linspace(max(dlo, eff.min()), min(dhi, eff.max()), 500)
        ax.plot(xs, cmn_fgm.fgm_dfe_pdf(xs, n, s, r), color=MODEL_COLOR, lw=2.0,
                label="FGM fit")
        ax.set_title(rf"{lab}  ($r = {r:.2f}$ [{r_ci[0]:.2f}, {r_ci[1]:.2f}])",
                     fontsize=12)
        ax.set_xlabel(r"Fitness effect $\Delta$")
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=10, loc="upper right")
    fig.suptitle(rf"$n = {n}$ [{round(n_ci[0])}, {round(n_ci[1])}],"
                 rf"  $\sigma = {s:.4f}$ [{s_ci[0]:.4f}, {s_ci[1]:.4f}]", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Text export of the Ascensao fits
# ══════════════════════════════════════════════════════════════════════════════
def _ci(c, p=3):
    return f"[{c[0]:.{p}f}, {c[1]:.{p}f}]"


def write_asencao_txt(per_dfe, shared, path):
    """Human-readable table of the FGM parameters fit to the Ascensao DFEs.

    ``per_dfe``/``shared`` are the JSON-public summary dicts. Both MAP (= MLE under
    the flat priors) and the marginal posterior median [95% CI] are reported.
    """
    asc = {k: v for k, v in per_dfe.items() if k.startswith("Asc")}
    ash = {k: v for k, v in shared.items() if k.startswith("Asc")}
    out = [
        "Fisher's Geometric Model DFE -- fitted parameters, Ascensao et al. data",
        "Generated by code_figs/figSX_experiment_bayes.py (see code_figs/fgm_dfe_inference.md)",
        "",
        "Model:  Delta = exp(-|r + delta|^2 / 2) - exp(-r^2 / 2),  delta ~ N(0, sigma^2 I_n)",
        "        n = phenotypic dimension (integer), sigma = mutation-step s.d., r = distance to optimum",
        "Priors: n ~ U{1..60}, sigma ~ U[0,10], r ~ U[0,100]  (flat -> MAP = maximum likelihood)",
        f"Tail trimming: deleterious {TRIM_ASENCAO[0]:.1%} of lower tail, "
        f"beneficial {TRIM_ASENCAO[1]:.1%} of upper tail (R = ancestor; L, S = evolved offshoots)",
        "Each cell:  MAP  median[95% credible interval]",
        "",
        "=" * 110,
        "Independent per-DFE fits  (every DFE has its own n, sigma, r)",
        "=" * 110,
        f"{'dataset':<13}{'N':>6}   {'n':<23}{'sigma':<32}{'r':<24}",
    ]
    for k in sorted(asc):
        e = asc[k]; mp, md, ci = e["map"], e["median"], e["ci95"]
        ncol = f"{mp['n']:<3} {md['n']:.1f} {_ci(ci['n'], 1)}"
        scol = f"{mp['sigma']:.4f}  {md['sigma']:.4f} {_ci(ci['sigma'], 4)}"
        rcol = f"{mp['r']:.3f}  {md['r']:.3f} {_ci(ci['r'])}"
        out.append(f"{k:<13}{e['data']['N']:>6}   {ncol:<23}{scol:<32}{rcol:<24}")

    out += ["", "=" * 110,
            "Shared-(n, sigma) fits  (ancestor R and one offshoot share n & sigma; r free per timepoint)",
            "=" * 110]
    for g in sorted(ash):
        sr = ash[g]; sh = sr["shared"]
        out.append("")
        out.append(f"{g}")
        out.append(f"    shared n     : MAP {sh['map']['n']}   median {sh['n']['median']:.2f} "
                   f"{_ci(sh['n']['ci95'], 2)}")
        out.append(f"    shared sigma : MAP {sh['map']['sigma']:.4f}   median {sh['sigma']['median']:.4f} "
                   f"{_ci(sh['sigma']['ci95'], 4)}")
        for lab in sr["timepoints"]:
            rr = sr["r"][lab]
            tag = "ancestor" if lab.endswith(" R") else "evolved "
            out.append(f"    r [{lab:<11}] ({tag}): MAP {rr['map']:.3f}   "
                       f"median {rr['median']:.3f} {_ci(rr['ci95'])}")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(FIGS, exist_ok=True)
    data_map = dict(load_couce() + load_asencao())

    # ── per-DFE fits (independent n, sigma, r) -- Ascensao only ───────────────
    # Couce is fit shared-only (per request), so it is skipped here.
    print(f"{'dataset':<16}{'N':>7}  {'n [95% CI]':<18}{'sigma [95% CI]':<24}{'r [95% CI]':<22}")
    print("-" * 90)
    per_dfe = {}
    for name, eff in load_asencao():
        res = infer(eff)
        md, ci = res["median"], res["ci95"]
        n_str = f"{int(round(md['n']))} [{ci['n'][0]:.1f}, {ci['n'][1]:.1f}]"
        print(f"{name:<16}{eff.size:>7}  {n_str:<18}"
              f"{_fmt(md['sigma'], ci['sigma']):<24}{_fmt(md['r'], ci['r']):<22}")
        per_dfe[name] = {k: v for k, v in res.items() if not k.startswith("_")}

    # ── shared-(n, sigma) fits per experiment (r free per timepoint) ──────────
    # Couce LTEE series + Ascensao ancestor->offshoot pairs.
    print("\nShared (n, sigma) per experiment; r free per timepoint")
    print("-" * 90)
    shared, shared_pub = {}, {}
    for g, labs in groups().items():
        sr = infer_shared([(l, data_map[l]) for l in labs])
        shared[g] = sr
        sh = sr["shared"]
        rstr = "  ".join(f"{l.split()[-1]}:{sr['r'][l]['median']:.3g}" for l in labs)
        print(f"{g:<14} n={sh['map']['n']} ({_fmt(sh['n']['median'], sh['n']['ci95'])})"
              f"  sigma={_fmt(sh['sigma']['median'], sh['sigma']['ci95'])}   r[ {rstr} ]")
        shared_pub[g] = {k: v for k, v in sr.items() if not k.startswith("_")}

    # posterior summaries -> JSON (Ascensao per-DFE + all shared fits)
    with open(JSON_PATH, "w") as f:
        json.dump({"per_dfe": per_dfe, "shared": shared_pub}, f, indent=2)
    print(f"\nSaved {JSON_PATH}")

    # Ascensao parameters -> human-readable text file
    write_asencao_txt(per_dfe, shared_pub, ASENCAO_TXT)

    # the one figure: Couce shared-(n, sigma) fit
    plot_shared(shared["Couce"], data_map, os.path.join(FIGS, "figSX_experiment_bayes.pdf"))


if __name__ == "__main__":
    main()
