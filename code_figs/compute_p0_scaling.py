"""Precompute P(0) of the peak DFE vs system size N, for all three models.

P(0) is the density of the final (peak) DFE at the boundary Delta=0 -- the
"pseudogap floor". It is estimated with a binning-free, theta-free
order-statistic edge estimator, validated against the known SK result:

    edge floor (per repeat) = k / (n_mut * u_(k))           [k = K_EDGE]

where u_(k) is the k-th smallest |Delta| among the n_mut deleterious effects of
that repeat's local optimum, and P(0) is the mean over the 10 repeats. Fixed
small k keeps the probe pinned to the near-zero floor as N grows; the resulting
exponent is k-stable for p-spin/FGM (a genuine power law) and reproduces the
Sherrington-Kirkpatrick value P(0) ~ N^{-1/2} for the pure 2-spin model.

Units match each model's fig4 main panel:
    FGM    : raw fitness effect;  "N" axis is m (number of sampled mutations)
    NK     : Delta * N            (same rescaling as panel D)
    p-spin : raw compute_dfe;     pure model, param = interaction order p

Run this once to (re)generate data/p0_scaling.json, which code_figs/fig4 reads.
The p-spin p=3 datasets are multi-GB, so they are loaded here and discarded,
keeping the figure script light.
"""
import json
import os
import pickle
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
for p in (SCRIPT_DIR, REPO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from cmn import cmn, cmn_pspin

DATA_DIR = os.path.join(REPO_DIR, "data")
OUT_PATH = os.path.join(DATA_DIR, "p0_scaling.json")

REPS = 10
N_VALUES = [100, 200, 300, 400, 500]
K_EDGE = 10                # order-statistic depth of the floor probe
N_BOOT = 400               # bootstrap resamples (over repeats) for exponent error
RNG = np.random.default_rng(0)

FGM_NS = [4, 8, 16, 32]
NK_KS = [4, 8, 16, 32]
PSPIN_PS = [2, 3]          # p=1 (additive) has no N-sweep -> excluded


# ── per-repeat deleterious effects |Delta| at the final (peak) DFE ─────────────
def fgm_rep_effects(n, N):
    path = os.path.join(DATA_DIR, "FGM", f"fgm_rps{REPS}_n{n}_N{N}_sig0.05.pkl")
    with open(path, "rb") as f:
        d = pickle.load(f)
    out = []
    for r in d:
        a = np.asarray(r["dfes"][-1], dtype=float)
        out.append(-a[a < 0])
    return out


def nk_rep_effects(K, N):
    path = os.path.join(DATA_DIR, "NK", f"N_{N}_K_{K}_repeats_{REPS}.pkl")
    with open(path, "rb") as f:
        d = pickle.load(f)
    out = []
    for r in d:
        a = np.asarray(r["dfes"][-1], dtype=float) * N     # panel-D units
        out.append(-a[a < 0])
    return out


def pspin_rep_effects(P, N):
    # the existing N=400 pure file is named differently from the N-sweep files
    name = (f"N400_P{P}_pure_repeats{REPS}.pkl" if N == 400
            else f"N{N}_P{P}_pure_repeats{REPS}.pkl")
    path = os.path.join(DATA_DIR, "PSPIN", name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        d = pickle.load(f)
    out = []
    for e in d:
        sigma = cmn.compute_sigma_from_hist(e["init_sigma"], e["flip_seq"])
        a = np.asarray(cmn_pspin.compute_dfe(sigma, e["J"]), dtype=float)
        out.append(-a[a < 0])
    return out


# ── estimator ─────────────────────────────────────────────────────────────────
def edge_floor_per_repeat(rep_effects, k=K_EDGE):
    """k / (n_mut * u_(k)) for every repeat (the near-zero density floor)."""
    vals = []
    for u in rep_effects:
        u = np.sort(u[(u > 0) & np.isfinite(u)])
        if u.size > k:
            vals.append(k / (u.size * u[k - 1]))
    return np.asarray(vals, dtype=float)


def powerlaw_fit(Ns, P0):
    Ns = np.asarray(Ns, float)
    P0 = np.asarray(P0, float)
    ok = np.isfinite(P0) & (P0 > 0)
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan
    a, b = np.polyfit(np.log(Ns[ok]), np.log(P0[ok]), 1)
    ly = np.log(P0[ok])
    pred = a * np.log(Ns[ok]) + b
    r2 = 1.0 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return a, np.exp(b), r2


def build_series(loader, params, param_name):
    series = {}
    for prm in params:
        Ns, P0, per_rep = [], [], {}
        for N in N_VALUES:
            reps = loader(prm, N)
            if reps is None:
                continue
            ef = edge_floor_per_repeat(reps)
            if ef.size == 0:
                continue
            Ns.append(N)
            P0.append(float(np.mean(ef)))
            per_rep[N] = ef
        alpha, coef, r2 = powerlaw_fit(Ns, P0)

        # bootstrap the exponent by resampling repeats within each N
        boot = []
        for _ in range(N_BOOT):
            bp = []
            for N in Ns:
                ef = per_rep[N]
                bp.append(float(np.mean(ef[RNG.integers(0, ef.size, ef.size)])))
            a, _, _ = powerlaw_fit(Ns, bp)
            if np.isfinite(a):
                boot.append(a)
        alpha_err = float(np.std(boot)) if boot else np.nan

        series[str(prm)] = {
            "N": Ns,
            "P0": P0,
            "alpha": float(alpha),
            "alpha_err": alpha_err,
            "coef": float(coef),
            "r2": float(r2),
        }
        print(f"  {param_name}={prm}: alpha={alpha:+.3f} ± {alpha_err:.3f}  "
              f"R2={r2:.3f}  N={Ns}")
    return series


def main():
    out = {
        "estimator": {"name": "edge_floor_order_statistic", "k": K_EDGE,
                      "reps": REPS, "definition": "P0 = mean_reps[ k / (n_mut * u_(k)) ]"},
    }
    print("FGM (param=n, N-axis = m = number of mutations; raw units)")
    out["FGM"] = {"param_name": "n", "N_label": "m", "units": "raw",
                  "series": build_series(fgm_rep_effects, FGM_NS, "n")}
    print("NK (param=K; Delta*N units)")
    out["NK"] = {"param_name": "K", "N_label": "N", "units": "Delta*N",
                 "series": build_series(nk_rep_effects, NK_KS, "K")}
    print("p-spin (param=p, pure; raw units)  [loads multi-GB p=3 files]")
    out["PSPIN"] = {"param_name": "p", "N_label": "N", "units": "raw",
                    "series": build_series(pspin_rep_effects, PSPIN_PS, "p")}

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
