"""Re-anchored *unconstrained* FGM adaptive walks: the dataset behind the
combined-timescale panel of figS7_fgm_scrambling.py, and behind its
radius/orientation panels.

The constrained sweeps of gen_dat_fgm_constrained.py each freeze one half of the
FGM geometry so the other half can be timed on its own. Here nothing is frozen:
we reanalyse the ordinary SSWM walks in data/sim/FGM/fgm_rps1000_n*_sig0.05.pkl, in
which the walk descends *and* rotates at once, and ask whether the combined
timescale

    tau_s^-1 = tau_as^-1 + tau_rs^-1,
    tau_as = 2 R~(0)^2 / (n - 1),   tau_rs = R~(0) / sqrt(pi/2),

is the one the DFE autocorrelation actually runs on.

Re-anchoring
------------
Every production walk starts far-field at R~(0) = sqrt(n)/sigma = 20 sqrt(n) and
descends to its optimum, so one walk is not an experiment at a single R~(0). To
probe a chosen starting radius we re-anchor each walk at the first step where its
radius drops to that value, freeze the mutation pool's effects there, and follow
the Pearson autocorrelation from that step. Sweeping the anchor over a grid turns
one set of walks into the whole R~(0) sweep that the constrained runs had to
simulate cell by cell.

What is recorded
----------------
cells  -- {(R~(0), n): {"time_points", "pearson"}}, deliberately the same shape
          the constrained sweeps use, so the figure reads it with the same code.
walks  -- {n: {"radius", "cosine"}}, the per-replicate radius R~(t) and the
          orientation overlap rhat(0).rhat(t) of every walk. These carry the
          CV^2, radial-law and angular-law panels, which until now re-read the
          multi-GB trajectory files twice between them.

Usage (from this directory):
    python gen_dat_fgm_reanchored.py
    python gen_dat_fgm_reanchored.py --n-values 4,8 --reps 200
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np

SIGMA = 0.05
N_VALUES = [4, 8, 16, 32]
# The anchor grid, in units of sigma. A walk can only be anchored at a radius it
# actually passes through, so each n keeps the anchors at or below its own start
# radius 20 sqrt(n): n = 4 stops at 32, n = 32 reaches 96. The grid straddles the
# crossover radius R~_c = (n-1)/sqrt(2 pi) for every n (R~_c is 1.2 at n = 4 and
# 12.4 at n = 32), which is the point -- neither pure law works on both sides of
# it, and the combined one is supposed to.
ANCHORS = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
MAX_WINDOW = 400          # cap on the recorded window, in steps
WINDOW_TAUS = 5.0         # record this many tau_s past the anchor
WINDOW_PAD = 10           # ... plus a few steps, so the short near-field windows
                          # still have a tail to read the floor off


def tau_s(r0_tilde, n):
    """The combined scrambling time: angular and radial *rates* add."""
    return 1.0 / ((n - 1.0) / (2.0 * r0_tilde ** 2)
                  + np.sqrt(np.pi / 2.0) / r0_tilde)


def window_steps(r0_tilde, n):
    """Recorded window for one cell, long enough that the walk has arrived: the
    figure divides out the correlation left at the end, and that is only a floor
    if the run actually reached it."""
    return int(min(np.ceil(WINDOW_TAUS * tau_s(r0_tilde, n)) + WINDOW_PAD,
                   MAX_WINDOW))


def resolve_fgm_dir():
    here = Path(__file__).resolve().parent
    candidates = [here.parent / "FGM", here.parent / "fgm"]
    return next((p for p in candidates if p.exists()), candidates[0])


def normalized_rows(dfes):
    """The (T, m) DFE history, mean-centred and scaled to unit norm per row, so a
    dot product of two rows is their Pearson correlation.

    The stored effects are the absolute differences w(r+delta) - w(r), whose
    global e^{-||r||^2} prefactor changes by ~e^n over a descent. That prefactor
    is irrelevant here: Pearson is invariant under a positive rescaling of either
    argument, so correlating the raw effects and correlating the selection
    coefficients s = w(r+delta)/w(r) - 1 give exactly the same number. (The EMD
    measurements elsewhere do have to convert; this one does not.)

    Rows with no spread -- a walk sitting at its optimum can produce one -- are
    left as zeros, so every correlation against them comes out 0 and is dropped
    downstream rather than raising.
    """
    x = np.asarray(dfes, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    good = norms[:, 0] > 1e-300
    x[good] /= norms[good]
    x[~good] = 0.0
    return x


def anchored_pearson(xn, radius, r0_tilde, n_steps):
    """rho(t) of one walk re-anchored at radius r0_tilde, over n_steps + 1 steps.

    Returns None if the walk never reaches that radius. Past the end of the walk
    the trace is *held* at its last value rather than padded with NaN: an SSWM
    walk that has run out of beneficial mutations sits at its optimum and freezes
    there, so holding it is exact, and it is what makes the end-of-window
    correlation a property of where the walk stopped rather than of when the
    recording did.
    """
    hits = np.nonzero(radius <= r0_tilde)[0]
    if hits.size == 0:
        return None
    k0 = int(hits[0])
    k_end = min(len(xn), k0 + n_steps + 1)
    seg = xn[k0:k_end] @ xn[k0]
    if seg.size == 0 or not np.isfinite(seg[0]) or seg[0] <= 0.0:
        return None
    out = np.empty(n_steps + 1)
    out[:seg.size] = seg
    out[seg.size:] = seg[-1]              # freeze at the peak floor
    return out


def process_n(n, reps, anchors, sigma, data_dir):
    """One dimensionality: the re-anchored rho traces for each admissible anchor,
    plus the radius and orientation history of every walk."""
    path = data_dir / f"fgm_rps{1000}_n{n}_sig{sigma:g}.pkl"
    if not path.exists():
        print(f"  n={n}: {path.name} not found -- skipping.")
        return None, None

    print(f"  n={n}: loading {path.name} "
          f"({path.stat().st_size / 1024 ** 3:.1f} GB) ...", flush=True)
    try:
        with open(path, "rb") as handle:
            data = pickle.load(handle)
    except MemoryError:
        print(f"  n={n}: MemoryError -- skipping.")
        return None, None

    use = [a for a in anchors if a <= 20.0 * np.sqrt(n)]
    windows = {a: window_steps(a, n) for a in use}
    rho_by_anchor = {a: [] for a in use}
    radius_traces, cosine_traces = [], []

    for i in range(min(reps, len(data))):
        rep = data[i]
        data[i] = None                    # release each walk as it is consumed
        if not isinstance(rep, dict) or "traj" not in rep or "dfes" not in rep:
            continue
        traj = np.asarray(rep["traj"], dtype=np.float64)
        dfes = rep["dfes"]
        if traj.ndim != 2 or len(dfes) == 0:
            continue

        norms = np.linalg.norm(traj, axis=1)
        radius = norms / sigma
        cosine = np.clip((traj @ (traj[0] / norms[0])) / norms, -1.0, 1.0)
        radius_traces.append(radius.astype(np.float32))
        cosine_traces.append(cosine.astype(np.float32))

        xn = normalized_rows(dfes)
        rad_dfe = radius[:len(xn)]
        for a in use:
            trace = anchored_pearson(xn, rad_dfe, a, windows[a])
            if trace is not None:
                rho_by_anchor[a].append(trace.astype(np.float32))

        if (i + 1) % 200 == 0:
            print(f"    {i + 1} replicates", flush=True)

    del data

    cells = {}
    for a in use:
        traces = rho_by_anchor[a]
        if not traces:
            continue
        cells[(float(a), int(n))] = {
            "time_points": np.arange(windows[a] + 1),
            "pearson": np.vstack(traces),
        }
        print(f"    R~(0)={a:g}: {len(traces)} walks, window {windows[a]} steps")

    walks = {
        "radius": pad_stack(radius_traces),
        "cosine": pad_stack(cosine_traces),
    }
    return cells, walks


def pad_stack(traces):
    """Pad a ragged set of per-walk traces into one (reps, max_len) array, NaN
    past each walk's own end."""
    if not traces:
        return np.empty((0, 0), dtype=np.float32)
    max_len = max(len(t) for t in traces)
    out = np.full((len(traces), max_len), np.nan, dtype=np.float32)
    for i, t in enumerate(traces):
        out[i, :len(t)] = t
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-anchor the unconstrained FGM adaptive walks onto a grid "
                    "of starting radii R~(0) and record the DFE autocorrelation "
                    "from each anchor, together with every walk's radius and "
                    "orientation history. Read by figS7_fgm_scrambling.py.")
    parser.add_argument("--n-values", default=",".join(str(n) for n in N_VALUES))
    parser.add_argument("--reps", type=int, default=1000,
                        help="Max replicates per n.")
    parser.add_argument("--anchors", default=",".join(f"{a:g}" for a in ANCHORS),
                        help="Comma-separated anchor radii R~(0) = R(0)/sigma.")
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    n_values = [int(x) for x in args.n_values.split(",") if x.strip()]
    anchors = [float(x) for x in args.anchors.split(",") if x.strip()]
    data_dir = resolve_fgm_dir()
    out_dir = Path(args.output_dir) if args.output_dir else data_dir

    print(f"--- re-anchored sweep: sigma={args.sigma}, reps={args.reps} ---")
    print(f"anchors R~(0): {anchors}")
    print(f"n            : {n_values}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fgm_reanchored_sweep_rps{args.reps}_sig{args.sigma:g}.pkl"

    # Written again after every n rather than once at the end: the n = 32 file is
    # 27 GB against 51 GB of RAM, and if that pass is killed outright there is no
    # exception to catch -- the smaller n should still be on disk when it is.
    cache = {
        "mode": "reanchored",
        "sigma": args.sigma,
        "reps": args.reps,
        "n_values": [],
        "r0_tilde_values": anchors,
        "cells": {},
        "walks": {},
    }
    for n in n_values:
        cells, walks = process_n(n, args.reps, anchors, args.sigma, data_dir)
        if cells:
            cache["cells"].update(cells)
        if walks is not None:
            cache["walks"][int(n)] = walks
        if cells or walks is not None:
            cache["n_values"].append(int(n))
        with open(out_path, "wb") as handle:
            pickle.dump(cache, handle)
        print(f"  cache written to {out_path} "
              f"({out_path.stat().st_size / 1024 ** 2:.1f} MB)", flush=True)

    print(f"Data saved to {out_path}")
