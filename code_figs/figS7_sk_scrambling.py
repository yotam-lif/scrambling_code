r"""Scrambling diagnostics for the pure p-spin (SK) adaptive walk (figS7).

A 4x3 figure:
  A  Per-spin summaries vs system size N (p=2): the final-to-initial Hamming
     distance d_H(t=0)/N and the walk length (number of accepted flips)/N.
  B  Hamming distance to the final configuration d_H(t) along the walk, one
     curve per N (p=2), with the linear fit used in panel A's slope.
  C  Far-field angular scrambling: log of the in-shell direction autocorrelation
     <u_hat(t_ref).u_hat(t)>, re-anchored at theta_0 = pi/4 (N=1000, p=2).
  D  Near-field angular scrambling, re-anchored at theta_0 = pi/12 (N=1000, p=2).
  E  Subset autocorrelation of the distribution of fitness effects (DFE), highest
     available N for p=2 (N=2000), re-anchored at 4 angles to the final config.
  F  Same subset DFE autocorrelation for p=3 (highest available N=500).
  G,H,I  Subset->total DFE earth-mover's-distance decay (figS3-style), N=1500 p=2,
     one re-anchoring angle per panel (EMD_ANCHOR_FRACS x theta_0).
  J,K,L  Same subset->total DFE EMD decay for p=3 (N=500).

Panels E/F track, from each of 4 anchors along the walk, the Pearson correlation between
the DFE at the anchor and the DFE later, restricted to the spins still in their anchor
state (flipped an even number of times since the anchor). The anchors are the steps where
the angle to the final config, theta(t) = arccos(sigma(t).sigma_f / N), reaches theta_0,
3 theta_0/4, theta_0/2, theta_0/4 (theta_0 = the start). A dashed -2(p-1)t/N reference line
is overlaid. Expensive quantities (the eigendecompositions and the multi-GB p=3 walk file)
are cached to data/.
"""

import os
import pickle
import sys
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import wasserstein_distance

# Repo root on path so we can drive the native p-spin DFE updates.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn import cmn_pspin

# ───────────────────────────────────── Style ─────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# ───────────────────────────────────── Configuration ─────────────────────────────────────
# p=2 walks used for the Hamming summaries (panels A, B).
FILE_PATHS = {
    300: "../data/PSPIN/N300_P2_pure_repeats10.pkl",
    400: "../data/PSPIN/N400_P2_pure_repeats10.pkl",
    500: "../data/PSPIN/N500_P2_pure_repeats10.pkl",
    1000: "../data/PSPIN/N1000_P2_pure_repeats10.pkl",
    1500: "../data/PSPIN/N1500_P2_pure_repeats10.pkl",
    2000: "../data/PSPIN/N2000_P2_pure_repeats10.pkl",
}
# N=1000 p=2 walk used for the angular scrambling panels (C, D).
GEOM_FILE = "../data/PSPIN/N1000_P2_pure_repeats10.pkl"
# Highest available N per interaction order, used for the Pearson DFE panels (E, F).
PEARSON_FILES = {
    2: "../data/PSPIN/N2000_P2_pure_repeats10.pkl",
    3: "../data/PSPIN/N500_P3_pure_repeats10.pkl",
}
CACHE_PATH = "../data/figS7_sk_scrambling_cache.pkl"

colors = sns.color_palette("CMRmap", 6)
HAMMING_COLOR_MAP = {300: colors[0], 400: colors[1], 500: colors[2],
                     1000: colors[3], 1500: colors[4], 2000: colors[5]}

# Angular-scrambling shells (re-anchor radius), far field then near field.
SHELL_THETAS = [np.pi / 4.0, np.pi / 12.0]
SHELL_TITLES = [r'$\theta_0 = \pi/4$', r'$\theta_0 = \pi/12$']
SHELL_COLOR = "m"
SHELL_TRUNC_THRESHOLD = -1.0

# Pearson DFE display.
PEARSON_FLOOR = 1e-3      # clip rho before taking the log
PEARSON_MIN_REPS = 3      # plot only steps still reached by at least this many walks
# Re-anchor the subset autocorrelation at 4 points along the walk, defined by where the angle
# to the final config, theta(t) = arccos(sigma(t).sigma_f / N), reaches these fractions of theta_0.
ANCHOR_FRACS = [1.0, 0.75, 0.50, 0.25]
ANCHOR_COLORS = [plt.get_cmap("viridis")(x) for x in (0.12, 0.40, 0.62, 0.84)]
ANCHOR_COLOR_BY_FRAC = dict(zip(ANCHOR_FRACS, ANCHOR_COLORS))
# Common steps-since-anchor window shown for every anchor curve, per interaction order.
PEARSON_WINDOW = {2: 20, 3: 20}

# Subset->total DFE EMD panels: figS3-style, one anchor angle per panel. Row G,H,I uses p=2
# (N=1500); row J,K,L is the same measure for p=3 (N=500).
# Tune EMD_ANCHOR_FRACS to move the anchor angle theta_i = frac * theta_0; labels and the
# cache key derive from it automatically, so changing it here is all that's needed.
EMD_FILE = "../data/PSPIN/N1500_P2_pure_repeats10.pkl"      # row G,H,I (p=2)
EMD_FILE_P3 = "../data/PSPIN/N500_P3_pure_repeats10.pkl"    # row J,K,L (p=3)
EMD_ANCHOR_FRACS = [1.0, 0.75, 0.25]
EMD_WINDOW = 50


def apply_axis_style(ax, label):
    ax.text(
        -0.08, 1.04, label,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    ax.tick_params(width=1.4, length=5, which="major")
    ax.tick_params(width=1.2, length=3, which="minor")
    ax.grid(False)


# ───────────────────────────────────── Data Loading ─────────────────────────────────────

def _reconstruct_J_matrix(J_data, N):
    """Reconstruct full (dense) pairwise J matrix from either dense or sparse format."""
    if isinstance(J_data, dict):
        # PSPIN sparse format (single p=2 sector for these files).
        J_full = np.zeros((N, N), dtype=float)
        for sector in J_data.get('sectors', []):
            spin_indices = sector['spin_indices']
            couplings = sector['couplings']
            for (i, j), coupling in zip(zip(spin_indices[0], spin_indices[1]), couplings):
                J_full[i, j] = coupling
                J_full[j, i] = coupling
        return J_full
    else:
        return np.asarray(J_data, dtype=float)


def load_sk_runs(file_path, n_repeats=None):
    """Load the first n_repeats SK trajectories as (sigma_initial, dense J, flip_seq)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Ensure data is present.")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    if n_repeats is None:
        n_repeats = len(data)
    elif n_repeats > len(data):
        raise ValueError(f"Requested n_repeats={n_repeats}, but file has only {len(data)} runs.")

    runs = []
    for k in range(n_repeats):
        entry = data[k]
        sigma_initial = np.asarray(entry.get("init_sigma", entry.get("init_alpha")), dtype=int)
        N = sigma_initial.shape[0]
        J = _reconstruct_J_matrix(entry["J"], N)
        flip_seq = np.asarray(entry["flip_seq"], dtype=int)
        runs.append((sigma_initial, J, flip_seq))

    return runs


# ───────────────────────────────────── Core Geometry ─────────────────────────────────────

def _theta_and_uhat(r_t, rhat_f, eps=1e-10):
    """Decompose r_t = ||r_t|| (cos(theta) rhat_f + sin(theta) uhat), uhat orthogonal to rhat_f."""
    nr = np.linalg.norm(r_t)
    if nr < eps:
        return np.nan, None

    cos_theta = np.dot(r_t, rhat_f) / nr
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    u_vec = r_t - np.dot(r_t, rhat_f) * rhat_f
    nu = np.linalg.norm(u_vec)
    if nu < eps:
        return theta, None
    return theta, u_vec / nu


def compute_hamming_distance_series(sigma_initial, flip_seq):
    """Hamming distance from the current state to the final state at each step. Shape (T+1,)."""
    T = len(flip_seq)
    sigma_t = sigma_initial.astype(int).copy()

    # Compute final state.
    sigma_f = sigma_t.copy()
    for t in range(T):
        i = int(flip_seq[t])
        sigma_f[i] *= -1

    hamming = np.zeros(T + 1, dtype=int)
    hamming[0] = np.sum(sigma_t != sigma_f)

    sigma_t = sigma_initial.astype(int).copy()
    for t in range(T):
        i = int(flip_seq[t])
        sigma_t[i] *= -1
        hamming[t + 1] = np.sum(sigma_t != sigma_f)

    return hamming.astype(float)


def analyze_run_time_series(sigma_initial, J, flip_seq, ref_percents, target_radii=None, eps=1e-10):
    """For one run, compute the shell-anchored direction autocorrelation.

    shell_corr_m(dt) = <uhat(t_ref_m), uhat(t_ref_m + dt)> where t_ref_m is the first step
    at which the angular radius R drops to target_radii[m]. (theta(t) and percent-aligned
    correlations are also returned but unused by the current figure.)
    """
    T = len(flip_seq)
    steps = np.arange(T + 1)
    if T == 0:
        frac = np.array([0.0], dtype=float)
    else:
        frac = steps / T * 100.0

    _, eigvecs = np.linalg.eigh(J)

    sigma_t = sigma_initial.astype(float).copy()
    r0 = eigvecs.T @ sigma_t
    n_dim = r0.shape[0]
    sphere_radius = np.sqrt(n_dim)
    target_radii = np.asarray([] if target_radii is None else target_radii, dtype=float)

    # Terminal point r_f in one pass.
    r_t = r0.copy()
    sigma_tmp = sigma_t.copy()
    for t in range(T):
        i = int(flip_seq[t])
        r_t += -2.0 * sigma_tmp[i] * eigvecs[i, :]
        sigma_tmp[i] *= -1.0
    rf = r_t

    nrf = np.linalg.norm(rf)
    if nrf < eps:
        theta = np.full(T + 1, np.nan, dtype=float)
        corr = np.full((len(ref_percents), T + 1), np.nan, dtype=float)
        shell_corr = np.full((len(target_radii), T + 1), np.nan, dtype=float)
        shell_ref_radii = np.full(len(target_radii), np.nan, dtype=float)
        return frac, theta, corr, shell_corr, shell_ref_radii

    rhat_f = rf / nrf

    theta = np.full(T + 1, np.nan, dtype=float)
    corr = np.full((len(ref_percents), T + 1), np.nan, dtype=float)
    shell_corr = np.full((len(target_radii), T + 1), np.nan, dtype=float)

    ref_percents = np.asarray(ref_percents, dtype=float)
    ref_set = np.zeros(len(ref_percents), dtype=bool)
    ref_vecs = np.zeros((len(ref_percents), r0.shape[0]), dtype=float)
    shell_ref_set = np.zeros(len(target_radii), dtype=bool)
    shell_ref_vecs = np.zeros((len(target_radii), r0.shape[0]), dtype=float)
    shell_ref_steps = np.full(len(target_radii), -1, dtype=int)
    shell_ref_radii = np.full(len(target_radii), np.nan, dtype=float)

    r_t = r0.copy()
    sigma_tmp = sigma_t.copy()

    theta[0], uhat = _theta_and_uhat(r_t, rhat_f, eps=eps)
    radius = sphere_radius * np.sin(theta[0]) if np.isfinite(theta[0]) else np.nan

    if uhat is not None:
        for j, rp in enumerate(ref_percents):
            if (not ref_set[j]) and (frac[0] >= rp):
                ref_set[j] = True
                ref_vecs[j, :] = uhat
                corr[j, 0] = 1.0

        for m, target_radius in enumerate(target_radii):
            if (not shell_ref_set[m]) and np.isfinite(radius) and (radius <= target_radius):
                shell_ref_set[m] = True
                shell_ref_steps[m] = 0
                shell_ref_vecs[m, :] = uhat
                shell_ref_radii[m] = radius
                shell_corr[m, 0] = 1.0

    for t in range(T):
        i = int(flip_seq[t])
        r_t += -2.0 * sigma_tmp[i] * eigvecs[i, :]
        sigma_tmp[i] *= -1.0

        idx = t + 1
        theta[idx], uhat = _theta_and_uhat(r_t, rhat_f, eps=eps)
        radius = sphere_radius * np.sin(theta[idx]) if np.isfinite(theta[idx]) else np.nan

        if uhat is not None:
            for j, rp in enumerate(ref_percents):
                if (not ref_set[j]) and (frac[idx] >= rp):
                    ref_set[j] = True
                    ref_vecs[j, :] = uhat
                    corr[j, idx] = 1.0

            if np.any(ref_set):
                corr[ref_set, idx] = ref_vecs[ref_set, :] @ uhat

            for m, target_radius in enumerate(target_radii):
                if (not shell_ref_set[m]) and np.isfinite(radius) and (radius <= target_radius):
                    shell_ref_set[m] = True
                    shell_ref_steps[m] = idx
                    shell_ref_vecs[m, :] = uhat
                    shell_ref_radii[m] = radius

                if shell_ref_set[m]:
                    dt = idx - shell_ref_steps[m]
                    shell_corr[m, dt] = np.dot(shell_ref_vecs[m, :], uhat)

    return frac, theta, corr, shell_corr, shell_ref_radii


def _finite_mean_std(values):
    """Columnwise mean/std ignoring NaNs, without all-NaN runtime warnings."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2D array")

    mask = np.isfinite(values)
    counts = mask.sum(axis=0)
    mean = np.full(values.shape[1], np.nan, dtype=float)
    std = np.full(values.shape[1], np.nan, dtype=float)

    valid = counts > 0
    if np.any(valid):
        safe_vals = np.where(mask[:, valid], values[:, valid], 0.0)
        mean_valid = safe_vals.sum(axis=0) / counts[valid]
        diff = np.where(mask[:, valid], values[:, valid] - mean_valid, 0.0)
        std_valid = np.sqrt((diff ** 2).sum(axis=0) / counts[valid])
        mean[valid] = mean_valid
        std[valid] = std_valid

    return mean, std, counts


def summarize_log_traces(values, tiny=1e-12):
    mean, std, counts = _finite_mean_std(values)
    log_mean = np.log(np.clip(mean, tiny, None))
    log_lower = np.log(np.clip(mean - std, tiny, None))
    log_upper = np.log(np.clip(mean + std, tiny, None))
    return log_mean, log_lower, log_upper, mean, std, counts


def _pad_traces(traces):
    if not traces:
        return np.empty((0, 0), dtype=float)

    max_len = max(len(trace) for trace in traces)
    padded = np.full((len(traces), max_len), np.nan, dtype=float)
    for idx, trace in enumerate(traces):
        padded[idx, :len(trace)] = trace
    return padded


def _truncate_log_trace(log_mean, *arrays, threshold=-1.1):
    """Truncate arrays at the first index where log_mean <= threshold, inclusive."""
    log_mean = np.asarray(log_mean, dtype=float)
    stop = len(log_mean)

    finite = np.flatnonzero(np.isfinite(log_mean))
    if finite.size:
        crossed = np.flatnonzero(log_mean[finite] <= threshold)
        if crossed.size:
            stop = finite[crossed[0]] + 1
        else:
            stop = finite[-1] + 1
    else:
        stop = 0

    truncated = [log_mean[:stop]]
    truncated.extend(np.asarray(arr)[:stop] for arr in arrays)
    return tuple(truncated)


def _subsample(last, n_markers=8):
    return np.arange(0, last + 1, max(1, (last + 1) // n_markers))


def _theta_frac_label(frac):
    """LaTeX label for a fraction of theta_0 (1 -> theta_0, 0.75 -> 3 theta_0/4, 0.5 -> theta_0/2)."""
    fr = Fraction(frac).limit_denominator(100)
    if fr == 1:
        return r"$\theta_0$"
    num, den = fr.numerator, fr.denominator
    return rf"$\theta_0/{den}$" if num == 1 else rf"${num}\theta_0/{den}$"


def _fracs_tag(fracs):
    """Stable cache-key fragment encoding a list of anchor fractions."""
    return "-".join(f"{f:g}" for f in fracs)


# ───────────────────────────────────── Computations ─────────────────────────────────────

def compute_hamming_stats(file_paths, n_repeats):
    """Per-N Hamming summaries (panel A) and full d_H(t) traces with a linear fit (panel B).

    Returns {'summary': {N: {...}}, 'datasets': [ {N, hamming_mean, ...}, ... ]}.
    """
    summary = {}
    datasets = []

    for N_val in sorted(file_paths.keys()):
        runs = load_sk_runs(file_paths[N_val], n_repeats=n_repeats)

        hamming_series = []
        dh0_norm = []
        steps_norm = []
        fit_slopes = []
        fit_intercepts = []

        for sigma0, _J, flip_seq in runs:
            hamming = compute_hamming_distance_series(sigma0, flip_seq)
            hamming_series.append(hamming)

            dh0_norm.append(hamming[0] / N_val)
            steps_norm.append(len(flip_seq) / N_val)

            # Linear fit over the first 70% of the walk (slope shown in panel B).
            T = len(hamming) - 1
            idx_70 = int(np.ceil(0.7 * T))
            t_fit = np.arange(idx_70 + 1)
            dh_fit = hamming[:idx_70 + 1]
            valid = np.isfinite(dh_fit)
            if valid.sum() > 1:
                coeffs = np.polyfit(t_fit[valid], dh_fit[valid], 1)
                fit_slopes.append(coeffs[0])
                fit_intercepts.append(coeffs[1])

        hamming_padded = _pad_traces(hamming_series)
        hamming_mean, hamming_std, _ = _finite_mean_std(hamming_padded)

        slope = float(np.mean(fit_slopes)) if fit_slopes else 0.0
        intercept = float(np.mean(fit_intercepts)) if fit_intercepts else 0.0

        dh0_norm = np.asarray(dh0_norm)
        steps_norm = np.asarray(steps_norm)
        summary[N_val] = {
            "dh0_mean": float(dh0_norm.mean()),
            "dh0_std": float(dh0_norm.std()),
            "steps_mean": float(steps_norm.mean()),
            "steps_std": float(steps_norm.std()),
        }

        datasets.append({
            "N": N_val,
            "hamming_mean": hamming_mean,
            "hamming_std": hamming_std,
            "hamming_fit_params": (slope, intercept),
            "color": HAMMING_COLOR_MAP[N_val],
        })

    return {"summary": summary, "datasets": datasets}


def compute_shell_panels(file_path, n_repeats):
    """Angular-scrambling panels (C, D) for one p=2 walk file."""
    runs = load_sk_runs(file_path, n_repeats=n_repeats)
    if not runs:
        raise RuntimeError(f"No SK runs were loaded from {file_path}.")

    n_dim = runs[0][0].size
    target_radii = np.sqrt(n_dim) * np.sin(np.asarray(SHELL_THETAS, dtype=float))
    ref_percents = [0, 20, 40, 60, 80]  # required by analyze_run_time_series's signature

    shell_traces = [[] for _ in target_radii]
    shell_ref_radii = [[] for _ in target_radii]

    for sigma0, J, flip_seq in runs:
        _, _, _, shell_corr, ref_radii = analyze_run_time_series(
            sigma0, J, flip_seq, ref_percents=ref_percents, target_radii=target_radii,
        )
        for m in range(len(target_radii)):
            valid_idx = np.flatnonzero(np.isfinite(shell_corr[m]))
            if valid_idx.size:
                shell_traces[m].append(shell_corr[m, :valid_idx[-1] + 1].copy())
                shell_ref_radii[m].append(ref_radii[m])

    panels = []
    for m in range(len(target_radii)):
        shell_stack = _pad_traces(shell_traces[m])
        if shell_stack.size == 0:
            print(f"Warning: no shell-aligned traces for shell index {m}; skipping.")
            continue

        log_mean, log_lower, log_upper, _, _, _ = summarize_log_traces(shell_stack)
        log_mean, log_lower, log_upper, time = _truncate_log_trace(
            log_mean, log_lower, log_upper, np.arange(shell_stack.shape[1]),
            threshold=SHELL_TRUNC_THRESHOLD,
        )

        mean_ref_radius = float(np.nanmean(shell_ref_radii[m]))
        if not np.isfinite(mean_ref_radius) or mean_ref_radius <= 0.0:
            print(f"Warning: invalid reference radius for shell index {m}; skipping.")
            continue

        panels.append({
            "title": SHELL_TITLES[m],
            "time": time,
            "log_mean": log_mean,
            "log_lower": log_lower,
            "log_upper": log_upper,
            "tau_theory": mean_ref_radius ** 2 / 2.0,
            "color": SHELL_COLOR,
        })

    return panels


def _replay_walk(entry):
    """Replay one stored walk: return (sig_hist, dfe_hist, theta, theta0).

    sig_hist[t], dfe_hist[t] are the spin configuration and full DFE at step t (DFE via the
    native incremental p-spin updates); theta(t) = arccos(sigma(t).sigma_f / N) is the angle to
    the final config, and theta0 = theta(0).
    """
    sigma0 = np.asarray(entry.get("init_sigma", entry.get("init_alpha")), dtype=np.int8)
    J = entry["J"]
    flip_seq = np.asarray(entry["flip_seq"], dtype=int)
    N = sigma0.shape[0]
    T = len(flip_seq)

    state = cmn_pspin._initialize_relaxation_state(sigma0, J)
    sig_hist = np.empty((T + 1, N), dtype=np.int8)
    dfe_hist = np.empty((T + 1, N), dtype=np.float32)
    sig_hist[0] = state["sigma"]
    dfe_hist[0] = state["dfe"]
    for j, site in enumerate(flip_seq, start=1):
        cmn_pspin._apply_flip(state, J, int(site))
        sig_hist[j] = state["sigma"]
        dfe_hist[j] = state["dfe"]

    # Angle to the final config (overlap angle; int32 avoids int8 overflow in the dot).
    sig_f = sig_hist[-1].astype(np.int32)
    theta = np.arccos(np.clip(sig_hist.astype(np.int32) @ sig_f / N, -1.0, 1.0))
    return sig_hist, dfe_hist, theta, float(theta[0])


def _subset_emd_from_anchor(dfe_hist, k0):
    """Normalized subset->total DFE EMD trace from anchor step k0 (figS3-style).

    M = spins beneficial (DFE > 0) at the anchor, with fixed indices tracked forward.
    seg[dt] = EMD(DFE_{k0+dt}[M], DFE_{k0+dt}) / EMD_anchor, so seg[0] = 1. Returns all-NaN if
    the anchor has no beneficial spins or a degenerate anchor EMD.
    """
    T = dfe_hist.shape[0] - 1
    mask = dfe_hist[k0] > 0.0
    n_steps = T - k0 + 1
    if mask.sum() < 1:
        return np.full(n_steps, np.nan)

    seg = np.empty(n_steps)
    for dt, t in enumerate(range(k0, T + 1)):
        x = dfe_hist[t].astype(float)
        seg[dt] = wasserstein_distance(x[mask], x)
    if not np.isfinite(seg[0]) or seg[0] <= 0.0:
        return np.full(n_steps, np.nan)
    return seg / seg[0]


def _subset_autocorr_from_anchor(sig_hist, dfe_hist, k0):
    """Subset DFE autocorrelation starting from anchor step k0.

    seg[dt] = corr(s_{k0}[subset], s_{k0+dt}[subset]) over the spins still in their anchor state
    (equal to sigma(k0), i.e. flipped an even number of times since k0). seg[0] = 1.
    """
    T = sig_hist.shape[0] - 1
    s_anchor = dfe_hist[k0].astype(float)
    sig_anchor = sig_hist[k0]

    seg = np.full(T - k0 + 1, np.nan)
    seg[0] = 1.0
    for dt, t in enumerate(range(k0 + 1, T + 1), start=1):
        subset = sig_hist[t] == sig_anchor
        if subset.sum() >= 2:
            a = s_anchor[subset]
            b = dfe_hist[t][subset].astype(float)
            if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                seg[dt] = float(np.corrcoef(a, b)[0, 1])
    return seg


def compute_pearson_dfe(file_path, n_repeats):
    """Subset DFE autocorrelation re-anchored at 4 angles along the walk (panels E, F).

    For each walk we replay the stored flip sequence, recording the full distribution of fitness
    effects (DFE) and the spin configuration at every step (the DFE comes from the native
    incremental p-spin updates). The angle to the final configuration is
    theta(t) = arccos(sigma(t) . sigma_f / N); we re-anchor the autocorrelation at the first step
    where theta drops to each fraction in ANCHOR_FRACS of theta_0 (fraction 1.0 == the start).

    From each anchor we track only spins still in their anchor state (flipped an even number of
    times since the anchor) and correlate their anchor DFE against their current DFE:
    rho(dt) = corr(s_anchor[subset], s_{anchor+dt}[subset]), rho(0) = 1. Returns one aggregated
    curve (columnwise mean/std of log rho vs steps-since-anchor, plus replicate counts) per anchor.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Ensure data is present.")

    size_gb = os.path.getsize(file_path) / 1024 ** 3
    print(f"  loading {os.path.basename(file_path)} ({size_gb:.2f} GB) for Pearson DFE, "
          f"up to {n_repeats} reps ...", flush=True)
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    n = min(n_repeats, len(data))
    anchor_traces = [[] for _ in ANCHOR_FRACS]
    theta0_vals = []

    for k in range(n):
        sig_hist, dfe_hist, theta, theta0 = _replay_walk(data[k])
        theta0_vals.append(theta0)
        for a, frac in enumerate(ANCHOR_FRACS):
            hits = np.flatnonzero(theta <= frac * theta0)
            k0 = int(hits[0]) if hits.size else 0
            anchor_traces[a].append(_subset_autocorr_from_anchor(sig_hist, dfe_hist, k0))
    del data  # free the (possibly multi-GB) walk file before returning

    anchors = []
    for a, frac in enumerate(ANCHOR_FRACS):
        arr = _pad_traces(anchor_traces[a])
        with np.errstate(divide="ignore", invalid="ignore"):
            logv = np.log(np.clip(arr, PEARSON_FLOOR, None))
        logv[~np.isfinite(arr) | (arr <= 0)] = np.nan
        mean_logv, std_logv, counts = _finite_mean_std(logv)
        anchors.append({
            "frac": frac,
            "label": _theta_frac_label(frac),
            "mean_logv": mean_logv,
            "std_logv": std_logv,
            "counts": counts,
            "n_reps": len(anchor_traces[a]),
        })

    return {"anchors": anchors, "theta0_mean": float(np.mean(theta0_vals))}


def compute_emd_subset(file_path, n_repeats):
    """Subset->total DFE EMD decay re-anchored at the EMD_ANCHOR_FRACS angles (panels G, H, I).

    figS3-style: at each anchor we fix M = spins beneficial at the anchor and track the normalized
    earth-mover's distance EMD(DFE_t[M], DFE_t) / EMD(anchor) between that subset's DFE and the
    total DFE as the background scrambles. Returns one aggregated curve (mean/std of
    log[EMD(t)/EMD(0)] vs steps-since-anchor, plus replicate counts) per anchor.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Ensure data is present.")

    size_gb = os.path.getsize(file_path) / 1024 ** 3
    print(f"  loading {os.path.basename(file_path)} ({size_gb:.2f} GB) for subset EMD, "
          f"up to {n_repeats} reps ...", flush=True)
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    n = min(n_repeats, len(data))
    anchor_traces = [[] for _ in EMD_ANCHOR_FRACS]
    theta0_vals = []

    for k in range(n):
        _, dfe_hist, theta, theta0 = _replay_walk(data[k])
        theta0_vals.append(theta0)
        for a, frac in enumerate(EMD_ANCHOR_FRACS):
            hits = np.flatnonzero(theta <= frac * theta0)
            k0 = int(hits[0]) if hits.size else 0
            anchor_traces[a].append(_subset_emd_from_anchor(dfe_hist, k0))
    del data  # free the walk file before returning

    anchors = []
    for a, frac in enumerate(EMD_ANCHOR_FRACS):
        arr = _pad_traces(anchor_traces[a])
        with np.errstate(divide="ignore", invalid="ignore"):
            logv = np.log(np.clip(arr, PEARSON_FLOOR, None))
        logv[~np.isfinite(arr) | (arr <= 0)] = np.nan
        mean_logv, std_logv, counts = _finite_mean_std(logv)
        anchors.append({
            "frac": frac,
            "label": _theta_frac_label(frac),
            "mean_logv": mean_logv,
            "std_logv": std_logv,
            "counts": counts,
            "n_reps": len(anchor_traces[a]),
        })

    return {"anchors": anchors, "theta0_mean": float(np.mean(theta0_vals))}


# ───────────────────────────────────── Cache ─────────────────────────────────────

def _load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def _cached(cache, key, n_repeats, compute_fn):
    """Return cache[key]['value'], recomputing if absent or generated with a different n_repeats."""
    entry = cache.get(key)
    if entry is not None and entry.get("n_repeats") == n_repeats:
        return cache[key]["value"], False
    value = compute_fn()
    cache[key] = {"n_repeats": n_repeats, "value": value}
    return value, True


# ───────────────────────────────────── Plotting ─────────────────────────────────────

def plot_panel_hamming_summary(ax, summary):
    """Panel A: d_H(0)/N and walk-length/N vs system size N."""
    Ns = sorted(summary.keys())
    dh0_mean = [summary[N]["dh0_mean"] for N in Ns]
    dh0_std = [summary[N]["dh0_std"] for N in Ns]
    steps_mean = [summary[N]["steps_mean"] for N in Ns]
    steps_std = [summary[N]["steps_std"] for N in Ns]

    # Limiting (largest-N) values, written into the legend.
    Nmax = Ns[-1]
    dh0_lim = summary[Nmax]["dh0_mean"]
    steps_lim = summary[Nmax]["steps_mean"]

    ax.errorbar(Ns, dh0_mean, yerr=dh0_std, fmt="o-", color=colors[1], lw=2.0,
                markersize=7, capsize=4, label=rf"$d_H(0)/N \to {dh0_lim:.2f}$")
    ax.errorbar(Ns, steps_mean, yerr=steps_std, fmt="s--", color=colors[4], lw=2.0,
                markersize=7, capsize=4, label=rf"$t_{{\mathrm{{max}}}}/N \to {steps_lim:.2f}$")

    ax.set_xlabel(r"$N$")
    ax.legend(frameon=False, loc="best")


def plot_panel_hamming_series(ax, datasets):
    """Panel B: Hamming distance to the final state d_H(t), one curve per N, with linear fits."""
    for ds in datasets:
        N = ds["N"]
        hamming_mean = ds["hamming_mean"]
        hamming_std = ds["hamming_std"]
        slope, intercept = ds["hamming_fit_params"]
        color = ds["color"]

        T = len(hamming_mean)
        t_steps = np.arange(T)
        ax.plot(t_steps, hamming_mean, lw=2.2, color=color, label=f"N={N}")
        ax.fill_between(
            t_steps,
            np.clip(hamming_mean - hamming_std, 0.0, N),
            np.clip(hamming_mean + hamming_std, 0.0, N),
            color=color, alpha=0.20, linewidth=0,
        )
        t_fit = np.linspace(0, T - 1, 100)
        ax.plot(t_fit, intercept + slope * t_fit, color=color, lw=1.4, ls=":")

    # Limiting (largest-N) slope of the linear fit.
    lim_slope = max(datasets, key=lambda d: d["N"])["hamming_fit_params"][0]
    ax.text(0.95, 0.50, rf"slope $\to {lim_slope:.2f}$", transform=ax.transAxes,
            ha="right", va="center", fontsize=13)

    ax.set_xlabel("Time (steps)")
    ax.set_ylabel(r"$d_H(t)$")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=11)


def plot_panel_shell(ax, panel):
    """Panels C/D: log in-shell direction autocorrelation with the linear angular timescale."""
    ax.plot(panel["time"], panel["log_mean"], lw=2.5, color=panel["color"], label="Simulation")
    ax.fill_between(panel["time"], panel["log_lower"], panel["log_upper"],
                    color=panel["color"], alpha=0.30, linewidth=0)
    ax.plot(panel["time"], -panel["time"] / panel["tau_theory"],
            color="black", lw=2.0, ls=":", label=r"Theory (***)")
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel(r'$\log (\hat{\boldsymbol{u}}(t_\mathrm{ref}) \cdot \hat{\boldsymbol{u}}(t))$')
    ax.set_title(panel["title"])
    ax.legend(frameon=False, loc="lower left")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_panel_pearson(ax, result, p, N):
    """Panels E/F: log subset DFE autocorrelation vs steps-since-anchor, one curve per anchor.

    Every anchor is shown over the same window (PEARSON_WINDOW[p] steps); a curve ends earlier
    only if the walk runs out of steps after that anchor (the near-optimum anchors).
    """
    window = PEARSON_WINDOW[p]
    for a, anchor in enumerate(result["anchors"]):
        mean_logv = anchor["mean_logv"]
        std_logv = anchor["std_logv"]
        counts = anchor["counts"]
        color = ANCHOR_COLORS[a]

        # Common window, but never past the last step still reached by >= PEARSON_MIN_REPS walks.
        enough = np.flatnonzero(counts >= PEARSON_MIN_REPS)
        last = int(enough[-1]) if enough.size else 1
        last = min(last, window)

        t = np.arange(last + 1)
        ax.plot(t, mean_logv[:last + 1], color=color, lw=2.0, label=anchor["label"])
        mk = _subsample(last, n_markers=6)
        ax.errorbar(t[mk], mean_logv[mk], yerr=std_logv[mk], fmt="o", color=color,
                    markersize=3.5, capsize=2.5, elinewidth=0.9, alpha=0.85)

    # Reference line f(t) = -2(p-1)/N * t from the anchor.
    t_line = np.linspace(0.0, float(window), 20)
    ax.plot(t_line, -2.0 * (p - 1) / N * t_line, color="black", lw=2.0, ls="--",
            label=r"$-2(p-1)\,t/N$")

    ax.set_xlim(0, window)
    ax.set_xlabel("Steps since anchor")
    ax.set_ylabel(r"$\log\,\rho$")
    ax.set_title(rf"$p={p}$, $N={N}$, $\theta_0={np.degrees(result['theta0_mean']):.0f}^\circ$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=11)


def plot_panel_emd(ax, anchor, N, theta0_mean, p):
    """EMD panels: log normalized subset->total DFE EMD vs steps-since-anchor (one anchor)."""
    mean_logv = anchor["mean_logv"]
    std_logv = anchor["std_logv"]
    counts = anchor["counts"]
    color = ANCHOR_COLOR_BY_FRAC.get(anchor["frac"], "steelblue")

    enough = np.flatnonzero(counts >= PEARSON_MIN_REPS)
    last = int(enough[-1]) if enough.size else 1
    last = min(last, EMD_WINDOW)

    t = np.arange(last + 1)
    y = mean_logv[:last + 1]
    ax.plot(t, y, color=color, lw=2.0, label="Simulation")
    mk = _subsample(last, n_markers=6)
    ax.errorbar(t[mk], mean_logv[mk], yerr=std_logv[mk], fmt="o", color=color,
                markersize=3.5, capsize=2.5, elinewidth=0.9, alpha=0.85)

    # Linear fit through the origin, log[EMD(t)/EMD(0)] = -(m/N) t (the curve is 0 at t=0).
    finite = np.isfinite(y)
    denom = float(np.sum(t[finite] ** 2))
    m = -N * float(np.sum(t[finite] * y[finite])) / denom if denom > 0 else np.nan
    ax.plot(t, -(m / N) * t, color="black", lw=1.8, ls="--",
            label=rf"$-(m/N)\,t,\ m={m:.1f}$")

    # Angular-scrambling timescale at the anchor angle theta_i = frac * theta_0:
    # tau = N sin^2(theta_i) / 2  (the perpendicular radius is sqrt(N) sin(theta_i), tau = R^2/2).
    theta_i = anchor["frac"] * theta0_mean
    tau = N * np.sin(theta_i) ** 2 / 2.0
    ax.plot(t, -t / tau, color="crimson", lw=2.0, ls=":",
            label=r"$-t/\tau,\ \tau=N\sin^2\!\theta_i/2$")

    ax.set_xlim(0, last)
    ax.set_xlabel("Steps since anchor")
    ax.set_ylabel(r"$\log[\mathrm{EMD}(t)/\mathrm{EMD}(0)]$")
    ax.set_title(rf"{anchor['label']},  $p={p}$,  $N={N}$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.legend(frameon=False, loc="lower left")


def make_combined_figure(hamming, shell_panels, pearson, emd_rows, out_path):
    """Assemble the 4x3 figure (A,B,C / D,E,F / G,H,I / J,K,L)."""
    fig, axes = plt.subplots(4, 3, figsize=(18.0, 20.0))
    fig.subplots_adjust(wspace=0.34, hspace=0.45)

    labels = [["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"], ["J", "K", "L"]]
    for r in range(4):
        for c in range(3):
            apply_axis_style(axes[r, c], labels[r][c])

    # Panel A — Hamming summaries vs N.
    plot_panel_hamming_summary(axes[0, 0], hamming["summary"])

    # Panel B — Hamming distance to final state along the walk.
    plot_panel_hamming_series(axes[0, 1], hamming["datasets"])

    # Panels C, D — angular scrambling shells (far field then near field).
    shell_axes = [axes[0, 2], axes[1, 0]]
    for ax, panel in zip(shell_axes, shell_panels):
        plot_panel_shell(ax, panel)
    for ax in shell_axes[len(shell_panels):]:
        ax.axis("off")

    # Panels E, F — Pearson DFE autocorrelation, p=2 then p=3.
    plot_panel_pearson(axes[1, 1], pearson[2]["curve"], 2, pearson[2]["N"])
    plot_panel_pearson(axes[1, 2], pearson[3]["curve"], 3, pearson[3]["N"])

    # Rows G,H,I (p=2) and J,K,L (p=3) — subset->total DFE EMD decay, one anchor angle each.
    for row_idx, emd in zip((2, 3), emd_rows):
        for ax, anchor in zip(axes[row_idx], emd["anchors"]):
            plot_panel_emd(ax, anchor, emd["N"], emd["theta0_mean"], emd["p"])
        for ax in axes[row_idx][len(emd["anchors"]):]:
            ax.axis("off")

    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {out_path}")


# ───────────────────────────────────── Main ─────────────────────────────────────

def _file_N(file_path):
    """Extract the integer N from a '..N1234_P..' PSPIN filename."""
    base = os.path.basename(file_path)
    return int(base.split("_")[0][1:])


def main(n_repeats=10):
    out_dir = "../figs_paper"
    os.makedirs(out_dir, exist_ok=True)

    cache = _load_cache()
    dirty = False

    hamming, d = _cached(cache, "hamming", n_repeats,
                         lambda: compute_hamming_stats(FILE_PATHS, n_repeats))
    dirty |= d

    shell_panels, d = _cached(cache, "shells_N1000", n_repeats,
                              lambda: compute_shell_panels(GEOM_FILE, n_repeats))
    dirty |= d

    pearson = {}
    pearson_tag = _fracs_tag(ANCHOR_FRACS)
    for p in (2, 3):
        curve, d = _cached(cache, f"pearson_anchors_p{p}_{pearson_tag}", n_repeats,
                           lambda p=p: compute_pearson_dfe(PEARSON_FILES[p], n_repeats))
        dirty |= d
        pearson[p] = {"curve": curve, "N": _file_N(PEARSON_FILES[p])}

    emd_tag = _fracs_tag(EMD_ANCHOR_FRACS)
    emd_rows = []
    for p, emd_file in ((2, EMD_FILE), (3, EMD_FILE_P3)):
        res, d = _cached(cache, f"emd_subset_p{p}_{emd_tag}", n_repeats,
                         lambda emd_file=emd_file: compute_emd_subset(emd_file, n_repeats))
        dirty |= d
        emd_rows.append({**res, "N": _file_N(emd_file), "p": p})

    if dirty:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)

    out_path = os.path.join(out_dir, "figS7_sk_scrambling.pdf")
    make_combined_figure(hamming, shell_panels, pearson, emd_rows, out_path)


if __name__ == "__main__":
    main(n_repeats=10)
