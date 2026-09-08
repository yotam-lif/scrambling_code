import argparse
import os
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ----------------------------------------------------------------
# 1. VISUAL STYLE CONFIGURATION
# ----------------------------------------------------------------
plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update({
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

CMR_COLORS = sns.color_palette("CMRmap", 4)
SQRT_HALF_PI = np.sqrt(np.pi / 2.0)   # |d<R~>/dt| of the far-field SSWM radial law

# The two constrained-walk sweeps behind panels C and D, generated once by
# data/gen_data/gen_dat_fgm_constrained.py. This script only reads them.
RADIAL_SWEEP_FILE = "fgm_wedge_sweep_rps500_sig0.05_phi0.6.pkl"
ANGULAR_SWEEP_FILE = "fgm_shell_sweep_rps1000_sig0.05_eps0.05.pkl"
ANGULAR_R0_TICKS = [8, 12, 16, 24, 32, 48]   # the R~(0) of the angular sweep

N_PATTERN = re.compile(r"_n(\d+)_")
# Finite-size sweep files (e.g. fgm_rps10_n4_N500_sig0.05.pkl) carry only ~10
# replicates each and belong to the floor study; the high-statistics production
# runs (fgm_rps1000_n*_sig*.pkl) have ~1000 replicates. Exclude the _N###_ sweep
# files so the CV^2 panel uses the smooth 1000-rep data, not the noisy 10-rep one.
NSWEEP_PATTERN = re.compile(r"_N\d+_")
# The constrained-walk sweeps (wedge = radial, shell = angular) also live in
# data/FGM; they hold summary arrays rather than trajectories, so keep them out
# of the trajectory globs.
CONSTRAINED_PATTERN = re.compile(r"wedge|shell")
SIG_PATTERN = re.compile(r"sig([0-9]*\.?[0-9]+)")
N_PERCENT_POINTS = 100


def apply_axis_style(ax, label):
    """Bold panel letter plus the spine/tick geometry shared with the other
    figures: only the left and bottom spines, both pushed outward."""
    ax.text(
        -0.08, 1.04, label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["left"].set_position(("outward", 10))
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6, which="major")
    ax.tick_params(width=1.5, length=3, which="minor")
    ax.grid(False)


def set_ylim_clipped(ax, ymin, ymax, pad=0.01, step=0.2, pad_top=0.0):
    """Give the curves a little room below ymin without drawing frame there: the
    view extends to ymin - pad, but the left spine and its ticks still stop at ymin."""
    ax.set_ylim(ymin - pad, ymax + pad_top)
    ax.set_yticks(np.arange(ymin, ymax + 1e-9, step))
    ax.spines["left"].set_bounds(ymin, ymax)


def pearson_theory_radial(t, r0_tilde, n):
    r"""Far-field prediction for the DFE autocorrelation of a purely radial descent.

    With the background pinned to the fixed axis, r = R rhat0, a mutation delta has
    Malthusian effect m(R) = -||delta||^2 / 2 - R delta_par. Across the pool the two
    terms are uncorrelated, with variances n sigma^4 / 2 and R^2 sigma^2, so

        rho(0, t) = (n/2 + R~0 R~t) / sqrt( (n/2 + R~0^2) (n/2 + R~t^2) ),

    evaluated on the SSWM radial law R~(t) = R~0 - t sqrt(pi/2). Shrinking the radius
    only rescales the delta_par term, so radial motion alone cannot scramble the DFE
    completely: rho bottoms out at sqrt(n/2) / sqrt(n/2 + R~0^2) at the optimum.
    """
    t = np.asarray(t, dtype=float)
    r_t = np.clip(r0_tilde - SQRT_HALF_PI * t, 0.0, None)
    half_n = 0.5 * n
    return (half_n + r0_tilde * r_t) / np.sqrt((half_n + r0_tilde ** 2)
                                               * (half_n + r_t ** 2))

# ----------------------------------------------------------------
# 1b. FGM RADIUS-CV^2 PANEL (loaded from saved trajectories)
# ----------------------------------------------------------------
def find_fgm_files(data_root):
    if not data_root.exists():
        return []
    files = [p for p in data_root.glob("*.pkl")
             if "fgm" in p.name.lower()
             and not NSWEEP_PATTERN.search(p.name)
             and not CONSTRAINED_PATTERN.search(p.name)]

    def extract_n(path):
        match = N_PATTERN.search(path.name)
        return int(match.group(1)) if match else 10 ** 9

    return sorted(files, key=lambda p: (extract_n(p), p.name))


def extract_n_from_name(path):
    match = N_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Could not parse n from filename: {path.name}")
    return int(match.group(1))


def extract_sigma_from_name(path, default=0.05):
    match = SIG_PATTERN.search(path.name)
    return float(match.group(1)) if match else default


def resolve_fgm_data_dir():
    script_dir = Path(__file__).resolve().parent
    data_candidates = [script_dir.parent / "data" / "fgm", script_dir.parent / "data" / "FGM"]
    return next((p for p in data_candidates if p.exists()), data_candidates[-1])


def sample_radius_vs_percent(traj, n_points=N_PERCENT_POINTS):
    arr = np.asarray(traj, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.full(n_points, np.nan)

    radii = np.linalg.norm(arr, axis=1)
    total_steps = len(radii) - 1
    percents = np.arange(1, n_points + 1, dtype=float)

    if total_steps <= 0:
        return np.full(n_points, radii[0], dtype=float)

    targets = (percents / 100.0) * total_steps
    idx = np.rint(targets).astype(int)
    idx = np.clip(idx, 0, total_steps)
    return radii[idx]


def load_fgm_cv2_metrics():
    data_dir = resolve_fgm_data_dir()

    coarse_by_n = {}
    for path in find_fgm_files(data_dir):
        n = extract_n_from_name(path)
        with path.open("rb") as handle:
            repeats = pickle.load(handle)

        percent_radii = []
        for rep in repeats:
            if not isinstance(rep, dict):
                continue
            traj = rep.get("traj")
            if traj is None or len(traj) == 0:
                continue
            sampled = sample_radius_vs_percent(traj, n_points=N_PERCENT_POINTS)
            if np.all(np.isfinite(sampled)):
                percent_radii.append(sampled)

        coarse_by_n[n] = (
            np.vstack(percent_radii)
            if len(percent_radii) > 0
            else np.empty((0, N_PERCENT_POINTS), dtype=float)
        )

    return dict(sorted(coarse_by_n.items()))


def cv2_over_percent_radius(coarse):
    if coarse.size == 0:
        return np.full(N_PERCENT_POINTS, np.nan, dtype=float)
    mean = np.nanmean(coarse, axis=0)
    std = np.nanstd(coarse, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv2 = (std / mean) ** 2
    cv2[~np.isfinite(cv2)] = np.nan
    return cv2


def plot_fgm_cv2_panel(ax):
    """Panel A: squared coefficient of variation of the FGM walk radius across
    replicates, versus walk progress, one curve per dimensionality n."""
    fgm_cv2_by_n = load_fgm_cv2_metrics()
    percent_axis = np.arange(1, N_PERCENT_POINTS + 1)
    for color, (n_val, coarse) in zip(CMR_COLORS, fgm_cv2_by_n.items()):
        cv2 = cv2_over_percent_radius(coarse)
        if np.all(np.isnan(cv2)):
            continue
        ax.plot(percent_axis, cv2, color=color, lw=2.3, label=rf"$n={n_val}$")

    ax.set_xlabel("Walk progress (%)")
    ax.set_ylabel(r"$CV^2(R(t))$")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")


# ----------------------------------------------------------------
# 1c. FGM RADIUS-VS-TIME PANEL (mean descent vs far-field ODE)
# ----------------------------------------------------------------
def load_fgm_radius_vs_time():
    """Per dimensionality n, the non-dimensional walk radius
    tilde_r = ||r|| / sigma for every replicate as a function of walk step,
    read from the high-statistics (1000-rep) production trajectories. Returns
    {n: array of shape (n_reps, max_steps)} with NaN padding past each walk's end.
    """
    data_dir = resolve_fgm_data_dir()
    radius_by_n = {}
    for path in find_fgm_files(data_dir):
        n = extract_n_from_name(path)
        sigma = extract_sigma_from_name(path)
        with path.open("rb") as handle:
            repeats = pickle.load(handle)

        tilde_r_traces = []
        for rep in repeats:
            if not isinstance(rep, dict):
                continue
            traj = rep.get("traj")
            if traj is None or len(traj) == 0:
                continue
            arr = np.asarray(traj, dtype=float)
            if arr.ndim != 2:
                continue
            tilde_r_traces.append(np.linalg.norm(arr, axis=1) / sigma)

        if not tilde_r_traces:
            continue

        max_steps = max(len(trace) for trace in tilde_r_traces)
        padded = np.full((len(tilde_r_traces), max_steps), np.nan)
        for i, trace in enumerate(tilde_r_traces):
            padded[i, :len(trace)] = trace
        radius_by_n[n] = padded

    return dict(sorted(radius_by_n.items()))


def realign_held_from_radius(radii, r_ref):
    """Re-segment each replicate so t'=0 is its first step with radius <= r_ref,
    then forward-fill the post-termination peak floor. Every larger-n walk passes
    through r_ref on its way down, so this places all n on a *common* start radius
    for a like-for-like descent. In SSWM a terminated walk has reached a local
    optimum and freezes there, so holding it at its final radius is exact, not a
    fudge. Returns a (n_reps, max_seg_len) array, each row starting near r_ref."""
    segments = []
    for row in radii:
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            continue
        hits = np.nonzero(finite <= r_ref)[0]
        idx0 = int(hits[0]) if hits.size else 0
        segments.append(finite[idx0:])
    if not segments:
        return np.empty((0, 0))

    max_len = max(len(s) for s in segments)
    held = np.empty((len(segments), max_len))
    for i, s in enumerate(segments):
        held[i, :len(s)] = s
        held[i, len(s):] = s[-1]          # freeze at the peak floor
    return held


def plot_fgm_radius_panel(ax):
    """Panel B: mean FGM walk radius vs time, re-segmented so every n starts from
    a *common* initial radius r_ref = the n=4 initial tilde_r_0 (~40). Each curve
    is then r(t)/r_ref, so the far-field ODE  d||r||/dt = -sqrt(pi/2) sigma
    collapses to a single line  r(t)/r_ref = 1 - sqrt(pi/2) t / tilde_r_0 shared
    by all n. The curves track it together and peel off (decelerate) once they
    reach the crossover radius tilde_R_c = (n-1)/sqrt(2 pi) -- larger n peeling
    earlier, since that radius is larger. One curve per n, colored as in panel A."""
    radius_by_n = load_fgm_radius_vs_time()
    slope = np.sqrt(np.pi / 2.0)

    # Common start radius: the smallest-n (n=4) initial tilde_r_0. Every larger-n
    # walk passes through it on the way down to the peak.
    smallest_n = min(radius_by_n)
    r_ref = float(np.nanmean(radius_by_n[smallest_n][:, 0]))

    right_edges = []
    n_handles = []
    for color, (n_val, radii) in zip(CMR_COLORS, radius_by_n.items()):
        held = realign_held_from_radius(radii, r_ref)
        if held.size == 0:
            continue
        mean_r = held.mean(axis=0)
        std_r = held.std(axis=0)
        n_steps = held.shape[1]
        t = np.arange(n_steps)

        # Normalize by the common start radius r_ref so all n share one theory line.
        norm_mean = mean_r / r_ref
        norm_std = std_r / r_ref

        line, = ax.plot(t, norm_mean, color=color, lw=2.2, label=rf"$n = {n_val}$")
        n_handles.append(line)
        ax.fill_between(t, norm_mean - norm_std, norm_mean + norm_std,
                        color=color, alpha=0.18, lw=0)
        right_edges.append(n_steps - 1)

        # Peel-off marker: the step at which the mean walk reaches the crossover
        # radius tilde_R_c = (n-1)/sqrt(2 pi), where the far-field ODE stops
        # holding and the descent decelerates.
        crossings = np.nonzero(mean_r <= (n_val - 1.0) / np.sqrt(2.0 * np.pi))[0]
        if crossings.size:
            k = crossings[0]
            ax.plot(t[k], norm_mean[k], marker="+", ms=16, mew=3.0, ls="none",
                    color=color, zorder=5)

    # One shared far-field line, from the common start radius r_ref; mask the part
    # below zero so the dashed line stops at the peak.
    t_max = max(right_edges) if right_edges else 1
    t_th = np.arange(0, t_max + 1)
    theory = 1.0 - slope * t_th / r_ref
    theory[theory < 0] = np.nan
    theory_line, = ax.plot(t_th, theory, color="black", lw=1.5, ls="--", alpha=0.9,
                           label="Eq. *")
    # Neutral proxy so the per-n peel-off markers get one legend entry.
    peel_proxy = mpl.lines.Line2D([], [], color="0.35", marker="+", ms=16,
                                  mew=3.0, ls="none",
                                  label=r"$\langle \tilde{R} \rangle = \tilde{R}_c$")

    ax.set_xlabel("Time (steps)")
    ax.set_ylabel(r"$\langle \tilde{R}(t) \rangle / \tilde{R}(0)$")
    ax.set_title(rf"$\tilde{{R}}(0) = {r_ref:.0f}$")
    set_ylim_clipped(ax, 0.0, 1.0)
    ax.set_xlim(0, 50)
    # Two legends: the n curves in the top-right corner, the theory line on its own
    # at the bottom left so it does not stretch the n block.
    theory_leg = ax.legend(handles=[theory_line, peel_proxy], frameon=False,
                           loc="lower left")
    ax.add_artist(theory_leg)
    ax.legend(handles=n_handles, frameon=False, loc="upper right")

# ----------------------------------------------------------------
# 2. 1/e COLLAPSE PANELS (decorrelation time vs the constrained-walk timescale)
# ----------------------------------------------------------------
def load_sweep(filename, mode):
    """One of the constrained-walk sweeps produced by
    data/gen_data/gen_dat_fgm_constrained.py. The figure never simulates."""
    path = resolve_fgm_data_dir() / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing sweep data: {path}\n"
            f"Generate it with:  cd data/gen_data && "
            f"python gen_dat_fgm_constrained.py --mode {mode}"
        )
    with open(path, "rb") as handle:
        cache = pickle.load(handle)
    print(f"Loaded {mode} sweep from {path}")
    return cache


def first_crossing(t, y, level):
    """Interpolated first time y drops below level, NaN if it never does."""
    below = np.nonzero(y < level)[0]
    if below.size == 0 or below[0] == 0:
        return np.nan
    k = int(below[0])
    return float(np.interp(level, [y[k], y[k - 1]], [t[k], t[k - 1]]))


def mean_alive(traces, min_alive_frac=0.5):
    """Replicate mean of rho(t), blanked where fewer than min_alive_frac of the
    walks are still stepping (past that point the mean is a handful of survivors)."""
    alive = np.sum(np.isfinite(traces), axis=0)
    keep = alive >= min_alive_frac * traces.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.where(keep, np.nanmean(traces, axis=0), np.nan), keep


def plot_1e_collapse_panel(ax, cache, n_values, tau_of, admissible, ylabel,
                           legend_loc="lower right", n_boot=150):
    r"""The time at which the DFE autocorrelation has decayed by one e-fold,
    log rho = -1, in units of the timescale of the constraint that the walk runs
    under -- one curve per n, swept over the starting radius R~(0).

    tau_of(R~(0), n) supplies that timescale: the radial collapse time
    tau_rs = sqrt(2/pi) R~(0) for the wedge walk, the rotational time
    tau_as = 2 R~(0)^2 / (n - 1) for the constant-radius walk. If the constrained
    motion is what scrambles the DFE, every curve lands on t_{1/e}/tau = 1 (the
    dashed line) once R~(0) is large enough for the finite-n corrections, which
    are of order n/R~(0)^2, to die out. admissible(R~(0), n) drops the cells where
    the constraint itself, rather than the geometry, sets the answer. Error bars
    bootstrap over replicates."""
    cells = cache["cells"]
    r0_values = sorted({r0 for r0, _ in cells})
    rng = np.random.default_rng(0)
    level = 1.0 / np.e                      # log rho = -1

    n_handles = []
    for color, n_val in zip(CMR_COLORS, n_values):
        xs, ys, errs = [], [], []
        for R0_tilde in r0_values:
            cell = cells.get((float(R0_tilde), int(n_val)))
            if cell is None or not admissible(R0_tilde, n_val):
                continue
            t = np.asarray(cell["time_points"], dtype=float)
            traces = np.asarray(cell["pearson"], dtype=float)
            mean_rho, keep = mean_alive(traces)
            t_1e = first_crossing(t, mean_rho, level)
            if not np.isfinite(t_1e):
                continue
            # Bootstrap the replicate set to get the error on the crossing time.
            draws = []
            for _ in range(n_boot):
                idx = rng.integers(0, traces.shape[0], traces.shape[0])
                boot_mean, _ = mean_alive(traces[idx])
                draws.append(first_crossing(t, np.where(keep, boot_mean, np.nan), level))
            tau = tau_of(R0_tilde, n_val)
            xs.append(R0_tilde)
            ys.append(t_1e / tau)
            errs.append(np.nanstd(draws) / tau)

        ax.errorbar(xs, ys, yerr=errs, marker="o", ms=6, lw=2.0, capsize=3,
                    color=color)
        # Plain line proxies so this legend reads like the ones in panels A and B,
        # rather than carrying the marker-and-cap glyph of the errorbar artist.
        n_handles.append(mpl.lines.Line2D([], [], color=color, lw=2.3,
                                          label=rf"$n = {n_val}$"))

    ax.axhline(1.0, color="black", ls="--", lw=1.5, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tilde{R}(0)$")
    ax.set_ylabel(ylabel)
    ax.legend(handles=n_handles, frameon=False, loc=legend_loc)


def plot_radial_collapse_panel(ax, n_values):
    """Panel C: the wedge walk, against the radial collapse time
    tau_rs = sqrt(2/pi) R~(0)."""
    cache = load_sweep(RADIAL_SWEEP_FILE, "radial")
    phi = float(cache["phi"])
    plot_1e_collapse_panel(
        ax, cache, n_values,
        # The wedge admits a mutation only while sin^2(phi) R~(0)^2 > ~n; at
        # smaller R~(0) the walks are pinned by the constraint, not by the FGM
        # geometry, so those cells say nothing about the radial timescale.
        tau_of=lambda r0, n: r0 / SQRT_HALF_PI,
        admissible=lambda r0, n: r0 > np.sqrt(n) / np.sin(phi),
        ylabel=r"$t_{1/e} \,/\, \tau_{rs}$",
    )
    ax.set_xlim(7, 150)
    ax.set_ylim(0.7, 1.1)
    ax.set_yticks(np.arange(0.7, 1.1001, 0.1))


def plot_angular_collapse_panel(ax, n_values):
    """Panel D: the constant-radius walk, against the rotational time
    tau_as = 2 R~(0)^2 / (n - 1)."""
    cache = load_sweep(ANGULAR_SWEEP_FILE, "angular")
    plot_1e_collapse_panel(
        ax, cache, n_values,
        # At fixed radius rho cannot fall below (n/2)/(n/2 + R~(0)^2), the part of
        # the DFE variance that carries no direction; one e-fold is out of reach
        # unless R~(0)^2 > (e-1) n / 2.
        tau_of=lambda r0, n: 2.0 * r0 ** 2 / (n - 1.0),
        admissible=lambda r0, n: r0 ** 2 > (np.e - 1.0) * n / 2.0,
        ylabel=r"$t_{1/e} \,/\, \tau_{as}$",
        legend_loc="upper right",
    )
    ax.set_xlim(7, 56)
    ax.set_ylim(0.9, 1.8)
    ax.set_yticks(np.arange(0.9, 1.8001, 0.1))
    # The sweep spans less than a decade, where the default log formatter labels
    # every minor tick and the labels collide: label the round decades only and
    # let the minor ticks mark the R~(0) actually simulated.
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([10, 20, 40]))
    ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(["10", "20", "40"]))
    ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(ANGULAR_R0_TICKS))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


# ----------------------------------------------------------------
# 3. MAIN EXPERIMENT
# ----------------------------------------------------------------
def run_experiment(n_values):
    print("--- figS7: radial scrambling (wedge-constrained SSWM) ---")
    print(f"Curves (n): {n_values}")

    # ------------------------------
    # Plotting: A = FGM radius CV^2, B = FGM mean radius vs time (far-field ODE
    # test), C = the 1/e decorrelation time of the wedge walk against tau_rs,
    # D = the same for the constant-radius walk against tau_as.
    # ------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.6))
    fig.subplots_adjust(wspace=0.3, hspace=0.45)

    apply_axis_style(axes[0, 0], "A")
    plot_fgm_cv2_panel(axes[0, 0])

    apply_axis_style(axes[0, 1], "B")
    plot_fgm_radius_panel(axes[0, 1])

    apply_axis_style(axes[1, 0], "C")
    plot_radial_collapse_panel(axes[1, 0], n_values)

    apply_axis_style(axes[1, 1], "D")
    plot_angular_collapse_panel(axes[1, 1], n_values)

    out_dir = "../figs_paper"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "figS7_radial_scrambling.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Radial scrambling in FGM. Panels A/B track the SSWM radial "
                    "descent itself; panels C and D measure the time at which the "
                    "DFE autocorrelation has decayed by one e-fold, in units of the "
                    "radial collapse time tau_rs = sqrt(2/pi) R~(0) (wedge-constrained "
                    "walk) and of the rotational time tau_as = 2 R~(0)^2 / (n-1) "
                    "(constant-radius walk). The constrained-walk sweeps are "
                    "generated by data/gen_data/gen_dat_fgm_constrained.py."
    )
    parser.add_argument(
        "--n-values",
        default="4,8,16,32",
        help="Comma-separated dimensionalities n, one curve each in every panel.",
    )
    args = parser.parse_args()
    run_experiment([int(x) for x in args.n_values.split(",") if x.strip()])
