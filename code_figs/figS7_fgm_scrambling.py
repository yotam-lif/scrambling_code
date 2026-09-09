import argparse
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
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

# The three sweeps behind the bottom row, generated once by the scripts in
# data/gen_data. This figure only ever reads them.
RADIAL_SWEEP_FILE = "fgm_frozen_sweep_rps500_sig0.05.pkl"
ANGULAR_SWEEP_FILE = "fgm_shell_sweep_rps1000_sig0.05_eps0.05.pkl"
REANCHORED_FILE = "fgm_reanchored_sweep_rps1000_sig0.05.pkl"
ANGULAR_R0_TICKS = [8, 12, 16, 24, 32, 48]   # the R~(0) of the angular sweep

# The re-anchored sweep also carries the radius and orientation history of every
# unconstrained walk, which is what the whole top row is drawn from. Reading it
# instead of the trajectories themselves is what keeps this figure off the ~40 GB
# of production pickles that panels A and B used to load twice between them.
N_PERCENT_POINTS = 100

# Panel C bins the walks by accumulated angular phase; this is the grid, and the
# smallest number of pooled (phase, cosine) samples a bin needs to be drawn.
PHASE_EDGES = np.linspace(0.0, 2.5, 26)
PHASE_MIN_SAMPLES = 200

# Panel A's CV^2 is a ratio of two replicate moments on a 100-point progress grid,
# so it carries visible sampling chatter, worst in the last decile where the walks
# are ending at different steps. Smooth it over a couple of grid points -- narrow
# enough to leave the peak height and position alone.
CV2_SMOOTH_POINTS = 2


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

    Panel D simulates exactly this: FisherFrozenAxis holds the direction fixed and
    lets only the radius move, so the walk is this law's own simulation rather than
    an approximation to it.
    """
    t = np.asarray(t, dtype=float)
    r_t = np.clip(r0_tilde - SQRT_HALF_PI * t, 0.0, None)
    half_n = 0.5 * n
    return (half_n + r0_tilde * r_t) / np.sqrt((half_n + r0_tilde ** 2)
                                               * (half_n + r_t ** 2))

# ----------------------------------------------------------------
# 1b. THE UNCONSTRAINED WALKS (top row), read from the re-anchored cache
# ----------------------------------------------------------------
def resolve_fgm_data_dir():
    script_dir = Path(__file__).resolve().parent
    data_candidates = [script_dir.parent / "data" / "fgm", script_dir.parent / "data" / "FGM"]
    return next((p for p in data_candidates if p.exists()), data_candidates[-1])


def sample_radius_vs_percent(radii, n_points=N_PERCENT_POINTS):
    """One walk's radius resampled onto a common 1..100% progress axis, so walks
    of different lengths can be averaged against each other."""
    radii = np.asarray(radii, dtype=float)
    radii = radii[np.isfinite(radii)]
    if radii.size == 0:
        return np.full(n_points, np.nan)

    total_steps = len(radii) - 1
    percents = np.arange(1, n_points + 1, dtype=float)

    if total_steps <= 0:
        return np.full(n_points, radii[0], dtype=float)

    targets = (percents / 100.0) * total_steps
    idx = np.rint(targets).astype(int)
    idx = np.clip(idx, 0, total_steps)
    return radii[idx]


_REANCHORED_CACHE = {}


def load_reanchored():
    """The re-anchored sweep of the *unconstrained* walks, produced once by
    data/gen_data/gen_dat_fgm_reanchored.py. It holds two things: `cells`, the
    DFE autocorrelation measured from a grid of starting radii (panel F, in the
    same layout the constrained sweeps use), and `walks`, every replicate's
    radius R~(t) and orientation overlap rhat(0).rhat(t) (panels A, B and C)."""
    if not _REANCHORED_CACHE:
        path = resolve_fgm_data_dir() / REANCHORED_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Missing re-anchored sweep: {path}\n"
                f"Generate it with:  cd data/gen_data && "
                f"python gen_dat_fgm_reanchored.py")
        with open(path, "rb") as handle:
            _REANCHORED_CACHE.update(pickle.load(handle))
        print(f"Loaded re-anchored sweep from {path}")
    return _REANCHORED_CACHE


def radius_by_n():
    """{n: (n_reps, max_steps) array of tilde_r = ||r|| / sigma}, NaN past each
    walk's own end."""
    walks = load_reanchored()["walks"]
    return {int(n): np.asarray(w["radius"], dtype=float)
            for n, w in sorted(walks.items())}


def load_fgm_cv2_metrics():
    coarse_by_n = {}
    for n, radii in radius_by_n().items():
        percent_radii = [sample_radius_vs_percent(row, n_points=N_PERCENT_POINTS)
                         for row in radii]
        percent_radii = [p for p in percent_radii if np.all(np.isfinite(p))]
        coarse_by_n[n] = (
            np.vstack(percent_radii)
            if len(percent_radii) > 0
            else np.empty((0, N_PERCENT_POINTS), dtype=float)
        )
    return coarse_by_n


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
        cv2 = gaussian_filter1d(cv2, CV2_SMOOTH_POINTS, mode="nearest")
        ax.plot(percent_axis, cv2, color=color, lw=2.3, label=rf"$n={n_val}$")

    ax.set_xlabel("Walk progress (%)")
    ax.set_ylabel(r"$CV^2(R(t))$")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")


# ----------------------------------------------------------------
# 1c. FGM RADIUS-VS-TIME PANEL (mean descent vs far-field ODE)
# ----------------------------------------------------------------
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
    radii_by_n = radius_by_n()
    slope = np.sqrt(np.pi / 2.0)

    # Common start radius: the smallest-n (n=4) initial tilde_r_0. Every larger-n
    # walk passes through it on the way down to the peak.
    smallest_n = min(radii_by_n)
    r_ref = float(np.nanmean(radii_by_n[smallest_n][:, 0]))

    right_edges = []
    for color, (n_val, radii) in zip(CMR_COLORS, radii_by_n.items()):
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

        ax.plot(t, norm_mean, color=color, lw=2.2)
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
    # The colours are the same n in every panel, so only panels A and D carry the
    # n key; this one is left with the theory line and the peel-off marker.
    ax.legend(handles=[theory_line, peel_proxy], frameon=False, loc="lower left")

# ----------------------------------------------------------------
# 1d. FGM ORIENTATION PANEL (angular law vs accumulated phase)
# ----------------------------------------------------------------
def accumulated_phase(radius, n):
    """The angular phase phi(t) = (n-1)/2 * integral_0^t dt'/R~(t')^2 accumulated
    along one walk, by the trapezoid rule on the walk's own radius track with a
    time step of one substitution.

    This is the exponent of the angular autocorrelation law: rotational diffusion
    on the sphere of radius R~ runs at rate (n-1)/(2 R~^2), so a *shrinking*
    radius has no single rate and no single timescale -- only the accumulated
    phase. Panel E's tau_as is the special case of a radius held fixed.
    """
    finite = radius[np.isfinite(radius)]
    if finite.size == 0:
        return finite
    rate = (n - 1.0) / (2.0 * finite ** 2)
    return np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]))])


def plot_angular_law_panel(ax):
    r"""Panel C: the angular counterpart of panel B -- the measured orientation
    overlap of an unconstrained walk against the law it is supposed to obey,

        E[ rhat(0).rhat(t) | R~ path ] = exp(-phi(t)),

    with phi the accumulated phase above. Every n collapses onto the single black
    line, which carries no fitted parameter.

    The walks are the same unconstrained production runs panels A and B use, taken
    whole: each replicate is followed from its own step 0, cos theta is measured
    against the direction it started in, and the phase is accumulated from there.
    Unlike panel B, they are *not* re-anchored onto a common starting radius --
    each n simply begins where its own runs begin, at R~(0) = sqrt(n)/sigma
    (40, 57, 80 and 113 for n = 4, 8, 16, 32). Nothing needs to be done about that
    here: the axis is phase rather than time, and the phase has already absorbed
    the entire radius history, so the law is the same one whatever radius the walk
    set out from. That the four n land on one line despite starting decades apart
    in tau_as is the statement being made.

    Two things make this the right way to read the law. The conditioning is on the
    radius path, so the pairing has to be per replicate: each walk contributes its
    own (phi, cos theta) samples and the average is taken *within* a phase bin.
    Averaging the phase first and exponentiating afterwards is a different and
    much worse approximation -- exp(-<phi>) is off by a factor of two once the
    walk is a few sigma from the peak, where <exp(-phi)> is still good to a few
    percent, because 1/R~^2 is convex and the radius spread is no longer
    negligible. Second, the samples are pooled over steps as well as replicates:
    what the law predicts is a relation between phase and overlap, not between
    time and overlap, so phase is the axis and time is integrated out.

    The law holds to about 1% out to phi ~ 0.5 and 5% at phi ~ 1, and gives out
    past phi ~ 1.5. That is the same near-field breakdown panel B shows: by then
    R~ is of order one, a single mutation is no longer a small step on the sphere,
    and the diffusion approximation behind the law has nothing left to stand on.
    """
    walks = load_reanchored()["walks"]
    centers = 0.5 * (PHASE_EDGES[1:] + PHASE_EDGES[:-1])

    # Small n in front: its curve is the one that leaves the law first, so it is
    # the one that must not be hidden. The black line goes over all of them.
    for depth, (color, (n_key, w)) in enumerate(zip(CMR_COLORS, sorted(walks.items()))):
        n_val = int(n_key)
        radii = np.asarray(w["radius"], dtype=float)
        cosines = np.asarray(w["cosine"], dtype=float)

        phis, coss = [], []
        for r_row, c_row in zip(radii, cosines):
            phi = accumulated_phase(r_row, n_val)
            if phi.size == 0:
                continue
            phis.append(phi)
            coss.append(c_row[np.isfinite(c_row)][:phi.size])
        if not phis:
            continue
        phi = np.concatenate(phis)
        cos = np.concatenate(coss)

        idx = np.digitize(phi, PHASE_EDGES)
        means = np.full(centers.size, np.nan)
        for b in range(1, PHASE_EDGES.size):
            sel = idx == b
            if np.count_nonzero(sel) >= PHASE_MIN_SAMPLES:
                means[b - 1] = np.mean(cos[sel])
        ax.plot(centers, means, marker="o", ms=5, lw=0.0, color=color,
                zorder=10 - depth)

    phi_line = np.linspace(PHASE_EDGES[0], PHASE_EDGES[-1], 200)
    theory_line, = ax.plot(phi_line, np.exp(-phi_line), color="black", lw=1.5,
                           ls="--", alpha=0.9, label="Eq. **", zorder=20)

    ax.set_yscale("log")
    ax.set_xlabel("Accumulated phase")
    ax.set_ylabel(r"$\langle \hat{r}(0) \cdot \hat{r}(t) \rangle$")
    ax.set_xlim(0, PHASE_EDGES[-1])
    ax.set_ylim(0.06, 1.15)
    ax.legend(handles=[theory_line], frameon=False, loc="lower left")


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


def strip_floor(traces, tail_frac=0.2):
    """Rescale rho(t) so that the part of it that does decay runs from 1 to 0:
    (rho - rho_end) / (1 - rho_end).

    Either constraint leaves a piece of the DFE untouched, so raw rho relaxes to a
    floor rather than to zero and the raw e-fold time inherits that offset. At
    fixed radius (panel E) the isotropic -||delta||^2/2 term carries no direction
    and never decorrelates; on a frozen axis (panel D) it is the other way round,
    the descent only rescales the R delta_par term and stops at a local optimum
    besides, which is the larger effect of the two -- the floor is 0.94 at n = 32,
    R~(0) = 8. Dividing it out is what makes the two panels one measurement.

    rho_end is measured, not taken from a closed form: it is the replicate mean
    over the last tail_frac of the recorded window, which both horizons place well
    past the point where the walk has arrived. The average has to be over
    replicates -- a single shell walk ends at a random direction, so its own last
    point is cos(theta) of a random orientation, a number with a spread of
    1/sqrt(n-1), not a floor."""
    k = max(1, int(round(tail_frac * traces.shape[1])))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        floor = float(np.nanmean(traces[:, -k:]))
    return (traces - floor) / (1.0 - floor)


def plot_1e_collapse_panel(ax, cache, n_values, tau_of, admissible, ylabel,
                           legend_loc="lower right", n_boot=150, floor=False,
                           legend=True):
    r"""The time at which the DFE autocorrelation has decayed by one e-fold,
    log rho = -1, in units of the timescale of the constraint that the walk runs
    under -- one curve per n, swept over the starting radius R~(0).

    tau_of(R~(0), n) supplies that timescale: the radial collapse time
    tau_rs = sqrt(2/pi) R~(0) for the frozen-axis walk, the rotational time
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
            # The floor is re-measured on every bootstrap draw, so its own
            # sampling error is inside the error bar.
            prep = strip_floor if floor else (lambda x: x)
            mean_rho, keep = mean_alive(prep(traces))
            t_1e = first_crossing(t, mean_rho, level)
            if not np.isfinite(t_1e):
                continue
            # Bootstrap the replicate set to get the error on the crossing time.
            draws = []
            for _ in range(n_boot):
                idx = rng.integers(0, traces.shape[0], traces.shape[0])
                boot_mean, _ = mean_alive(prep(traces[idx]))
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
    # One n key for the bottom row, in panel D; the colours mean the same thing in
    # all three.
    if legend:
        ax.legend(handles=n_handles, frameon=False, loc=legend_loc)


def plot_radial_collapse_panel(ax, n_values):
    """Panel D: the frozen-axis descent, against the radial collapse time
    tau_rs = sqrt(2/pi) R~(0). Read on the part of rho that decays: shrinking the
    radius only rescales the R delta_par term of the effect and leaves the
    isotropic -||delta||^2/2 term alone, so radial motion by itself cannot
    decorrelate the DFE and rho stops at a floor -- 0.94 at n = 32, R~(0) = 8 --
    which raw rho only gets under 1/e for n <~ 8.

    The curves approach 1 from above, by as much as 50% at the smallest R~(0), and
    that excess is in the descent rather than in the correlation. It factors as
    (1 - R~*/R~(0)) x (t_1e / [(R~(0) - R~*) / sqrt(pi/2)]), R~* being the radius
    at the crossing: the first factor is mild (0.98 at large R~(0), 0.44 only for
    n = 32 at R~(0) = 8), the second carries the deviation. sqrt(pi/2) is the
    R~ -> infinity descent speed; the true speed is 0.55 sqrt(pi/2) at R~ = 2 and
    0.78 at R~ = 4, for every n, so the last few sigma of any descent cost extra
    time, and at small R~(0) that stretch is most of the walk. Pushing the other
    way at large R~(0), the SSWM weight is expm1(s) rather than s, which favours
    big steps and runs the walk *faster* than the far-field law (1.12 sqrt(pi/2)
    at R~(0) = 256) -- which is why the curves cross the dashed line rather than
    settling onto it from above.

    Both effects come out of the exact step law with no free parameters: landing
    from R~ puts you at y ~ noncentral chi2(df = n, nc = R~^2), the move is
    beneficial iff y < R~^2, and SSWM picks with weight expm1(sigma^2 (R~^2-y)/2),
    so the speed is E[(R~ - sqrt(y)) w] / E[w] over that range. It reproduces the
    measured initial descent rate to 1-4% over the whole grid, and integrating it
    reproduces this panel to a median 1.2%; it runs high at the smallest R~(0),
    worst at n = 4 (11%) and least at n = 32 (2.5%), because a mean-field ODE
    cannot follow the last few stochastic jumps into the peak -- and only the
    small-n walks get there, R~ at the end of the run being 0.10 at n = 4 against
    3.76 at n = 32."""
    cache = load_sweep(RADIAL_SWEEP_FILE, "radial")
    plot_1e_collapse_panel(
        ax, cache, n_values,
        # tau_rs is the collapse time of the *far-field* descent, so a walk has to
        # start above the crossover radius R~_c = (n-1)/sqrt(2 pi) -- the radius at
        # which the descent decelerates, marked in panel B -- for that to be the
        # timescale it runs on at all.
        tau_of=lambda r0, n: r0 / SQRT_HALF_PI,
        admissible=lambda r0, n: r0 > (n - 1.0) / np.sqrt(2.0 * np.pi),
        ylabel=r"$t_{1/e} \,/\, \tau_{rs}$",
        legend_loc="upper right",
        floor=True,
    )
    ax.set_xlim(7, 300)
    ax.set_ylim(0.9, 1.7)
    ax.set_yticks(np.arange(0.9, 1.7001, 0.2))


def plot_angular_collapse_panel(ax, n_values):
    """Panel E: the constant-radius walk, against the rotational time
    tau_as = 2 R~(0)^2 / (n - 1). Read on the part of rho that decays: at fixed
    radius the walk can only turn the DFE, never shrink it, so rho settles onto a
    floor of order n / R~(0)^2 and the raw e-fold time carries that offset."""
    cache = load_sweep(ANGULAR_SWEEP_FILE, "angular")
    plot_1e_collapse_panel(
        ax, cache, n_values,
        # Every cell is usable once the floor is divided out: the decaying part of
        # rho runs from 1 to 0 by construction, so it always reaches 1/e -- unlike
        # raw rho, which only does so when R~(0)^2 > (e-1) n / 2.
        tau_of=lambda r0, n: 2.0 * r0 ** 2 / (n - 1.0),
        admissible=lambda r0, n: True,
        ylabel=r"$t_{1/e} \,/\, \tau_{as}$",
        floor=True,
        legend=False,
    )
    ax.set_xlim(7, 56)
    ax.set_ylim(0.8, 1.2)
    ax.set_yticks(np.arange(0.8, 1.2001, 0.1))
    # The sweep spans less than a decade, where the default log formatter labels
    # every minor tick and the labels collide: label the round decades only and
    # let the minor ticks mark the R~(0) actually simulated.
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([10, 20, 40]))
    ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(["10", "20", "40"]))
    ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(ANGULAR_R0_TICKS))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def plot_combined_collapse_panel(ax, n_values):
    r"""Panel F: the unconstrained walk -- descending and rotating at once --
    against the combined timescale

        tau_s^-1 = tau_as^-1 + tau_rs^-1,

    the two scrambling *rates* added. Panels D and E each freeze half the geometry
    to time the other half; this one freezes nothing, and is the only panel of the
    three measured on the ordinary SSWM walks the rest of the paper uses. The
    R~(0) sweep comes from re-anchoring: each walk is picked up again at the first
    step where its radius reaches the anchor, so one set of walks supplies every
    column of the sweep.

    Unlike panel D, no cell is excluded. The point of adding the rates is that the
    result should hold on *both* sides of the crossover radius R~_c, where neither
    pure law does: over this grid t_1e/tau_as ranges from 0.04 to 0.75 and
    t_1e/tau_rs from 0.24 to 0.97, while the ratio plotted here stays between 0.80
    and 1.10. The residual structure is a mild and readable one -- the smallest
    anchors sit a little below 1, where the walk is close enough to the peak that
    the descent has slowed below sqrt(pi/2) and tau_rs is an underestimate, and
    the largest anchors at n = 32 sit ~9% above it.
    """
    cache = load_reanchored()
    plot_1e_collapse_panel(
        ax, cache, n_values,
        tau_of=lambda r0, n: 1.0 / ((n - 1.0) / (2.0 * r0 ** 2)
                                    + SQRT_HALF_PI / r0),
        admissible=lambda r0, n: True,
        ylabel=r"$t_{1/e} \,/\, \tau_{s}$",
        floor=True,
        legend=False,
    )
    # The sweep spans more than a decade, so the default log formatter labels the
    # round decades and nothing else -- the same axis panel D gets.
    ax.set_xlim(3.5, 115)
    ax.set_ylim(0.7, 1.2)
    ax.set_yticks(np.arange(0.7, 1.2001, 0.1))


# ----------------------------------------------------------------
# 3. MAIN EXPERIMENT
# ----------------------------------------------------------------
def run_experiment(n_values):
    print("--- figS7: FGM scrambling timescales ---")
    print(f"Curves (n): {n_values}")

    # ------------------------------
    # Two rows. The top one is the mean-field description of an unconstrained
    # walk: A that the radius is sharp enough for a mean-field law to exist at
    # all, B the radial law it follows, C the angular law it follows. The bottom
    # one is the timescale each of those hands the DFE autocorrelation: D the
    # radial one alone (frozen axis), E the angular one alone (fixed radius),
    # F the two together on the unconstrained walk.
    # ------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 9.6))
    fig.subplots_adjust(wspace=0.32, hspace=0.45)

    apply_axis_style(axes[0, 0], "A")
    plot_fgm_cv2_panel(axes[0, 0])

    apply_axis_style(axes[0, 1], "B")
    plot_fgm_radius_panel(axes[0, 1])

    apply_axis_style(axes[0, 2], "C")
    plot_angular_law_panel(axes[0, 2])

    apply_axis_style(axes[1, 0], "D")
    plot_radial_collapse_panel(axes[1, 0], n_values)

    apply_axis_style(axes[1, 1], "E")
    plot_angular_collapse_panel(axes[1, 1], n_values)

    apply_axis_style(axes[1, 2], "F")
    plot_combined_collapse_panel(axes[1, 2], n_values)

    out_dir = "../figs_paper"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "figS7_fgm_scrambling.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrambling timescales in FGM. Panels A-C track the motion of "
                    "an unconstrained SSWM walk itself: the radius is sharp (A) and "
                    "follows the far-field radial law (B), while the orientation "
                    "follows the angular law in accumulated phase (C). Panels D-F "
                    "measure the time at which the DFE autocorrelation has decayed "
                    "by one e-fold, in units of the radial collapse time "
                    "tau_rs = sqrt(2/pi) R~(0) (frozen-axis descent), the rotational "
                    "time tau_as = 2 R~(0)^2 / (n-1) (constant-radius walk), and "
                    "their combination tau_s^-1 = tau_as^-1 + tau_rs^-1 (the "
                    "unconstrained walk, re-anchored). The sweeps are generated by "
                    "data/gen_data/gen_dat_fgm_constrained.py and "
                    "gen_dat_fgm_reanchored.py."
    )
    parser.add_argument(
        "--n-values",
        default="4,8,16,32",
        help="Comma-separated dimensionalities n, one curve each in every panel.",
    )
    args = parser.parse_args()
    run_experiment([int(x) for x in args.n_values.split(",") if x.strip()])
