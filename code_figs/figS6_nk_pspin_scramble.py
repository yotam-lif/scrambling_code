import seaborn as sns

r"""Collapse of the subset DFE-autocorrelation across interaction strength, at two anchor points,
for the NK model (top row) and the pure p-spin model (bottom row).

A 2x2 figure. Each panel plots the log subset-autocorrelation log rho of the DFE against a
connectivity-rescaled step, and overlays the fixed reference line -t/tau_s.

  A  NK, anchored at the start of the walk (t_0 = 0%),  x = t*K/N,  K in {4, 8, 16, 32, 64}, N=500.
  B  NK, anchored near the optimum   (t_0 = 75%),       x = t*K/N.
  C  p-spin, t_0 = 0%,   x = 2(p-1)*t/N,  p in {2, 3, 4} (N=500 for p=2,3; N=300 for p=4).
  D  p-spin, t_0 = 75%,  x = 2p*t/N.

The connectivity is K for NK and p-1 for the p-spin (the number of other spins coupled to a given
site in one interaction). A clean collapse onto -x means the decorrelation timescale is tau_s = 3N/4K (t_0=0) resp. N/K
(t_0=75%) for the NK model, and N/2(p-1) (t_0=0) resp. N/2p (t_0=75%) for the p-spin.

Measure
-------
For one walk, anchor at step k0 (a fixed fraction of the walk length T) and keep the subset of
sites still in their anchor state at step k0+t (flipped an even number of times since the anchor).
rho(t) = corr(dF[k0][subset], dF[k0+t][subset]), rho(0) = 1, averaged over replicates in log space.
The correlation is taken on the raw fitness effects dF_i. Using the selection coefficient
s_i = dF_i / F instead would rescale each step by a constant, which leaves |rho| unchanged but
flips its sign wherever F < 0 -- and a p-spin walk starts at F ~ -O(sqrt(N)) -- so the raw effects
are the only choice that keeps the t_0 = 0% anchor meaningful.

Data and cache
--------------
Built from the stored adaptive walks: data/NK/N_500_K_*_repeats_10.pkl (which carry the DFE at
every step, so the NK walks are read, not replayed) and the pure p-spin runs (which carry J and
the flip sequence, so each walk is replayed through cmn_pspin's incremental updates) --
data/PSPIN/N500_P{2,3}_pure_repeats10.pkl, plus data/PSPIN/N300_P4_pure_repeats5/ for p=4, one
pickle per replicate because a single N=300 p=4 landscape is ~9 GB and must be streamed alone. The aggregated per-anchor curves are cached in data/cache/figS6_nk_pspin_cache.pkl;
delete it or pass --refresh to rebuild.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

# Repo root on path so we can drive the native p-spin DFE updates.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn import cmn_pspin


class _FixedOrderFormatter(ScalarFormatter):
    """ScalarFormatter that pins the shared 10^n multiplier to a fixed power (e.g. 10^-2), so
    ticks read 2, 4, 6 with a '×10⁻²' offset rather than 0.02, 0.04, 0.06."""

    def __init__(self, order, **kwargs):
        super().__init__(**kwargs)
        self._fixed_order = order

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self._fixed_order


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

# ───────────────────────────────────── Style ─────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams.update({
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

# ───────────────────────────────────── Configuration ─────────────────────────────────────
# NK: N=500, several K; connectivity = K. The stored walks carry their own DFE history.
NK_N = 500
K_VALUES = [4, 8, 16, 32, 64]
NK_FILES = {k: f"../data/NK/N_{NK_N}_K_{k}_repeats_10.pkl" for k in K_VALUES}
K_COLORS = dict(zip(K_VALUES, sns.color_palette("viridis", len(K_VALUES))))

# Pure p-spin: connectivity = p-1. The stored walks are replayed through cmn_pspin.
PSPIN = {2: 500, 3: 500, 4: 300}     # p -> N of the stored run
# A path is either a single pickle holding a list of walks, or a directory holding one pickle per
# replicate (p=4, whose ~9 GB landscapes have to be loaded and freed one at a time).
PSPIN_FILES = {2: "../data/PSPIN/N500_P2_pure_repeats10.pkl",
               3: "../data/PSPIN/N500_P3_pure_repeats10.pkl",
               4: "../data/PSPIN/N300_P4_pure_repeats5"}
P_COLORS = dict(zip(sorted(PSPIN), sns.color_palette("CMRmap", len(PSPIN) + 1)))

# Aggregated per-anchor curves, keyed by model, size, connectivity, replicates and anchor set.
CACHE_PATH = "../data/cache/figS6_nk_pspin_cache.pkl"
N_REPEATS = 10
# p=4 was run with fewer replicates than the rest (each landscape is ~9 GB).
PSPIN_REPEATS = {2: N_REPEATS, 3: N_REPEATS, 4: 5}
# Anchors computed and cached (a superset of the two that are plotted), as fractions of the walk
# length: anchor step k0 = round(frac * T).
ANCHOR_FRACS = [0.0, 0.25, 0.5, 0.75]
PEARSON_FLOOR = 1e-3      # clip rho before taking the log
# Minimum number of walks that must still reach a step for it to be plotted (used to trim each
# anchor series). Matches the convention of the sibling scrambling scripts.
PEARSON_MIN_REPS = 3

# Anchors shown, and their index into ANCHOR_FRACS.
PANEL_FRACS = [0.0, 0.75]
ANCHOR_IDX = {f: i for i, f in enumerate(ANCHOR_FRACS)}
# Fixed x-window for both p-spin (bottom-row) panels, and the forced x-axis 10^n multiplier.
PSPIN_XMAX = 0.16
PSPIN_XORDER = -1
# Same explicit control for the NK (top-row) panels, per anchor fraction (their ranges differ).
NK_XMAX = {0.0: 0.4, 0.75: 0.4}
NK_XORDER = {0.0: -1, 0.75: -1}
# Fixed (not fitted) NK decorrelation timescale per anchor: tau_s = NK_TAU_CONST * N/K, so the
# reference line has slope -1/NK_TAU_CONST in x = tK/N units.
NK_TAU_CONST = {0.0: 3.0 / 4.0, 0.75: 1.0}
NK_TAU_LABEL = {0.0: r"$\tau_s = 3N/4K$", 0.75: r"$\tau_s = N/K$"}
# Forced 10^n y-axis multiplier per panel, so every panel carries an explicit x10^-n on the
# y-axis: the NK panels span ~[-0.6,0] and the p-spin t_0=75% panel ~[-0.3,0] (order -1), while
# the p-spin t_0=0 panel only reaches ~-0.2 and reads better as x10^-2.
NK_YORDER = {0.0: -1, 0.75: -1}
PSPIN_YORDER = {0.0: -2, 0.75: -2}

OUT_PATH = "../figs_paper/figS6_nk_pspin_scramble.pdf"


# ───────────────────────────────────── Autocorrelation ─────────────────────────────────────

def _finite_mean_std(values):
    """Columnwise mean/std ignoring NaNs, without all-NaN runtime warnings."""
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    counts = mask.sum(axis=0)
    mean = np.full(values.shape[1], np.nan, dtype=float)
    std = np.full(values.shape[1], np.nan, dtype=float)

    valid = counts > 0
    if np.any(valid):
        safe_vals = np.where(mask[:, valid], values[:, valid], 0.0)
        mean_valid = safe_vals.sum(axis=0) / counts[valid]
        diff = np.where(mask[:, valid], values[:, valid] - mean_valid, 0.0)
        mean[valid] = mean_valid
        std[valid] = np.sqrt((diff ** 2).sum(axis=0) / counts[valid])
    return mean, std, counts


def _pad_traces(traces):
    """Stack ragged per-walk traces into one array, NaN-padded to the longest."""
    if not traces:
        return np.empty((0, 0), dtype=float)
    max_len = max(len(trace) for trace in traces)
    padded = np.full((len(traces), max_len), np.nan, dtype=float)
    for idx, trace in enumerate(traces):
        padded[idx, :len(trace)] = trace
    return padded


def _sigma_history(init_sigma, flip_seq):
    """(T+1, N) spin configurations along the stored flip sequence."""
    sigma = np.asarray(init_sigma, dtype=np.int8).copy()
    hist = np.empty((len(flip_seq) + 1, sigma.size), dtype=np.int8)
    hist[0] = sigma
    for j, site in enumerate(flip_seq, start=1):
        sigma[int(site)] *= -1
        hist[j] = sigma
    return hist


def _subset_autocorr_from_anchor(sig_hist, dfe_hist, k0):
    """seg[t] = corr(dF_{k0}[subset], dF_{k0+t}[subset]) over the sites still in their anchor
    state at step k0+t (flipped an even number of times since the anchor). seg[0] = 1."""
    T = sig_hist.shape[0] - 1
    dfe_anchor = dfe_hist[k0]
    sig_anchor = sig_hist[k0]

    seg = np.full(T - k0 + 1, np.nan)
    seg[0] = 1.0
    for dt, t in enumerate(range(k0 + 1, T + 1), start=1):
        subset = sig_hist[t] == sig_anchor
        if subset.sum() >= 2:
            a = dfe_anchor[subset]
            b = dfe_hist[t][subset]
            if np.all(np.isfinite(a)) and np.all(np.isfinite(b)) \
                    and np.std(a) > 1e-12 and np.std(b) > 1e-12:
                seg[dt] = float(np.corrcoef(a, b)[0, 1])
    return seg


def _anchor_curves(walks):
    """Aggregate the per-anchor subset autocorrelation over an iterable of (sig_hist, dfe_hist).

    Returns {"anchors": [...]}, one entry per ANCHOR_FRACS: the columnwise mean/std of log rho
    against steps-since-anchor, plus the number of walks still contributing at each step.
    """
    anchor_traces = [[] for _ in ANCHOR_FRACS]
    for sig_hist, dfe_hist in walks:
        T = sig_hist.shape[0] - 1
        for a, frac in enumerate(ANCHOR_FRACS):
            k0 = min(T, max(0, int(round(frac * T))))
            anchor_traces[a].append(_subset_autocorr_from_anchor(sig_hist, dfe_hist, k0))

    anchors = []
    for a, frac in enumerate(ANCHOR_FRACS):
        arr = _pad_traces(anchor_traces[a])
        with np.errstate(divide="ignore", invalid="ignore"):
            logv = np.log(np.clip(arr, PEARSON_FLOOR, None))
        logv[~np.isfinite(arr) | (arr <= 0)] = np.nan
        mean_logv, std_logv, counts = _finite_mean_std(logv)
        anchors.append({"frac": frac, "mean_logv": mean_logv, "std_logv": std_logv,
                        "counts": counts, "n_reps": len(anchor_traces[a])})
    return {"anchors": anchors}


# ───────────────────────────────────── Walk readers ─────────────────────────────────────

def _nk_walks(k, n_repeats):
    """(sig_hist, dfe_hist) per stored NK walk. The DFE at every step is stored, not recomputed.

    The DFEs in data/NK/ are intensive (fitness = mean of f_i over loci), unlike the p-spin's
    extensive ones -- except N_500_K_64_repeats_10.pkl, generated after cmn_nk switched to the
    extensive convention, whose effects are ~N times larger than its siblings'. A Pearson
    correlation is invariant to either constant factor, so this figure is unaffected; anything
    that compares NK effect *sizes* across files is not.
    """
    path = NK_FILES[k]
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    for entry in data[:n_repeats]:
        flip_seq = np.asarray(entry["flip_seq"], dtype=int)
        dfe_hist = np.asarray(entry["dfes"], dtype=float)
        yield _sigma_history(entry["init_sigma"], flip_seq), dfe_hist


def _rep_sort_key(name):
    """Order per-replicate filenames by their 'rep<k>' index when present, else lexically."""
    stem = os.path.splitext(name)[0]
    for part in stem.split("_"):
        if part.startswith("rep") and part[3:].isdigit():
            return (0, int(part[3:]))
    return (1, name)


def _pspin_entries(path, n_repeats):
    """Yield up to n_repeats stored walks, loading lazily to bound memory.

    Two on-disk layouts are supported: a single pickle holding a list of walks, and a directory
    holding one pickle per replicate (loaded and freed one at a time, so only a single ~9 GB
    landscape is resident at once).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    if os.path.isdir(path):
        names = sorted((f for f in os.listdir(path) if f.endswith(".pkl")), key=_rep_sort_key)
        yielded = 0
        for name in names:
            if yielded >= n_repeats:
                break
            full = os.path.join(path, name)
            print(f"    loading {name} ({os.path.getsize(full) / 1024 ** 3:.2f} GB) ...",
                  flush=True)
            with open(full, "rb") as f:
                loaded = pickle.load(f)
            # The generator writes a list of walks even for a single replicate.
            for entry in (loaded if isinstance(loaded, list) else [loaded]):
                if yielded >= n_repeats:
                    break
                yield entry
                yielded += 1
            del loaded
    else:
        print(f"    loading {os.path.basename(path)} "
              f"({os.path.getsize(path) / 1024 ** 3:.2f} GB) ...", flush=True)
        with open(path, "rb") as f:
            data = pickle.load(f)
        for entry in data[:n_repeats]:
            yield entry
        del data


def _pspin_walks(p, n_repeats):
    """(sig_hist, dfe_hist) per stored pure p-spin walk, replayed through cmn_pspin.

    Only J and the flip sequence are stored, so each walk is stepped through the native
    incremental updates, which carry the full DFE along with the configuration.
    """
    for entry in _pspin_entries(PSPIN_FILES[p], n_repeats):
        sigma0 = np.asarray(entry.get("init_sigma", entry.get("init_alpha")), dtype=np.int8)
        J = entry["J"]
        flip_seq = np.asarray(entry["flip_seq"], dtype=int)
        state = cmn_pspin._initialize_relaxation_state(sigma0, J)
        sig_hist = np.empty((flip_seq.size + 1, sigma0.size), dtype=np.int8)
        dfe_hist = np.empty((flip_seq.size + 1, sigma0.size), dtype=np.float64)
        sig_hist[0], dfe_hist[0] = state["sigma"], state["dfe"]
        for j, site in enumerate(flip_seq, start=1):
            cmn_pspin._apply_flip(state, J, int(site))
            sig_hist[j], dfe_hist[j] = state["sigma"], state["dfe"]
        yield sig_hist, dfe_hist
        del state, J, entry


# ───────────────────────────────────── Cache ─────────────────────────────────────

def _fracs_tag(fracs):
    """Stable cache-key fragment encoding a list of anchor fractions."""
    return "-".join(f"{f:g}" for f in fracs)


def _cached(cache, key, build, refresh, want):
    """Cache entries are reused only if built with at least ``want`` replicates."""
    hit = cache.get(key)
    if hit is not None and not refresh and hit["n_repeats"] >= want:
        return hit["value"], False
    print(f"  computing {key} ...", flush=True)
    value = build()
    cache[key] = {"n_repeats": want, "value": value}
    return value, True


def load_curves(refresh=False):
    """{"nk": {K: result}, "pspin": {p: result}}, computed on a cache miss and written back."""
    cache = {}
    if os.path.exists(CACHE_PATH) and not refresh:
        try:
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    tag = _fracs_tag(ANCHOR_FRACS)
    curves = {"nk": {}, "pspin": {}}
    dirty = False
    for k in K_VALUES:
        curves["nk"][k], built = _cached(
            cache, f"nk_anchors_N{NK_N}_K{k}_{tag}",
            lambda k=k: _anchor_curves(_nk_walks(k, N_REPEATS)), refresh, N_REPEATS)
        dirty |= built
    for p, n in sorted(PSPIN.items()):
        reps = PSPIN_REPEATS[p]
        curves["pspin"][p], built = _cached(
            cache, f"pspin_anchors_p{p}_N{n}_{tag}",
            lambda p=p, reps=reps: _anchor_curves(_pspin_walks(p, reps)), refresh, reps)
        dirty |= built

    if dirty:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
        print(f"  cache written to {CACHE_PATH}")
    return curves


# ───────────────────────────────────── Plotting ─────────────────────────────────────

def _trim_anchor(anchors, anchor_idx):
    """Return (t, mean_logv) for one anchor, trimmed to steps reached by >= PEARSON_MIN_REPS reps."""
    a = anchors[anchor_idx]
    counts = a["counts"]
    enough = np.flatnonzero(counts >= PEARSON_MIN_REPS)
    last = int(enough[-1]) if enough.size else 0
    t = np.arange(last + 1)
    return t, a["mean_logv"][:last + 1]


def plot_panel(ax, series, a, title, xlabel, tau_text, xmax, show_ylabel=True, xorder=None,
               yorder=None):
    """One panel: log rho vs the rescaled step, plus the -a x reference line."""
    series_handles = []
    for label, tau, y, color in series:
        m = np.isfinite(y) & (tau <= xmax)
        (h,) = ax.plot(tau[m], y[m], color=color, lw=2.0, marker="o", markersize=4,
                       label=label)
        series_handles.append(h)

    xs = np.linspace(0.0, xmax, 40)
    (fit_h,) = ax.plot(xs, -a * xs, color="black", lw=2.0, ls="--", label=r"$-t/\tau_s$")

    ax.set_xlim(0, xmax)
    # rho(t_0) = 1 by construction, so the curves start at log rho = 0: pin the top of the view
    # there and let the bottom autoscale.
    ax.set_ylim(top=0.0)
    # Shared 10^n x-axis multiplier. xorder pins the power (e.g. -2); otherwise auto sci notation.
    if xorder is not None:
        fmt = _FixedOrderFormatter(xorder, useMathText=True)
        fmt.set_scientific(True)
        ax.xaxis.set_major_formatter(fmt)
    else:
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    # Same shared 10^n multiplier on the y-axis. yorder pins the power; otherwise auto sci notation.
    if yorder is not None:
        fmt = _FixedOrderFormatter(yorder, useMathText=True)
        fmt.set_scientific(True)
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(r"$\log\,\rho(t_0, t_0 + t)$")
    ax.set_title(title)

    # Series (K / p) legend, single column, lower left.
    series_leg = ax.legend(handles=series_handles, frameon=False, loc="lower left", ncol=1)
    ax.add_artist(series_leg)
    # Fit legend: a single "-t/tau_s" line (so the dashed handle sits level with it) at the top
    # right, with the explicit tau_s written as text just beneath the legend box.
    fit_leg = ax.legend(handles=[fit_h], frameon=False, loc="upper right")
    ax.figure.canvas.draw()
    bb = fit_leg.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(bb.x1, bb.y0 - 0.005, tau_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=mpl.rcParams["legend.fontsize"])


# ───────────────────────────────────── Data assembly ─────────────────────────────────────

def _nk_series(curves, frac):
    """(series, xmax) for the NK panel at anchor fraction ``frac``. x = t*K/N."""
    series = []
    for k in K_VALUES:
        t, y = _trim_anchor(curves["nk"][k]["anchors"], ANCHOR_IDX[frac])
        series.append((rf"$K={k}$", t * k / NK_N, y, K_COLORS[k]))
    return series, NK_XMAX[frac]


def _pspin_series(curves, frac):
    """(series, xmax, xlabel, tau_expr) for the p-spin panel at anchor fraction ``frac``.

    The collapse variable folds in the expected p-spin decorrelation rate, which differs by regime:
      * early (t_0=0):    x = 2(p-1) t / N   (unflipped-subset rate near the start)
      * late  (t_0=0.75): x = 2 p     t / N   (rate near the optimum)
    so a clean collapse onto -x confirms the expected rate.
    """
    early = (frac == 0.0)
    factor = (lambda pp: 2 * (pp - 1)) if early else (lambda pp: 2 * pp)
    xlabel = r"$2(p-1)\,t/N$" if early else r"$2p\,t/N$"
    # Fixed theory reference line (slope -1 in these rescaled units); no fitting for p-spin.
    tau_expr = r"$\tau_s = N/2(p-1)$" if early else r"$\tau_s = N/2p$"

    series = []
    for p in sorted(PSPIN):
        n = PSPIN[p]
        t, y = _trim_anchor(curves["pspin"][p]["anchors"], ANCHOR_IDX[frac])
        label = rf"$p={p}$" if n == 500 else rf"$p={p}$ ($N={n}$)"
        series.append((label, t * factor(p) / n, y, P_COLORS[p]))
    return series, PSPIN_XMAX, xlabel, tau_expr


def main(refresh=False):
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    curves = load_curves(refresh=refresh)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.6))
    fig.subplots_adjust(wspace=0.28, hspace=0.45)
    for ax, label in zip(axes.flat, ("A", "B", "C", "D")):
        apply_axis_style(ax, label)

    # Top row: NK. Bottom row: p-spin.
    for col, frac in enumerate(PANEL_FRACS):
        nk_series, nk_xmax = _nk_series(curves, frac)
        # NK: fixed (not fitted) timescale tau_s = NK_TAU_CONST * N/K, i.e. log rho = -t/tau_s =
        # -a x with a = 1/NK_TAU_CONST and x = tK/N.
        a = 1.0 / NK_TAU_CONST[frac]
        plot_panel(axes[0, col], nk_series, a,
                   title=rf"$t_0={100 * frac:g}\%$",
                   xlabel=r"$t\,K/N$", tau_text=NK_TAU_LABEL[frac],
                   xmax=nk_xmax, show_ylabel=(col == 0), xorder=NK_XORDER[frac],
                   yorder=NK_YORDER[frac])

        # Bottom row: p-spin, no fitting -- draw the theory line (slope -1 in rescaled units).
        ps_series, ps_xmax, ps_xlabel, ps_tau = _pspin_series(curves, frac)
        plot_panel(axes[1, col], ps_series, 1.0,
                   title=rf"$t_0={100 * frac:g}\%$",
                   xlabel=ps_xlabel, tau_text=ps_tau,
                   xmax=ps_xmax, show_ylabel=(col == 0), xorder=PSPIN_XORDER,
                   yorder=PSPIN_YORDER[frac])

    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="Recompute the autocorrelation curves even if cached.")
    main(refresh=parser.parse_args().refresh)
