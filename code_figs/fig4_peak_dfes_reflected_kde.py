import os
import sys
import pickle

import cmasher as cmr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde


NUM_REPS_EVOL = 10
NUM_REPS_FINAL = 10

# Log-log histogram defaults for panels B-D.
NUM_LOG_BINS = 22
U_MIN_QUANTILE = 0.005      # drop the lowest 0.5% of |Δ| (sample noise / ties at 0)
MIN_COUNTS_PER_BIN = 5

# Fit model:  p(u) = p_0 + C * u^theta,  with u = -Δ > 0.
# p_0 lets the density approach a finite nonzero value at the boundary;
# theta is the edge exponent of the *correction* to that plateau.
FIT_U_MAX_QUANTILE = 0.5    # fit only on the lower half of u (near-boundary regime)
PRINT_EDGE_FIT = True
SHOW_FIT_OVERLAY = True

# A parameter is treated as "not significantly different from 0" — i.e. pinned
# to the bound, so its reported error is from a singular Hessian — when the
# fitted value is below this many times its standard error.
SIG_THRESHOLD = 3.0

# u_max quantiles for the diagnostic scan. Useful for spotting non-asymptotic
# regimes: if theta drifts as u_max shrinks the data isn't in a clean
# p_0 + C u^θ window.
SCAN_U_MAX_QUANTILES = (0.05, 0.10, 0.20, 0.30, 0.50)
PRINT_SCAN = True


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)
for path in (SCRIPT_DIR, REPO_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from cmn import cmn, cmn_pspin


plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update(
    {
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
    }
)

CMR_COLORS = sns.color_palette("CMRmap", 5)
PEAK4_COLORS = cmr.take_cmap_colors("cmr.emerald", 4, cmap_range=(0.3, 1.0))
PEAK3_COLORS = cmr.take_cmap_colors("cmr.emerald", 3, cmap_range=(0.3, 1.0))
PERCENTS = [0, 25, 50, 75, 100]


# ── Ridge / joy-plot panel (panel A — unchanged) ─────────────────────────────
def _two_sided_kde(data, bw_method, num_points=400, min_fraction=0.02):
    """Boundary-corrected KDE on each side of 0, weighted by empirical fraction.

    Each side is estimated with the reflection trick so there is no KDE leakage
    across 0. The density values just below and just above 0 are independent,
    making any discontinuity at 0 visible in the resulting curve.
    """
    neg = data[data <= 0]
    pos = data[data > 0]
    n = len(data)
    parts_x, parts_y = [], []

    if neg.size / n >= min_fraction and neg.size >= 2 and not np.allclose(neg.min(), neg.max()):
        kde = gaussian_kde(np.concatenate([neg, -neg]), bw_method=bw_method)
        x = np.linspace(neg.min(), 0.0, num_points // 2)
        parts_x.append(x)
        parts_y.append(2.0 * kde.evaluate(x) * (neg.size / n))

    if pos.size / n >= min_fraction and pos.size >= 2 and not np.allclose(pos.min(), pos.max()):
        kde = gaussian_kde(np.concatenate([pos, -pos]), bw_method=bw_method)
        x = np.linspace(0.0, pos.max(), num_points // 2)
        skip = 1 if parts_x else 0
        parts_x.append(x[skip:])
        parts_y.append(2.0 * kde.evaluate(x)[skip:] * (pos.size / n))

    if not parts_x:
        return None
    return np.concatenate(parts_x), np.concatenate(parts_y)


def ridge_plot_panel(ax, time_datasets, colors, labels, bw_method=0.4,
                     xlabel=None, title=None, overlap=0.6):
    kdes = []
    xmin_g, xmax_g = np.inf, -np.inf
    max_y = 0.0

    for i, data in enumerate(time_datasets):
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if data.size < 2 or np.allclose(data.min(), data.max()):
            kdes.append(None)
            continue
        if i == 0:
            kde = gaussian_kde(data, bw_method=bw_method)
            x_pts = np.linspace(data.min(), data.max(), 400)
            y_pts = kde.evaluate(x_pts)
        else:
            result = _two_sided_kde(data, bw_method)
            if result is None:
                kdes.append(None)
                continue
            x_pts, y_pts = result
        kdes.append((x_pts, y_pts))
        xmin_g = min(xmin_g, x_pts.min())
        xmax_g = max(xmax_g, x_pts.max())
        max_y = max(max_y, y_pts.max())

    if max_y == 0.0 or not np.isfinite(xmin_g):
        return

    n = len(kdes)
    step = max_y * (1.0 - overlap)
    x_full = np.linspace(xmin_g, xmax_g, 600)
    trans = blended_transform_factory(ax.transAxes, ax.transData)

    offsets = [(n - 1 - i) * step for i in range(n)]

    for i, (kde_result, color) in enumerate(zip(kdes, colors)):
        offset = offsets[i]
        base_z = 10 * i
        if kde_result is None:
            ax.axhline(offset, color="black", lw=0.8, zorder=base_z + 3)
            continue
        x_k, y_k = kde_result
        y_full = np.interp(x_full, x_k, y_k, left=0.0, right=0.0)
        ax.fill_between(x_full, offset, y_full + offset,
                        color="white", zorder=base_z, lw=0)
        ax.fill_between(x_full, offset, y_full + offset,
                        color=color, alpha=0.75, zorder=base_z + 1, lw=0)
        ax.plot(x_full, y_full + offset, color="black", lw=0.8, zorder=base_z + 2)
        ax.axhline(offset, color="black", lw=0.8, zorder=base_z + 3)

    label_z = 10 * n + 5
    for i, (color, label) in enumerate(zip(colors, labels)):
        ax.text(0.02, offsets[i] + step * 0.45, label, transform=trans,
                ha="left", va="center", fontsize=13, color=color,
                fontweight="bold", zorder=label_z)

    ax.set_xlim(xmin_g, xmax_g)
    ax.set_yticks([])
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6, which="major")
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontsize=18, pad=10)


# ── Log-binned density on u = -Δ ──────────────────────────────────────────────
def edge_distances(samples, edge=0.0):
    """u = edge - x for x < edge, then drop u <= 0."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    u = edge - samples[samples < edge]
    return u[u > 0]


def log_binned_pdf(u, num_bins=NUM_LOG_BINS, u_min_quantile=U_MIN_QUANTILE,
                   min_counts=MIN_COUNTS_PER_BIN):
    """Density estimate on log-spaced bins.

    Returns (centers, density, counts) with bins below `min_counts` removed.
    Density is counts / bin_width / total_sample_size so it integrates to 1
    over the original u support.
    """
    u = np.asarray(u, dtype=float)
    u = u[(u > 0) & np.isfinite(u)]
    if u.size < 20:
        return None

    u_min = max(np.quantile(u, u_min_quantile), 1e-300)
    u_max = u.max()
    if u_min >= u_max:
        return None

    bins = np.logspace(np.log10(u_min), np.log10(u_max), num_bins + 1)
    counts, edges = np.histogram(u, bins=bins)
    widths = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    density = counts / widths / u.size

    mask = counts >= min_counts
    if mask.sum() < 3:
        return None
    return centers[mask], density[mask], counts[mask]


# ── Fit p(u) = p_0 + C * u^theta ──────────────────────────────────────────────
def fit_p0_plus_power(u, u_max_quantile=FIT_U_MAX_QUANTILE):
    """Fit p(u) = p_0 + C * u^theta on the near-boundary part of the data.

    p_0   — plateau value at the boundary (allows finite P(Δ=0^-)).
    C, θ  — power-law correction away from the plateau.

    Fit is done in log-density space using a log-binned histogram so each
    decade of u contributes comparably (otherwise the abundant large-u bins
    dominate and the near-boundary signal is lost).
    """
    u = np.asarray(u, dtype=float)
    u = u[(u > 0) & np.isfinite(u)]
    if u.size < 50:
        return None

    u_max_fit = np.quantile(u, u_max_quantile)
    u_fit = u[u <= u_max_fit]
    if u_fit.size < 50:
        return None

    res = log_binned_pdf(u_fit)
    if res is None:
        return None
    centers, density, _ = res
    if centers.size < 5:
        return None

    # log-space model so fit weights decades equally
    def log_model(u_arr, p0, C, theta):
        return np.log(np.maximum(p0 + C * u_arr ** theta, 1e-300))

    head = max(3, density.size // 4)
    tail = max(3, density.size // 4)
    p0_init = float(density[:head].mean())
    # rough θ from low/high slope
    if centers[-tail:].mean() > centers[:head].mean() and density[-tail:].mean() > p0_init:
        theta_init = (
            np.log(density[-tail:].mean() - 0.5 * p0_init)
            - np.log(max(density[head:2 * head].mean() - 0.5 * p0_init, 1e-12))
        ) / (np.log(centers[-tail:].mean()) - np.log(centers[head:2 * head].mean() + 1e-300))
        theta_init = float(np.clip(theta_init, 0.1, 5.0))
    else:
        theta_init = 1.0
    C_init = max(
        (density[-tail:].mean() - p0_init) / centers[-1] ** theta_init, 1e-12
    )

    try:
        popt, pcov = curve_fit(
            log_model, centers, np.log(density),
            p0=[p0_init, C_init, theta_init],
            bounds=([0.0, 0.0, 0.01], [np.inf, np.inf, 20.0]),
            maxfev=20000,
        )
    except Exception:
        return None

    perr = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
    return {
        "p0": float(popt[0]), "C": float(popt[1]), "theta": float(popt[2]),
        "p0_err": float(perr[0]), "C_err": float(perr[1]), "theta_err": float(perr[2]),
        "u_max_fit": float(u_max_fit),
        "u_min_fit": float(centers[0]),
        "n_in_fit": int(u_fit.size),
    }


def _format_fit_row(fit, sig_threshold=SIG_THRESHOLD):
    """Render a fit dict as three lines with degeneracy annotations.

    `curve_fit` returns covariances from a Hessian that goes singular when a
    parameter is pinned to a bound, so the reported errors are not meaningful
    there. The flags below tell the reader which entries are reliable.
    """
    p0_pinned = fit["p0"] < sig_threshold * max(fit["p0_err"], 1e-300)
    C_pinned = fit["C"] < sig_threshold * max(fit["C_err"], 1e-300)

    p0_tag = "  [pinned to 0; no plateau detected]" if p0_pinned else ""
    if C_pinned:
        C_tag = "  [pinned to 0; no power-law correction]"
        theta_tag = "  [undetermined: C ≈ 0]"
    else:
        C_tag = ""
        theta_tag = ""

    return (
        f"  p_0    = {fit['p0']:.4g} ± {fit['p0_err']:.2g}{p0_tag}\n"
        f"  C      = {fit['C']:.4g} ± {fit['C_err']:.2g}{C_tag}\n"
        f"  theta  = {fit['theta']:.4f} ± {fit['theta_err']:.4f}{theta_tag}"
    )


def print_edge_fit(name, samples):
    u = edge_distances(samples)
    fit = fit_p0_plus_power(u)
    if fit is None:
        print(f"\n{name}: p_0 + C u^θ fit failed")
        return
    print(
        f"\n{name}:  p(u) = p_0 + C * u^θ   "
        f"(u in [{fit['u_min_fit']:.3g}, {fit['u_max_fit']:.3g}], n={fit['n_in_fit']})\n"
        f"{_format_fit_row(fit)}"
    )

    if PRINT_SCAN:
        print(f"  scan over u_max quantiles "
              f"(drift in θ ⇒ non-asymptotic regime):")
        print(f"    {'q_max':>6}  {'u_max':>10}  {'p_0':>10}  "
              f"{'C':>10}  {'theta':>10}")
        for q in SCAN_U_MAX_QUANTILES:
            scan_fit = fit_p0_plus_power(u, u_max_quantile=q)
            if scan_fit is None:
                print(f"    {q:>6.2f}  {'-':>10}  {'-':>10}  {'-':>10}  {'-':>10}")
                continue
            print(
                f"    {q:>6.2f}  {scan_fit['u_max_fit']:>10.3g}  "
                f"{scan_fit['p0']:>10.3g}  {scan_fit['C']:>10.3g}  "
                f"{scan_fit['theta']:>10.3f}"
            )


def plot_log_log_density(ax, samples, color, label,
                         show_fit=SHOW_FIT_OVERLAY, num_bins=NUM_LOG_BINS):
    """Plot log-binned PDF of u = -Δ on the current (log-log) axes."""
    u = edge_distances(samples)
    res = log_binned_pdf(u, num_bins=num_bins)
    if res is None:
        return None

    centers, density, _ = res
    ax.plot(centers, density, marker="o", color=color, label=label,
            lw=2, ms=6, mfc=color, mec=color)

    fit_info = None
    if show_fit:
        fit_info = fit_p0_plus_power(u)
        if fit_info is not None:
            u_grid = np.logspace(
                np.log10(max(centers[0], 1e-300)),
                np.log10(fit_info["u_max_fit"]),
                200,
            )
            y_fit = fit_info["p0"] + fit_info["C"] * u_grid ** fit_info["theta"]
            ax.plot(u_grid, y_fit, color=color, ls="--", lw=1.5, alpha=0.85)
    return fit_info


# ── Data loaders ──────────────────────────────────────────────────────────────
def load_fgm_data():
    fgm_ns = [4, 8, 16, 32]
    fgm_data = {}
    for n_val in fgm_ns:
        for path in [
            f"../data/FGM/fgm_rps1000_n{n_val}_sig0.05_m2000.pkl",
            f"../data/FGM/fgm_rps1000_n{n_val}_sig0.05.pkl",
        ]:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    fgm_data[n_val] = pickle.load(f)
                break
        else:
            fgm_data[n_val] = []

    final = {}
    for n_val, rep_list in fgm_data.items():
        all_last = []
        for rep in rep_list[:NUM_REPS_FINAL]:
            if not isinstance(rep, dict):
                continue
            dfes = rep.get("dfes")
            if dfes:
                all_last.extend(dfes[-1])
        final[n_val] = all_last
    return final


def load_pspin_data():
    file_paths = {
        1: "../data/PSPIN/N400_P1_pure_repeats10.pkl",
        2: "../data/PSPIN/N400_P2_pure_repeats10.pkl",
        3: "../data/PSPIN/N400_P3_pure_repeats10.pkl",
    }
    pspin_data = {}
    for order, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"PSPIN data file not found: {path}")
        with open(path, "rb") as f:
            pspin_data[order] = pickle.load(f)
    return pspin_data


def load_nk_data():
    res_directory = "../data/NK"
    k_values = [4, 8, 16, 32]
    data_arr = []
    for k in k_values:
        path = os.path.join(res_directory, f"N_2000_K_{k}_repeats_100.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                data_arr.append(pickle.load(f))
        else:
            data_arr.append([])
    return data_arr, k_values


# ── Panel builders ────────────────────────────────────────────────────────────
def pspin_p1_ridge_panel(ax, pspin_data):
    num_repeats = min(len(pspin_data[1]), NUM_REPS_EVOL)
    combined = [[] for _ in PERCENTS]
    for repeat in range(num_repeats):
        entry = pspin_data[1][repeat]
        flip_seq = entry["flip_seq"]
        ts = [int((len(flip_seq) - 1) * pct / 100) for pct in PERCENTS]
        sigma_list = cmn.curate_sigma_list(entry["init_sigma"], flip_seq, ts)
        for idx, sigma in enumerate(sigma_list):
            combined[idx].extend(cmn_pspin.compute_dfe(sigma, entry["J"]))

    labels = [f"$t={p}\\%$" for p in PERCENTS]
    ridge_plot_panel(ax, combined, CMR_COLORS, labels,
                     bw_method=0.4, xlabel=r"Fitness effect $(\Delta)$",
                     title="Additive model: DFE evolution")


def _finalize_log_log_panel(ax, title, legend_loc="lower right"):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|\Delta|$")
    ax.set_ylabel(r"$P(|\Delta|)$")
    ax.set_title(title, fontsize=18, pad=10)
    ax.legend(frameon=False, loc=legend_loc)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6, which="major")
    ax.tick_params(width=1.5, length=3, which="minor")
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.5)


def fgm_final_panel(ax, final):
    for idx, (n_val, dfe) in enumerate(final.items()):
        plot_log_log_density(
            ax, dfe, PEAK4_COLORS[idx % len(PEAK4_COLORS)], f"$n={n_val}$"
        )
        if PRINT_EDGE_FIT:
            print_edge_fit(f"FGM n={n_val}", dfe)
    _finalize_log_log_panel(ax, "FGM: final DFE")


def pspin_final_panel(ax, pspin_data):
    for idx, order in enumerate(sorted(pspin_data)):
        dfe = []
        for entry in pspin_data[order][:NUM_REPS_FINAL]:
            sigma = cmn.compute_sigma_from_hist(entry["init_sigma"], entry["flip_seq"])
            dfe.extend(cmn_pspin.compute_dfe(sigma, entry["J"]))

        plot_log_log_density(
            ax, dfe, PEAK3_COLORS[idx % len(PEAK3_COLORS)], f"$p={order}$"
        )
        if PRINT_EDGE_FIT:
            print_edge_fit(f"p-spin p={order}", dfe)
    _finalize_log_log_panel(ax, "p-spin: final DFE")


def nk_final_panel(ax, data_arr, k_values):
    for idx, k_val in enumerate(k_values):
        combined = []
        for entry in data_arr[idx][:NUM_REPS_FINAL]:
            combined.extend(entry["dfes"][-1])
        dfe_arr = np.asarray(combined, dtype=float) * 2000

        plot_log_log_density(
            ax, dfe_arr, PEAK4_COLORS[idx % len(PEAK4_COLORS)], f"$K={k_val}$"
        )
        if PRINT_EDGE_FIT:
            print_edge_fit(f"NK K={k_val}", dfe_arr)
    _finalize_log_log_panel(ax, "NK: final DFE")


def main():
    print("Loading FGM data...")
    fgm_final = load_fgm_data()

    print("Loading PSPIN data...")
    pspin_data = load_pspin_data()

    print("Loading NK data...")
    nk_data_arr, nk_k_values = load_nk_data()

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.subplots_adjust(hspace=0.38, wspace=0.22)

    # A: ridge plot — unchanged from the linear-scale figure.
    pspin_p1_ridge_panel(axes[0, 0], pspin_data)

    # B, C, D: log-binned PDFs of |Δ| on log-log axes, with a
    # p(u) = p_0 + C·u^θ fit overlaid on the near-boundary range.
    fgm_final_panel(axes[0, 1], fgm_final)
    pspin_final_panel(axes[1, 0], pspin_data)
    nk_final_panel(axes[1, 1], nk_data_arr, nk_k_values)

    for panel_label, ax in zip(["A", "B", "C", "D"], axes.flat):
        ax.text(-0.1, 1.05, panel_label, transform=ax.transAxes,
                fontsize=18, fontweight="bold", va="bottom", ha="left")

    out_dir = os.path.join("..", "figs_paper")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig4_peak_dfes.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
