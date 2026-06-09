import os
import sys
import pickle

import cmasher as cmr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import blended_transform_factory
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde


NUM_REPS_EVOL = 10
NUM_REPS_FINAL = 10


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
        "legend.fontsize": 16,
    }
)

CMR_COLORS = sns.color_palette("CMRmap", 5)
PEAK4_COLORS = cmr.take_cmap_colors('cmr.emerald', 4, cmap_range=(0.3, 1.0))
PEAK3_COLORS = cmr.take_cmap_colors('cmr.emerald', 3, cmap_range=(0.3, 1.0))
PERCENTS = [0, 25, 50, 75, 100]


# ── Ridge / joy-plot panel ────────────────────────────────────────────────────
def _two_sided_kde(data, bw_method, num_points=400, min_fraction=0.02):
    """Boundary-corrected KDE on each side of 0, weighted by empirical fraction.

    Each side is estimated with the reflection trick so there is no KDE leakage
    across 0. The density values just below and just above 0 are independent,
    making any discontinuity at 0 visible in the resulting curve.
    A side is only plotted if it contains at least min_fraction of the data,
    preventing artefact bumps from a handful of near-zero outliers.
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
        # Skip x=0 when negative side already claimed it
        skip = 1 if parts_x else 0
        parts_x.append(x[skip:])
        parts_y.append(2.0 * kde.evaluate(x)[skip:] * (pos.size / n))

    if not parts_x:
        return None
    return np.concatenate(parts_x), np.concatenate(parts_y)


def ridge_plot_panel(ax, time_datasets, colors, labels, bw_method=0.4,
                     xlabel=None, title=None, overlap=0.6):
    """Draw a joy/ridge plot for DFE time evolution in a single matplotlib axes.

    t=0% row sits at the top; t=100% row sits at the bottom and is drawn in
    front (highest z-order). Baselines are drawn above all fills so they are
    always visible.
    """
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

    # t=0% (i=0) → top row (highest offset); t=100% (i=n-1) → bottom (offset=0)
    offsets = [(n - 1 - i) * step for i in range(n)]

    # Draw t=0% first (back) through t=100% last (front).
    # Within each curve: fill → outline → baseline, so the baseline is in front
    # of that curve's fill but behind the next curve's fill.
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

    # Labels on top of everything
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


# ── KDE line plot (used for final-DFE panels) ─────────────────────────────────

def plot_kde(ax, samples, color, label, bw_method, offset=0.0,
             num_points=400, reflect_negative=False):
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size < 2:
        return

    if reflect_negative:
        samples = samples[samples <= 0]
        if samples.size < 2:
            return
        mirrored = np.concatenate([samples, -samples])
        if np.allclose(mirrored.min(), mirrored.max()):
            return
        kde = gaussian_kde(mirrored, bw_method=bw_method)
        x_grid = np.linspace(samples.min(), 0.0, num_points)
        y_grid = 2.0 * kde.evaluate(x_grid)
    else:
        if np.allclose(samples.min(), samples.max()):
            return
        kde = gaussian_kde(samples, bw_method=bw_method)
        x_grid = np.linspace(samples.min(), samples.max(), num_points)
        y_grid = kde.evaluate(x_grid)

    x_plot = np.concatenate([[x_grid[0]], x_grid, [x_grid[-1]]])
    y_plot = np.concatenate([[0.0], y_grid, [0.0]])
    ax.plot(x_plot, y_plot + offset, lw=3, color=color, label=label)


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
    # ax.set_ylabel(r"$P(\Delta, t)$")


def _set_xlim_with_epsilon(ax):
    """Set right xlim to 15% of the data range past 0."""
    xmin = ax.get_xlim()[0]
    ax.set_xlim(None, 0.15 * abs(xmin))


def fgm_final_panel(ax, final):
    for idx, (n_val, dfe) in enumerate(final.items()):
        plot_kde(ax, dfe, PEAK4_COLORS[idx % len(PEAK4_COLORS)],
                 f"$n={n_val}$", bw_method=0.3)
    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    # ax.set_ylabel(r"$P(\Delta, t=100\%)$")
    _set_xlim_with_epsilon(ax)
    ax.legend(frameon=False, loc="upper left")


def pspin_final_panel(ax, pspin_data):
    for idx, order in enumerate(sorted(pspin_data)):
        dfe = []
        for entry in pspin_data[order][:NUM_REPS_FINAL]:
            sigma = cmn.compute_sigma_from_hist(entry["init_sigma"], entry["flip_seq"])
            dfe.extend(cmn_pspin.compute_dfe(sigma, entry["J"]))
        plot_kde(ax, dfe, PEAK3_COLORS[idx % len(PEAK3_COLORS)],
                 f"$p={order}$", bw_method=0.4, reflect_negative=(order == 1))
    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    # ax.set_ylabel(r"$P(\Delta, t=100\%)$")
    _set_xlim_with_epsilon(ax)
    ax.legend(loc="upper left", frameon=False)


def nk_final_panel(ax, data_arr, k_values):
    for idx, k_val in enumerate(k_values):
        combined = []
        for entry in data_arr[idx][:NUM_REPS_FINAL]:
            combined.extend(entry["dfes"][-1])
        dfe_arr = np.asarray(combined, dtype=float) * 2000
        plot_kde(ax, dfe_arr, PEAK4_COLORS[idx % len(PEAK4_COLORS)],
                 f"$K={k_val}$", bw_method=0.25)
    ax.set_xlabel(r"Fitness effect $(\Delta)$")
    # ax.set_ylabel(r"$P(\Delta, t=100\%)$")
    _set_xlim_with_epsilon(ax)
    ax.legend(frameon=False, loc="upper left")


class FixedPowerFormatter(ScalarFormatter):
    """ScalarFormatter that locks the y-axis exponent to a chosen power."""
    def __init__(self, power):
        super().__init__(useMathText=True)
        self._fixed_power = power
        self.set_scientific(True)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self._fixed_power

    def _set_orderOfMagnitude(self, range_):
        self.orderOfMagnitude = self._fixed_power

    def __call__(self, x, pos=None):
        return f"{x / 10**self._fixed_power:.0f}"


def style_axis(ax):
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6, which="major")
    ax.tick_params(width=1.5, length=3, which="minor")


def main():
    print("Loading FGM data...")
    fgm_final = load_fgm_data()

    print("Loading PSPIN data...")
    pspin_data = load_pspin_data()

    print("Loading NK data...")
    nk_data_arr, nk_k_values = load_nk_data()

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.subplots_adjust(hspace=0.38, wspace=0.15)

    # A: SK P=1 evolution (ridge plot)
    pspin_p1_ridge_panel(axes[0, 0], pspin_data)

    # B, C, D: final DFEs — FGM, SK, NK
    fgm_final_panel(axes[0, 1], fgm_final)
    axes[0, 1].set_title("FGM: final DFE", fontsize=18, pad=10)

    pspin_final_panel(axes[1, 0], pspin_data)
    axes[1, 0].set_title("p-spin: final DFE", fontsize=18, pad=10)

    nk_final_panel(axes[1, 1], nk_data_arr, nk_k_values)
    axes[1, 1].set_title("NK: final DFE", fontsize=18, pad=10)

    for panel_label, ax in zip(["A", "B", "C", "D"], axes.flat):
        ax.text(-0.1, 1.05, panel_label, transform=ax.transAxes,
                fontsize=18, fontweight="bold", va="bottom", ha="left")
        if panel_label != "A":
            style_axis(ax)

    axes[1, 1].yaxis.set_major_formatter(FixedPowerFormatter(-2))

    out_dir = os.path.join("..", "figs_paper")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig4_peak_dfes.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
