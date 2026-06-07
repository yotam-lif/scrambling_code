import os
import sys
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.transforms import blended_transform_factory
from scipy.stats import gaussian_kde


NUM_REPS_EVOL = 10

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
    }
)

CMR_COLORS = sns.color_palette("CMRmap", 5)
PERCENTS = [0, 25, 50, 75, 100]


def ridge_plot_panel(ax, time_datasets, colors, labels, bw_method=0.4,
                     xlabel=None, title=None, overlap=0.6):
    """Draw a joy/ridge plot in a single matplotlib axes."""
    kdes = []
    xmin_g, xmax_g = np.inf, -np.inf
    max_y = 0.0

    for data in time_datasets:
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if data.size < 2 or np.allclose(data.min(), data.max()):
            kdes.append(None)
            continue
        kde = gaussian_kde(data, bw_method=bw_method)
        x_pts = np.linspace(data.min(), data.max(), 400)
        y_pts = kde(x_pts)
        kdes.append((x_pts, y_pts))
        xmin_g = min(xmin_g, x_pts.min())
        xmax_g = max(xmax_g, x_pts.max())
        max_y = max(max_y, y_pts.max())

    if max_y == 0.0 or not np.isfinite(xmin_g):
        return

    step = max_y * (1.0 - overlap)
    x_full = np.linspace(xmin_g, xmax_g, 600)
    trans = blended_transform_factory(ax.transAxes, ax.transData)

    for i, (kde_result, color, label) in enumerate(zip(kdes, colors, labels)):
        offset = i * step
        base_z = 10 * i

        ax.axhline(offset, color="black", lw=0.4, alpha=0.4, zorder=base_z - 1)

        if kde_result is None:
            ax.text(0.02, offset + step * 0.5, label, transform=trans,
                    ha="left", va="center", fontsize=13, color=color, fontweight="bold")
            continue

        x_k, y_k = kde_result
        y_full = np.interp(x_full, x_k, y_k, left=0.0, right=0.0)

        ax.fill_between(x_full, offset, y_full + offset,
                        color="white", zorder=base_z, lw=0)
        ax.fill_between(x_full, offset, y_full + offset,
                        color=color, alpha=0.75, zorder=base_z + 1, lw=0)
        ax.plot(x_full, y_full + offset, color="black", lw=0.8, zorder=base_z + 2)

        ax.text(0.02, offset + step * 0.5, label, transform=trans,
                ha="left", va="center", fontsize=13, color=color, fontweight="bold")

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


def extract_fgm_ridge_data(reps):
    combined = [[] for _ in PERCENTS]
    for rep in reps[:NUM_REPS_EVOL]:
        if not isinstance(rep, dict):
            continue
        walk_length = len(rep["dfes"])
        for idx, pct in enumerate(PERCENTS):
            t_idx = int(pct * (walk_length - 1) / 100)
            combined[idx].extend(rep["dfes"][t_idx])
    return combined


def extract_pspin_ridge_data(data):
    num_repeats = min(len(data), NUM_REPS_EVOL)
    combined = [[] for _ in PERCENTS]
    for repeat in range(num_repeats):
        entry = data[repeat]
        flip_seq = entry["flip_seq"]
        ts = [int((len(flip_seq) - 1) * pct / 100) for pct in PERCENTS]
        sigma_list = cmn.curate_sigma_list(entry["init_sigma"], flip_seq, ts)
        for idx, sigma in enumerate(sigma_list):
            combined[idx].extend(cmn_pspin.compute_dfe(sigma, entry["J"]))
    return combined


def extract_nk_ridge_data(nk_data):
    combined = [[] for _ in PERCENTS]
    for repeat in nk_data[:NUM_REPS_EVOL]:
        flip_seq = repeat["flip_seq"]
        num_flips = len(flip_seq)
        ts = [int(pct * num_flips / 100) for pct in PERCENTS]
        for idx, t_idx in enumerate(ts):
            combined[idx].extend(repeat["dfes"][t_idx])
    return [np.asarray(d, dtype=float) * 2000 for d in combined]


def load_fgm_reps():
    for path in [
        "../data/FGM/fgm_rps1000_n4_sig0.05_m2000.pkl",
        "../data/FGM/fgm_rps1000_n4_sig0.05.pkl",
    ]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return []


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


def load_nk_single_k():
    path = "../data/NK/N_2000_K_8_repeats_100.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return []


def main():
    print("Loading FGM data...")
    fgm_reps = load_fgm_reps()

    print("Loading PSPIN data...")
    pspin_data = load_pspin_data()

    print("Loading NK data...")
    nk_data = load_nk_single_k()

    labels = [f"$t={p}\\%$" for p in PERCENTS]

    fgm_datasets = extract_fgm_ridge_data(fgm_reps)
    pspin_datasets = extract_pspin_ridge_data(pspin_data[2])
    nk_datasets = extract_nk_ridge_data(nk_data)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(wspace=0.3)

    ridge_plot_panel(axes[0], fgm_datasets, CMR_COLORS, labels,
                     bw_method=0.5, xlabel=r"Fitness effect $(\Delta)$", title="FGM")
    ridge_plot_panel(axes[1], pspin_datasets, CMR_COLORS, labels,
                     bw_method=0.4, xlabel=r"Fitness effect $(\Delta)$", title="SK")
    ridge_plot_panel(axes[2], nk_datasets, CMR_COLORS, labels,
                     bw_method=0.5, xlabel=r"Fitness effect $(\Delta)$", title="NK")

    for panel_label, ax in zip(["A", "B", "C"], axes):
        ax.text(-0.1, 1.05, panel_label, transform=ax.transAxes,
                fontsize=18, fontweight="bold", va="bottom", ha="left")

    out_dir = os.path.join("..", "figs_paper")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "figS13_dfe_dynamics.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()