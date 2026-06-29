import os
import sys
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import seaborn as sns
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


def _two_sided_kde(data, bw_method, num_points=400, min_fraction=0.02):
    """Boundary-corrected KDE on each side of 0, weighted by empirical fraction.

    A side is only plotted if it contains at least min_fraction of the data,
    preventing artefact bumps from a handful of near-zero outliers. For data
    that is entirely <=0 (the t=100% slice) this reduces to mirroring the
    negative values around zero and plotting only the negative side.
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


def waterfall_plot_panel(ax, time_datasets, colors, time_values,
                         bw_method=0.4, elev=15, azim=-75, title=None):
    """Draw a 3D waterfall of the DFE time evolution on a 3d matplotlib axes.

    Each time slice is placed at its own depth (y = time percentage) so the
    slices recede into the page: t=0% sits at the back, t=100% is drawn in
    front nearest the viewer. Every slice uses a plain KDE except the final
    t=100% slice. There the population sits at the peak, every fitness effect
    is <=0, and we use the same boundary-corrected reflect-negative KDE as the
    final (t=100%) slice of fig4 panel A: the negative values are mirrored
    around Δ=0, a KDE is fit to the symmetrised sample, and only the negative
    side is drawn, giving a clean vertical cut at Δ=0.
    """
    ax.computed_zorder = False

    kdes = []
    xmin_g, xmax_g = np.inf, -np.inf
    max_y = 0.0

    last_idx = len(time_datasets) - 1
    for i, data in enumerate(time_datasets):
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if data.size < 2 or np.allclose(data.min(), data.max()):
            kdes.append(None)
            continue
        if i == last_idx:
            # t=100%: same treatment as fig4 panel A's final slice.
            result = _two_sided_kde(data, bw_method)
            if result is None:
                kdes.append(None)
                continue
            x_pts, y_pts = result
        else:
            kde = gaussian_kde(data, bw_method=bw_method)
            x_pts = np.linspace(data.min(), data.max(), 400)
            y_pts = kde.evaluate(x_pts)
        kdes.append((x_pts, y_pts))
        xmin_g = min(xmin_g, x_pts.min())
        xmax_g = max(xmax_g, x_pts.max())
        max_y = max(max_y, y_pts.max())

    if max_y == 0.0 or not np.isfinite(xmin_g):
        return

    x_full = np.linspace(xmin_g, xmax_g, 600)
    for i, (kde_result, color) in enumerate(zip(kdes, colors)):
        y_t = time_values[i]
        if kde_result is None:
            ax.plot([xmin_g, xmax_g], [y_t, y_t], [0.0, 0.0],
                    color=color, lw=3, zorder=i)
            continue
        x_k, y_k = kde_result
        y_full = np.interp(x_full, x_k, y_k, left=0.0, right=0.0)
        verts = [(x_full[0], y_t, 0.0)]
        verts += list(zip(x_full, np.full_like(x_full, y_t), y_full))
        verts += [(x_full[-1], y_t, 0.0)]
        fill = Poly3DCollection([verts], facecolors=[(*mpl.colors.to_rgb(color), 0.30)],
                                edgecolors="none")
        fill.set_zorder(i - 0.1)
        ax.add_collection3d(fill)
        ax.plot(x_full, np.full_like(x_full, y_t), y_full,
                color=color, lw=3, zorder=i)

    ax.set_xlim(xmin_g, xmax_g)
    ax.set_ylim(time_values[-1], time_values[0])  # t=100% toward the viewer
    ax.set_zlim(0.0, None)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(None, zoom=1.4)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_yticks(list(time_values))
    ax.set_yticklabels([])
    ax.set_zticks([])
    ax.tick_params(axis="both", labelsize=16, pad=0.01)
    ax.tick_params(width=3, length=10, which="major")
    ax.tick_params(width=3, length=5, which="minor")
    # Offset the time labels just past the right edge of the data, scaled to
    # the panel's own fitness-effect range (an absolute offset would overflow
    # into the neighbouring panel for the small-range models, e.g. FGM).
    dx = 0.03 * (xmax_g - xmin_g)
    for v in time_values:
        ax.text(xmax_g + dx, v, 0.0, f" {v}%", fontsize=16, ha="left", va="center")
    ax.set_xlabel(r"Fitness effect $(\Delta)$", labelpad=2)
    ax.set_ylabel(r"$t$", labelpad=7)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis.line.set_linewidth(1.5)
    ax.grid(False)
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


def _content_bbox(fig, pad=0.1):
    """Tight crop box (in inches) computed from the rendered pixels.

    `bbox_inches="tight"` crops a 3d axes at its subplot-cell edge and ignores
    `ax.text` artists placed in data coordinates, so the rightmost panel's time
    labels get clipped. Measuring the non-white bounding box of the drawn canvas
    instead captures every visible mark (labels included), regardless of artist
    type, and is converted back to an inch-space Bbox for savefig.
    """
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    h, w = buf.shape[:2]
    nonwhite = (buf[:, :, :3] < 250).any(axis=2)
    rows = np.where(nonwhite.any(axis=1))[0]
    cols = np.where(nonwhite.any(axis=0))[0]
    dpi = fig.dpi
    x0, x1 = cols.min() / dpi, cols.max() / dpi
    # canvas rows run top→bottom; inch-space y runs bottom→top.
    y0, y1 = (h - rows.max()) / dpi, (h - rows.min()) / dpi
    return Bbox([[x0 - pad, y0 - pad], [x1 + pad, y1 + pad]])


def main():
    print("Loading FGM data...")
    fgm_reps = load_fgm_reps()

    print("Loading PSPIN data...")
    pspin_data = load_pspin_data()

    print("Loading NK data...")
    nk_data = load_nk_single_k()

    fgm_datasets = extract_fgm_ridge_data(fgm_reps)
    pspin_datasets = extract_pspin_ridge_data(pspin_data[2])
    nk_datasets = extract_nk_ridge_data(nk_data)

    fig = plt.figure(figsize=(24, 7))
    fig.subplots_adjust(wspace=0.30, left=0.02, right=0.78)
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    waterfall_plot_panel(ax1, fgm_datasets, CMR_COLORS, PERCENTS,
                         bw_method=0.5, title="FGM")
    waterfall_plot_panel(ax2, pspin_datasets, CMR_COLORS, PERCENTS,
                         bw_method=0.4, title="p-spin")
    waterfall_plot_panel(ax3, nk_datasets, CMR_COLORS, PERCENTS,
                         bw_method=0.5, title="NK")

    label_kw = dict(fontsize=18, fontweight="bold", va="bottom", ha="left")
    for panel_label, ax in zip(["A", "B", "C"], [ax1, ax2, ax3]):
        ax.text2D(-0.05, 1.05, panel_label, transform=ax.transAxes, **label_kw)

    out_dir = os.path.join("..", "figs_paper")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "figS13_dfe_dynamics.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches=_content_bbox(fig, pad=0.1))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()