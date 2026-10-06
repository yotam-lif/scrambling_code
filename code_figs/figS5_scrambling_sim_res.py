r"""Figure S5: simulated scrambling -- fig2 row 1 for the SK, NK and Fisher landscapes.

One row per model -- SK (A-C), NK (D-F), FGM (G-I) -- each drawn by the same code as fig2 row 1
(``cmn/cmn_plots.py``).  There are no panel titles; the caption names the rows.

    left    fate bars: what the mutations beneficial at one time are at the other, as
            deleterious / beneficial there.  See NO NEUTRAL CLASS below.
    middle  forward: the ancestral beneficial DFE in front, where those mutations land in the
            evolved background on the raised back layer, and the full evolved DFE as a line.
    right   backward: the evolved beneficial DFE, where those mutations sat in the ancestral
            background, and the full ancestral
            DFE.

NO NEUTRAL CLASS.  Fig2 calls an effect neutral when |s| <= 0.015, Couce et al.'s own cut, set
by what their assay can resolve.  The simulations have no measurement noise, so that rationale
has no analogue here, and any band would be a free choice.  The bars therefore split at zero:
beneficial above it, deleterious below, and the legend names only those two classes.  No
simulated effect is exactly zero.

Run from anywhere:  python code_figs/figS5_scrambling_sim_res.py
Output:             figs_paper/figS5_scrambling_sim_res.pdf
"""

import os
import pickle
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn.cmn import compute_sigma_from_hist  # noqa: E402
from cmn.cmn_fgm import Fisher  # noqa: E402
from cmn.cmn_plots import (  # noqa: E402  (shared with fig2 row 1)
    FATE_CLASSES, create_fate_bars, create_overlapping_dfes, plain_histogram,
    style_dfe_axes,
)
from cmn import cmn_pspin, cmn_scatter  # noqa: E402


# Every DFE is plotted as the raw (absolute) fitness effect ΔF, NOT as a selection
# coefficient -- no effect is divided by the fitness at that point in the walk. Per model:
#   * FGM   : ΔF_i = w(r + delta_i) - w(r), taken straight from the live SSWM walk
#             (dfes[k] is the DFE at phenotype traj[k]).
#   * p-spin: ΔF_i for flipping each spin i, recomputed from the stored landscape J
#             via cmn_pspin.compute_dfe(sigma, J).
#   * NK    : the landscape is NOT stored, only the precomputed ΔF DFEs. These pickles
#             predate cmn_nk's switch to an extensive fitness, so they hold INTENSIVE
#             effects (fitness = mean of f_i over loci), ~N times smaller than the
#             p-spin's extensive sum over interactions. They are multiplied by N to put
#             both models on the same scale -- see the note at the NK data block.


def auto_xlim(dfe1, dfe2, q=99.0):
    """Pick a symmetric x-limit from the beneficial (positive) effects of both DFEs.

    Uses the ``q``-th percentile so a single outlier does not blow up the box.
    Returns a positive float; falls back to 1.0 if there are no positive effects.
    """
    vals = np.concatenate([np.asarray(dfe1, float), np.asarray(dfe2, float)])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 1.0
    return float(np.percentile(vals, q))


# FGM parameters
FGM_N = 4
FGM_SIGMA = 0.05
FGM_M = 4 * 10 ** 3
FGM_RANDOM_STATE = 1
FGM_T1 = 0.8
FGM_T2 = 0.9
FGM_XLIM = 0.08  # x-axis half-width of the fitness-effect (ΔF) box
FGM_AXIS_ORDER = -2  # power of ten pulled out of the x tick labels

# p-spin parameters
SK_FILE = "N2000_P2_pure_repeats10.pkl"
SK_ENTRY = 1
SK_T1 = 0.05
SK_T2 = 0.5
SK_XLIM = 12  # x-axis half-width of the fitness-effect (ΔF) box
SK_AXIS_ORDER = 0

# NK parameters
NK_FILE = "N_2000_K_32_repeats_100.pkl"
NK_ENTRY = 2
NK_T1 = 0.05
NK_T2 = 0.5
NK_XLIM = 40  # x-axis half-width of the fitness-effect (ΔF) box, after the xN rescale
NK_AXIS_ORDER = 0

# Output parameters
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figs_paper")
OUTPUT_FILE = "figS5_scrambling_sim_res.pdf"


# Histogram bins per curve: the simulated DFEs hold ~2000 effects and their beneficial
# subsets a few hundred, so fewer bins than fig2's tens of thousands of Couce segments.
SIM_BINS = {"forward_subset": 15, "forward_anchor": 8, "forward_backdrop": 15,
            "backward_subset": 10, "backward_anchor": 10, "backward_backdrop": 15}
# No measurement noise, so every curve is a plain density histogram.
SIM_HISTOGRAMS = {role: plain_histogram for role in SIM_BINS}
# The two walk snapshots (T1, T2 above) are the ancestral and evolved backgrounds, named as
# in fig2.
BACKGROUND_LABELS = ("anc.", "evo.")


def fate_caption(forward):
    """The fate bar's title, worded as fig2's: where the effects are read, and selected."""
    target, source = BACKGROUND_LABELS[::-1] if forward else BACKGROUND_LABELS
    return (f"Mutation effects measured in {target} background,\n"
            f"conditioned on being beneficial in {source} background")


cmn_scatter.apply_style()

# fig2's row-1 layout, three rows deep: explicit spacer rows rather than hspace, and the fate
# bars widened into the free left margin.
fig = plt.figure(figsize=(18, 16.5))
gs = GridSpec(5, 3, figure=fig, wspace=0.36, hspace=0.0,
              height_ratios=(1.0, 0.28, 1.0, 0.28, 1.0))
axes = np.array([[fig.add_subplot(gs[2 * row, col]) for col in range(3)]
                 for row in range(3)])
panel_positions = [ax.get_position().frozen() for ax in axes.ravel()]
for row in range(3):
    fate_position = axes[row, 0].get_position()
    extra_width = 0.055
    axes[row, 0].set_position([fate_position.x0 - extra_width, fate_position.y0,
                               fate_position.width + extra_width, fate_position.height])

# FGM simulation
fgm = Fisher(n=FGM_N, sigma=FGM_SIGMA, m=FGM_M, random_state=FGM_RANDOM_STATE)
flips, traj, dfes = fgm.relax()
ind1 = int(FGM_T1 * (len(dfes) - 1))
ind2 = int(FGM_T2 * (len(dfes) - 1))
# Raw fitness effects ΔF_i = w(r + delta_i) - w(r); dfes[k] is the DFE recorded at
# phenotype traj[k] along the walk.
fgm_dfe1 = dfes[ind1]
fgm_dfe2 = dfes[ind2]

# SK data
res_directory = os.path.join(os.path.dirname(__file__), "..", "data", "sim", "PSPIN")
data_file_sk = os.path.join(res_directory, SK_FILE)
with open(data_file_sk, "rb") as f:
    data_sk = pickle.load(f)
data_entry = data_sk[SK_ENTRY]
init_sigma = data_entry["init_sigma"]
J = data_entry["J"]
flip_seq = data_entry["flip_seq"]
ind1 = int(SK_T1 * (len(flip_seq) - 1))
ind2 = int(SK_T2 * (len(flip_seq) - 1))
sig1 = compute_sigma_from_hist(init_sigma, flip_seq, t=ind1)
sig2 = compute_sigma_from_hist(init_sigma, flip_seq, t=ind2)
# Raw fitness effects ΔF_i for flipping each spin i, at the two points in the walk.
sk_dfe1 = cmn_pspin.compute_dfe(sig1, J)
sk_dfe2 = cmn_pspin.compute_dfe(sig2, J)

# NK data
res_directory = os.path.join(os.path.dirname(__file__), "..", "data", "sim", "NK")
data_file_nk = os.path.join(res_directory, NK_FILE)
with open(data_file_nk, "rb") as f:
    data_nk = pickle.load(f)
data_entry = data_nk[NK_ENTRY]
flip_seq = data_entry["flip_seq"]
dfes = data_entry["dfes"]
ind1 = int(NK_T1 * (len(flip_seq) - 1))
ind2 = int(NK_T2 * (len(flip_seq) - 1))
# The pickles under data/sim/NK/ were generated BEFORE cmn_nk switched to the extensive
# convention, so their effects are intensive (fitness = mean of f_i over loci) and ~N
# times smaller than the p-spin's. Rescale by N to match; each stored DFE holds one entry
# per locus, so N is just its length.
# IMPORTANT: cmn_nk.NK.compute_fitness is now EXTENSIVE. If data/sim/NK/ is ever regenerated
# with the current code its effects are already extensive -- DROP this * nk_n rescale, or
# they will be inflated by a further factor of N.
nk_n = len(dfes[ind1])
nk_dfe1 = np.asarray(dfes[ind1]) * nk_n
nk_dfe2 = np.asarray(dfes[ind2]) * nk_n

# One row per model: (name for the printout, ancestral DFE, evolved DFE, x half-width,
# x tick order).
ROWS = (
    ("SK", sk_dfe1, sk_dfe2, SK_XLIM, SK_AXIS_ORDER),
    ("NK", nk_dfe1, nk_dfe2, NK_XLIM, NK_AXIS_ORDER),
    ("FGM", fgm_dfe1, fgm_dfe2, FGM_XLIM, FGM_AXIS_ORDER),
)
fate_summaries = []
for row_axes, (model, dfe1, dfe2, xlim, order) in zip(axes, ROWS):
    fate_summaries.append((model, create_fate_bars(
        row_axes[0], dfe1, dfe2, labels=BACKGROUND_LABELS, neutral_band=None,
        caption=fate_caption)))
    create_overlapping_dfes(
        row_axes[1], row_axes[2], dfe1, dfe2, SIM_HISTOGRAMS,
        headroom=lambda ylim: 0.1 * ylim, xlim=xlim, bins=SIM_BINS, axis_order=order,
        xlabel=r"Fitness effect $(\Delta)$")

for index, (ax, label) in enumerate(zip(axes.ravel(), "ABCDEFGHI")):
    position = panel_positions[index]
    fig.text(position.x0 - 0.30 * position.width, 1.15, label,
             transform=mpl.transforms.blended_transform_factory(fig.transFigure, ax.transAxes),
             fontsize=18, fontweight="heavy", va="top", ha="left")
    style_dfe_axes(ax)

# Save the figure
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
fig.savefig(out_path, format="pdf", bbox_inches="tight")
plt.close(fig)

for model, summary in fate_summaries:
    for n, counts, src, dst in summary:
        print(f"{model}: beneficial at {src} (n={n}), measured at {dst}:  "
              + "  ".join(f"{name} {c} ({100 * c / n:.1f}%)"
                          for name, c in zip(FATE_CLASSES, counts) if name != "Neutral"))
print(f"Saved: {out_path}")
