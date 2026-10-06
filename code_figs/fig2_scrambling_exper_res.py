r"""Figure 2: experimental scrambling in two independent LTEE knockout panels.

Panel A shows the cross-background effects of beneficial insertions as one 100% bar
per direction.  Panels B and C show the underlying effect distributions.

Two rows, three columns.

Row 1 (A-C)  Couce Ara+2, generation 0K -> 2K.

    A  Fate of the knockouts beneficial at one timepoint, read in the other background: the
       fraction that is deleterious / neutral / beneficial there.  Top bar forward (beneficial
       at 0, read at 2K), bottom bar backward (beneficial at 2K, read at 0).  The classes are
       Couce et al.'s own: beneficial s > 0.015, the cut in their
       ``beneficials_accross_backgrounds.R``, and neutral |s| <= 0.015.  With these the
       forward fates come out 6.1 / 76.5 / 17.4 % (ben. / neu. / del.), against the paper's
       quoted 5.9 / 76.9 / 17.2 %; the small gap is their per-site de-duplication, which this
       repo does per segment (see ``cmn/cmn_exper.py``).
    B  Forward: the ancestor's beneficial DFE (grey) and where those same knockouts
       land in the evolved background (magenta), against the evolved full DFE (line).
    C  Backward: the evolved clone's beneficial DFE (magenta) and where those same
       knockouts sat in the ancestor (grey), against the ancestor's full DFE (line).

Row 2 (D-F)  Paired-effect scatters with nested ancestor-defined tail exclusions.

    Each occupied hexagon is colored by the fraction of the panel's mutants it contains,
    on a logarithmic scale specific to each panel.  Singleton bins share the lightest
    color, and each panel's most populated bin uses the darkest color.
    Each panel uses the same number of
    bins across its own axis span.  Blue rectangles enclose every retained bulk point.
    Every panel's axes start at ``SCATTER_FLOOR``; the few points below it are not drawn but
    still count toward the densities and the Pearson r.

    D  Limdi REL607 green- versus red-reference estimates.  Zero evolution, so whatever
       decorrelation it shows is measurement error, not epistasis; D calibrates E and F.
    E  Limdi REL607 -> Ara+2.
    F  Couce 0K -> 2K.

    Each reports Pearson r over every pair (r_All) and again over the pairs whose x effect
    is smallest in ABSOLUTE value (r_Bulk) -- one partition per panel.  D and E drop the
    largest p* of |s|, the half-max edge of -dr/dp of the pooled r(p) over the ten retained
    Limdi lineages, read from Table S5 (``data/exper/TableS5_limdi_half_max.csv``, row
    ``pooled``, column ``p_star``; run ``code_figs/TableS5_limdi_half_max.py`` first).
    One panel-wide cut rather than Ara+2's own edge, so D is cut at the same fraction as the
    transition it calibrates and neither rests on one lineage's noisier edge.  F drops the
    largest 3%:
    the Couce r(p) is close to a power law with no edge to locate, and its effects are compact
    enough that 10% would reach well inside the bulk.  The exclusion is defined only from the x
    (ancestor/control) measurement and never from y, so the retained subset is not
    conditioned on the outcome whose correlation is reported.  Ranking on |s| rather than on
    s keeps the retained set symmetric about zero: a knockout is dropped for being large, not
    for being deleterious.

Data conventions
----------------
Couce genes are matched on allele name after dropping duplicate fitted values and keeping
abundance > 1, as in the published panel.  Limdi genes are matched on metadata row index --
the shared gene identity across the Limdi matrices; see the block comment in
``cmn/cmn_exper.py`` for why the labelled CSV must not be used for this.

Nothing is cut from any statistic.  Row 1 shows the full measured range.  Row 2 runs from
``SCATTER_FLOOR`` up to the top of each panel's own data envelope, with x and y sharing those
limits so the identity line is the panel diagonal; F borrows E's, so the two evolved panels
read on one scale.  Points below the floor are hidden only.  The Limdi
counterpart of row 1, which needs a nonlethal cut and window-clipped histograms because
its deleterious tail is real where Couce's is not, lives in
``code_tmp/fig1_limdi_clones.py``.

Run from anywhere:  python code_figs/fig2_scrambling_exper_res.py
Output:             figs_paper/fig2_scrambling_exper_res.pdf
"""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn import cmn_exper, cmn_scatter  # noqa: E402
from cmn.cmn_plots import (  # noqa: E402  (row 1 is shared with fig S5)
    FATE_CLASSES, arrow_title, create_fate_bars, create_overlapping_dfes, style_dfe_axes,
    thresholded_histogram,
)
from cmn.cmn_scatter import (  # noqa: E402  (row 2 is shared with figs S1-S4)
    R_LABELS, SCATTER_FLOOR, envelope_limits, finish_density_panel, limdi_half_max_cut,
    print_correlations, scatter_panel,
)

# ───────────────────────────────────── Style ─────────────────────────────────────
cmn_scatter.apply_style()

# ─────────────────────────────────── Parameters ──────────────────────────────────
XLIM = 0.06                 # half-width of the plotted fitness-effect window

# Panel A classes: beneficial s > NEUTRAL_BAND, neutral |s| <= NEUTRAL_BAND, deleterious
# s < -NEUTRAL_BAND.  0.015 is Couce et al.'s own beneficial cut.
NEUTRAL_BAND = 0.015

OUT_DIR = os.path.join(_REPO_ROOT, "figs_paper")

# Row 2 cuts: the fraction of largest |x| moved from r_All to r_Bulk.  D and E use the
# pooled Limdi half-max edge p* (``cmn_scatter.limdi_half_max_cut``).
COUCE_CUT = 0.03                    # F
# Bulk-label leaders, (rise, run) in points; see ``cmn_scatter.annotate_bulk``.
# D's label runs out flatter and further left, clear of the diagonal cloud.
BULK_LEADER_D = (-10, 40)
# E's rectangle sits in the dense upper-right cloud, so its label goes up and further out.
BULK_LEADER_E = (8, 55)
# F's label goes up and left, rising as far as E's.
BULK_LEADER_F = (BULK_LEADER_E[0], 40)

# Row 1 spans +-0.06, so every tick label would read "-0.02", "0.00", ... .  Pulling a
# fixed 10^-2 out into the axis offset text turns those into "-2", "0", ... , which is
# both shorter and easier to read.
EFFECT_AXIS_ORDER = -2


def fate_caption(forward):
    """Panel A's per-bar title: which background the effects are read in, and selected in."""
    target, source = ("evo.", "anc.") if forward else ("anc.", "evo.")
    return (f"Insertion effects measured in {target} background, \n"
            f"conditioned on being beneficial in {source} background")


# ─────────────────────────────────── Data loading ─────────────────────────────────
def load_couce_pair(ancestor="0K", evolved="2K"):
    """Matched Couce segment effects for one transition.

    Segments are keyed on ``alle`` ("<ORF>-<segment>"), the sub-genic unit the authors'
    own scripts match on and the only key comparable across independently mutagenised
    libraries -- see the block comment in ``cmn/cmn_exper.py``.  The cleaning (abundance
    above one, duplicate and failed fits dropped) lives in that loader so every figure and
    table in the repo starts from the same rows.
    """
    early = cmn_exper.load_couce_segment_series(ancestor)
    late = cmn_exper.load_couce_segment_series(evolved)
    shared = early.index.intersection(late.index)
    return (early.loc[shared].to_numpy(float),
            late.loc[shared].to_numpy(float))


# ───────────────────────────────────  Histograms  ─────────────────────────────────
def couce_histogram(threshold):
    """Row 1: fig1's hand-set count thresholds, applied to the unclipped data."""
    def histogram(data, final_bins):
        return thresholded_histogram(data, threshold, final_bins)
    return histogram


# ─────────────────────────────────────  Figure  ───────────────────────────────────
def main():
    limdi_exclusions = (0.0, limdi_half_max_cut())
    couce_exclusions = (0.0, COUCE_CUT)
    couce_anc, couce_evo = load_couce_pair()
    control_green, control_red = cmn_exper.limdi_channel_series("REL607")
    control_green = np.asarray(control_green, dtype=float)
    control_red = np.asarray(control_red, dtype=float)

    scatter_anc = cmn_exper.limdi_gene_series("REL607")
    scatter_evo = cmn_exper.limdi_gene_series("Ara+2")
    shared = scatter_anc.index.intersection(scatter_evo.index)
    scatter_anc = scatter_anc.loc[shared].to_numpy(float)
    scatter_evo = scatter_evo.loc[shared].to_numpy(float)

    fig = plt.figure(figsize=(18, 12.5))
    # An explicit spacer row rather than hspace, so the gap can be set independently of
    # the row heights -- row 2 is the taller of the two.
    gs = GridSpec(3, 3, figure=fig, wspace=0.36, hspace=0.0,
                  height_ratios=(1.0, 0.28, 1.32))
    panel_rows = (0, 2)
    axes = np.array([[fig.add_subplot(gs[panel_rows[row], col]) for col in range(3)]
                     for row in range(2)])
    # Use the free left margin for longer fate bars; retain the shared column
    # positions below for aligned panel letters across the two rows.
    panel_positions = [ax.get_position().frozen() for ax in axes.ravel()]
    fate_position = axes[0, 0].get_position()
    extra_width = 0.055
    axes[0, 0].set_position([fate_position.x0 - extra_width, fate_position.y0,
                             fate_position.width + extra_width, fate_position.height])

    # Row 1: Couce, fig1's hand-set per-curve histogram thresholds.
    couce_histograms = {
        "forward_subset": couce_histogram(3), "forward_anchor": couce_histogram(3),
        "forward_backdrop": couce_histogram(6), "backward_subset": couce_histogram(2),
        "backward_anchor": couce_histogram(3), "backward_backdrop": couce_histogram(8),
    }
    fate_summary = create_fate_bars(axes[0, 0], couce_anc, couce_evo, labels=('0', '2K'),
                                    neutral_band=NEUTRAL_BAND, caption=fate_caption)
    create_overlapping_dfes(axes[0, 1], axes[0, 2], couce_anc, couce_evo,
                            couce_histograms, headroom=lambda ylim: 10.0, xlim=XLIM,
                            axis_order=EFFECT_AXIS_ORDER)
    for ax in axes[0]:
        arrow_title(ax, "ARA+2 (DM25), 0K", "2K")

    # Row 2: paired-effect scatters, all floored at SCATTER_FLOOR.  D and E top out at the
    # envelope of their own data, F at E's.
    control_limits = (SCATTER_FLOOR, envelope_limits(control_green, control_red)[1])
    ara2_limits = (SCATTER_FLOOR, envelope_limits(scatter_anc, scatter_evo)[1])
    control_results, _ = scatter_panel(
        axes[1, 0], control_green, control_red, "Isogenic control (REL607, LB)",
        r"Fitness effect $(s)$, measurement 1",
        r"Fitness effect $(s)$, measurement 2",
        control_limits,
        exclusions=limdi_exclusions, r_labels=R_LABELS, show_inset=False)
    ara2_results, _ = scatter_panel(
        axes[1, 1], scatter_anc, scatter_evo,
        "",     # arrow title set below
        r"Ancestral effect $(s)$", r"Evolved effect $(s)$",
        ara2_limits,
        exclusions=limdi_exclusions, r_labels=R_LABELS, show_inset=False)
    couce_results, _ = scatter_panel(
        axes[1, 2], couce_anc, couce_evo,
        "",     # arrow title set below
        r"Ancestral effect $(s)$", r"Evolved effect $(s)$",
        ara2_limits,
        marker_size=6.0, exclusions=couce_exclusions, r_labels=R_LABELS,
        show_inset=False)

    arrow_title(axes[1, 1], "ARA+2 (LB), 0K", "50K")
    arrow_title(axes[1, 2], "ARA+2 (DM25), 0K", "2K")

    # Per panel: the bulk-label leader.  Every Pearson block hangs just below y = 0, and every
    # panel's density key is normalized on its own singleton and peak bins.
    for ax, x, y, results, leader in (
            (axes[1, 0], control_green, control_red, control_results, BULK_LEADER_D),
            (axes[1, 1], scatter_anc, scatter_evo, ara2_results, BULK_LEADER_E),
            (axes[1, 2], couce_anc, couce_evo, couce_results, BULK_LEADER_F)):
        finish_density_panel(ax, x, y, results, leader=leader)

    for index, (ax, label) in enumerate(zip(axes.ravel(), "ABCDEF")):
        position = panel_positions[index]
        fig.text(position.x0 - 0.30 * position.width,
                 1.15, label,
                 transform=mpl.transforms.blended_transform_factory(
                     fig.transFigure, ax.transAxes),
                 fontsize=18, fontweight="heavy", va="top", ha="left")
        if index >= 3:      # row 2: open, detached frame, matching B and C
            cmn_scatter.style_scatter_axes(ax)
        else:
            style_dfe_axes(ax)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "fig2_scrambling_exper_res.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches='tight')
    plt.close(fig)

    for n, counts, src, dst in fate_summary:
        print(f"Beneficial at {src} (n={n}), measured at {dst}:  "
              + "  ".join(f"{name} {c} ({100 * c / n:.1f}%)"
                          for name, c in zip(FATE_CLASSES, counts)))

    print_correlations("REL607 green vs red", control_results)
    print_correlations("REL607 -> Ara+2", ara2_results)
    print_correlations("Couce 0K -> 2K", couce_results)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
