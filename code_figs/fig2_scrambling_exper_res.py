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

import csv
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.patches import Rectangle

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn import cmn_exper, cmn_scatter  # noqa: E402
from cmn.cmn_scatter import (  # noqa: E402  (row 2 is shared with figs S1-S4)
    envelope_limits, print_correlations, scatter_panel,
)

# ───────────────────────────────────── Style ─────────────────────────────────────
cmn_scatter.apply_style()

EVO_FILL = mpl.colors.to_rgba("#CF0282", alpha=0.8)
ANC_FILL = mpl.colors.to_rgba("#FAF9F6", alpha=0.6)

# ─────────────────────────────────── Parameters ──────────────────────────────────
XLIM = 0.06                 # half-width of the plotted fitness-effect window
SHIFT_FRAC = 0.025          # sideways offset between the paired histograms

# Panel A classes: beneficial s > NEUTRAL_BAND, neutral |s| <= NEUTRAL_BAND, deleterious
# s < -NEUTRAL_BAND.  0.015 is Couce et al.'s own beneficial cut.
NEUTRAL_BAND = 0.015
# Ordered from negative to positive effect, left to right.
FATE_CLASSES = ("Deleterious", "Neutral", "Beneficial")
# Both directions share one diverging set: dark red for deleterious, light grey for
# neutral, dark navy for beneficial.  The navy is far darker than row 2's light bulk blue,
# and none of the three comes near the grey ANC_FILL or the orange EVO_FILL in B/C.
FATE_COLORS = ("#8C1C1C", "#BDBDBD", "#1F3F7A")
# In-bar percentage colors: white on the dark ends, dark on the grey.
FATE_TEXT_COLORS = ("white", "#222222", "white")

OUT_DIR = os.path.join(_REPO_ROOT, "figs_paper")

# Row 2 cuts: the fraction of largest |x| moved from r_All to r_Bulk.
HALF_MAX_TABLE = os.path.join(_REPO_ROOT, "data", "exper", "TableS5_limdi_half_max.csv")
LIMDI_CUT_LINEAGE = "pooled"        # D and E both use the pooled half-max edge
COUCE_CUT = 0.03                    # F
# Lower limit of both axes in every row-2 panel; points below it are not drawn.
SCATTER_FLOOR = -0.6
R_LABELS = ("All", "Bulk")
HEX_GRIDSIZE = 75          # fine enough to retain isolated occupied bins
# Bulk-percentage leader, in points: a 45-degree segment off the rectangle's upper-left
# corner, then a horizontal run to the label.  (rise, run); a negative rise heads down.
BULK_LEADER = (-22, 18)
# D's label runs out flatter and further left, clear of the diagonal cloud.
BULK_LEADER_D = (-10, 40)
# E's rectangle sits in the dense upper-right cloud, so its label goes up and further out.
BULK_LEADER_E = (8, 55)
# F's label goes up and left, rising as far as E's.
BULK_LEADER_F = (BULK_LEADER_E[0], 40)
# The Pearson block hangs just below the y = 0 guide, by this many points.
PEARSON_BELOW_ZERO = 6
# Row 2's retained bulk: rectangle, its label and the r_Bulk line, in the shared blue.
BULK_COLOR = cmn_scatter.RETAINED_COLOR
HEX_DENSITY_CMAP =mpl.colors.LinearSegmentedColormap.from_list(
    "purple_hex_density", ("#C5B6DF", cmn_scatter.EXCLUDED_COLOR, "#21103F"))

# Row 1 spans +-0.06, so every tick label would read "-0.02", "0.00", ... .  Pulling a
# fixed 10^-2 out into the axis offset text turns those into "-2", "0", ... , which is
# both shorter and easier to read.
EFFECT_AXIS_ORDER = -2


class FixedOrderFormatter(mticker.ScalarFormatter):
    """ScalarFormatter with the power-of-ten offset pinned rather than auto-chosen."""

    def __init__(self, order):
        super().__init__(useOffset=False, useMathText=True)
        self._fixed_order = order
        self.set_scientific(True)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self._fixed_order


def scale_effect_axis(ax):
    ax.xaxis.set_major_formatter(FixedOrderFormatter(EFFECT_AXIS_ORDER))
    ax.xaxis.get_offset_text().set_fontsize(14)


# Panel titles read "<population>, <from> -> <to>".  The arrow is drawn as its own text in
# a larger font than the title around it, so it reads at a glance.
TITLE_ARROW_SIZE = 26       # points; the title text itself is 16
TITLE_ARROW_GAP = 3         # points of space either side of the arrow
TITLE_ARROW_FONT = "Times New Roman"    # a thin arrow with an open head; DejaVu's is blunt


def arrow_title(ax, before, after, pad=8):
    """Set ``before -> after`` as the title of ``ax``, with an enlarged arrow.

    ``before`` is the ordinary title; the arrow and ``after`` are chained to its right edge.
    The whole group is then centred by shifting ``before`` left in points, not in axes
    fraction, so it stays centred when an equal-aspect panel shrinks its box at draw time.
    """
    title = ax.set_title(before, pad=pad)
    arrow = ax.annotate("→", xy=(1, 0.5), xycoords=title, xytext=(TITLE_ARROW_GAP, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=TITLE_ARROW_SIZE, family=TITLE_ARROW_FONT)
    tail = ax.annotate(after, xy=(1, 0.5), xycoords=arrow, xytext=(TITLE_ARROW_GAP, 0),
                       textcoords="offset points", ha="left", va="center",
                       fontsize=title.get_fontsize())
    renderer = ax.figure.canvas.get_renderer()
    trailing_px = (arrow.get_window_extent(renderer).width
                   + tail.get_window_extent(renderer).width)
    trailing_in = trailing_px / ax.figure.dpi + 2 * TITLE_ARROW_GAP / 72
    title.set_transform(title.get_transform() + mpl.transforms.ScaledTranslation(
        -trailing_in / 2, 0, ax.figure.dpi_scale_trans))


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
def thresholded_histogram(data, threshold, final_bins):
    """Drop bins holding fewer than ``threshold`` points, then re-bin what survives."""
    counts, bin_edges = np.histogram(data, bins=10 * final_bins)
    valid_data = []
    for i, keep in enumerate(counts >= threshold):
        if keep:
            bin_mask = (data >= bin_edges[i]) & (data < bin_edges[i + 1])
            valid_data.append(data[bin_mask])
    if not valid_data:
        raise ValueError("No bins passed the threshold.")
    cleaned_data = np.concatenate(valid_data)
    final_counts, final_edges = np.histogram(
        cleaned_data, bins=final_bins, density=True)
    return final_counts, final_edges


def couce_histogram(threshold):
    """Row 1: fig1's hand-set count thresholds, applied to the unclipped data."""
    def histogram(data, final_bins):
        return thresholded_histogram(data, threshold, final_bins)
    return histogram


# ─────────────────────────────────  Row 1 panels  ─────────────────────────────────
def create_overlapping_dfes(ax_left, ax_right, dfe_anc, dfe_evo, histograms, headroom):
    """Forward (left) and backward (right) subset-DFE panels for one transition.

    ``histograms`` maps a curve role to the histogram callable for that curve, so row 1
    keeps fig1's per-curve hand-set thresholds while row 2 uses the rescaled clipped one.
    ``headroom`` turns a panel's y-limit into the extra space left above it.
    """
    z_frac = 0.1
    lw_main = 1.0
    valid = np.isfinite(dfe_anc) & np.isfinite(dfe_evo)
    dfe_anc, dfe_evo = dfe_anc[valid], dfe_evo[valid]

    def draw_custom_segments(ax, _xlim, _ylim):
        z = _ylim * z_frac * 1.1
        ax.plot([-_xlim * 0.9, _xlim * 0.9], [z, z],
                linestyle=":", color="grey", lw=lw_main)
        for (x0, y0), (x1, y1) in [
            ((-_xlim, -0.75), (-_xlim * 0.9, z)),
            ((_xlim, -0.75), (_xlim * 0.9, z)),
            ((-_xlim / 2, -0.75), (-_xlim / 2 * 0.9, z)),
            ((_xlim / 2, -0.75), (_xlim / 2 * 0.9, z)),
            ((0, -0.75), (0, z)),
        ]:
            ax.plot([x0, x1], [y0, y1], linestyle=":", color="grey", lw=lw_main)

    bdfe_anc = dfe_anc[dfe_anc > 0]
    bdfe_evo = dfe_evo[dfe_evo > 0]
    prop_bdfe_anc = dfe_evo[dfe_anc > 0]     # ancestor-beneficial, read on the evolved
    prop_bdfe_evo = dfe_anc[dfe_evo > 0]     # evolved-beneficial, read on the ancestor

    def frame(ax, ylim):
        ax.set_xlim(-XLIM, XLIM)
        ax.set_ylim(0, ylim + headroom(ylim))
        ax.tick_params(labelsize=16)
        scale_effect_axis(ax)

    # ── Left panel: forward propagation ──
    counts, bin_edges = histograms["forward_subset"](prop_bdfe_anc, 25)
    anc_counts, anc_bin_edges = histograms["forward_anchor"](bdfe_anc, 20)
    dfe_counts, dfe_bin_edges = histograms["forward_backdrop"](dfe_evo, 30)
    bin_edges = bin_edges - XLIM * SHIFT_FRAC
    dfe_bin_edges = dfe_bin_edges - XLIM * SHIFT_FRAC
    anc_bin_edges = anc_bin_edges + XLIM * SHIFT_FRAC

    ylim = max(counts.max(), anc_counts.max(), dfe_counts.max()) * (1 + z_frac)
    z = ylim * z_frac
    frame(ax_left, ylim)
    ax_left.stairs(values=counts + z, edges=bin_edges, baseline=0, fill=True,
                   facecolor=EVO_FILL, edgecolor="black", lw=1.1, label="Evo.")
    ax_left.stairs(values=dfe_counts + z, edges=dfe_bin_edges, baseline=0, fill=False,
                   edgecolor="black", linestyle="-.", lw=1.5, label="DFE Evo.")
    ax_left.add_patch(Rectangle((-XLIM, 0), 2 * XLIM, z,
                                facecolor="white", edgecolor="none"))
    draw_custom_segments(ax_left, XLIM, ylim)
    ax_left.stairs(values=anc_counts, edges=anc_bin_edges, baseline=0, fill=True,
                   facecolor=ANC_FILL, edgecolor="black", lw=1.1, label="Anc.")
    ax_left.legend(frameon=False)
    ax_left.set_xlabel(r'Fitness effect $(s)$')
    ax_left.set_ylabel('Density')

    # ── Right panel: backward propagation ──
    counts2, bin_edges2 = histograms["backward_subset"](bdfe_evo, 12)
    anc2_counts, anc2_bin_edges = histograms["backward_anchor"](prop_bdfe_evo, 22)
    dfe2_counts, dfe2_bin_edges = histograms["backward_backdrop"](dfe_anc, 24)
    bin_edges2 = bin_edges2 + XLIM * SHIFT_FRAC
    dfe2_bin_edges = dfe2_bin_edges - XLIM * SHIFT_FRAC
    anc2_bin_edges = anc2_bin_edges - XLIM * SHIFT_FRAC

    ylim = max(counts2.max(), anc2_counts.max(), dfe2_counts.max()) * (1 + z_frac)
    z = ylim * z_frac
    frame(ax_right, ylim)
    ax_right.stairs(values=counts2 + z, edges=bin_edges2, baseline=0, fill=True,
                    facecolor=EVO_FILL, edgecolor="black", lw=1.1, label="Evo.")
    ax_right.add_patch(Rectangle((-XLIM, 0), 2 * XLIM, z,
                                 facecolor="white", edgecolor="none"))
    draw_custom_segments(ax_right, XLIM, ylim)
    ax_right.stairs(values=anc2_counts, edges=anc2_bin_edges, baseline=0, fill=True,
                    facecolor=ANC_FILL, edgecolor="black", lw=1.1, label="Anc.")
    ax_right.stairs(values=dfe2_counts, edges=dfe2_bin_edges, baseline=0,
                    edgecolor="black", linestyle="-.", lw=1.5, label="DFE Anc.")
    ax_right.legend(frameon=False)
    ax_right.set_xlabel(r'Fitness effect $(s)$')
    ax_right.set_ylabel('Density')

    for ax in (ax_left, ax_right):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')


def fate_fractions(effects):
    """Fractions of ``effects`` that are deleterious / neutral / beneficial, in that order."""
    return np.array([np.mean(effects < -NEUTRAL_BAND),
                     np.mean(np.abs(effects) <= NEUTRAL_BAND),
                     np.mean(effects > NEUTRAL_BAND)])


def create_fate_bars(ax, dfe_anc, dfe_evo, labels):
    """One 100% bar per direction: what the beneficial set is in the other background.

    Stacked left to right deleterious / neutral / beneficial, the same way round as a fitness
    axis.  Each bar carries its own title above it and there is no x axis: the bars are
    percentages of one set, labelled in place.  Returns ``(n, counts, source, target)`` per
    direction for the printed summary.
    """
    valid = np.isfinite(dfe_anc) & np.isfinite(dfe_evo)
    dfe_anc, dfe_evo = dfe_anc[valid], dfe_evo[valid]
    early, late = labels

    # (beneficial-in background, read-in background), forward on top.
    directions = ((dfe_anc, dfe_evo, early, late, 1.05),
                  (dfe_evo, dfe_anc, late, early, -0.30))
    height = 0.52
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.05, 2.10)
    renderer = ax.figure.canvas.get_renderer()
    summary = []
    for source, target, src_label, dst_label, y in directions:
        selected = source > NEUTRAL_BAND
        n = int(selected.sum())
        fractions = fate_fractions(target[selected])
        summary.append((n, np.rint(fractions * n).astype(int), src_label, dst_label))

        left = 0.0
        for class_index, (fraction, color) in enumerate(zip(100 * fractions, FATE_COLORS)):
            ax.barh(y, fraction, height, left=left, color=color, edgecolor="black",
                    linewidth=1.0, clip_on=False)
            center = left + fraction / 2
            label = ax.text(center, y, f"{fraction:.1f}%", ha="center",
                            va="center", fontsize=16,
                            color=FATE_TEXT_COLORS[class_index])
            # Measure the actual glyph width, including padding, in display pixels.
            segment_width = (ax.transData.transform((left + fraction, y))[0]
                             - ax.transData.transform((left, y))[0])
            if (class_index == 2
                    or label.get_window_extent(renderer).width + 3 > segment_width):
                right_side = center >= 50
                sign = -1
                elbow = 91 if right_side else -2
                end = elbow + sign * 4
                label_y = y - height / 2 - 0.16
                label.set_position((end + sign * 1.5, label_y))
                label.set_ha("right")
                label.set_color("#222222")
                label.set_clip_on(False)
                ax.plot([center, elbow, end],
                        [y - height * 0.30, label_y, label_y],
                        color="#555555", linewidth=0.9, clip_on=False,
                        solid_capstyle="round", solid_joinstyle="round")
            left += fraction

        ax.text(0, y - height / 2 - 0.04, f"n = {n}",
                ha="left", va="top", fontsize=12, fontstyle="italic",
                color="#aaaaaa")

        source_name = "anc." if src_label == early else "evo."
        target_name = "evo." if dst_label == late else "anc."
        ax.text(0, y + height / 2 + 0.08,
                f"Insertions beneficial in {source_name} background\n"
                f"when measured in {target_name} background",
                ha="left", va="bottom", fontsize=12, fontstyle="italic",
                color="black", linespacing=1.15)

    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", linewidth=1.0)
               for c in FATE_COLORS]
    ax.legend(handles, FATE_CLASSES, loc="upper left",
              bbox_to_anchor=(0, directions[-1][-1] - height / 2 - 0.28),
              bbox_transform=ax.transData, ncol=3,
              frameon=False, fontsize=12, handlelength=1.0,
              handletextpad=0.4, columnspacing=0.9, borderaxespad=0)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return summary


# ─────────────────────────────────────  Figure  ───────────────────────────────────
def draw_density_hexagons(ax, x, y):
    """Fraction of a panel's mutants per hexagon, retaining singleton bins.

    Only points inside the axis limits are binned, but the fractions are of every finite
    pair, so hiding the points below the floor does not inflate the rest.
    """
    for collection in list(ax.collections):
        collection.remove()
    x, y = np.asarray(x, float), np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    lo, hi = ax.get_xlim()
    shown = finite & (x >= lo) & (y >= lo) & (x <= hi) & (y <= hi)
    hexagons = ax.hexbin(x[shown], y[shown], gridsize=HEX_GRIDSIZE, extent=(lo, hi, lo, hi),
                        mincnt=1, cmap=HEX_DENSITY_CMAP, linewidths=0.15,
                        edgecolors="face", zorder=3)
    counts = hexagons.get_array()
    hexagons.set_array(counts / finite.sum())
    return hexagons


def annotate_bulk(ax, x, y, results, leader=BULK_LEADER):
    """Outline every retained point and label its percentage off the upper-left corner.

    The label hangs from the corner on an angled-then-horizontal leader, the same shape as
    the out-of-bar labels in panel A, running left and, per ``leader = (rise, run)`` in
    points, down (rise < 0) or up (rise > 0).
    """
    x, y = np.asarray(x), np.asarray(y)
    bulk = results[-1]
    retained = np.argsort(np.abs(x), kind="stable")[:bulk["kept"]]
    color = BULK_COLOR
    # Pad the data bounds slightly so the outline clears the occupied hexagons.
    x_pad = 0.008 * np.ptp(ax.get_xlim())
    y_pad = 0.008 * np.ptp(ax.get_ylim())
    left, right = np.min(x[retained]) - x_pad, np.max(x[retained]) + x_pad
    bottom, top = np.min(y[retained]) - y_pad, np.max(y[retained]) + y_pad
    ax.add_patch(Rectangle((left, bottom), right - left, top - bottom,
                           fill=False, edgecolor=color, linewidth=1.1, zorder=9))
    percentage = 100 * bulk["kept"] / results[0]["kept"]
    rise, run = leader
    ax.annotate(f"{percentage:.1f}% of data", xy=(left, top),
                xytext=(-run - abs(rise), rise),
                textcoords="offset points", ha="right", va="center",
                fontsize=14, color=color, zorder=9, annotation_clip=False,
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.88),
                arrowprops=dict(arrowstyle="-", color=color, linewidth=0.9,
                                shrinkA=3, shrinkB=0, relpos=(1, 0.5),
                                connectionstyle=f"angle,angleA=0,angleB={45 if rise < 0 else -45}",
                                capstyle="round", joinstyle="round"))


def limdi_half_max_cut(lineage=LIMDI_CUT_LINEAGE):
    """The half-max edge p* of a Table S5 row, as a fraction."""
    with open(HALF_MAX_TABLE, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["lineage"] == lineage:
                return float(row["p_star"]) / 100
    raise KeyError(f"{lineage} not in {HALF_MAX_TABLE}")


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
    fate_summary = create_fate_bars(axes[0, 0], couce_anc, couce_evo, labels=('0', '2K'))
    create_overlapping_dfes(axes[0, 1], axes[0, 2], couce_anc, couce_evo,
                            couce_histograms, headroom=lambda ylim: 10.0)
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
        exclusions=limdi_exclusions, r_labels=R_LABELS, show_inset=False,
        bulk_color=BULK_COLOR)
    ara2_results, _ = scatter_panel(
        axes[1, 1], scatter_anc, scatter_evo,
        "",     # arrow title set below
        r"Ancestral effect $(s)$", r"Evolved effect $(s)$",
        ara2_limits,
        exclusions=limdi_exclusions, r_labels=R_LABELS, show_inset=False,
        bulk_color=BULK_COLOR)
    couce_results, _ = scatter_panel(
        axes[1, 2], couce_anc, couce_evo,
        "",     # arrow title set below
        r"Ancestral effect $(s)$", r"Evolved effect $(s)$",
        ara2_limits,
        marker_size=6.0, exclusions=couce_exclusions, r_labels=R_LABELS,
        show_inset=False, bulk_color=BULK_COLOR)

    arrow_title(axes[1, 1], "ARA+2 (LB), 0K", "50K")
    arrow_title(axes[1, 2], "ARA+2 (DM25), 0K", "2K")

    # Per panel: the bulk-label leader.  Every Pearson block hangs just below y = 0.
    density_hexagons = []
    for ax, x, y, results, leader in (
            (axes[1, 0], control_green, control_red, control_results, BULK_LEADER_D),
            (axes[1, 1], scatter_anc, scatter_evo, ara2_results, BULK_LEADER_E),
            (axes[1, 2], couce_anc, couce_evo, couce_results, BULK_LEADER_F)):
        density_hexagons.append(draw_density_hexagons(ax, x, y))
        # These panels contain the two zero guides and the identity line.
        for line in ax.lines:
            is_axis = np.ptp(line.get_xdata()) == 0 or np.ptp(line.get_ydata()) == 0
            line.set_linestyle(":" if is_axis else "--")
        for artist in ax.artists:
            if isinstance(artist, AnchoredOffsetbox):
                # x in axes fraction, y in data, then nudged down in points.
                below_zero = (mpl.transforms.blended_transform_factory(
                    ax.transAxes, ax.transData)
                    + mpl.transforms.ScaledTranslation(
                        0, -PEARSON_BELOW_ZERO / 72, fig.dpi_scale_trans))
                artist.loc = AnchoredOffsetbox.codes["upper left"]
                artist.set_bbox_to_anchor((0.02, 0.0), transform=below_zero)
        annotate_bulk(ax, x, y, results, leader=leader)

    # Normalize each panel separately so singleton and peak bins share endpoint colors.
    for ax, hexagons in zip(axes[1], density_hexagons):
        density_min = float(hexagons.get_array().min())
        density_max = float(hexagons.get_array().max())
        hexagons.set_norm(mpl.colors.LogNorm(vmin=density_min, vmax=density_max))
        density_order = int(np.ceil(np.log10(density_min)))
        key_ax = ax.inset_axes((0.59, 0.08, 0.36, 0.03))
        key = fig.colorbar(hexagons, cax=key_ax, orientation="horizontal",
                           ticks=[10.0 ** order for order in range(density_order, 1)
                                  if 10.0 ** order <= density_max],
                           format=mticker.LogFormatterMathtext())
        key.ax.minorticks_off()
        key.ax.tick_params(labelsize=13, length=3, pad=2)
        key.ax.set_title("Density", fontsize=15, pad=6)

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
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            ax.tick_params(axis='both', which='major', length=10, width=1.5)
            ax.tick_params(axis='both', which='minor', length=5, width=1.6)

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
