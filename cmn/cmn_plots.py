r"""Subset-DFE panels shared by fig2 row 1 (experiment) and fig S5 (simulations).

Each row of either figure shows one transition t1 -> t2 three ways:

    * the fate bars -- what the knockouts beneficial at one time are at the other, as one
      100% bar per direction (:func:`create_fate_bars`),
    * forward and backward layered histograms -- the beneficial DFE at one time in front,
      where those same knockouts land at the other time on a raised back layer, and the full
      DFE there as a line (:func:`create_overlapping_dfes`),

each titled ``<population>, <from> -> <to>`` with an enlarged arrow (:func:`arrow_title`).
The experimental and simulated rows differ only in their data, their histogram rules and how
wide the neutral band is, which are all arguments.
"""

import matplotlib as mpl
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Rectangle

# ───────────────────────────────────── Style ─────────────────────────────────────
EVO_FILL = mpl.colors.to_rgba("#CF0282", alpha=0.8)
ANC_FILL = mpl.colors.to_rgba("#FAF9F6", alpha=0.6)

# Fate classes, ordered from negative to positive effect, left to right.
FATE_CLASSES = ("Deleterious", "Neutral", "Beneficial")
# Both directions share one diverging set: dark red for deleterious, light grey for
# neutral, dark navy for beneficial.  The navy is far darker than the scatter panels' light
# bulk blue, and none of the three comes near the grey ANC_FILL or the magenta EVO_FILL.
FATE_COLORS = ("#8C1C1C", "#BDBDBD", "#1F3F7A")
# In-bar percentage colors: white on the dark ends, dark on the grey.
FATE_TEXT_COLORS = ("white", "#222222", "white")

SHIFT_FRAC = 0.025          # sideways offset between the paired histograms

# Panel titles read "<population>, <from> -> <to>".  The arrow is drawn as its own text in
# a larger font than the title around it, so it reads at a glance.
TITLE_ARROW_SIZE = 26       # points; the title text itself is 16
TITLE_ARROW_GAP = 3         # points of space either side of the arrow
TITLE_ARROW_FONT = "Times New Roman"    # a thin arrow with an open head; DejaVu's is blunt

# Fig2's bin counts per curve, (forward: subset, anchor, backdrop; backward: same).
DEFAULT_BINS = {"forward_subset": 25, "forward_anchor": 20, "forward_backdrop": 30,
                "backward_subset": 12, "backward_anchor": 22, "backward_backdrop": 24}


class FixedOrderFormatter(mticker.ScalarFormatter):
    """ScalarFormatter with the power-of-ten offset pinned rather than auto-chosen."""

    def __init__(self, order):
        super().__init__(useOffset=False, useMathText=True)
        self._fixed_order = order
        self.set_scientific(True)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self._fixed_order


def scale_effect_axis(ax, order):
    """Pull a fixed ``10^order`` out of the x tick labels into the axis offset text."""
    ax.xaxis.set_major_formatter(FixedOrderFormatter(order))
    ax.xaxis.get_offset_text().set_fontsize(14)


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


def plain_histogram(data, final_bins):
    """A density histogram with no thresholding, for the noise-free simulations."""
    return np.histogram(data, bins=final_bins, density=True)


# ─────────────────────────────────────  Panels  ───────────────────────────────────
def create_overlapping_dfes(ax_left, ax_right, dfe_anc, dfe_evo, histograms, headroom,
                            xlim, bins=None, axis_order=-2,
                            xlabel=r'Fitness effect $(s)$'):
    """Forward (left) and backward (right) subset-DFE panels for one transition.

    ``histograms`` maps a curve role to the histogram callable ``(data, bins)`` for that
    curve, so fig2 keeps its per-curve hand-set thresholds while the simulations use plain
    ones; ``bins`` maps the same roles to bin counts (default: fig2's).  ``headroom`` turns a
    panel's y-limit into the extra space left above it.  ``xlim`` is the half-width of the
    effect window and ``axis_order`` the power of ten pulled out of its tick labels.
    """
    bins = {**DEFAULT_BINS, **(bins or {})}
    z_frac = 0.1
    lw_main = 1.0
    valid = np.isfinite(dfe_anc) & np.isfinite(dfe_evo)
    dfe_anc, dfe_evo = dfe_anc[valid], dfe_evo[valid]

    def draw_custom_segments(ax, _xlim, _ylim):
        z = _ylim * z_frac * 1.1
        base = -0.0075 * _ylim      # just below the axis, so the guides meet it cleanly
        ax.plot([-_xlim * 0.9, _xlim * 0.9], [z, z],
                linestyle=":", color="grey", lw=lw_main)
        for (x0, y0), (x1, y1) in [
            ((-_xlim, base), (-_xlim * 0.9, z)),
            ((_xlim, base), (_xlim * 0.9, z)),
            ((-_xlim / 2, base), (-_xlim / 2 * 0.9, z)),
            ((_xlim / 2, base), (_xlim / 2 * 0.9, z)),
            ((0, base), (0, z)),
        ]:
            ax.plot([x0, x1], [y0, y1], linestyle=":", color="grey", lw=lw_main)

    bdfe_anc = dfe_anc[dfe_anc > 0]
    bdfe_evo = dfe_evo[dfe_evo > 0]
    prop_bdfe_anc = dfe_evo[dfe_anc > 0]     # ancestor-beneficial, read on the evolved
    prop_bdfe_evo = dfe_anc[dfe_evo > 0]     # evolved-beneficial, read on the ancestor

    def frame(ax, ylim):
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(0, ylim + headroom(ylim))
        ax.tick_params(labelsize=16)
        scale_effect_axis(ax, axis_order)

    # ── Left panel: forward propagation ──
    counts, bin_edges = histograms["forward_subset"](prop_bdfe_anc, bins["forward_subset"])
    anc_counts, anc_bin_edges = histograms["forward_anchor"](
        bdfe_anc, bins["forward_anchor"])
    dfe_counts, dfe_bin_edges = histograms["forward_backdrop"](
        dfe_evo, bins["forward_backdrop"])
    bin_edges = bin_edges - xlim * SHIFT_FRAC
    dfe_bin_edges = dfe_bin_edges - xlim * SHIFT_FRAC
    anc_bin_edges = anc_bin_edges + xlim * SHIFT_FRAC

    ylim = max(counts.max(), anc_counts.max(), dfe_counts.max()) * (1 + z_frac)
    z = ylim * z_frac
    frame(ax_left, ylim)
    ax_left.stairs(values=counts + z, edges=bin_edges, baseline=0, fill=True,
                   facecolor=EVO_FILL, edgecolor="black", lw=1.1, label="Evo.")
    ax_left.stairs(values=dfe_counts + z, edges=dfe_bin_edges, baseline=0, fill=False,
                   edgecolor="black", linestyle="-.", lw=1.5, label="DFE Evo.")
    ax_left.add_patch(Rectangle((-xlim, 0), 2 * xlim, z,
                                facecolor="white", edgecolor="none"))
    draw_custom_segments(ax_left, xlim, ylim)
    ax_left.stairs(values=anc_counts, edges=anc_bin_edges, baseline=0, fill=True,
                   facecolor=ANC_FILL, edgecolor="black", lw=1.1, label="Anc.")
    ax_left.legend(frameon=False)
    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel('Density')

    # ── Right panel: backward propagation ──
    counts2, bin_edges2 = histograms["backward_subset"](bdfe_evo, bins["backward_subset"])
    anc2_counts, anc2_bin_edges = histograms["backward_anchor"](
        prop_bdfe_evo, bins["backward_anchor"])
    dfe2_counts, dfe2_bin_edges = histograms["backward_backdrop"](
        dfe_anc, bins["backward_backdrop"])
    bin_edges2 = bin_edges2 + xlim * SHIFT_FRAC
    dfe2_bin_edges = dfe2_bin_edges - xlim * SHIFT_FRAC
    anc2_bin_edges = anc2_bin_edges - xlim * SHIFT_FRAC

    ylim = max(counts2.max(), anc2_counts.max(), dfe2_counts.max()) * (1 + z_frac)
    z = ylim * z_frac
    frame(ax_right, ylim)
    ax_right.stairs(values=counts2 + z, edges=bin_edges2, baseline=0, fill=True,
                    facecolor=EVO_FILL, edgecolor="black", lw=1.1, label="Evo.")
    ax_right.add_patch(Rectangle((-xlim, 0), 2 * xlim, z,
                                 facecolor="white", edgecolor="none"))
    draw_custom_segments(ax_right, xlim, ylim)
    ax_right.stairs(values=anc2_counts, edges=anc2_bin_edges, baseline=0, fill=True,
                    facecolor=ANC_FILL, edgecolor="black", lw=1.1, label="Anc.")
    ax_right.stairs(values=dfe2_counts, edges=dfe2_bin_edges, baseline=0,
                    edgecolor="black", linestyle="-.", lw=1.5, label="DFE Anc.")
    ax_right.legend(frameon=False)
    ax_right.set_xlabel(xlabel)
    ax_right.set_ylabel('Density')

    for ax in (ax_left, ax_right):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')


def fate_fractions(effects, neutral_band):
    """Fractions of ``effects`` that are deleterious / neutral / beneficial, in that order."""
    return np.array([np.mean(effects < -neutral_band),
                     np.mean(np.abs(effects) <= neutral_band),
                     np.mean(effects > neutral_band)])


def create_fate_bars(ax, dfe_anc, dfe_evo, labels, neutral_band, caption):
    """One 100% bar per direction: what the beneficial set is in the other background.

    Stacked left to right deleterious / neutral / beneficial, the same way round as a fitness
    axis.  Each bar carries its own title above it, ``caption(forward)``, and there is no x
    axis: the bars are percentages of one set, labelled in place.  A class with no members is
    left out of the bar but kept in the legend.

    ``neutral_band = None`` drops the neutral class altogether -- for the noise-free
    simulations, where no resolution sets a band: an effect is beneficial above zero and
    deleterious below it, and the legend names those two classes only.  An effect of exactly
    zero would then belong to neither, so it is an error.  Returns ``(n, counts, source,
    target)`` per direction for the printed summary.
    """
    with_neutral = neutral_band is not None
    neutral_band = neutral_band if with_neutral else 0.0
    valid = np.isfinite(dfe_anc) & np.isfinite(dfe_evo)
    dfe_anc, dfe_evo = dfe_anc[valid], dfe_evo[valid]
    early, late = labels

    # (beneficial-in background, read-in background, forward?), forward on top.
    directions = ((dfe_anc, dfe_evo, early, late, True, 1.05),
                  (dfe_evo, dfe_anc, late, early, False, -0.30))
    height = 0.52
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.05, 2.10)
    renderer = ax.figure.canvas.get_renderer()
    summary = []
    for source, target, src_label, dst_label, forward, y in directions:
        selected = source > neutral_band
        n = int(selected.sum())
        if not with_neutral and np.any(target[selected] == 0):
            raise ValueError("an effect of exactly zero has no class without a neutral band")
        fractions = fate_fractions(target[selected], neutral_band)
        summary.append((n, np.rint(fractions * n).astype(int), src_label, dst_label))

        left = 0.0
        overflow = []       # (segment centre, label) for labels wider than their segment
        for class_index, (fraction, color) in enumerate(zip(100 * fractions, FATE_COLORS)):
            if fraction <= 0:
                continue
            ax.barh(y, fraction, height, left=left, color=color, edgecolor="black",
                    linewidth=1.0, clip_on=False)
            center = left + fraction / 2
            label = ax.text(center, y, f"{fraction:.1f}%", ha="center",
                            va="center", fontsize=16,
                            color=FATE_TEXT_COLORS[class_index])
            # Measure the actual glyph width, including padding, in display pixels.
            segment_width = (ax.transData.transform((left + fraction, y))[0]
                             - ax.transData.transform((left, y))[0])
            if label.get_window_extent(renderer).width + 3 > segment_width:
                overflow.append((center, label))
            left += fraction

        # A label that does not fit its segment hangs off it on an angled-then-horizontal
        # leader, right-aligned just left of the elbow.  The rightmost goes below the bar; a
        # second goes above it, mirrored, and the bar's title is lifted clear of it.
        if len(overflow) > 2:
            raise ValueError(f"{len(overflow)} fate labels do not fit their segments; "
                             "there is room for two outside the bar")
        for side, (center, label) in zip((-1, 1), sorted(overflow, key=lambda o: -o[0])):
            elbow = min(center + 3, 91)
            end = elbow - 4
            label_y = y + side * (height / 2 + 0.16)
            label.set_position((end - 1.5, label_y))
            label.set_ha("right")
            label.set_color("#222222")
            label.set_clip_on(False)
            ax.plot([center, elbow, end],
                    [y + side * height * 0.30, label_y, label_y],
                    color="#555555", linewidth=0.9, clip_on=False,
                    solid_capstyle="round", solid_joinstyle="round")

        ax.text(0, y - height / 2 - 0.04, f"n = {n}",
                ha="left", va="top", fontsize=12, fontstyle="italic",
                color="#aaaaaa")
        title_lift = 0.30 if len(overflow) == 2 else 0.0
        ax.text(0, y + height / 2 + 0.08 + title_lift, caption(forward),
                ha="left", va="bottom", fontsize=12, fontstyle="italic",
                color="black", linespacing=1.15)

    shown = [i for i in range(len(FATE_CLASSES)) if with_neutral or FATE_CLASSES[i] != "Neutral"]
    handles = [Rectangle((0, 0), 1, 1, facecolor=FATE_COLORS[i], edgecolor="black",
                         linewidth=1.0) for i in shown]
    ax.legend(handles, [FATE_CLASSES[i] for i in shown], loc="upper left",
              bbox_to_anchor=(0, directions[-1][-1] - height / 2 - 0.28),
              bbox_transform=ax.transData, ncol=3,
              frameon=False, fontsize=12, handlelength=1.0,
              handletextpad=0.4, columnspacing=0.9, borderaxespad=0)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return summary


def style_dfe_axes(ax):
    """The heavier spines and long ticks of the row-1 panels."""
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', length=10, width=1.5)
    ax.tick_params(axis='both', which='minor', length=5, width=1.6)
