r"""Paired-effect scatter panels: one shared implementation for fig1 row 2 and figs S1-S4.

The same panel is drawn eleven times across the paper -- once per pair of fitness-effect
measurements of the same knockouts on two backgrounds (or on the same background twice, for
a control).  Every instance shows

    * the raw pair cloud, split into the near-neutral bulk and the excluded large-effect
      points, which are the colour split and nothing else -- no rule is drawn, because the
      partition is on |s| and so falls on BOTH sides of zero,
    * the identity line and the two zero axes,
    * Pearson r over every pair and again over the pairs whose x effect is smallest in
      ABSOLUTE value -- one partition per panel, dropping the largest 10% or 2% of |x|
      depending on how far that dataset's effects run, and
    * a hexbin inset zoomed on the dense core near the origin.

so it lives here rather than being copied per figure.

Excluding on |x| rather than on x alone is what makes the second r a statement about the
near-neutral bulk.  Pearson r is dominated by whatever sits farthest from the origin, and in
these datasets that is a strongly conserved set of large effects -- deleterious in the Limdi
panels, but beneficial as well in the Ascensao ones, where an ancestral knockout at s = +0.5
would otherwise carry the correlation on its own.  Ranking on the signed effect removed only
the deleterious half of that leverage.

The exclusion is defined only from the x (ancestor / first-measurement) side and never from
y, so the retained subset is not conditioned on the outcome whose correlation is being
reported.  Nothing is clipped: every panel's limits are the envelope of its own data, and x
and y share them so the identity line is the diagonal -- see :func:`envelope_limits`.

Nothing here is rasterized.  Matplotlib renders rasterized artists at ``figure.dpi`` (100),
which leaves these clouds visibly soft next to the vector text and axes; drawing all ~16k
markers as vector costs about 0.15 MB per figure.
"""

import cmasher  # noqa: F401  (registers the cmr.* colormaps with matplotlib)
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from scipy.stats import pearsonr

# ───────────────────────────────────── Style ─────────────────────────────────────
IDENTITY_COLOR = (0.18, 0.18, 0.18)

# Every panel reports ONE partition, so the palette is two bands: the near-neutral bulk and
# the excluded large-|s| points.  Positions along cmr.freeze, which runs near-black ->
# purple -> indigo -> blue -> cyan -> white.  0.60 for the bulk because the pale cyan end of
# that ramp left the cloud reading as absent; 0.30 for the tail because it lands between the
# near-black and the indigo an earlier three-band palette used, so it is unmistakably neither
# while staying dark enough to read as the excluded ends of the distribution.
BAND_POSITIONS = (0.30, 0.60)          # (excluded large effects, retained bulk)
EXCLUDED_COLOR, RETAINED_COLOR = (
    mpl.colors.to_hex(mpl.colormaps["cmr.freeze"](position))
    for position in BAND_POSITIONS)
# The inset is a density map of the SAME points the retained band draws in the panel around it,
# so both ends of its ramp come from that band's own colour: full strength where the core is
# densest, mixed toward white where it is sparse.  Nothing here is an independently picked hex,
# so the inset cannot drift out of the palette when the bands are retuned -- and it cannot pick
# up a hue that belongs to one of the excluded bands, which reads as a meaning it does not have.
#
# The light end is the end that matters.  The inset hexbin is drawn on a LogNorm over counts,
# and about 70% of its hexagons hold few enough points to land in the bottom third of the ramp,
# so a near-white light end washes the whole inset out however dark the other end is.
INSET_LIGHT_TINT = 0.70          # how far the sparse end sits from the band colour toward white


def _tint(color, fraction):
    """``color`` mixed ``fraction`` of the way to white, as hex."""
    rgb = np.asarray(mpl.colors.to_rgb(color))
    return mpl.colors.to_hex(rgb + (1.0 - rgb) * fraction)


DENSITY_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "moderate_density",
    (_tint(RETAINED_COLOR, INSET_LIGHT_TINT), RETAINED_COLOR))

# Retained bulk first, excluded large effects second: darker, larger and more opaque, on top.
BAND_COLORS = (RETAINED_COLOR, EXCLUDED_COLOR)
BAND_STYLES = ((0.38, 1.00), (0.84, 1.45))


def apply_style():
    """The shared rcParams for every figure built out of these panels."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 16
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['legend.fontsize'] = 14


# ─────────────────────────────────── Parameters ──────────────────────────────────
# One partition per panel: the fraction of pairs with the LARGEST |x| that is dropped.  Limdi
# knockouts spread over |s| up to 0.65, so 10% still leaves the cut well outside the bulk.  The
# Couce and Ascensao effect distributions are an order of magnitude more compact -- their 5th
# percentile already sits inside the bulk -- so those panels drop 2% instead, the largest
# fraction there that is still only large-effect points.
MAGNITUDE_EXCLUSIONS = (0.00, 0.10)
SHALLOW_MAGNITUDE_EXCLUSIONS = (0.00, 0.02)

INSET_LIMITS = (-0.05, 0.05)

PEARSON_FONTSIZE = 17


def envelope_limits(x, y, pad_frac=0.04, step=0.005):
    """Square limits containing every finite point of both measurements.

    Padded by ``pad_frac`` of the range and rounded outwards to a multiple of ``step``, so
    panels of similar scale end up on identical limits without hand-setting each one.
    """
    finite = np.concatenate([np.asarray(x, float)[np.isfinite(x)],
                             np.asarray(y, float)[np.isfinite(y)]])
    lo, hi = float(finite.min()), float(finite.max())
    pad = pad_frac * (hi - lo)
    return (float(np.floor((lo - pad) / step) * step),
            float(np.ceil((hi + pad) / step) * step))


def matched_ticks(lo, hi, max_ticks=6):
    """One tick set for BOTH axes of a square panel.

    x and y already share ``limits`` (see :func:`envelope_limits`), but matplotlib locates each
    axis independently and the two boxes are not the same length in points, so it can and does
    choose different steps for the same range -- 0.2 across x against 0.1 up y.  On a panel whose
    whole point is that the identity line is the diagonal, mismatched ticks make equal distances
    on the two axes look unequal.  Locating once and applying the result to both fixes that.
    """
    ticks = mpl.ticker.MaxNLocator(nbins=max_ticks, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    tol = 1e-9 * max(1.0, abs(hi - lo))       # tick_values rounds outward and drifts in float
    return ticks[(ticks >= lo - tol) & (ticks <= hi + tol)]


# ────────────────────────────────── Correlations ─────────────────────────────────
def magnitude_exclusion_correlations(ancestor_effect, comparison_effect, exclusions):
    """Pearson r after removing exact fractions of the LARGEST |ancestor effect|.

    Ranking on magnitude rather than on the signed effect keeps the retained subset symmetric
    about zero: a knockout is dropped for being large, not for being deleterious.  Only the
    ancestor side enters the ranking, so the subsets remain nested and are never conditioned
    on the outcome being predicted.  ``cut_abs`` is the smallest EXCLUDED magnitude, i.e. the
    |s| threshold above which pairs are dropped.
    """
    magnitude = np.abs(ancestor_effect)
    order = np.argsort(magnitude, kind="stable")          # ascending |s|
    n = ancestor_effect.size
    results = []
    for fraction in exclusions:
        n_remove = int(np.floor(fraction * n))
        kept = order[:n - n_remove]
        results.append({
            "fraction": fraction,
            "removed": n_remove,
            "kept": kept.size,
            "cut_abs": (float("inf") if n_remove == 0
                        else float(magnitude[order[n - n_remove]])),
            "r": float(pearsonr(ancestor_effect[kept], comparison_effect[kept]).statistic),
        })
    return results


def print_correlations(name, results):
    values = "  ".join(
        f"remove {100 * item['fraction']:4.3g}%: r={item['r']:.4f} (n={item['kept']})"
        for item in results)
    print(f"{name:30s} {values}")


# ──────────────────────────────────── Drawing ────────────────────────────────────
def draw_scatter_points(ax, x, y, marker_size, exclusions):
    """Layer the points by the single nested exclusion the panel reports.

    Two layers: the near-neutral bulk, and the excluded large-|x| points drawn on top of it in
    the darker, larger, more opaque style.  Because the split is on magnitude, the second layer
    appears at BOTH ends of the x range -- which is the whole point, and the reason no cutoff
    rule is drawn: a single vertical line would describe only half of it.

    ``exclusions`` is ``(0.0, cut)``; every panel in the paper reports one partition, so a
    longer ladder is a mistake to raise on rather than a case to interpolate the palette
    through.
    """
    if len(exclusions) != 2 or exclusions[0] != 0.0:
        raise ValueError(f"expected one partition as (0.0, cut), got {exclusions}")
    order = np.argsort(np.abs(x), kind="stable")
    split = x.size - int(np.floor(exclusions[1] * x.size))
    for depth, indices in enumerate((order[:split], order[split:])):
        alpha, scale = BAND_STYLES[depth]
        ax.scatter(x[indices], y[indices], s=marker_size * scale,
                   color=BAND_COLORS[depth], alpha=alpha, linewidths=0,
                   zorder=3 + depth)


def scatter_panel(ax, x, y, title, xlabel, ylabel, limits,
                  marker_size=8.0, exclusions=MAGNITUDE_EXCLUSIONS,
                  inset_limits=INSET_LIMITS, inset_rect=(0.52, 0.04, 0.38, 0.38),
                  r_labels=None):
    """One paired-effect scatter with a colour-split cloud, Pearson block and density inset.

    ``limits`` is applied to BOTH axes, so the panel is square and the identity line is its
    diagonal; :func:`envelope_limits` builds it from the panel's own data, and
    :func:`matched_ticks` puts the same ticks on x and y so equal distances look equal.  Returns
    ``(correlations, density)`` -- the latter so a caller can put every inset in a figure
    on one shared colour norm via :func:`share_density_norm`.  ``r_labels`` names the Pearson
    lines instead of the retained percentage; see :func:`draw_pearson_block`.
    """
    lo, hi = limits

    ax.axhline(0.0, color="grey", lw=0.75, ls="--", zorder=1)
    ax.axvline(0.0, color="grey", lw=0.75, ls="--", zorder=1)
    ax.plot([lo, hi], [lo, hi], color=IDENTITY_COLOR, lw=1.2, zorder=2)
    draw_scatter_points(ax, x, y, marker_size, exclusions)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = matched_ticks(lo, hi)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    correlations = magnitude_exclusion_correlations(x, y, exclusions)
    draw_pearson_block(ax, correlations, labels=r_labels)

    # Right edge at 0.90, not 0.96: the right-hand y labels need room inside the parent
    # frame, and inward ticks keep the tick marks off the inset spines' outer side.
    inset = ax.inset_axes(list(inset_rect))
    inset.set_zorder(20)
    inset.patch.set_facecolor("white")
    inset.patch.set_alpha(1.0)
    ilo, ihi = inset_limits
    inset.axhline(0.0, color="grey", lw=0.55, ls="--", zorder=1)
    inset.axvline(0.0, color="grey", lw=0.55, ls="--", zorder=1)
    inset.plot([ilo, ihi], [ilo, ihi], color=IDENTITY_COLOR, lw=0.8, zorder=2)
    density = inset.hexbin(x, y, gridsize=28, extent=(ilo, ihi, ilo, ihi), mincnt=1,
                           cmap=DENSITY_CMAP, linewidths=0.25, edgecolors="face",
                           zorder=3)
    # Spines and ticks default to zorder 2.5, so the hexagons would paint over them.
    for spine in inset.spines.values():
        spine.set_zorder(6)
    inset.xaxis.set_zorder(6)
    inset.yaxis.set_zorder(6)
    inset.set_xlim(ilo, ihi)
    inset.set_ylim(ilo, ihi)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([ilo, 0.0, ihi])
    inset.set_yticks([ilo, 0.0, ihi])
    inset.set_xticklabels([f"{ilo:.2f}", "0.00", f"{ihi:.2f}"])
    # The x-axis lower-bound label also denotes the shared square-inset limit; suppress
    # its duplicate on y, which would otherwise collide at a corner.
    inset.set_yticklabels(["", "0.00", f"{ihi:.2f}"])
    inset.yaxis.set_ticks_position("right")
    inset.tick_params(axis="y", labelleft=False, labelright=True)
    inset.tick_params(labelsize=9, length=2.5, width=0.8, pad=2, direction="in")
    return correlations, density


def draw_pearson_block(ax, correlations, loc="upper left", anchor=(0.02, 0.985), labels=None):
    """The Pearson ladder, one line per subset, each in the colour of the points it covers.

    Built with an offsetbox rather than one multi-line ``Text`` because matplotlib cannot
    colour parts of a single text artist: ``VPacker`` stacks a ``TextArea`` per line and the
    frame draws one white background behind the lot, so the lines can differ in colour without
    the per-line bounding boxes that would otherwise tile visibly on top of the data.

    The all-pairs r takes the excluded points' colour and the retained-subset r the bulk's:
    those points are precisely what the first number has and the second does not, so the pair
    of colours reads as which points went into each.

    Each line is subscripted with the percentage retained (``r_90%``) unless ``labels`` gives
    one name per line (``("All", "Bulk")`` -> ``r_All``, ``r_Bulk``), for a cut that is not a
    round number.
    """
    colors = (IDENTITY_COLOR, EXCLUDED_COLOR, RETAINED_COLOR)
    lines = [TextArea("Pearson", textprops={"color": colors[0],
                                            "fontsize": PEARSON_FONTSIZE})]
    if labels is None:
        labels = [rf"{int(100 * (1 - result['fraction']))}\%" for result in correlations]
    else:
        labels = [rf"\mathrm{{{label}}}" for label in labels]
    for result, label, color in zip(correlations, labels, colors[1:]):
        lines.append(TextArea(
            rf"$r_{{{label}}}={result['r']:.3f}$",
            textprops={"color": color, "fontsize": PEARSON_FONTSIZE}))
    block = AnchoredOffsetbox(
        loc=loc, child=VPacker(children=lines, pad=0.0, sep=3.0, align="left"),
        pad=0.32, borderpad=0.0, frameon=True,
        bbox_to_anchor=anchor, bbox_transform=ax.transAxes)
    block.patch.set(facecolor="white", edgecolor="none", alpha=0.88)
    block.set_zorder(8)
    ax.add_artist(block)
    return block


def share_density_norm(densities):
    """Put every inset hexbin on one shared log colour norm."""
    densities = list(densities)
    shared = mpl.colors.LogNorm(
        vmin=1.0, vmax=max(float(np.max(plot.get_array())) for plot in densities))
    for plot in densities:
        plot.set_norm(shared)
    return shared


def style_scatter_axes(ax):
    """The open, detached frame these panels share with fig1's DFE panels."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_position(("outward", 10))
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', length=10, width=1.5)
    ax.tick_params(axis='both', which='minor', length=5, width=1.6)


def panel_label(ax, label):
    """The heavy A/B/C/... tag, at the same offset every panel in the paper uses."""
    ax.text(-0.13, 1.15, label, transform=ax.transAxes, fontsize=18,
            fontweight='heavy', va='top', ha='left')


def draw_panel_grid(axes, panels, exclusions=MAGNITUDE_EXCLUSIONS,
                    inset_limits=INSET_LIMITS, marker_size=8.0):
    """Fill a grid of axes with paired scatters, A/B/C/... labelled and normed together.

    ``panels`` is a sequence of dicts with ``name`` (for the printed correlation ladder),
    ``x``, ``y``, ``title``, ``xlabel``, ``ylabel``, ``limits`` and an optional
    ``overrides`` dict forwarded to :func:`scatter_panel`.  Returns ``[(name, results)]``
    in panel order.
    """
    ladders, densities = [], []
    for ax, label, panel in zip(np.ravel(axes), "ABCDEFGH", panels):
        kwargs = {"exclusions": exclusions, "inset_limits": inset_limits,
                  "marker_size": marker_size}
        kwargs.update(panel.get("overrides", {}))
        correlations, density = scatter_panel(
            ax, panel["x"], panel["y"], panel["title"], panel["xlabel"], panel["ylabel"],
            panel["limits"], **kwargs)
        panel_label(ax, label)
        style_scatter_axes(ax)
        ladders.append((panel["name"], correlations))
        densities.append(density)
    share_density_norm(densities)
    return ladders
