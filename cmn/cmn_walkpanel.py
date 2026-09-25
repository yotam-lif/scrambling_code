#!/usr/bin/env python3
r"""One autocorrelation panel: simulated walks, their noisy re-measurement, measured stars.

Every figure that puts a DFE autocorrelation against fixed background mutations draws the same
panel -- Figure 4's bottom row for three hand-picked columns, and the supplementary sheet for
all twelve LB lineages at once.  The panel lives here so those two cannot drift apart, in the
same way ``cmn_scatter.scatter_panel`` holds the scatter panels the two of them share.

WHAT A PANEL SHOWS

  solid    the LATENT correlation: the model's own effects, no measurement error anywhere.
  dashed   the same walks re-measured with rank-matched published errors.
  band     the 16-84% interval over walk x noise replicates of that noisy measurement.
  stars    the MEASURED correlation, on the same |s|-ranked ladder as the curves.

Two subsets are drawn per panel by default: the whole library, and the library minus its
largest-|s| ancestral fraction.  Which fraction is the caller's business -- 10% for Limdi,
whose effects run out to |s| = 0.65, and 2% for Couce, whose effects are compact enough that a
10% cut reaches inside the bulk -- and it is named as a FRACTION, never as a column index.
``cmn_walkcache.column_for`` then finds it in whatever ladder that cache happens to hold.

SMOOTHING.  Curves are Gaussian-filtered along the step axis for display, with BOTH endpoints
pinned to their raw values: t = 0 because r = 1 there by construction, and the last step
because that is where the measured stars sit and where the plateau is read off.  This is
cosmetic.  The step-to-step wiggle it removes is Monte-Carlo noise and nothing else -- splitting
the walks into disjoint halves gives medians that disagree by 0.01-0.02 at exactly the steps
where the kinks appear.  The honest fix is more walks, which means regenerating a cache, not
editing a figure.

A trace carrying a NaN is left alone: NaN is what a cache whose walks have run out of
survivors looks like, and the filter would smear it across a whole kernel width.
"""

from __future__ import annotations

import os
import sys
import warnings

import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
from scipy.ndimage import gaussian_filter1d

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cmn import cmn_scatter, cmn_walkcache  # noqa: E402

SMOOTH_SIGMA = 1.6

# A star's points are thin, so at a given nominal area it reads smaller than a blob of the
# same number -- hence an area well above the 210 the scatter markers use, to keep the same
# visual weight.  Line2D takes a diameter, so a legend handle is sized as sqrt(area).
MEASURED_MARKER_AREA = 330.0

# Light enough to read past: the grid runs under curves, bands and markers alike, and is a
# reading aid rather than anything a panel claims.
GRID_COLOR = "#EBEBEB"
GRID_LINEWIDTH = 0.6

# The two band colours of Figure 1's lower row, with the same meaning: the all-pairs r in the
# colour of the points its subset adds, the retained-subset r in the colour of the bulk it
# covers.  A panel with more than two subsets extends along the same ramp rather than picking
# new hexes, so the set cannot drift out of gamut relative to itself.
CURVE_COLORS = (cmn_scatter.EXCLUDED_COLOR, cmn_scatter.RETAINED_COLOR)


def colors_for(number: int) -> tuple[str, ...]:
    """``number`` curve colours, evenly spaced along ``cmn_scatter``'s own band ramp.

    At two subsets this is exactly ``CURVE_COLORS``, the pair Figure 1 uses, since the ramp is
    sampled between the same two positions.  More subsets extend along cmr.freeze rather than
    reaching for new hexes -- and deliberately NOT along the cmr.fall ramp the DFE row uses,
    because the two rows say different things with colour and a shared palette would invite
    the reader to carry a meaning across.
    """
    if number < 1:
        raise ValueError("a panel needs at least one subset")
    ramp = mpl.colormaps["cmr.freeze"]
    return tuple(mpl.colors.to_hex(ramp(position))
                 for position in np.linspace(*cmn_scatter.BAND_POSITIONS, number))


def cut_key(excluded: float) -> str:
    """``r100``/``r98``/``r90``: the retained percentage, not the dropped one."""
    return f"r{int(round(100 * (1.0 - excluded)))}"


def smooth(trace: np.ndarray) -> np.ndarray:
    """Gaussian filter along the step axis, per subset, with both endpoints held exact."""
    if not np.isfinite(trace).all():
        return trace
    smoothed = gaussian_filter1d(trace, SMOOTH_SIGMA, axis=0, mode="nearest")
    smoothed[0], smoothed[-1] = trace[0], trace[-1]
    return smoothed


def curves(path: str, last_time: int, fractions) -> dict:
    """Median latent and observed traces plus the observed 16-84% band, for one cache.

    ``fractions`` are exclusion fractions, resolved to columns BY VALUE, so the returned
    arrays are already reduced to exactly the subsets asked for and in that order.  Nothing
    downstream can pick the wrong one.
    """
    cache = cmn_walkcache.read(path)
    columns = [cmn_walkcache.column_for(cache, fraction) for fraction in fractions]
    latent = cache["latent"][:, :, columns]
    observed = cache["observed"][:, :, :, columns]
    steps = min(last_time, latent.shape[1] - 1)
    latent = latent[:, : steps + 1, :]
    pooled = observed[:, :, : steps + 1, :].reshape(-1, steps + 1, len(columns))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {
            "times": np.arange(steps + 1),
            "keys": [cut_key(fraction) for fraction in fractions],
            "surviving": np.isfinite(latent[:, :, 0]).sum(axis=0),
            "walks": latent.shape[0],
            "metadata": cache["metadata"],
            "latent": smooth(np.nanmedian(latent, axis=0)),
            "observed": smooth(np.nanmedian(pooled, axis=0)),
            "lower": smooth(np.nanquantile(pooled, 0.16, axis=0)),
            "upper": smooth(np.nanquantile(pooled, 0.84, axis=0)),
        }


def style_axis(axis) -> None:
    """The spine, tick and offset geometry the autocorrelation panels share with Figure 1."""
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_position(("outward", 10))
    axis.spines["left"].set_position(("outward", 10))
    axis.xaxis.set_ticks_position("bottom")
    axis.yaxis.set_ticks_position("left")
    for spine in axis.spines.values():
        spine.set_linewidth(1.5)
    axis.tick_params(axis="both", which="major", length=10, width=1.5)
    axis.tick_params(axis="both", which="minor", length=5, width=1.6)


def draw(axis, panel_curves, ladders=(), *, colors=None, band_alpha=0.12) -> float:
    """Draw one panel's bands, curves and measured stars; return the lowest value drawn.

    ``ladders`` are ``(time, {cut key: r}, note)`` triples.  A vertical rule marks each, so a
    star's position on the step axis is legible as the claim it is -- and where that position
    is a plateau reference rather than a step count, the note says so on the face of the panel.
    """
    times, keys = panel_curves["times"], panel_curves["keys"]
    colors = colors or colors_for(len(keys))
    for column, color in enumerate(colors):
        axis.fill_between(times, panel_curves["lower"][:, column],
                          panel_curves["upper"][:, column],
                          color=color, alpha=band_alpha, linewidth=0)
        axis.plot(times, panel_curves["latent"][:, column], color=color, linewidth=2.7)
        axis.plot(times, panel_curves["observed"][:, column], color=color, linewidth=2.5,
                  linestyle=(0, (4.0, 2.4)))

    floor = min(float(np.nanmin(panel_curves["lower"])),
                float(np.nanmin(panel_curves["latent"])))
    last_time = int(times[-1])
    for time, ladder, note in ladders:
        if time > last_time:
            raise SystemExit(
                f"measured marker at t={time} is outside the {last_time}-step frame")
        missing = [key for key in keys if key not in ladder]
        if missing:
            raise SystemExit(f"marker at t={time} has no value for {', '.join(missing)}")
        axis.axvline(time, color="#777777", linewidth=1.3,
                     linestyle=(0, (2.0, 2.5)), zorder=1)
        for key, color in zip(keys, colors):
            # clip_on stays off so a marker sitting on the frame edge is drawn whole.
            axis.scatter([time], [ladder[key]], s=MEASURED_MARKER_AREA, marker="*",
                         facecolor=color, edgecolor="white", linewidth=1.0,
                         zorder=7, clip_on=False)
        if note:
            axis.annotate(note, xy=(time, max(ladder[key] for key in keys)),
                          xytext=(-4, 16), textcoords="offset points", ha="right",
                          va="bottom", fontsize=12.5, color="#555555",
                          annotation_clip=False)
        floor = min(floor, min(ladder[key] for key in keys))
    return floor


def finish_axis(axis, last_time: int, *, xlabel="Fixed background mutations",
                ylabel=None, title=None, limits=(0.0, 1.0)) -> None:
    """Frame, ticks and the grid every panel carries, applied after ``draw``.

    A light rule at every major tick of both axes, so a plateau height and the step it is
    reached at can be read off without tracking a value back to the spine.  It is pushed
    behind everything: ``set_axisbelow`` drops the whole axis layer below the artists, which
    is what the filled bands need, and the per-line zorder puts it behind anything given a
    zorder of its own below that.
    """
    if title:
        axis.set_title(title, pad=10)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)
    axis.set_xlim(0, last_time)
    axis.set_ylim(*limits)
    spacing = 5 if last_time <= 25 else 10
    axis.xaxis.set_major_locator(FixedLocator(list(range(0, last_time + 1, spacing))))
    style_axis(axis)
    axis.set_axisbelow(True)
    axis.grid(True, which="major", color=GRID_COLOR, linewidth=GRID_LINEWIDTH,
              alpha=1.0, zorder=0)
    for gridline in axis.get_xgridlines() + axis.get_ygridlines():
        gridline.set_zorder(0)


def cut_legend(axis, fractions, colors=None, labels=None, **placement):
    """Colour key: which ancestral subset each band is.

    Labelled by the retained percentage (``r_90%``) unless ``labels`` names each subset
    (``("All", "Bulk")`` -> ``r_All``, ``r_Bulk``), for a cut that is not a round number.
    """
    colors = colors or colors_for(len(fractions))
    if labels is None:
        labels = [rf"{int(round(100 * (1.0 - excluded)))}\%" for excluded in fractions]
    else:
        labels = [rf"\mathrm{{{label}}}" for label in labels]
    return axis.legend(
        [Line2D([], [], color=color, linewidth=2.7) for color in colors],
        [rf"$r_{{{label}}}$" for label in labels],
        loc="lower left", frameon=False, handlelength=2.0, labelspacing=0.35, **placement)


def style_legend(axis, **placement):
    """Style key: latent against noisy against measured.  Identical in every panel."""
    return axis.legend(
        [Line2D([], [], color="#555555", linewidth=2.7),
         Line2D([], [], color="#555555", linewidth=2.5, linestyle=(0, (4.0, 2.4))),
         Line2D([], [], marker="*", linestyle="none",
                markersize=np.sqrt(MEASURED_MARKER_AREA),
                markerfacecolor="#777777", markeredgecolor="white")],
        ["Latent", "Noisy", "Measured"],
        loc="lower left", frameon=False, handlelength=2.4, labelspacing=0.35, **placement)


def place_beside(axis, anchor_legend, fractions, colors=None, gap=0.045, labels=None):
    """Set a colour key down just clear of the style key it sits beside.

    The gap is measured, not guessed: how wide the style key is in axes coordinates is known
    only after a layout pass, and it moves with the font size and the panel width.  Call after
    ``figure.canvas.draw()``.
    """
    right = anchor_legend.get_window_extent().transformed(axis.transAxes.inverted()).x1
    return cut_legend(axis, fractions, colors, labels, bbox_to_anchor=(right + gap, 0.0),
                      bbox_transform=axis.transAxes)
