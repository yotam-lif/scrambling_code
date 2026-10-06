#!/usr/bin/env python3
r"""Figure S8: the autocorrelation panel of Figure 4, for every lineage it does not draw.

Figure 4 shows three columns, chosen because each has a fitted ancestral DFE drawn above it:
REL606 -> Ara-1, REL607 -> Ara+2, and the Couce 0K -> 15K span.  This sheet is the same panel
for the REST of the walk registry, which is what says whether those three were representative
or lucky.  Nothing is drawn twice: the three Figure 4 transitions are left out here, as are

  * Ara-2 and Ara+4 -- the two populations Limdi et al. themselves drop, on measurement-quality
    grounds they state before any cross-background comparison (a few insertions swept and
    biased the Ara-2 assay; Ara+4's technical replicates are poor).  TableS1 carries the
    verification, and TableS1/TableS2 still report them.
  * DM25 0K -> 2K and 0K -> 15K -- both start from the 0K landscape Figure 4 already shows.
    2K -> 15K is the one DM25 interval on a landscape no other panel draws, since a walk out
    of 2K starts where the population actually was at 2K and so carries its own fit.

    Row 1  REL606 -> Ara-3, Ara-4, Ara-5, Ara-6   (LB, Limdi)
    Row 2  REL607 -> Ara+1, Ara+3, Ara+5, Ara+6   (LB, Limdi)
    Row 3  ARA+2 2K -> 15K                        (DM25, Couce)

Every ROW is one FGM fit, which is why the DM25 interval gets a row to itself rather than a
fifth cell in an LB row.  All four clones in an LB row descend from the same ancestor, so they
are simulated on the same landscape and differ only in their observation layer -- the matched
gene set and the published per-gene errors of that particular clone, so the spread between
the four curves in a row is measurement noise and nothing else.

TITLES.  Figure 4's: the registry label, ``ARA-3``, ``ARA+2 2K->15K``, minus the medium tag the
block header already carries.

LEGEND.  One boxed key for the whole sheet, in the empty cells of the DM25 row: simulations /
measured, then r_All / r_Bulk as fig2 and Figure 4 name the subsets.  Figure 4 keeps its keys
inside a panel, but at four panels to a row there is no corner of a panel the curves leave
free.  One key serves every row because the subsets are named, not
numbered: r_Bulk is the p* cut in LB and the 2% cut in DM25, in the same colour either way,
and the caption gives both fractions.

WHAT EACH PANEL DRAWS.  Figure 4's bottom row, through the same ``cmn_walkpanel.draw``: the
solid curve is the simulated walks re-measured with rank-matched published errors, the band
their 16-84% interval, and the stars are measured.  As in Figure 4, the latent (noise-free)
correlation is not drawn; it is still in the caches.

WHERE THE STARS SIT.  The t = 0 star is the ISOGENIC control -- one library against its own
two channels, nothing fixed in between -- recomputed on the panel's own |s|-ranked ladder, as
Figure 4 does; it is TableS1's ``<ancestor> green -> red`` row, and agrees with that table's
r_89 column.  It is the ceiling the panel's curves start from.  The
Couce release publishes no replicate of a timepoint, so the DM25 panel has no t = 0 star; see
the FIT-VARIANT ROWS block in TableS2 for why its fitted1/fitted2 pair is not one.

The right-hand star is placed differently in the two blocks, and the difference is the point.
An LTEE clone at 50K carries between 70 and 2600 mutations, almost all of them hitchhikers no
SSWM walk would fix, so the LB rows put their measured star at the right-hand EDGE (t = 20, as
in Figure 4), labelled with the real count: it is a plateau reference, not a claim about how
many mutations fixed.
The 2K -> 15K interval is short enough to sit inside the walk, so its star goes at the
substitution count itself.

SUBSETS.  Two per panel, as in Figure 4: the whole library (r_All), and the library minus its
largest-|s| ancestral fraction (r_Bulk) -- in LB the pooled Limdi half-max edge p* from Table S5
(also fig2 D-E and figs S1-S2); in DM25 2%, since the Couce effects are compact enough that a
10% cut would reach inside the bulk.

Reads the caches written by ``code_figs/sim_walk_caches.py``; it does not re-run them.

Run from anywhere:  python code_figs/figS8_lineage_autocorr.py
Output:             figs_paper/figS8_lineage_autocorr.pdf
"""

import os
import sys

import cmasher  # noqa: F401  (registers the cmr.* colormaps with matplotlib)
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cmn import cmn_exper, cmn_walkpanel, cmn_walksim  # noqa: E402
from cmn.cmn_scatter import R_LABELS, limdi_half_max_cut  # noqa: E402

# Figure 4's typography.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16

OUT_DIR = os.path.join(_REPO_ROOT, "figs_paper")

# Per-medium display rules, Figure 4's: the pooled p* in LB and 2% in DM25.  ``steps`` is
# Figure 4's 20 for LB -- the walks peak after a median of about 18 steps and write NaN
# afterwards, so the last few steps already run over fewer than all 500, which the console
# reports -- while the DM25 walks are held at their peak and all contribute.
DISPLAY = {
    "LB":   {"fractions": (0.00, limdi_half_max_cut()), "steps": 20},
    # 25 rather than Figure 4's 30: the only DM25 interval left here is 2K -> 15K, whose
    # substitution count is 22, and a frame wider than the marker it carries is empty space.
    "DM25": {"fractions": (0.00, 0.02), "steps": 25},
}

# Frame reaching below zero rather than clipping a star off the bottom of the sheet.  r = 1 is
# still where every curve starts and r = 0 is still no correlation left, so the scale is read
# against the same two landmarks Figure 4 uses.
Y_LIMITS = (-0.1, 1.0)

# What this sheet leaves out, and why.  Named here rather than imported, since figure and table
# scripts in this repo do not import one another; the module docstring argues each entry.
LIMDI_OMITTED = {"Ara-1": "drawn in Figure 4",
                 "Ara-2": "biased by sweeping insertions",
                 "Ara+2": "drawn in Figure 4",
                 "Ara+4": "poor technical replicates"}
COUCE_OMITTED = {("0K", "2K"): "starts from the 0K landscape Figure 4 draws",
                 ("0K", "15K"): "drawn in Figure 4"}

# Four LB clones per row, and the single DM25 interval alone on the row below them.  The three
# unused cells of that row are simply not created.
COLUMNS = 4
ROWS = 3
DM25_ROW = 2
# The two LB rows and the DM25 row are laid out as two nested grids rather than three rows of
# one, because a single uniform hspace does not put the same amount of clear space above each
# block header: only the lower LB row carries an x-axis name, and that name eats into the gap
# the DM25 header then has to sit in.  ROW_GAP is the inner (LB) gap; OUTER_GAP is set so the
# gap below the LB block comes out a little wider, by exactly what that x label takes.
ROW_GAP = 0.62
OUTER_GAP = 0.44
# The sheet's one key has a row's worth of empty cells to itself, so it is set larger than the
# panel text.
KEY_FONTSIZE = 22


def control_ladder(transition, fractions):
    """The isogenic t = 0 ceiling for one ancestor: its green -> red channel pair.

    Recomputed on the same |s|-ranked ladder as every other marker, as Figure 4 does.  It is
    TableS1's ``<ancestor> green -> red`` row and agrees with that table's columns.
    """
    green, red = cmn_exper.limdi_channel_series(transition.ancestor)
    green, red = green.to_numpy(float), red.to_numpy(float)
    finite = np.isfinite(green) & np.isfinite(red)
    green, red = green[finite], red[finite]
    order = np.argsort(np.abs(green), kind="stable")
    ladder = {}
    for excluded in fractions:
        kept = order[: green.size - int(np.floor(excluded * green.size))]
        ladder[cmn_walkpanel.cut_key(excluded)] = float(
            np.corrcoef(green[kept], red[kept])[0, 1])
    return ladder


def markers_for(transition, profile, fractions, last_time):
    """``(time, ladder, note)`` triples for one panel.

    LB transitions get the isogenic ceiling at t = 0 and a plateau reference at the right-hand
    edge, labelled with the substitution count so its position cannot be read as a step count.
    DM25 intervals get their star at the substitution count itself, which is inside the walk.
    """
    measured = cmn_walksim.empirical_ladder(profile, fractions)
    if transition.read_at == "substitutions":
        return [(transition.substitutions, measured, None)]
    # No inline note on the star: at four panels to a row there is no room above it, so the
    # count is written into the panel's own corner by ``build`` instead.
    return [(0, control_ladder(transition, fractions), None),
            (last_time, measured, None)]


def panel_title(transition):
    """The registry label without its medium tag, which the block header already carries.

    Stripped here rather than in the registry: the tag is right for a table row or a one-panel
    figure, and is only redundant because this sheet groups its panels by medium.
    """
    suffix = f" ({transition.medium})"
    if not transition.label.endswith(suffix):
        raise SystemExit(f"{transition.key}: label {transition.label!r} does not end in "
                         f"{suffix!r}, so the medium tag cannot be stripped")
    return transition.label[: -len(suffix)]


def legend_entries(axis, fractions):
    """The style key's and the colour key's handles and labels, for one figure-level key.

    Built by ``cmn_walkpanel``'s own legend functions and then removed again, so the sheet's key
    cannot drift from the one Figure 4 draws.
    """
    handles, labels = [], []
    for probe in (cmn_walkpanel.style_legend(axis, latent=False),
                  cmn_walkpanel.cut_legend(axis, fractions, labels=R_LABELS)):
        handles += probe.legend_handles
        labels += [text.get_text() for text in probe.get_texts()]
        probe.remove()
    return handles, labels


def panel_grid():
    """Which registry entry sits in each cell, as ``{(row, column): transition}``.

    Rows 1 and 2 are the two LB ancestors in clone order, minus the omitted populations; the
    single surviving DM25 interval opens row 3 on its own.
    """
    grid = {}
    for row, ancestor in enumerate(("REL606", "REL607")):
        lineages = [t for t in cmn_walksim.TRANSITIONS.values()
                    if t.dataset == "limdi" and t.ancestor == ancestor
                    and t.evolved not in LIMDI_OMITTED]
        if len(lineages) != COLUMNS:
            raise SystemExit(f"{ancestor} keeps {len(lineages)} clones after the omissions; "
                             f"the grid is built for {COLUMNS}")
        for column, transition in enumerate(lineages):
            grid[(row, column)] = transition
    couce = [t for t in cmn_walksim.TRANSITIONS.values()
             if t.dataset == "couce" and (t.ancestor, t.evolved) not in COUCE_OMITTED]
    if len(couce) != 1:
        raise SystemExit(f"{len(couce)} DM25 intervals survive the omissions; the grid holds "
                         f"exactly one")
    grid[(DM25_ROW, 0)] = couce[0]
    return grid


def build(path):
    grid = panel_grid()
    figure = plt.figure(figsize=(19.5, 14.4))
    outer = figure.add_gridspec(2, 1, hspace=OUTER_GAP,
                                height_ratios=(2.0 + ROW_GAP, 1.0))
    cells = {}
    block = outer[0].subgridspec(DM25_ROW, COLUMNS, hspace=ROW_GAP, wspace=0.14)
    for row in range(DM25_ROW):
        for column in range(COLUMNS):
            cells[(row, column)] = block[row, column]
    for column, cell in enumerate(outer[1].subgridspec(1, COLUMNS, wspace=0.14)):
        cells[(DM25_ROW, column)] = cell

    axes, shared = {}, None
    for cell in sorted(grid):
        axes[cell] = shared = figure.add_subplot(cells[cell], sharey=shared)

    floor = 1.0
    for (row, column), transition in sorted(grid.items()):
        axis = axes[(row, column)]
        rules = DISPLAY[transition.medium]
        fractions = rules["fractions"]
        profile = cmn_walksim.noise_profile(transition)
        curves = cmn_walkpanel.curves(
            cmn_walksim.find_cache(transition), rules["steps"], fractions)
        last_time = int(curves["times"][-1])
        ladders = markers_for(transition, profile, fractions, last_time)
        floor = min(floor, cmn_walkpanel.draw(axis, curves, ladders, latent=False))
        # Only the lower row of the LB block names the x axis; the top row keeps its tick
        # labels but would otherwise repeat the name straight into the block header below it.
        cmn_walkpanel.finish_axis(
            axis, last_time, title=panel_title(transition),
            xlabel="" if row == 0 else "Fixed background mutations",
            ylabel="Pearson autocorrelation" if column == 0 else None,
            limits=Y_LIMITS)
        # The substitution count belongs in the corner, not on the star: at t = 20 the star is
        # a plateau reference, and writing the count beside it would read as a claim that this
        # many mutations fixed in twenty steps.
        # Under the title rather than in the lower corner: the lower left of each row's first
        # panel is where the keys go, and the r100 curves plateau below 0.9, so the top strip
        # is the one place on every LB panel that nothing else is using.
        if transition.read_at == "plateau":
            axis.text(0.97, 0.995, f"measured at $t = {transition.substitutions}$",
                      transform=axis.transAxes, ha="right", va="top",
                      fontsize=11.5, color="#555555")
        # finish_axis restores the shared axis's tick labels; switch them off again.
        axis.tick_params(axis="y", labelleft=(column == 0))
        axis.tick_params(axis="x", labelbottom=True)
        measured = ladders[-1][1]
        simulated = curves["observed"][ladders[-1][0]]
        print(f"{transition.key:22s} {transition.medium:5s} "
              + "  ".join(f"{key}: meas {measured[key]:+.3f} / sim {value:+.3f}"
                          for key, value in zip(curves["keys"], simulated))
              + f"   walks alive {curves['surviving'][-1]}/{curves['walks']}")

    axes[(0, 0)].set_ylim(*Y_LIMITS)
    if floor < Y_LIMITS[0]:
        print(f"note: lowest drawn value is {floor:+.3f}, clipped by the "
              f"[{Y_LIMITS[0]}, {Y_LIMITS[1]}] y limits")

    for row, text in enumerate(("REL606 ancestor (LB)", "REL607 ancestor (LB)",
                                "ARA+2 lineage (DM25)")):
        axis = axes[(row, 0)]
        axis.text(-0.28, 1.20, text, transform=axis.transAxes,
                  ha="left", va="bottom", fontsize=18, fontweight="heavy")

    # One boxed key for the sheet, centred in the empty cells of the DM25 row.
    empty = [cells[(DM25_ROW, column)].get_position(figure) for column in range(1, COLUMNS)]
    centre = ((empty[0].x0 + empty[-1].x1) / 2, (empty[0].y0 + empty[0].y1) / 2)
    key = figure.legend(*legend_entries(axes[(0, 0)], DISPLAY["LB"]["fractions"]),
                        loc="center", bbox_to_anchor=centre, ncol=2, frameon=True,
                        fontsize=KEY_FONTSIZE, markerscale=1.3, handlelength=3.0,
                        columnspacing=3.0, labelspacing=0.8, borderpad=1.1,
                        fancybox=False)
    key.get_frame().set(edgecolor="#555555", linewidth=1.0)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {path}")


def main():
    build(os.path.join(OUT_DIR, "figS8_lineage_autocorr.pdf"))


if __name__ == "__main__":
    main()
