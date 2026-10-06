r"""Figure S2: isogenic Limdi controls -- four replicates of fig2 D.

Fig2 D shows one paired-effect scatter of a Limdi library measured twice, green- against
red-reference, in a single background with zero evolution between the two numbers.  Whatever
decorrelation it shows is therefore assay noise, not epistasis, and it is what calibrates the
evolved panels beside it.  This figure repeats that control in four independent backgrounds --
the REL606 ancestor and three 50K LTEE clones -- so the noise floor behind fig S1 is not read
off a single library.

    A  REL606      B  ARA+2      C  ARA-3      D  ARA-6

Panels are drawn by the same code as fig2 row 2 (``cmn/cmn_scatter.py``): a hexagonal map of
the fraction of mutants per bin on a per-panel log scale, the identity line, an outlined
near-neutral bulk, and Pearson r over every pair (r_All) and again after dropping the largest
p* of |first measurement| (r_Bulk) -- the pooled Limdi half-max edge from Table S5, the same cut
as fig2 D-E, so each control is cut at the same fraction as the transitions it calibrates.  The exclusion is defined only from the x measurement and never from
y, so the retained subset is not conditioned on the outcome whose correlation is reported, and
ranking on |s| keeps it symmetric about zero.

Genes are matched on metadata row index, the shared gene identity across the Limdi matrices;
see the block comment in ``cmn/cmn_exper.py`` for why the labelled CSV must not be used.  The
two channels of one library cover exactly the same genes -- the missing-value sentinel marks a
gene absent from both at once, never from just one -- so no intersection is needed here.

As in fig2, every panel's axes start at ``SCATTER_FLOOR`` (-0.6) and run up to the top of its
own data envelope, with x and y sharing those limits so the identity line is the panel
diagonal.  Points below the floor are hidden only; they still count toward the densities and
the Pearson r.

Run from anywhere:  python code_figs/figS2_limdi_controls.py
Output:             figs_paper/figS2_limdi_controls.pdf
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn import cmn_exper, cmn_scatter  # noqa: E402
from cmn.cmn_scatter import (  # noqa: E402
    SCATTER_FLOOR, envelope_limits, limdi_half_max_cut,
)

cmn_scatter.apply_style()

OUT_DIR = os.path.join(_REPO_ROOT, "figs_paper")

# (library, panel title), titled like fig2 D.  REL606 is the Ara- founder; the other three
# are 50K clones.
CONTROLS = (
    ("REL606", "Isogenic control (REL606, LB)"),
    ("Ara+2",  "Isogenic control (ARA+2 50K, LB)"),
    ("Ara-3",  "Isogenic control (ARA-3 50K, LB)"),
    ("Ara-6",  "Isogenic control (ARA-6 50K, LB)"),
)


def control_pair(pop):
    """The two unaveraged technical replicates of one Limdi library, as finite arrays."""
    green, red = cmn_exper.limdi_channel_series(pop)
    x = green.to_numpy(float)
    y = red.to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def main():
    panels = []
    for pop, title in CONTROLS:
        x, y = control_pair(pop)
        panels.append({
            "name": f"{pop} green vs red", "x": x, "y": y, "title": title,
            "xlabel": r"Fitness effect $(s)$, measurement 1",
            "ylabel": r"Fitness effect $(s)$, measurement 2",
            "limits": (SCATTER_FLOOR, envelope_limits(x, y)[1]),
        })

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 13.5))
    fig.subplots_adjust(wspace=0.32, hspace=0.34)
    ladders = cmn_scatter.draw_panel_grid(axes, panels,
                                          exclusions=(0.0, limdi_half_max_cut()))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "figS2_limdi_controls.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    for name, results in ladders:
        cmn_scatter.print_correlations(name, results)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
