r"""Figure S3: ancestor-to-evolved Ascensao scatters -- fig1 E in a second dataset.

Each panel pairs a knockout's effect in the REL606 ancestor against its effect in one of the
two diversified Ara-2 ecotypes, both measured in the same condition, so the only difference
between the two axes is 6.5K generations of evolution:

    A  L (SLR)     B  S (SLR)
    C  S (MNO)     D  L (GHI)

``S`` and ``L`` are the two ecotypes that diversified from the Ara-2 ancestor REL606.  The title
names the evolved side only, since the ancestor is REL606 in every panel and the axis labels
already say which side is which; fig S4 uses the same ``genotype (experiment)`` format for its
controls, where both axes are the one strain.  Each ecotype appears twice, so the loss of
correlation is not a property of one lineage or one medium.  A and B are the two of them in the
SAME experiment, off the same ancestor measurement, which is why they lead and why they come out
on identical limits: everything separating those two panels is which ecotype the genes were
re-measured in.  Two of the four have their own control drawn in fig S4 -- S in MNO and L in GHI
-- so for those the comparison against zero evolution is panel-for-panel rather than by analogy.  Read against fig S4, whose panels are single strains
fit twice in the same experiment with no evolution in between and which uses the same title
format, the drop from r ~ 0.90-0.95 there to what these panels show is epistasis rather than
assay noise.  The experiment code in each title is the whole condition label -- the five
monoculture regimes, GHI on acetate and MNO on DM27.8 at 1:10 among them, are documented in
``cmn/cmn_exper.py`` under ``ASENCAO_MONO``, and repeating the medium in the title would only
crowd it.

Panels are drawn by the same code as fig1 row 2 (``cmn/cmn_scatter.py``).  The partition drops the largest 2% of
|ancestral effect| rather than fig1 E's 10%: the Ascensao DFEs are an order of magnitude more
compact than Limdi's -- their 5th percentile already sits inside the bulk -- so anything deeper
would reach past the large-effect points.  The exclusion is defined only from the x (ancestral) measurement and never from y.

Rows are keyed on ``gene_ID``, never on row position -- gene sets differ substantially between
strains, and L is covered in ~700 fewer genes than R, which is why the two L panels carry
n = 2329 (A) and 2211 (D) against 2842-2914 for S -- so the two indices are intersected per
panel.  Nothing is clipped: each panel's limits are the envelope of its own data, which is why
they differ; A and B coincide only because they share an experiment and hence an ancestor axis.
D is the widest, set by the GHI ancestor's most deleterious knockout (ECB_01353, s = -0.336,
reading -0.064 in L), and C is the narrowest in x and the deepest in y -- the 1:10 regime
compresses the ancestral effects while S still reaches -0.283.

Run from anywhere:  python code_figs/figS3_ascensao_evolved.py
Output:             figs_paper/figS3_ascensao_evolved.pdf
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn import cmn_exper, cmn_scatter  # noqa: E402
from cmn.cmn_scatter import SHALLOW_MAGNITUDE_EXCLUSIONS, envelope_limits  # noqa: E402

cmn_scatter.apply_style()

OUT_DIR = os.path.join(_REPO_ROOT, "figs_paper")

# Genotype label used in the panel titles, matching figS4 and the TableS3 ``comparison`` column.
# The release calls the ancestor REL606; "R" is what fits a title beside S and L.
ECOTYPE_LABEL = {"REL606": "R"}

# (ancestor letter, evolved letter) per panel.  Letters are unique across the release and each
# pair is one experiment, so both members share a condition: R/L and R/S = SLR, O/M = MNO,
# I/H = GHI.  A and B are the two ecotypes in the SAME experiment, which is why they lead.
TRANSITIONS = (("R", "L"), ("R", "S"), ("O", "M"), ("I", "H"))

# The Ascensao core is an order of magnitude tighter than Limdi's, so the inset zooms harder.
ASENCAO_INSET_LIMITS = (-0.03, 0.03)


def transition_pair(ancestor_letter, evolved_letter):
    """Matched combined fits for one ancestor -> evolved pair, intersected on gene_ID."""
    anc = cmn_exper.asencao_mono_series(ancestor_letter)
    evo = cmn_exper.asencao_mono_series(evolved_letter)
    shared = anc.index.intersection(evo.index)
    x = anc.loc[shared].to_numpy(float)
    y = evo.loc[shared].to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def main():
    panels = []
    for ancestor_letter, evolved_letter in TRANSITIONS:
        # ``ancestor`` is REL606 in every panel, so it stays out of the title and survives only
        # in ``name``, the console line, where it identifies the transition being printed.
        folder, ancestor = cmn_exper.ASENCAO_MONO[ancestor_letter][:2]
        ancestor = ECOTYPE_LABEL.get(ancestor, ancestor)
        evolved = cmn_exper.ASENCAO_MONO[evolved_letter][1]
        evolved = ECOTYPE_LABEL.get(evolved, evolved)
        x, y = transition_pair(ancestor_letter, evolved_letter)
        panels.append({
            "name": f"{ancestor} -> {evolved} ({folder})", "x": x, "y": y,
            "title": f"{evolved} ({folder})",
            "xlabel": r"Ancestral effect $(s)$",
            "ylabel": r"Evolved effect $(s)$",
            "limits": envelope_limits(x, y),
        })

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 13.5))
    fig.subplots_adjust(wspace=0.32, hspace=0.34)
    ladders = cmn_scatter.draw_panel_grid(
        axes, panels, exclusions=SHALLOW_MAGNITUDE_EXCLUSIONS,
        inset_limits=ASENCAO_INSET_LIMITS)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "figS3_ascensao_evolved.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    for (name, results), panel in zip(ladders, panels):
        cmn_scatter.print_correlations(name, results)
        print(f"{'':30s} limits {panel['limits']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
