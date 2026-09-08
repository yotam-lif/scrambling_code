r"""Figure 4: ancestral DFEs and the autocorrelation of the walks that leave them.

Six panels on a 2 x 3 grid.  Every COLUMN is one ancestor: its whole measured DFE on top,
and below it the DFE autocorrelation along an adaptive walk started from that same ancestor.

    A  REL606 (LB)   -- Limdi ancestor, 3488 genes above the cut.
    B  REL607 (LB)   -- Limdi ancestor, 3497 genes above the cut.
    C  REL607 (DM25) -- Couce 0K ancestor, 13258 segments above the cut.

    D  ARA-1  (LB)   -- Limdi REL606 -> Ara-1 at 50K, assayed in LB.
    E  ARA+2  (LB)   -- Limdi REL607 -> Ara+2 at 50K, assayed in LB.
    F  ARA+2  (DM25) -- Couce Ara+2, 0K -> 15K, assayed in DM25.

The column pairing is not decorative.  The walk in each bottom panel is simulated in the
heavy-tailed FGM whose parameters were fitted to the DFE drawn directly above it, so a
column reads as one claim: this is the landscape the fit sees, and this is what a walk on
that landscape does to the correlation between an early and a late measurement of the same
mutation.  The walk below and the curve above it are the SAME fit: both read
data/fig3_fgm_fits.json, and each cache records the parameters it used in its own metadata
block, so the two rows of a column cannot come to describe different landscapes.


ROW 1 -- the fitted DFEs
------------------------
The three panels share a y axis, so only the leftmost carries tick labels and only the
leftmost carries the legend; the fitted values are repeated in every panel.

Each panel spans the deleterious tail, the bulk and the beneficial tail at once.  A
single bin width cannot do that -- the bulk needs bins the deep tail would leave empty --
so the histogram is adaptive: fine bins are merged left to right until each carries at
least MIN_COUNT observations.  Horizontal bars are the bin extents, so a wide point in
the tail reads as the wide bin it is.

Two models, both fitted by full unbinned maximum likelihood directly to the measured
effects.  Neither is convolved with the published measurement errors, so each curve is a
prediction for the pooled histogram it is drawn against.

    canonical    Fisher's Geometric Model with isotropic Gaussian mutation vectors:
                 delta ~ N(0, sigma^2 I_n), s = -x.delta - ||delta||^2 / 2.
    heavy-tailed The same geometry with a beta-prime radial mixture on the mutation
                 length -- delta = sqrt(q) omega with q = sigma^2 T and
                 T ~ BetaPrime(n/2, mu).  mu = 1/2 recovers a radial Cauchy; the
                 canonical model is the mu -> infinity limit at fixed sigma^2.

Both likelihoods are conditional on the observed effect clearing LOWER_CUT = -0.5, which
is where both assays stop resolving deleterious effects.

Both fits use every retained effect -- no tail trimming -- so the two logliks are
maximised on the same sample and are directly comparable.

Fits are cached in data/fig3_fgm_fits.json; pass --refit to recompute them.  The file keeps
its name because TableS4_fgm_params.py, cmn/cmn_walksim.py and code_tmp/poster_fig5_couce_noise.py
all read it by that path, and it holds one entry (couce_2K) that no panel here draws but the
2K -> 15K walk starts from.


ROW 2 -- the simulated and measured autocorrelations
----------------------------------------------------
Each panel shows SSWM adaptive walks in the heavy-tailed FGM fitted to that column's
ancestor *without* measurement-error convolution.  Two probe subsets are drawn per panel:
all probes (r100) and the probes surviving a cut of the largest-|s| ancestral effects,
defined once from the noisy ancestral measurement and then held fixed.

The cut follows cmn_scatter, so each panel matches its own scatter panels in fig1 and figs
S1-S4: 10% for the Limdi data, whose effects run out to |s| = 0.65, and 2% for the Couce
data, whose effects are compact enough that a 10% cut would reach inside the bulk.  Hence
r90 in panels D and E and r98 in panel F.  All three rank on |s| and drop the
LARGEST-magnitude fraction, not the signed effect.  Every cache carries a wider ladder than
any one panel draws, and a panel names the FRACTION it wants rather than a column number, so
a dot and the curve it sits on cannot come to mean different things.

Curves are smoothed for display, both endpoints pinned to their raw values; the filter and
the argument that it is cosmetic are in ``cmn/cmn_walkpanel.py``, which draws the panels.

Solid curves are the latent correlation -- the model's own effects, with no measurement
error anywhere.  Dashed curves add rank-matched measurement noise: each simulated mutation
is assigned the published error of the empirical gene at its own effect rank, drawn fresh at
every step for the endpoint and once per replicate for the ancestor.  The band is the
16-84% interval over walk x noise replicates.  Filled stars are the measured
correlations.

Where the markers sit
---------------------
Panels D and E carry a t = 0 marker, F does not.  The t = 0 marker is the ISOGENIC CONTROL:
the same genotype assayed twice, so nothing has fixed between the two measurements and the
only thing separating them is the assay itself.  It is the ceiling the panel's curves start
from -- the simulation asserts r = 1 at t = 0, and the control says what the measurement can
actually deliver there.  Both are read straight out of TableS1 (``REL606 green -> red`` and
``REL607 green -> red``) rather than recomputed, on the same |s|-ranked ladder as every other
marker, so the figure and that table cannot drift apart.

Panel F has no such row to draw.  The Couce release publishes no replicate of a timepoint;
its only same-background pair is ``fitted1`` against ``fitted2``, two fits of the SAME five
read counts.  Their disagreement is 0.29x the published per-segment error where a genuine
replicate scores about 1 (Limdi's green/red channels give 1.5-1.8), so their r = 0.98 is an
upper bound on a ceiling, not a control -- see the FIT-VARIANT ROWS block in
``code_figs/TableS2_couce_autocorr.py``.  Drawn at t = 0 it would sit pinned at the top of
the frame and read as an assay far more reproducible than Limdi's, when it is really the only
column whose t = 0 would not be a replicate.  Better absent than misleading.

Panel F carries two sets of dots.  The right-hand set sits at 22 fixed mutations, the count
the 0K -> 15K walk cache was built around; the left-hand set at 9 is the short 0K -> 2K leg.
(The main text quotes roughly 8 fixed mutations for the 0-2K interval and 22 for 2K-15K, so
the right-hand dots are the later interval rather than the cumulative total; keep the two
consistent when the caption is written.)  The t = 9 correlations are stated in ``PANELS``
rather than recomputed here -- see the comment on that marker.

Panels D and E instead put their dots at the right-hand edge, labelled with the substitution
count they really correspond to, so that their position is not read as a claim about how many
mutations fixed.  Ara-1 is a point-mutator carrying of order 1100 mutations by 50K, almost
all of them hitchhikers that SSWM would never fix; Ara+2 is a non-mutator and carries about
70.  Neither number is a step count.  Fifteen steps is inside the range over which both
simulated walks reach their plateau, so those dots are plateau references placed where the
curves have levelled off.

The panels stop at different times.  The Couce cache holds a walk at its peak once it runs
out of beneficial mutations, so all 500 walks contribute at every step.  The Limdi caches
write NaN instead, and those walks peak after a median of about 19 steps, so beyond about 15
the median is taken over a shrinking and increasingly atypical set of survivors and starts to
rattle.  Panels D and E therefore stop at 15, where 446 and 496 of the 500 walks are still
going -- and stopping both at the same step is also what makes them readable side by side.

Both datasets now ascertain their probe libraries the same way -- inside the measured effect
window of the matched empirical library.  That is not cosmetic: a Pearson r over the whole
library is anchored by its most extreme |s| probes, so admitting simulated mutations the assay
could never have reported moves r100 hard and the cut subsets barely at all (0.117 on Couce
r100 against 0.0002 on r90).  See ``cmn/cmn_walksim.py``.

Reads cached walks from data/FGM_HEAVY_TAILED; it does not re-run them.  Regenerate with
``python code_figs/sim_walk_caches.py``.

Run from anywhere:  python code_figs/fig4_autocorrelation.py
Output:             figs_paper/fig4_autocorrelation.pdf
"""

import argparse
import csv
import json
import os
import sys
import time

import cmasher  # noqa: F401  (registers the cmr.* colormaps with matplotlib)
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import minimize

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn import (  # noqa: E402
    cmn_exper, cmn_fgm, cmn_walkpanel, cmn_walksim,
)
from cmn.cmn_cauchy_fgm import (  # noqa: E402
    cauchy_fgm_dfe_logpdf, cauchy_fgm_dfe_pdf, cauchy_fgm_survival,
)

# ───────────────────────────────────── Style ─────────────────────────────────────
# Typography, spine weight and tick geometry are all kept identical to fig1 so the
# figures read as one set.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 14

DATA_COLOR = "#666666"
DATA_EDGE = "#4A4A4A"

# Row 1 draws two models over one histogram, so its palette is two bands -- exactly the
# structure cmn_scatter's bands have, and picked the same way: two positions along a single
# cmasher ramp rather than two independently chosen hexes, so the pair cannot drift out of
# gamut relative to each other when either is retuned.  cmr.fall runs deep plum -> maroon ->
# rust -> amber -> olive.  0.30 for the heavy-tailed model, because the dark maroon end
# carries the curve that is the figure's claim and reads over the grey data at every density;
# 0.60 for the canonical model, far enough along the ramp to be unmistakably the other colour
# while staying warm rather than turning to the pale olive the top of the ramp ends in.
MODEL_POSITIONS = (0.30, 0.60)         # (heavy-tailed, canonical)
HEAVY_COLOR, CANONICAL_COLOR = (
    mpl.colors.to_hex(mpl.colormaps["cmr.fall"](position))
    for position in MODEL_POSITIONS)

# Row 2's palette, its grid and its marker geometry all live in ``cmn_walkpanel``, which
# draws the panel itself.  They are deliberately a different ramp from row 1's: the two
# rows say different things with colour -- which MODEL up top, which SUBSET below -- and
# reusing one palette across both would invite the reader to carry a meaning between them.

# ─────────────────────────────── Fit configuration ────────────────────────────────
LOWER_CUT = -0.5             # both assays stop resolving below this
N_BOUNDS = (2.0, 500.0)
R_BOUNDS = (1.0e-5, 5.0)
C_BOUNDS = (1.0e-6, 0.2)     # C = n sigma^2
A_BOUNDS = (1.0e-5, 0.2)     # A = r sigma
MU_BOUNDS = (0.10, 5.0)
PLOT_DX = 1.0e-4
DFE_Y_LIMITS = (1.0e-2, 1.0e2)   # shared by all three row-1 panels
MAX_INTEGER_EVALUATIONS = 10  # ceiling on the integer-n profile search

# The fit cache keeps its old name: TableS4_fgm_params.py, cmn/cmn_walksim.py and
# code_tmp/poster_fig5_couce_noise.py all read it by this path, and renaming it would silently
# strand them on a stale copy.
FIT_JSON = os.path.join(_REPO_ROOT, "data", "fig3_fgm_fits.json")
# Caches are located through the registry rather than by filename: the stems carry the
# matched-gene and replicate counts, and no figure should have to track either.
WALK_DIR = cmn_walksim.WALK_DIR
# The t = 0 isogenic controls are read from the supplementary table rather than recomputed,
# so the stars on this figure and the numbers in that table cannot drift apart.
LIMDI_TABLE = os.path.join(_REPO_ROOT, "data", "TableS1_limdi_autocorr.csv")
OUT_DIR = os.path.join(_REPO_ROOT, "figs_paper")


# ─────────────────────────────────── Data loading ─────────────────────────────────
def limdi_effects(population="REL607"):
    effects = cmn_exper.limdi_gene_series(population).to_numpy(float)
    return effects[np.isfinite(effects) & (effects >= LOWER_CUT)]


def couce_effects(timepoint="0K"):
    effects = cmn_exper.load_couce_segment_series(timepoint).to_numpy(float)
    return effects[np.isfinite(effects) & (effects >= LOWER_CUT)]


PLOTTED = ("limdi_REL606", "limdi_REL607", "couce_0K")

DATASETS = {
    "limdi_REL607": {
        "title": "Limdi REL607 ancestor",
        "load": limdi_effects,
        "panel_title": "REL607 (LB)",
        "xlim": (-0.52, 0.106),
        "fine_bin_width": 0.0025,
        "min_count": 8,
    },
    "couce_0K": {
        "title": "Couce 0K ancestor",
        "load": couce_effects,
        "panel_title": "REL607 (DM25)",
        "xlim": (-0.245, 0.102),
        "fine_bin_width": 0.0020,
        "min_count": 12,
    },
    # REL606 is also the ancestor whose no-error heavy-tailed MLE the REL606 -> Ara-1
    # walks in panel D are simulated from.
    "limdi_REL606": {
        "title": "Limdi REL606 ancestor",
        "load": lambda: limdi_effects("REL606"),
        "panel_title": "REL606 (LB)",
        "xlim": (-0.52, 0.106),
        "fine_bin_width": 0.0025,
        "min_count": 8,
    },
    # Fitted but not plotted.  2K is an EVOLVED background, not an ancestor, so it does not
    # belong in a row of ancestral DFEs -- but the 2K -> 15K walk has to start from the
    # landscape the population was actually on at 2K, and that means its own MLE rather than
    # the 0K one.  It is also the only place in either dataset where the same lineage is
    # fitted twice, so the pair 0K/2K is the one direct check on whether the fitted radius
    # really falls as the population climbs.
    "couce_2K": {
        "title": "Couce 2K background",
        "load": lambda: couce_effects("2K"),
        "panel_title": "ARA+2 2K (DM25)",
        "xlim": (-0.245, 0.102),
        "fine_bin_width": 0.0020,
        "min_count": 12,
    },
}


# ─────────────────────────────────── Likelihoods ──────────────────────────────────
class CanonicalLikelihood:
    """Exact conditional Gaussian-mutation FGM likelihood, no error convolution."""

    def __init__(self, effects):
        self.effects = np.asarray(effects, dtype=float)

    @staticmethod
    def survival(n, r, sigma):
        return float(cmn_fgm.fgm_fitness_survival_many_eps(
            LOWER_CUT, n=n, sigma=sigma, r=r, eps=np.array([0.0]))[0])

    def loglik(self, n, r, sigma):
        log_density = cmn_fgm.fgm_fitness_dfe_logpdf(
            self.effects, n=n, sigma=sigma, r=r)
        survival = self.survival(n=n, r=r, sigma=sigma)
        if (np.any(~np.isfinite(log_density))
                or not np.isfinite(survival) or survival <= 0.0):
            return -np.inf
        return float(np.sum(log_density) - self.effects.size * np.log(survival))

    def pdf(self, grid, n, r, sigma):
        return (cmn_fgm.fgm_fitness_dfe_pdf(grid, n=n, sigma=sigma, r=r)
                / self.survival(n=n, r=r, sigma=sigma))


class HeavyLikelihood:
    """Exact conditional beta-prime radial FGM likelihood, no error convolution."""

    def __init__(self, effects):
        self.effects = np.asarray(effects, dtype=float)

    @staticmethod
    def survival(n, r, sigma, mu):
        return float(cauchy_fgm_survival(
            LOWER_CUT, n=n, sigma=sigma, r=r, eps=0.0, mu=mu))

    def loglik(self, n, r, sigma, mu):
        log_density = cauchy_fgm_dfe_logpdf(
            self.effects, n=n, sigma=sigma, r=r, mu=mu)
        survival = self.survival(n=n, r=r, sigma=sigma, mu=mu)
        if (np.any(~np.isfinite(log_density))
                or not np.isfinite(survival) or survival <= 0.0):
            return -np.inf
        return float(np.sum(log_density) - self.effects.size * np.log(survival))

    def pdf(self, grid, n, r, sigma, mu):
        return (cauchy_fgm_dfe_pdf(grid, n=n, sigma=sigma, r=r, mu=mu)
                / self.survival(n=n, r=r, sigma=sigma, mu=mu))


# ──────────────────────────────────── Optimisers ──────────────────────────────────
# Both models are fitted in log(n, C, A[, mu]) coordinates.  Bounds are used only to
# discover local basins; each interior candidate is then polished without bounds, so the
# reported point is a genuine stationary point and not a boundary artefact.

def unpack(theta, heavy, fixed_n=None):
    """Read a log-coordinate vector.  With ``fixed_n`` set, n is pinned and dropped
    from the vector, which then holds log(C, A[, mu]) alone."""
    values = np.exp(np.asarray(theta, dtype=float))
    if fixed_n is None:
        n, c_scale, a_scale = values[:3]
        rest = values[3:]
    else:
        n, (c_scale, a_scale), rest = float(fixed_n), values[:2], values[2:]
    sigma = np.sqrt(c_scale / n)
    parameters = {"n": float(n), "r": float(a_scale / sigma), "sigma": float(sigma)}
    if heavy:
        parameters["mu"] = float(rest[0])
    return parameters


def bounds_for(heavy, fixed_n=None):
    box = [] if fixed_n is not None else [(np.log(N_BOUNDS[0]), np.log(N_BOUNDS[1]))]
    box += [(np.log(C_BOUNDS[0]), np.log(C_BOUNDS[1])),
            (np.log(A_BOUNDS[0]), np.log(A_BOUNDS[1]))]
    if heavy:
        box.append((np.log(MU_BOUNDS[0]), np.log(MU_BOUNDS[1])))
    return box


def multistart_interior(likelihood, starts, heavy, label, fixed_n=None):
    box = bounds_for(heavy, fixed_n)
    lower, upper = np.asarray(box)[:, 0], np.asarray(box)[:, 1]

    def objective(theta):
        parameters = unpack(theta, heavy, fixed_n)
        if not R_BOUNDS[0] <= parameters["r"] <= R_BOUNDS[1]:
            return 1.0e100
        value = likelihood.loglik(**parameters)
        return -value if np.isfinite(value) else 1.0e100

    def interior(result):
        theta = np.asarray(result.x, dtype=float)
        parameters = unpack(theta, heavy, fixed_n)
        margin = 2.0e-3
        return bool(np.isfinite(result.fun)
                    and np.all(theta > lower + margin)
                    and np.all(theta < upper - margin)
                    and (fixed_n is not None
                         or N_BOUNDS[0] * 1.002 < parameters["n"] < N_BOUNDS[1] / 1.002)
                    and R_BOUNDS[0] * 1.002 < parameters["r"] < R_BOUNDS[1] / 1.002)

    bounded = [minimize(objective, np.clip(np.log(start), lower, upper),
                        method="L-BFGS-B", bounds=box,
                        options={"ftol": 1.0e-12, "gtol": 1.0e-7,
                                 "maxiter": 2000, "maxls": 60})
               for start in starts]
    candidates = [result for result in bounded if interior(result)]
    if not candidates:
        raise RuntimeError(f"{label}: no finite interior maximum was found.")

    polished = [minimize(objective, result.x, method="BFGS",
                         options={"gtol": 2.0e-6, "maxiter": 1500})
                for result in candidates]
    eligible = [result for result in polished if interior(result)] or candidates
    best = min(eligible, key=lambda result: result.fun)

    fit = unpack(best.x, heavy, fixed_n)
    fit["loglik"] = float(-best.fun)
    fit["C_n_sigma2"] = fit["n"] * fit["sigma"] ** 2
    fit["A_r_sigma"] = fit["r"] * fit["sigma"]
    diagnostics = {
        "converged": bool(best.success),
        "message": str(best.message),
        "starts": len(starts),
        "interior_bounded_candidates": len(candidates),
        "interior_polished_candidates": len(
            [result for result in polished if interior(result)]),
        "selection_rule": ("highest interior local maximum from multistart; "
                           "final polish has no parameter bounds"),
    }
    return fit, diagnostics


CANONICAL_STARTS = ((2.2, 3.0e-2, 4.0e-2), (3.0, 2.0e-2, 3.0e-2),
                    (5.0, 1.0e-2, 2.0e-2), (10.0, 5.0e-3, 1.0e-2),
                    (30.0, 5.0e-3, 1.0e-2), (100.0, 5.0e-3, 1.0e-2),
                    (5.0, 1.0e-3, 1.0e-2), (12.0, 2.0e-3, 1.5e-2))

HEAVY_STARTS = ((4.0, 1.0e-4, 4.0e-3, 0.13), (4.0, 1.0e-4, 4.0e-3, 0.25),
                (6.0, 1.2e-4, 2.0e-3, 0.13), (7.0, 1.2e-4, 2.0e-3, 0.21),
                (10.0, 2.5e-4, 4.0e-3, 0.20), (12.0, 2.0e-4, 2.5e-3, 0.35),
                (30.0, 2.0e-4, 2.0e-3, 0.20), (10.0, 9.0e-3, 1.6e-2, 1.30),
                (6.0, 2.6e-3, 1.0e-2, 1.02), (20.0, 5.0e-3, 1.2e-2, 0.80))


def integer_n_fit(likelihood, continuous, heavy, label):
    """Refit with n pinned to each integer neighbouring the continuous MLE.

    n counts phenotypic dimensions, so only integers are realisable.  Pinning it to
    floor and ceil of the continuous optimum and re-maximising the remaining parameters
    at each is a profile likelihood in n; the better of the two is the integer fit.
    """
    seed = [continuous["C_n_sigma2"], continuous["A_r_sigma"]]
    if heavy:
        seed.append(continuous["mu"])
    # The profile at a pinned integer sits right next to the continuous optimum, so a
    # modest spread of perturbations around it is enough to rule out a second basin.
    starts = [tuple(seed)]
    for c_factor in (0.5, 1.0, 2.0):
        for a_factor in (0.7, 1.4):
            for mu_factor in ((0.6, 1.5) if heavy else (1.0,)):
                scaled = [seed[0] * c_factor, seed[1] * a_factor]
                if heavy:
                    scaled.append(seed[2] * mu_factor)
                starts.append(tuple(scaled))

    candidates = {}

    def evaluate(n):
        if n in candidates or not N_BOUNDS[0] <= n <= N_BOUNDS[1]:
            return
        fit, diagnostics = multistart_interior(
            likelihood, starts, heavy, f"{label} at n={n}", fixed_n=n)
        candidates[n] = {"fit": fit, "diagnostics": diagnostics}

    neighbours = sorted({int(np.floor(continuous["n"])), int(np.ceil(continuous["n"]))})
    for n in neighbours:
        evaluate(n)
    if not candidates:
        raise RuntimeError(f"{label}: no integer neighbour of n is inside the bounds.")

    def loglik(n):
        return candidates[n]["fit"]["loglik"]

    # The two neighbours are the answer whenever the continuous search really found the
    # maximum.  When it did not -- it can stall in a poor basin -- both neighbours can
    # sit on the same side of the integer profile's peak, so climb outward from the
    # better of them until neither step up nor step down improves.
    chosen = max(candidates, key=loglik)
    improved = True
    while improved and len(candidates) < MAX_INTEGER_EVALUATIONS:
        improved = False
        for step in (chosen - 1, chosen + 1):
            evaluate(step)
            if step in candidates and loglik(step) > loglik(chosen):
                chosen, improved = step, True

    print(f"  {label} integer n: "
          + ",  ".join(f"n={n} loglik={loglik(n):.2f}" for n in sorted(candidates))
          + f"   -> n={chosen}", flush=True)
    return {
        "continuous_n": continuous["n"],
        "neighbours_of_continuous_n": neighbours,
        "n_tested": sorted(candidates),
        "chosen_n": chosen,
        # True when the integer profile beat the continuous optimum, i.e. the continuous
        # search had stalled below the real maximum.
        "beat_continuous_mle": bool(loglik(chosen) > continuous["loglik"]),
        "candidates": {str(n): entry for n, entry in candidates.items()},
        "fit": candidates[chosen]["fit"],
    }


def fit_dataset(label, config):
    effects = config["load"]()
    print(f"{label}: N={effects.size} "
          f"range=({effects.min():.4f}, {effects.max():.4f})", flush=True)

    canonical_fit, canonical_diagnostics = multistart_interior(
        CanonicalLikelihood(effects), CANONICAL_STARTS,
        heavy=False, label=f"{label} canonical")
    # Seed the heavy search with the canonical solution as well as the fixed starts.
    heavy_starts = list(HEAVY_STARTS) + [(
        canonical_fit["n"], canonical_fit["C_n_sigma2"],
        max(canonical_fit["A_r_sigma"], A_BOUNDS[0]), 0.25)]
    heavy_fit, heavy_diagnostics = multistart_interior(
        HeavyLikelihood(effects), heavy_starts, heavy=True,
        label=f"{label} heavy-tailed")

    print(f"  canonical    n={canonical_fit['n']:.3f}  r={canonical_fit['r']:.4f}  "
          f"sigma={canonical_fit['sigma']:.5f}  loglik={canonical_fit['loglik']:.2f}")
    print(f"  heavy-tailed n={heavy_fit['n']:.3f}  r={heavy_fit['r']:.4f}  "
          f"sigma={heavy_fit['sigma']:.5f}  mu={heavy_fit['mu']:.3f}  "
          f"loglik={heavy_fit['loglik']:.2f}", flush=True)

    # n is a dimension count, so the reported fit is the better of its two integer
    # neighbours, with the remaining parameters re-maximised at each.
    canonical_integer = integer_n_fit(
        CanonicalLikelihood(effects), canonical_fit, heavy=False,
        label=f"{label} canonical")
    heavy_integer = integer_n_fit(
        HeavyLikelihood(effects), heavy_fit, heavy=True,
        label=f"{label} heavy-tailed")

    # Both models are maximised on the same sample, so their logliks compare directly.
    comparison = {
        "sample": "all effects above the cut",
        "N": int(effects.size),
        "canonical_loglik": canonical_integer["fit"]["loglik"],
        "heavy_loglik": heavy_integer["fit"]["loglik"],
        "continuous_n_canonical_loglik": canonical_fit["loglik"],
        "continuous_n_heavy_loglik": heavy_fit["loglik"],
    }
    comparison["heavy_minus_canonical"] = (
        comparison["heavy_loglik"] - comparison["canonical_loglik"])
    print(f"  heavy - canonical at integer n = "
          f"{comparison['heavy_minus_canonical']:+.1f} nats "
          f"(N={comparison['N']})", flush=True)
    return {
        "dataset": {
            "name": config["title"],
            "N": int(effects.size),
            "observed_lower_cut": LOWER_CUT,
            "minimum": float(effects.min()),
            "maximum": float(effects.max()),
        },
        "canonical_full_mle": {"fit": canonical_fit,
                               "diagnostics": canonical_diagnostics},
        "heavy_tailed_full_mle": {"fit": heavy_fit, "diagnostics": heavy_diagnostics},
        "canonical_integer_n": canonical_integer,
        "heavy_tailed_integer_n": heavy_integer,
        "loglik_comparison": comparison,
    }


def load_or_fit(refit):
    """Return the stored fits, fitting only the datasets that are missing.

    Fitting is INCREMENTAL rather than all-or-nothing.  Adding a dataset used to
    invalidate the whole cache, so every already-published number was re-optimised
    alongside the new one -- and a multistart search is not bit-reproducible across
    SciPy versions, so figures and tables that quote these fits could move for a reason
    that has nothing to do with the change being made.  Only ``--refit`` refits
    everything now; otherwise a stored entry is kept exactly as it is.
    """
    stored = None
    if os.path.exists(FIT_JSON):
        with open(FIT_JSON, encoding="utf-8") as handle:
            stored = json.load(handle)
    entries = dict(stored.get("datasets", {})) if (stored and not refit) else {}
    missing = [label for label in DATASETS
               if label not in entries or "canonical_integer_n" not in entries[label]]
    if not missing:
        print(f"Loaded cached fits from {FIT_JSON}")
        return stored
    if entries:
        print(f"Cached: {', '.join(sorted(entries))}; fitting: {', '.join(missing)}")

    started = time.perf_counter()
    for label in missing:
        entries[label] = fit_dataset(label, DATASETS[label])
    payload = {
        "analysis": ("Canonical Gaussian and heavy-tailed beta-prime FGM DFE fits, "
                     "full unbinned MLE, no measurement-error convolution"),
        "likelihood": {
            "measurement_error_convolution": False,
            "conditional_on_s_at_least": LOWER_CUT,
            "lower_tail_trim": None,
        },
        "datasets": entries,
    }
    payload["elapsed_seconds"] = time.perf_counter() - started
    os.makedirs(os.path.dirname(FIT_JSON), exist_ok=True)
    with open(FIT_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved {FIT_JSON}  (fitted {len(missing)} of {len(DATASETS)} datasets)")
    return payload


# ══════════════════════════════ Row 2: cached walks ═══════════════════════════════
# Each panel names a transition in ``cmn_walksim.TRANSITIONS`` and the ancestral subsets it
# draws.  Subsets are named as EXCLUDED FRACTIONS, not as column indices: the caches carry one
# ladder wide enough for every consumer, and ``cmn_walkpanel`` resolves a fraction to whatever
# column holds it, so a panel cannot silently draw a different subset under the old label.
#
# The cut follows cmn_scatter, so each panel matches its own scatter panels in fig1 and figs
# S1-S4: 10% for the Limdi data, whose effects run out to |s| = 0.65, and 2% for the Couce
# data, whose effects are compact enough that a 10% cut would reach inside the bulk.  Hence
# r90 in panels D and E and r98 in panel F.
PANELS = (
    {
        "transition": "limdi_REL606_Ara-1",
        "title": "ARA-1 (LB)",
        # Limdi effects run out to |s| = 0.65, so cmn_scatter drops 10%.
        "fractions": (0.00, 0.10),
        # These walks peak after a median of about 19 steps; past 15 the median runs out of
        # surviving walks and becomes noise.
        "display_steps": 15,
        # The leftmost panel is the one a reader meets first, so it carries both keys: the
        # style key for the whole row in the corner, its own colour key beside it.
        "legend": "both",
        # A plateau reference, not a substitution count -- see the module docstring.  The
        # note says so on the face of the figure, since the marker's position would
        # otherwise read as a claim that Ara-1 fixed 15 mutations.
        "markers": ({"time": 0, "control": (LIMDI_TABLE, "REL606 green -> red")},
                    {"time": 15, "pair": ("limdi", "REL606", "Ara-1"),
                     "note": "Measured at $t = 1100$ $\\rightarrow$"}),
    },
    {
        "transition": "limdi_REL607_Ara+2",
        "title": "ARA+2 (LB)",
        "fractions": (0.00, 0.10),
        # 496 of the 500 walks are still going at step 15; stopping here as well as in
        # panel D also keeps the two Limdi panels on one x range.
        "display_steps": 15,
        "legend": "cuts",
        # Ara+2 is a non-mutator and carries about 70 mutations by 50K -- three fewer orders
        # of magnitude than Ara-1's hitchhiker load, and still not a step count.
        "markers": ({"time": 0, "control": (LIMDI_TABLE, "REL607 green -> red")},
                    {"time": 15, "pair": ("limdi", "REL607", "Ara+2"),
                     "note": "Measured at $t = 70$ $\\rightarrow$"}),
    },
    {
        "transition": "couce_0K_15K",
        "title": "ARA+2 (DM25)",
        # Couce effects are compact, so cmn_scatter drops 2% -- see SHALLOW_MAGNITUDE_EXCLUSIONS.
        "fractions": (0.00, 0.02),
        # Terminated walks are held at their peak, so all 500 contribute throughout.
        "display_steps": 25,
        # Its cut is 2% where the two Limdi panels are 10%, so it needs its own colour key.
        "legend": "cuts",
        # The fixed-mutation count the 0K -> 15K cache was built around, plus the
        # shorter 0K -> 2K leg.  The second marker's correlations are supplied
        # directly rather than recomputed from the segment tables: they come from
        # the same ranked-|s| ladder, so they carry the same meaning as the ones
        # ``empirical_ladder`` returns, and they are keyed by retained fraction to
        # match this panel's cuts (r100 and r98).
        # NO t = 0 marker here, unlike D and E.  The Couce release publishes no replicate
        # of a timepoint; its only same-background pair is fitted1 against fitted2, two fits
        # of the SAME read counts, whose disagreement is 0.29x the published per-segment
        # error (a real replicate scores about 1).  It correlates at r = 0.98 and would sit
        # pinned at the top of the frame, reading as an assay far more reproducible than
        # Limdi's when it is really the only column whose t = 0 is not a replicate at all.
        "markers": ({"time": 22, "pair": ("couce", "0K", "15K")},
                    {"time": 9, "ladder": {"r100": 0.48, "r98": 0.25}}),
    },
)


# ─────────────────────────────── Measured correlations ────────────────────────────
def couce_pair(ancestor, evolved):
    """Matched Couce segment effects for two timepoints."""
    early = cmn_exper.load_couce_segment_series(ancestor)
    late = cmn_exper.load_couce_segment_series(evolved)
    shared = early.index.intersection(late.index)
    return early.loc[shared].to_numpy(float), late.loc[shared].to_numpy(float)


def limdi_pair(founder, clone):
    """Matched Limdi per-gene effects, on the genes measured in both libraries.

    Matching is on the row index of the .npy fitness matrices, NOT on the ``Genes`` column
    of ``dfe_data_pandas.csv`` -- that column is mislabelled upstream by a pandas
    index-alignment slip, so matching on it pairs genes by row position.  See the block
    comment in ``cmn/cmn_exper.py``.  Duplicated from TableS1_limdi_autocorr.py rather than
    imported: figure and table scripts in this repo do not import one another.
    """
    early = cmn_exper.limdi_gene_series(founder)
    late = cmn_exper.limdi_gene_series(clone)
    shared = early.index.intersection(late.index)
    return early[shared].to_numpy(float), late[shared].to_numpy(float)


def measured_pair(spec):
    """``(ancestor, evolved)`` effect arrays for one panel's marker."""
    source, early, late = spec
    return {"couce": couce_pair, "limdi": limdi_pair}[source](early, late)


def empirical_ladder(spec, exclusions):
    """Measured r for each retained fraction, ranked on |ancestral effect| as in fig1.

    The largest-|s| ``exclusions`` fraction is dropped, defined from the ancestor side only so
    the retained subset is not conditioned on the outcome whose correlation is reported.  This
    is the same rule cmn_scatter applies, and the same rule the cached walks were simulated
    under, so a marker and the curve it sits on mean the same thing.
    """
    ancestor, evolved = measured_pair(spec)
    finite = np.isfinite(ancestor) & np.isfinite(evolved)
    ancestor, evolved = ancestor[finite], evolved[finite]
    order = np.argsort(np.abs(ancestor), kind="stable")
    ladder = {}
    for excluded in exclusions:
        kept = order[: ancestor.size - int(np.floor(excluded * ancestor.size))]
        ladder[cmn_walkpanel.cut_key(excluded)] = float(
            np.corrcoef(ancestor[kept], evolved[kept])[0, 1])
    return ladder


def control_ladder(table, transition, exclusions):
    """Measured r for each retained fraction, read out of a supplementary table row.

    The t = 0 markers are the ISOGENIC controls: the same genotype measured twice, so the
    only thing separating the two assays is measurement noise and no background mutation
    has fixed between them.  They are the ceiling every curve in the panel starts from --
    r = 1 is what the simulation asserts at t = 0, and these say what the assay itself can
    actually deliver there.

    Read from the table rather than recomputed here because the ranked-|s| ladder in
    TableS1/TableS2 already applies exactly the rule ``empirical_ladder`` applies, on the
    same green/red replicate pairing; recomputing it in this script would duplicate that
    pairing logic for no gain and let the figure drift away from the tables.
    """
    with open(table, encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["transition"] == transition]
    if len(rows) != 1:
        raise SystemExit(f"{os.path.basename(table)} holds {len(rows)} rows for "
                         f"{transition!r}; expected exactly one")
    row = rows[0]
    ladder = {}
    for excluded in exclusions:
        key = cmn_walkpanel.cut_key(excluded)
        column = f"r_{key[1:]}"
        if not row.get(column):
            raise SystemExit(f"{os.path.basename(table)} row {transition!r} has no "
                             f"{column} column for the {key} cut")
        ladder[key] = float(row[column])
    return ladder


def marker_ladder(marker, exclusions):
    """One marker's ``{cut key: r}``: stated in the panel, tabulated, or measured here."""
    if "ladder" in marker:
        missing = [cmn_walkpanel.cut_key(excluded) for excluded in exclusions
                   if cmn_walkpanel.cut_key(excluded) not in marker["ladder"]]
        if missing:
            raise SystemExit(f"Marker at t={marker['time']} has no value for "
                             + ", ".join(missing))
        return marker["ladder"]
    if "control" in marker:
        return control_ladder(*marker["control"], exclusions)
    return empirical_ladder(marker["pair"], exclusions)


# ══════════════════════════════════ Row 1 drawing ═════════════════════════════════
def adaptive_histogram(effects, fine_bin_width, min_count):
    """Variable-width histogram: fine in the bulk, merged where the data run out.

    Fine bins of ``fine_bin_width`` are merged left to right until each carries at least
    ``min_count`` observations.  The bulk keeps the fine width -- it has counts to spare
    -- while the deleterious and beneficial tails coarsen until their Poisson errors mean
    something, which is what lets one panel show the whole DFE on a log axis.
    """
    left = fine_bin_width * np.floor(float(effects.min()) / fine_bin_width)
    right = fine_bin_width * np.ceil(float(effects.max()) / fine_bin_width)
    fine_edges = np.arange(left, right + 1.0001 * fine_bin_width, fine_bin_width)
    fine_counts, fine_edges = np.histogram(effects, bins=fine_edges)

    edges, counts, running = [float(fine_edges[0])], [], 0
    for index, count in enumerate(fine_counts):
        running += int(count)
        if running >= min_count:
            edges.append(float(fine_edges[index + 1]))
            counts.append(running)
            running = 0
    if running > 0 and counts:          # fold the short last bin into its neighbour
        counts[-1] += running
        edges[-1] = float(fine_edges[-1])
    elif running > 0:
        edges.append(float(fine_edges[-1]))
        counts.append(running)

    edges = np.asarray(edges, dtype=float)
    counts = np.asarray(counts, dtype=float)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = counts / (effects.size * widths)
    error = np.sqrt(counts) / (effects.size * widths)
    return centers, density, error, counts, widths


def draw_dfe_panel(axis, effects, grid, canonical, heavy, config):
    xlim = config["xlim"]
    centers, density, error, counts, widths = adaptive_histogram(
        effects, config["fine_bin_width"], config["min_count"])
    visible = (counts > 0) & (centers >= xlim[0]) & (centers <= xlim[1])
    axis.errorbar(centers[visible], density[visible], yerr=error[visible],
                  xerr=0.5 * widths[visible], fmt="o",
                  ms=4.2, mfc=DATA_COLOR, mec=DATA_EDGE, mew=0.45, ecolor=DATA_COLOR,
                  elinewidth=0.8, capsize=0, alpha=0.9, zorder=5)

    shown = (grid >= xlim[0]) & (grid <= xlim[1])
    axis.plot(grid[shown], canonical[shown], color=CANONICAL_COLOR, lw=2.4, zorder=3)
    axis.plot(grid[shown], heavy[shown], color=HEAVY_COLOR, lw=2.6, zorder=4)

    axis.set_yscale("log")
    axis.set_xlim(*xlim)
    cmn_walkpanel.style_axis(axis)


def parameter_block(fit, heavy):
    """Fitted values only -- the colour matches the curve's legend entry."""
    lines = [rf"$n={int(round(fit['n']))}$", rf"$\sigma={fit['sigma']:.2f}$",
             rf"$r={fit['r']:.2f}$"]
    if heavy:
        lines.append(rf"$\mu={fit['mu']:.2f}$")
    return "\n".join(lines)


def build_dfe_row(axes, payload):
    """Draw the three ancestral DFEs.

    Returns the row's one legend and the per-panel fits, because the fitted-value blocks
    are positioned relative to where that legend ends and so cannot be placed until the
    figure has been laid out once.  ``place_parameter_blocks`` finishes the job.
    """
    grid = np.arange(-0.56, 0.1201, PLOT_DX)

    blocks = []
    for axis, label in zip(axes, PLOTTED):
        config = DATASETS[label]
        entry = payload["datasets"][label]
        effects = config["load"]()
        canonical_fit = entry["canonical_integer_n"]["fit"]
        heavy_fit = entry["heavy_tailed_integer_n"]["fit"]

        canonical = CanonicalLikelihood(effects).pdf(
            grid, n=canonical_fit["n"], r=canonical_fit["r"],
            sigma=canonical_fit["sigma"])
        heavy = HeavyLikelihood(effects).pdf(
            grid, n=heavy_fit["n"], r=heavy_fit["r"], sigma=heavy_fit["sigma"],
            mu=heavy_fit["mu"])

        draw_dfe_panel(axis, effects, grid, canonical, heavy, config)
        blocks.append((axis, heavy_fit, canonical_fit))
        # set_ticks_position("left") undoes the shared-axis label suppression, so the
        # labels have to be switched off again on every panel but the first.
        if axis is not axes[0]:
            axis.tick_params(axis="y", labelleft=False)
        axis.set_xlabel(r"Fitness effect $(s)$")
        axis.set_title(config["panel_title"], pad=10)

    # One shared y range.  Both models dive by decades past the edges of the measured
    # support; the curves are allowed to leave the frame rather than set the scale.
    axes[0].set_ylim(*DFE_Y_LIMITS)
    axes[0].set_ylabel("Probability density")

    # The legend goes in the first panel only; the fitted values are repeated in each,
    # in the colour of the curve they belong to, so the blocks need no headings.
    legend = axes[0].legend(
        [Line2D([], [], color=HEAVY_COLOR, lw=2.6),
         Line2D([], [], color=CANONICAL_COLOR, lw=2.4),
         Line2D([], [], marker="o", linestyle="none", ms=4.2, mfc=DATA_COLOR,
                mec=DATA_EDGE)],
        ["HT", "Canonical", "Data"],
        loc="upper left", bbox_to_anchor=(0.012, 0.99), frameon=False,
        fontsize=14, handlelength=2.0, labelspacing=0.42,
        handletextpad=0.7, borderpad=0.0)
    return legend, blocks


def place_parameter_blocks(axes, legend, blocks):
    """Line the fitted-value blocks up across the row, just under where the legend ends."""
    below = legend.get_window_extent().transformed(axes[0].transAxes.inverted()).y0
    for axis, heavy_fit, canonical_fit in blocks:
        for offset, (fit, is_heavy, colour) in enumerate((
                (heavy_fit, True, HEAVY_COLOR),
                (canonical_fit, False, CANONICAL_COLOR))):
            axis.text(0.040 + 0.285 * offset, below - 0.035,
                      parameter_block(fit, heavy=is_heavy),
                      transform=axis.transAxes, ha="left", va="top",
                      fontsize=13.5, linespacing=1.35, color=colour)


# ══════════════════════════════════ Row 2 drawing ═════════════════════════════════
# The panel itself -- bands, curves, stars, smoothing, legends and grid -- is drawn by
# ``cmn_walkpanel``, which the all-lineages supplement calls the same way.  What stays here is
# the three-column arrangement: which transitions, which subsets, where the markers go, and
# which panel carries which legend.
def build_autocorr_row(axes):
    """Draw the three autocorrelation panels.

    Returns the panels that carry two legends, for ``place_side_legends`` to finish once
    the figure has been laid out.
    """
    floor, side_by_side = 1.0, []
    for index, (axis, panel) in enumerate(zip(axes, PANELS)):
        transition = cmn_walksim.TRANSITIONS[panel["transition"]]
        fractions = panel["fractions"]
        curves = cmn_walkpanel.curves(
            cmn_walksim.find_cache(transition, WALK_DIR),
            panel["display_steps"], fractions)
        ladders = [(marker["time"], marker_ladder(marker, fractions), marker.get("note"))
                   for marker in panel["markers"]]
        floor = min(floor, cmn_walkpanel.draw(axis, curves, ladders))
        last_time = int(curves["times"][-1])
        # A fixed [0, 1] frame: r = 1 is the value every curve starts at and r = 0 is no
        # correlation left, so the row is read against the two ends of the scale rather than
        # against whatever the lowest band happened to reach.  One shared y axis across the
        # row, so it is named and ticked once, on D.
        cmn_walkpanel.finish_axis(
            axis, last_time, title=panel["title"],
            ylabel="Pearson autocorrelation" if index == 0 else None)
        if index != 0:
            # finish_axis calls set_ticks_position("left"), which undoes the shared axis's
            # label suppression, so the labels have to be switched off again afterwards.
            axis.tick_params(axis="y", labelleft=False)

        for (time, ladder, _), marker in zip(ladders, panel["markers"]):
            if "pair" in marker:
                source = "/".join(marker["pair"])
            elif "control" in marker:
                source = f"control {marker['control'][1]}"
            else:
                source = "stated"
            print(f"{panel['title']} {source} at t={time}: measured "
                  + ", ".join(f"{key}={value:+.3f}" for key, value in ladder.items())
                  + "   simulated observed "
                  + ", ".join(f"{value:+.3f}" for value in curves["observed"][time]))
        print(f"  walks alive at t={last_time}: "
              f"{curves['surviving'][-1]}/{curves['walks']}")

        # Every panel names its own subsets, because 10% and 2% are different cuts and
        # cannot share a label.  The style key -- what solid, dashed and the stars mean,
        # which is the same in all three -- is stated once, in D, with D's colour key
        # immediately to its right.
        if panel["legend"] == "both":
            # Re-adding the style key as an artist keeps the second ``legend`` call on the
            # same axes from replacing it.  The colour key beside it cannot be placed yet --
            # see ``place_side_legends``.
            style = cmn_walkpanel.style_legend(axis)
            axis.add_artist(style)
            side_by_side.append((axis, style, fractions))
        else:
            cmn_walkpanel.cut_legend(axis, fractions)

    # Anything that dips below the frame is therefore cut off, which is worth saying out loud
    # rather than leaving to the eye.
    if floor < 0.0:
        print(f"  note: lowest drawn value is {floor:+.3f}, clipped by the [0, 1] y limits")
    return side_by_side


def place_side_legends(side_by_side):
    """Set each colour key down just clear of the style key it sits beside."""
    for axis, style, fractions in side_by_side:
        cmn_walkpanel.place_beside(axis, style, fractions)


# ════════════════════════════════════ Assembly ════════════════════════════════════
def build(payload, path):
    figure = plt.figure(figsize=(15.0, 12.1))
    # The rows are laid out separately -- each runs its own scale across its three panels,
    # and row 2 gives each panel its own step range -- but with only one set of y tick
    # labels per row they take the same wspace, so the columns line up down the figure.
    outer = figure.add_gridspec(2, 1, hspace=0.36, height_ratios=(1.0, 0.98))
    dfe_grid = outer[0].subgridspec(1, 3, wspace=0.13)
    walk_grid = outer[1].subgridspec(1, 3, wspace=0.13)

    dfe_axes = []
    for column in range(3):
        dfe_axes.append(figure.add_subplot(
            dfe_grid[0, column], sharey=dfe_axes[0] if dfe_axes else None))
    walk_axes = []
    for column in range(3):
        walk_axes.append(figure.add_subplot(
            walk_grid[0, column], sharey=walk_axes[0] if walk_axes else None))

    legend, blocks = build_dfe_row(dfe_axes, payload)
    side_by_side = build_autocorr_row(walk_axes)

    # Both of these are positioned relative to an already-drawn legend, so the canvas has
    # to have been laid out once before they can be placed.
    figure.canvas.draw()
    place_parameter_blocks(dfe_axes, legend, blocks)
    place_side_legends(side_by_side)

    # Sit the letters above the frame, level with the titles, so all six share one offset
    # from their panel -- A no longer has to dodge the 10^2 tick label, and D sits directly
    # under A.
    for axes, tags in ((dfe_axes, "ABC"), (walk_axes, "DEF")):
        for axis, tag in zip(axes, tags):
            axis.text(-0.075, 1.035, tag, transform=axis.transAxes, ha="left",
                      va="bottom", fontsize=18, fontweight="heavy")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit", action="store_true",
                        help="Recompute the DFE fits instead of using the cached JSON.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    payload = load_or_fit(args.refit)
    build(payload, os.path.join(OUT_DIR, "fig4_autocorrelation.pdf"))


if __name__ == "__main__":
    main()
