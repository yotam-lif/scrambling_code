#!/usr/bin/env python3
r"""SSWM adaptive walks in the fitted heavy-tailed FGM, re-measured with published errors.

One walk cache answers one question: if a population really is climbing the heavy-tailed
FGM that was fitted to its ancestor's DFE, how fast does the correlation between an early
and a late measurement of the SAME mutation decay, and how much of that decay is the assay
rather than the landscape?

The engine here replaces two scripts that had diverged -- ``code_tmp/poster_fig5_limdi_noise.py``
and ``code_tmp/poster_fig5_couce_noise.py`` -- whose only real differences were the two knobs
below.  It reproduces both of them bit-for-bit when handed their own settings, which is what
makes the merge auditable rather than a rewrite.


WHAT ONE REPLICATE DOES
-----------------------
A probe library of ``profile.number`` mutation vectors is drawn once, at the ancestor, and
then measured again after every fixation.  The library is the simulation's stand-in for the
assayed knockout/insertion library, so it is drawn ONCE and never redrawn: the whole point is
to follow the same mutations along the walk.

Adaptation runs on a SEPARATE, unconditioned pool of ``BACKGROUND_MUTATIONS`` vectors.  At each
step one beneficial member of that pool fixes, without replacement, with probability
proportional to its latent effect -- the SSWM weighting.  The probes never fix; the background
never gets measured.

Every effect is Malthusian, on ``F(x) = -|x|^2 / 2``:  s = -x.delta - |delta|^2 / 2.


THE TWO KNOBS
-------------
``probe_rule``   which invented mutations are allowed into the measured library.

    observed_window  keep latent s inside the exact [min, max] of the matched empirical
                     library.  This is the rule the whole registry now uses.
    lower_cut        keep latent s >= OBSERVED_LOWER_CUT, with no upper bound.  Retained
                     ONLY so the pre-merge Limdi caches can be reproduced.

    This is not a cosmetic choice.  The Pearson r over the full library is anchored by its
    most extreme |s| probes, so admitting mutations the assay could never have reported
    moves r_100 hard while leaving the cut subsets alone.  Measured on a paired comparison
    (same mutation pool, same background, same fixation stream, only the rule changed):
    Couce 0K -> 15K at t = 15 gives r_100 = 0.730 under ``lower_cut`` against 0.614 under
    ``observed_window`` -- a 0.117 gap, thirty times the Monte-Carlo noise -- while r_90
    moves by 0.0002.  Limdi shifts the other way and far less, +0.012 on r_100, because its
    measured window bottom (-0.5498) sits just BELOW the -0.5 assay cut and so admits a few
    extra deleterious anchors.

``termination``  what a walk does once no beneficial background mutation is left.

    hold  freeze the genotype and keep measuring.  Every walk then contributes at every step,
          so a fixed-time median is over all of them.
    nan   stop writing.  Past the median walk length a fixed-time median is then taken over a
          shrinking and increasingly atypical set of survivors -- which is why TableS1 reads
          these caches at ``time=None``, each walk at its own last step.

    Kept per-dataset rather than harmonised: the tables and Figure 4 already read the two
    conventions correctly through ``cmn_walkcache``, and the difference is a deliberate,
    documented one.


THE OBSERVATION LAYER
---------------------
Evolution NEVER sees measurement noise; it always uses latent effects.  The noise is applied
on top, and only to what gets reported:

  * each simulated mutation is assigned the published error of the empirical gene at its own
    effect rank -- exactly, rank for rank, not by binning or interpolation;
  * the ancestral measurement is drawn once per noise replicate and held fixed along the
    curve, because the real experiment measured its ancestor once;
  * the endpoint measurement is redrawn at every step, because each later timepoint is its
    own assay;
  * the retained subsets are defined once from the NOISY ancestral measurement and then held
    fixed, because that is all an experimenter could have ranked on.


SUBSETS
-------
``TAIL_EXCLUSIONS`` are fractions of the library dropped from the ancestral side, ranked on
|s| and dropping the LARGEST -- ``EXCLUSION_MODE = "magnitude"``, the rule ``cmn_scatter``
applies in Figure 1.  One ladder is written for every cache, a superset of what TableS1
(100/95/90), TableS2 (100/98/95/90) and Figure 4 (100 + one cut) each read, so a transition
has exactly one cache and consumers select rungs by VALUE through
``cmn_walkcache.column_for``, never by column index.


PARAMETERS
----------
Every model comes from ``data/fig3_fgm_fits.json``, the same file Figure 4's top row draws its
DFE curves from, at ``heavy_tailed_integer_n.fit``.  The pre-merge caches were built from two
other fit files holding independent runs of the same multistart search; they agreed on the
integer dimension in every case and differed by about 1% in r and sigma, which moved the
simulated ladders by roughly 0.01.  One file, so a walk and the DFE drawn above it cannot
drift apart.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cmn import cmn_exper  # noqa: E402

DATA_DIR = os.path.join(_REPO_ROOT, "data")
WALK_DIR = os.path.join(DATA_DIR, "sim", "FGM_HT")
FIT_JSON = os.path.join(DATA_DIR, "fig3_fgm_fits.json")

# Bumped when the arrays or their meaning change, so a stale cache cannot be read as current.
# 4 is the first version written by the merged engine.
CACHE_VERSION = 4

# 0.112 is the pooled half-max edge p* of the ten retained Limdi lineages (Table S5, row
# ``pooled``), the Limdi cut of Figures 2 and 4; 0.03 is the Couce cut of Figure 2.  Figure 4
# reads p* from Table S5 and ``cmn_walkcache.column_for`` fails if this ladder no longer holds it.
TAIL_EXCLUSIONS = (0.00, 0.02, 0.03, 0.05, 0.10, 0.112)
EXCLUSION_MODE = "magnitude"

DEFAULT_REPLICATES = 500
DEFAULT_NOISE_REPLICATES = 10
DEFAULT_MAX_WALK_STEPS = 60
DEFAULT_ERROR_SCALE = 1.0
BACKGROUND_MUTATIONS = 3500
# Where both assays stop resolving deleterious effects.  Used only by the legacy
# ``lower_cut`` probe rule; ``observed_window`` takes its bounds from the data.
OBSERVED_LOWER_CUT = -0.5

# The two legacy seed roots, kept so that regenerating a pre-merge cache under its own
# settings reproduces it exactly.  Changing either silently reshuffles every published walk.
LIMDI_MASTER_SEED = 260821
COUCE_MASTER_SEED = 260826


# ────────────────────────────────── Fitted model ──────────────────────────────────
@dataclass(frozen=True)
class ModelConfig:
    """One fitted FGM parameter set, with the dimension rounded to a realisable integer."""

    key: str
    title: str
    n_fit: float
    n: int
    radius: float
    sigma: float
    mu: float | None

    def serializable(self) -> dict:
        return {"key": self.key, "title": self.title, "n_fit": self.n_fit,
                "n_simulation": self.n, "radius": self.radius,
                "sigma": self.sigma, "mu": self.mu}


def load_model(fit_key: str, fit_json: str = FIT_JSON) -> ModelConfig:
    """The heavy-tailed integer-n MLE stored under ``fit_key``.

    ``heavy_tailed_integer_n`` rather than ``heavy_tailed_full_mle``: n counts phenotypic
    dimensions, so the reported fit is the better of the two integer neighbours with the
    remaining parameters re-maximised there, and that is the point the walk must start from.
    """
    with open(fit_json, encoding="utf-8") as handle:
        entries = json.load(handle).get("datasets", {})
    if fit_key not in entries:
        raise RuntimeError(f"{os.path.relpath(fit_json, _REPO_ROOT)} has no fit {fit_key!r}; "
                           f"available: {', '.join(sorted(entries))}")
    fit = entries[fit_key]["heavy_tailed_integer_n"]["fit"]
    return ModelConfig(key="heavy_tailed", title="Heavy-tailed",
                       n_fit=float(fit["n"]), n=int(round(float(fit["n"]))),
                       radius=float(fit["r"]), sigma=float(fit["sigma"]),
                       mu=float(fit["mu"]))


# ─────────────────────────────── The empirical profile ────────────────────────────
@dataclass(frozen=True)
class NoiseProfile:
    """One matched ancestor/evolved pair with its published per-gene errors."""

    ancestor: str
    evolved: str
    ancestor_effects: np.ndarray
    ancestor_errors: np.ndarray
    evolved_effects: np.ndarray
    evolved_errors: np.ndarray

    @property
    def number(self) -> int:
        return int(self.ancestor_effects.size)

    @property
    def window(self) -> tuple[float, float]:
        return (float(np.min(self.ancestor_effects)), float(np.max(self.ancestor_effects)))

    @property
    def ancestor_errors_by_rank(self) -> np.ndarray:
        return self.ancestor_errors[np.argsort(self.ancestor_effects, kind="stable")]

    @property
    def evolved_errors_by_rank(self) -> np.ndarray:
        return self.evolved_errors[np.argsort(self.evolved_effects, kind="stable")]

    def serializable(self) -> dict:
        return {
            "ancestor": self.ancestor,
            "evolved": self.evolved,
            "transition": f"{self.ancestor} -> {self.evolved}",
            "matched_genes": self.number,
            "ancestor_effect_range": list(self.window),
            "evolved_effect_range": [float(np.min(self.evolved_effects)),
                                     float(np.max(self.evolved_effects))],
            "ancestor_error_quantiles": np.quantile(
                self.ancestor_errors, (0.0, 0.16, 0.5, 0.84, 1.0)).tolist(),
            "evolved_error_quantiles": np.quantile(
                self.evolved_errors, (0.0, 0.16, 0.5, 0.84, 1.0)).tolist(),
            "profile_hash": profile_hash(self),
        }


def profile_hash(profile: NoiseProfile) -> str:
    """Stable digest of the four empirical arrays the observation layer consumes."""
    digest = hashlib.sha256()
    for array in (profile.ancestor_effects, profile.ancestor_errors,
                  profile.evolved_effects, profile.evolved_errors):
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _finite_pair(a, a_error, b, b_error, label):
    """Drop rows without a usable effect and a strictly positive error on BOTH sides."""
    a, a_error, b, b_error = (np.asarray(v, dtype=float) for v in (a, a_error, b, b_error))
    valid = (np.isfinite(a) & np.isfinite(b) & np.isfinite(a_error) & np.isfinite(b_error)
             & (a_error > 0.0) & (b_error > 0.0))
    if int(np.sum(valid)) < 3:
        raise RuntimeError(f"{label} has fewer than three usable measurements")
    return a[valid], a_error[valid], b[valid], b_error[valid]


def noise_profile(transition: "Transition") -> NoiseProfile:
    """The matched empirical pair behind one transition.

    Matching is on the loaders' own index in both datasets -- the Limdi row index of the
    ``.npy`` fitness matrices (NOT the mislabelled ``Genes`` column; see the block comment in
    ``cmn_exper``) and the Couce ``alle`` segment id -- which is the same matching TableS1 and
    TableS2 apply, so a simulated curve and the measured number printed beside it are built on
    the same set of genes.
    """
    label = f"{transition.dataset} {transition.ancestor} -> {transition.evolved}"
    if transition.dataset == "limdi":
        a, a_error = cmn_exper.limdi_gene_series(transition.ancestor, errors=True)
        b, b_error = cmn_exper.limdi_gene_series(transition.evolved, errors=True)
        index = a.index.intersection(b.index)
        arrays = (a[index], a_error[index], b[index], b_error[index])
    elif transition.dataset == "couce":
        a = cmn_exper.load_couce_segment_series(transition.ancestor)
        a_error = cmn_exper.load_couce_segment_errors(transition.ancestor)
        b = cmn_exper.load_couce_segment_series(transition.evolved)
        b_error = cmn_exper.load_couce_segment_errors(transition.evolved)
        index = a.index.intersection(b.index).sort_values()
        arrays = (a.loc[index], a_error.loc[index], b.loc[index], b_error.loc[index])
    else:
        raise ValueError(f"no profile loader for dataset {transition.dataset!r}")
    values = _finite_pair(*(x.to_numpy(float) for x in arrays), label=label)
    return NoiseProfile(transition.ancestor, transition.evolved, *values)


# ──────────────────────────────── The registry ────────────────────────────────────
@dataclass(frozen=True)
class Transition:
    """One ancestor -> evolved pair, and everything needed to simulate and read it.

    ``substitutions`` is the count reported for the interval in the source paper.  It is a
    COUNT, never a step index: an LTEE clone at 50K carries of order a thousand mutations,
    almost all of them hitchhikers no SSWM walk would fix.  ``read_at`` says which of the two
    it is safe to use -- "substitutions" when the count is small enough to sit inside the
    walk, "plateau" when it is not and each walk must be read at its own last step.
    """

    key: str
    dataset: str
    medium: str
    ancestor: str
    evolved: str
    fit_key: str
    label: str
    probe_rule: str = "observed_window"
    termination: str = "hold"
    substitutions: int | None = None
    read_at: str = "plateau"


# The 0 -> 50K complement of each Limdi population, as TableS1 reports it.  Mutator lines carry
# of order a thousand; the non-mutators tens.  Neither is a step count -- see ``read_at``.
_LIMDI_SUBSTITUTIONS = {
    "Ara-1": 1100, "Ara-2": 1000, "Ara-3": 800, "Ara-4": 1300, "Ara-5": 90, "Ara-6": 90,
    "Ara+1": 125, "Ara+2": 70, "Ara+3": 1800, "Ara+4": 70, "Ara+5": 80, "Ara+6": 2600,
}
# Mutations fixed DURING each Couce interval, from ltee_muts as quoted in the main text:
# roughly 8 over 0K -> 2K and 22 over 2K -> 15K, hence 30 cumulative.  Small enough to sit
# inside the walk, so these transitions are read at a fixed step.
_COUCE_SUBSTITUTIONS = {("0K", "2K"): 8, ("0K", "15K"): 30, ("2K", "15K"): 22}


def _build_registry() -> dict[str, Transition]:
    registry: dict[str, Transition] = {}
    # LB.  Both ancestors were assayed in LB and every 50K clone descends from one of them,
    # so the walk model is set by the ancestor and only the observation layer differs between
    # the six clones sharing it.  Walks write NaN past their peak; a 50K clone's substitution
    # count is far outside any SSWM walk, so these are read at the plateau.
    for ancestor, clones in cmn_exper.LIMDI_EVOLVED.items():
        for clone in clones:
            key = f"limdi_{ancestor}_{clone}"
            registry[key] = Transition(
                key=key, dataset="limdi", medium="LB", ancestor=ancestor, evolved=clone,
                fit_key=f"limdi_{ancestor}", label=f"{clone.upper()} (LB)",
                probe_rule="observed_window", termination="nan",
                substitutions=_LIMDI_SUBSTITUTIONS[clone], read_at="plateau")
    # DM25.  One lineage, three sequenced timepoints, so the transitions are the two
    # consecutive intervals plus the whole span.  A walk out of 2K starts on the landscape the
    # population was actually on at 2K, hence its own fit rather than the 0K one.
    for early, late in (("0K", "2K"), ("0K", "15K"), ("2K", "15K")):
        key = f"couce_{early}_{late}"
        registry[key] = Transition(
            key=key, dataset="couce", medium="DM25", ancestor=early, evolved=late,
            fit_key=f"couce_{early}", label=f"ARA+2 {early}→{late} (DM25)",
            probe_rule="observed_window", termination="hold",
            substitutions=_COUCE_SUBSTITUTIONS[(early, late)], read_at="substitutions")
    return registry


TRANSITIONS = _build_registry()


def select(expression: str | None = None) -> list[Transition]:
    """Transitions matching ``field=value`` (e.g. ``medium=DM25``), a key, or everything.

    The point of the field form is that a question like "every DM25 lineage" is answered by
    the registry rather than by a list someone has to keep in step with it.
    """
    values = list(TRANSITIONS.values())
    if not expression or expression == "all":
        return values
    if "=" in expression:
        field, wanted = expression.split("=", 1)
        if field not in Transition.__dataclass_fields__:
            raise SystemExit(f"unknown selector field {field!r}; choose from "
                             f"{', '.join(Transition.__dataclass_fields__)}")
        chosen = [t for t in values if str(getattr(t, field)) == wanted]
    else:
        chosen = [t for t in values if t.key == expression]
    if not chosen:
        raise SystemExit(f"no transition matches {expression!r}")
    return chosen


def safe_label(value: str) -> str:
    """Filesystem-safe population spelling, matching the pre-merge cache stems."""
    return value.replace("+", "_plus_").replace("-", "_minus_")


def cache_path(transition: Transition, profile: NoiseProfile, replicates: int,
               noise_replicates: int, directory: str = WALK_DIR) -> str:
    """Where one transition's cache lives.

    The stem carries the sample size because it is the one number that changes when the
    upstream release does, and a cache built against a different gene set must not be picked
    up silently by a glob written for this one.
    """
    stem = (f"walk_{transition.dataset}_{safe_label(transition.ancestor)}_to_"
            f"{safe_label(transition.evolved)}_w{replicates}_e{noise_replicates}_"
            f"m{profile.number}")
    return os.path.join(directory, f"{stem}.npz")



def cache_glob(transition: Transition) -> str:
    """The filename pattern for one transition, wild over the run size and sample size.

    Figures and tables should not have to know how many genes matched or how many walks were
    run -- both change when the upstream release or the replicate count does -- so they glob
    on the transition alone and ``cmn_walkcache.locate`` refuses an ambiguous match.
    """
    return (f"walk_{transition.dataset}_{safe_label(transition.ancestor)}_to_"
            f"{safe_label(transition.evolved)}_w*_e*_m*.npz")


def find_cache(transition: Transition, directory: str = WALK_DIR) -> str:
    """The single cache on disk for one transition; ambiguity or absence is an error."""
    from cmn import cmn_walkcache
    return cmn_walkcache.locate(directory, cache_glob(transition))

# ──────────────────────────────── FGM primitives ──────────────────────────────────
def draw_mutations(rng: np.random.Generator, number: int, model: ModelConfig) -> np.ndarray:
    """Isotropic Gaussian vectors, or the beta-prime radial mixture when ``mu`` is set.

    delta = sqrt(q) * omega with q = sigma^2 T and T ~ BetaPrime(n/2, mu), realised as a
    Gaussian divided by sqrt(2 * Gamma(mu)).  mu = 1/2 is a radial Cauchy; the canonical
    Gaussian model is the mu -> infinity limit at fixed sigma^2.
    """
    deltas = rng.normal(loc=0.0, scale=model.sigma, size=(number, model.n))
    if model.mu is not None:
        scale = rng.gamma(shape=model.mu, scale=1.0, size=(number, 1))
        # Underflow is not expected at these library sizes, but a strictly positive
        # denominator keeps the generator well defined regardless.
        deltas /= np.sqrt(2.0 * np.maximum(scale, np.finfo(float).tiny))
    return deltas


def mutation_effects(position: np.ndarray, deltas: np.ndarray,
                     squared_lengths: np.ndarray | None = None) -> np.ndarray:
    """Malthusian FGM effects of a fixed mutation matrix at one position."""
    if squared_lengths is None:
        with np.errstate(over="ignore", invalid="ignore"):
            squared_lengths = np.einsum("ij,ij->i", deltas, deltas, optimize=True)
    with np.errstate(over="ignore", invalid="ignore"):
        return -(deltas @ position) - 0.5 * squared_lengths


def subset_indices(values, exclusions, mode=EXCLUSION_MODE):
    """Kept-index arrays per exclusion fraction, over the last axis of ``values``.

    Works for a single ancestral trace (1-D) and a stack of noisy ancestral measurements
    (2-D, one row per noise replicate) alike.
    """
    if mode not in ("signed", "magnitude"):
        raise ValueError(f"unknown exclusion mode {mode!r}")
    values = np.asarray(values, dtype=float)
    number = values.shape[-1]
    ranked = np.argsort(values if mode == "signed" else np.abs(values),
                        axis=-1, kind="stable")
    kept = []
    for fraction in exclusions:
        dropped = int(np.floor(fraction * number))
        # signed: the dropped effects are the most deleterious, at the head of the order.
        # magnitude: they are the largest in absolute value, at the tail of it.
        kept.append(ranked[..., dropped:] if mode == "signed"
                    else ranked[..., : number - dropped])
    return kept


def assign_errors_by_rank(values: np.ndarray, errors_by_rank: np.ndarray) -> np.ndarray:
    """Give each value the published error at the identical empirical effect rank."""
    values = np.asarray(values, dtype=float)
    errors_by_rank = np.asarray(errors_by_rank, dtype=float)
    if values.ndim != 1 or errors_by_rank.shape != values.shape:
        raise ValueError("values and errors_by_rank must be equal-length 1-D arrays")
    assigned = np.empty_like(errors_by_rank)
    assigned[np.argsort(values, kind="stable")] = errors_by_rank
    return assigned


def pearson_1d(a: np.ndarray, b: np.ndarray) -> float:
    da, db = a - np.mean(a), b - np.mean(b)
    denominator = np.sqrt(np.sum(da * da) * np.sum(db * db))
    return float(np.sum(da * db) / denominator) if denominator > 0.0 else np.nan


def pearson_rows_at_indices(a: np.ndarray, b: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Row-wise Pearson after selecting row-specific column indices."""
    sa, sb = np.take_along_axis(a, indices, axis=1), np.take_along_axis(b, indices, axis=1)
    da = sa - np.mean(sa, axis=1, keepdims=True)
    db = sb - np.mean(sb, axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(da * da, axis=1) * np.sum(db * db, axis=1))
    result = np.full(a.shape[0], np.nan, dtype=float)
    np.divide(np.sum(da * db, axis=1), denominator, out=result, where=denominator > 0.0)
    return result



def empirical_ladder(profile: NoiseProfile, exclusions=TAIL_EXCLUSIONS,
                     mode=EXCLUSION_MODE) -> dict[str, float]:
    """The MEASURED r for each retained fraction, ranked on the ancestral |effect|.

    Ranked from the ancestor side alone, so the retained subset is never conditioned on the
    outcome whose correlation is being reported.  Same rule the walks are simulated under and
    the same rule ``cmn_scatter`` applies, which is what lets a measured dot sit on a
    simulated curve and mean the same thing.
    """
    kept = subset_indices(profile.ancestor_effects, exclusions, mode)
    return {cut_key(excluded): pearson_1d(profile.ancestor_effects[keep],
                                          profile.evolved_effects[keep])
            for excluded, keep in zip(exclusions, kept)}


def cut_key(excluded: float) -> str:
    """``r100``/``r98``/``r90``: the retained percentage, not the dropped one."""
    return f"r{int(round(100 * (1.0 - excluded)))}"

# ──────────────────────────────── One replicate ───────────────────────────────────
def draw_probe_library(rng, model, position, number, probe_rule, window):
    """Accept-reject a probe library under the transition's ascertainment rule."""
    if probe_rule == "observed_window":
        lower, upper = window
    elif probe_rule == "lower_cut":
        lower, upper = OBSERVED_LOWER_CUT, np.inf
    else:
        raise ValueError(f"unknown probe rule {probe_rule!r}")
    accepted, have = [], 0
    while have < number:
        batch = draw_mutations(rng, max(4096, 2 * (number - have)), model)
        effects = mutation_effects(position, batch)
        keep = np.isfinite(effects) & (effects >= lower) & (effects <= upper)
        if np.any(keep):
            selected = batch[keep]
            accepted.append(selected)
            have += selected.shape[0]
    probes = np.concatenate(accepted, axis=0)[:number]
    return probes, np.einsum("ij,ij->i", probes, probes, optimize=True)


def simulate_replicate(model, profile, biological_seed, observation_seed, *,
                       probe_rule, termination, exclusions=TAIL_EXCLUSIONS,
                       mode=EXCLUSION_MODE, noise_replicates=DEFAULT_NOISE_REPLICATES,
                       error_scale=DEFAULT_ERROR_SCALE,
                       max_walk_steps=DEFAULT_MAX_WALK_STEPS,
                       background_mutations=BACKGROUND_MUTATIONS):
    """One latent walk and its rank-matched noisy observations.

    Two independent generators, so the observation layer cannot perturb the biology: adding
    or removing noise replicates leaves the walk itself untouched.
    """
    biological = np.random.default_rng(biological_seed)
    observation = np.random.default_rng(observation_seed)

    position = np.zeros(model.n, dtype=float)
    position[0] = model.radius
    probes, probe_q = draw_probe_library(
        biological, model, position, profile.number, probe_rule, profile.window)
    ancestral = mutation_effects(position, probes, probe_q)

    latent_indices = subset_indices(ancestral, exclusions, mode)
    ancestor_sigma = assign_errors_by_rank(ancestral, profile.ancestor_errors_by_rank)
    observed_ancestor = (
        ancestral[np.newaxis, :]
        + error_scale * ancestor_sigma[np.newaxis, :]
        * observation.normal(size=(noise_replicates, profile.number)))
    observed_indices = subset_indices(observed_ancestor, exclusions, mode)

    # The ascertainment window describes the ASSAYED library, not the mutations available to
    # adaptation, so the background pool stays unconditioned whatever the probe rule is.
    background = draw_mutations(biological, background_mutations, model)
    background_q = np.einsum("ij,ij->i", background, background, optimize=True)
    available = np.ones(background_mutations, dtype=bool)

    latent = np.full((max_walk_steps + 1, len(exclusions)), np.nan, dtype=float)
    observed = np.full((noise_replicates, max_walk_steps + 1, len(exclusions)),
                       np.nan, dtype=np.float32)
    steps_fixed, finished = 0, False

    for time in range(max_walk_steps + 1):
        current = mutation_effects(position, probes, probe_q)
        evolved_sigma = assign_errors_by_rank(current, profile.evolved_errors_by_rank)
        observed_current = (
            current[np.newaxis, :]
            + error_scale * evolved_sigma[np.newaxis, :]
            * observation.normal(size=(noise_replicates, profile.number)))
        for cut, (latent_kept, observed_kept) in enumerate(
                zip(latent_indices, observed_indices)):
            latent[time, cut] = pearson_1d(ancestral[latent_kept], current[latent_kept])
            observed[:, time, cut] = pearson_rows_at_indices(
                observed_ancestor, observed_current, observed_kept)

        if time == max_walk_steps:
            if not finished:
                steps_fixed = max_walk_steps
            break
        if finished:
            # The genotype no longer changes, but fresh endpoint noise is still drawn above,
            # which is what exposes the true steady plateau rather than a frozen number.
            continue

        candidate = mutation_effects(position, background, background_q)
        beneficial = np.flatnonzero(
            available & np.isfinite(candidate) & (candidate > 0.0))
        weights = candidate[beneficial]
        total = float(np.sum(weights)) if beneficial.size else 0.0
        if beneficial.size == 0 or not np.isfinite(total) or total <= 0.0:
            steps_fixed, finished = time, True
            if termination == "nan":
                break
            continue
        chosen = int(biological.choice(beneficial, p=weights / total))
        position += background[chosen]
        available[chosen] = False
        steps_fixed = time + 1

    return latent, observed, steps_fixed, float(np.linalg.norm(position))


# ───────────────────────────────── Seed trees ─────────────────────────────────────
def seed_pairs(transition: Transition, replicates: int):
    """``(biological, observation)`` seeds per replicate, on the legacy per-dataset roots.

    These are kept exactly as the two pre-merge scripts built them so that regenerating an
    old cache under its own settings reproduces it bit-for-bit.  The Couce key carries the
    endpoint generation, and the early one too whenever it is not 0K -- without that,
    0K -> 15K and 2K -> 15K would draw the same walks.
    """
    if transition.dataset == "limdi":
        # The legacy driver spawned one sub-sequence per model and ran a single model.
        root = np.random.SeedSequence(LIMDI_MASTER_SEED).spawn(1)[0]
    elif transition.dataset == "couce":
        generations = {"0K": 0, "2K": 2000, "15K": 15000}
        key = [COUCE_MASTER_SEED, generations[transition.evolved] // 1000]
        if transition.ancestor != "0K":
            key.append(generations[transition.ancestor] // 1000)
        root = np.random.SeedSequence(key)
    else:
        raise ValueError(f"no seed rule for dataset {transition.dataset!r}")
    for sequence in root.spawn(replicates):
        biological, observation = sequence.spawn(2)
        yield (int(biological.generate_state(1, dtype=np.uint64)[0]),
               int(observation.generate_state(1, dtype=np.uint64)[0]))


# ─────────────────────────────── Run and cache ────────────────────────────────────
def build_metadata(transition, model, profile, *, replicates, noise_replicates,
                   max_walk_steps, error_scale, exclusions, mode) -> dict:
    """Complete cache identity: everything a reader must check before trusting the arrays."""
    return {
        "cache_version": CACHE_VERSION,
        "transition_key": transition.key,
        "dataset": transition.dataset,
        "medium": transition.medium,
        "label": transition.label,
        "fit_path": os.path.relpath(FIT_JSON, _REPO_ROOT),
        # TableS2 checks this: a 2K -> 15K walk that accidentally started at 0K would
        # otherwise be reported as if it had started on the 2K landscape.
        "fit_dataset": transition.fit_key,
        "model": model.serializable(),
        "profile": profile.serializable(),
        "substitutions": transition.substitutions,
        "read_at": transition.read_at,
        "replicates": replicates,
        "noise_replicates_per_walk": noise_replicates,
        "max_walk_steps": max_walk_steps,
        "retained_fractions": [1.0 - fraction for fraction in exclusions],
        "tail_exclusions": list(exclusions),
        "exclusion_mode": mode,
        "probe_mutations": profile.number,
        "background_mutations": BACKGROUND_MUTATIONS,
        "probe_rule": transition.probe_rule,
        "probe_ancestral_effect_window": list(profile.window),
        "probe_ascertainment": (
            "latent ancestral s inside the exact [min, max] of the matched empirical library"
            if transition.probe_rule == "observed_window"
            else f"latent ancestral s >= {OBSERVED_LOWER_CUT}"),
        "termination": transition.termination,
        "background_ascertainment": "unconditioned",
        "background_protocol": (
            "independent finite mutation pool; beneficial mutations fixed without "
            "replacement with probability proportional to latent s"),
        "fitness_convention": "F(x)=-|x|^2/2",
        "effect_convention": "s=-x.delta-|delta|^2/2",
        "error_scale": error_scale,
        "measurement_noise": (
            "independent Gaussian with the published per-gene error assigned by exact "
            "effect rank"),
        "ancestor_measurement": "drawn once per noise replicate and held fixed along the curve",
        "endpoint_measurement": "fresh draw at every walk step",
        "ancestor_error_assignment": "exact empirical ancestor-effect rank",
        "endpoint_error_assignment": "exact empirical current-effect rank",
        "observed_subset_definition": "ranked once from the noisy ancestor and held fixed",
        "evolution_uses_measurement_noise": False,
    }


def run(transition, model, profile, *, replicates=DEFAULT_REPLICATES,
        noise_replicates=DEFAULT_NOISE_REPLICATES, max_walk_steps=DEFAULT_MAX_WALK_STEPS,
        error_scale=DEFAULT_ERROR_SCALE, exclusions=TAIL_EXCLUSIONS, mode=EXCLUSION_MODE,
        progress=True) -> dict:
    """Every replicate of one transition, as the four arrays a cache holds."""
    latent = np.full((replicates, max_walk_steps + 1, len(exclusions)), np.nan, dtype=float)
    observed = np.full((replicates, noise_replicates, max_walk_steps + 1, len(exclusions)),
                       np.nan, dtype=np.float32)
    walk_lengths = np.zeros(replicates, dtype=np.int16)
    endpoint_radii = np.zeros(replicates, dtype=float)

    if progress:
        print(f"  {replicates} walks x {noise_replicates} observations, "
              f"N={profile.number}, n={model.n}, probes={transition.probe_rule}, "
              f"termination={transition.termination}", flush=True)
    for replicate, (biological_seed, observation_seed) in enumerate(
            seed_pairs(transition, replicates)):
        latent[replicate], observed[replicate], length, radius = simulate_replicate(
            model, profile, biological_seed, observation_seed,
            probe_rule=transition.probe_rule, termination=transition.termination,
            exclusions=exclusions, mode=mode, noise_replicates=noise_replicates,
            error_scale=error_scale, max_walk_steps=max_walk_steps)
        walk_lengths[replicate], endpoint_radii[replicate] = length, radius
        if progress and ((replicate + 1) % 100 == 0 or replicate + 1 == replicates):
            print(f"    {replicate + 1}/{replicates}", flush=True)
    return {"latent_correlations": latent, "observed_correlations": observed,
            "walk_lengths": walk_lengths, "endpoint_radii": endpoint_radii}


def validate(arrays, *, replicates, noise_replicates, max_walk_steps, exclusions,
             error_scale, profile) -> None:
    """Fail on broken shapes, a non-unit t=0 latent correlation, or a dead observation layer."""
    if arrays["latent_correlations"].shape != (
            replicates, max_walk_steps + 1, len(exclusions)):
        raise RuntimeError("unexpected latent-correlation shape")
    if arrays["observed_correlations"].shape != (
            replicates, noise_replicates, max_walk_steps + 1, len(exclusions)):
        raise RuntimeError("unexpected observed-correlation shape")
    if not np.allclose(arrays["latent_correlations"][:, 0, :], 1.0, atol=2.0e-12):
        raise RuntimeError("latent ancestral correlations are not all one")
    if not np.all(np.isfinite(arrays["observed_correlations"][:, :, 0, :])):
        raise RuntimeError("a noisy ancestral correlation is non-finite")
    if error_scale > 0.0 and np.allclose(
            arrays["observed_correlations"][:, :, 0, :], 1.0, atol=1.0e-6):
        raise RuntimeError("nonzero measurement noise left every t=0 correlation at one")
    # The rank assignment is the one step where an off-by-one would be invisible downstream.
    probe = np.linspace(-1.0, 1.0, profile.number)[::-1]
    assigned = assign_errors_by_rank(probe, profile.ancestor_errors_by_rank)
    if not np.array_equal(assigned[np.argsort(probe, kind="stable")],
                          profile.ancestor_errors_by_rank):
        raise RuntimeError("exact rank assignment failed")


def load_or_run(transition, *, replicates=DEFAULT_REPLICATES,
                noise_replicates=DEFAULT_NOISE_REPLICATES,
                max_walk_steps=DEFAULT_MAX_WALK_STEPS, error_scale=DEFAULT_ERROR_SCALE,
                exclusions=TAIL_EXCLUSIONS, mode=EXCLUSION_MODE, directory=WALK_DIR,
                force=False, progress=True) -> tuple[str, dict, dict]:
    """The cache for one transition, simulated only if it is missing or out of date.

    Identity is the whole metadata block, so any change of parameters, ladder, ascertainment
    or sample size recomputes rather than silently reusing a file that no longer means what
    its name says.
    """
    model = load_model(transition.fit_key)
    profile = noise_profile(transition)
    metadata = build_metadata(
        transition, model, profile, replicates=replicates,
        noise_replicates=noise_replicates, max_walk_steps=max_walk_steps,
        error_scale=error_scale, exclusions=exclusions, mode=mode)
    metadata_text = json.dumps(metadata, sort_keys=True)
    path = cache_path(transition, profile, replicates, noise_replicates, directory)

    if os.path.exists(path) and not force:
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["metadata"].item()) == metadata_text:
                if progress:
                    print(f"  cached: {os.path.basename(path)}", flush=True)
                return path, metadata, {key: np.asarray(cached[key]) for key in (
                    "latent_correlations", "observed_correlations",
                    "walk_lengths", "endpoint_radii")}
        if progress:
            print("  cache metadata differ; recomputing", flush=True)

    arrays = run(transition, model, profile, replicates=replicates,
                 noise_replicates=noise_replicates, max_walk_steps=max_walk_steps,
                 error_scale=error_scale, exclusions=exclusions, mode=mode,
                 progress=progress)
    validate(arrays, replicates=replicates, noise_replicates=noise_replicates,
             max_walk_steps=max_walk_steps, exclusions=exclusions,
             error_scale=error_scale, profile=profile)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, metadata=np.array(metadata_text), **arrays)
    if progress:
        print(f"  saved: {os.path.basename(path)}", flush=True)
    return path, metadata, arrays
