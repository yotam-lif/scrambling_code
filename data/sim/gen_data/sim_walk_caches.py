#!/usr/bin/env python3
r"""Generate the adaptive-walk caches every autocorrelation figure and table reads.

One cache per transition in ``cmn_walksim.TRANSITIONS`` -- twelve LB lineages from Limdi and
three DM25 intervals from Couce.  The simulation itself lives in ``cmn/cmn_walksim.py``; this
script is the driver, the summaries and the console report.

    python code_figs/sim_walk_caches.py                    # everything that is missing
    python code_figs/sim_walk_caches.py --list             # what the registry holds
    python code_figs/sim_walk_caches.py --select medium=DM25
    python code_figs/sim_walk_caches.py --select dataset=limdi --force
    python code_figs/sim_walk_caches.py --select limdi_REL606_Ara-1
    python code_figs/sim_walk_caches.py --verify-legacy    # audit against the pre-merge caches

``--select`` takes a registry key, ``field=value`` over any field of ``Transition``, or
nothing at all.  That is the whole point of the registry: "every DM25 lineage" is a query, not
a list somebody has to keep in step by hand.  Adding a lineage means adding a registry entry,
not writing another script.

Nothing is recomputed unless it has to be.  Cache identity is the full metadata block, so an
existing file is reused only when every parameter, the subset ladder, the ascertainment rule
and the sample size all still match; change any of them and that transition -- and only that
transition -- runs again.

WHAT IT COSTS.  About 15 s per LB transition and 100 s per DM25 one at the default 500 walks,
so the whole registry is roughly eight minutes on one core.  Cheap enough that regenerating
everything is always preferable to reasoning about which caches are stale.

``--verify-legacy`` is the audit that made the merge safe: it re-runs a handful of walks under
the settings the two pre-merge drivers used (their own fit files, their own ascertainment
rules and ladders) and requires the result to match the caches those drivers wrote, exactly.
It is not part of a normal run -- it needs the old ``poster_fig5_*`` files still on disk -- but
while they are there it proves the engine was merged rather than rewritten.

Output:  data/sim/FGM_HT/walk_<dataset>_<ancestor>_to_<evolved>_w500_e10_m<N>.npz
         and the matching _summary.json
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cmn import cmn_walksim as walksim  # noqa: E402

# Times reported in every summary, clipped to the walk.  0 is r = 1 by construction and the
# rest bracket the stretch over which these walks reach their plateau.
SUMMARY_TIMES = (0, 1, 2, 5, 9, 10, 15, 20, 22, 25)

# The pre-merge caches and the settings that produced them, for --verify-legacy.  Each entry
# is (registry key, glob, overrides, ladder).  The parameters come out of the legacy file's
# own metadata, so this compares the ENGINE, not the fit source.
LEGACY_CASES = (
    ("limdi_REL606_Ara-1",
     "poster_fig5_limdi_REL606_to_Ara_minus_1_without_errors_rank_noise_k1_w*_e*_m*_tbl.npz",
     {"probe_rule": "lower_cut", "termination": "nan"}, (0.00, 0.05, 0.10)),
    ("limdi_REL607_Ara+2",
     "poster_fig5_limdi_REL607_to_Ara_plus_2_without_errors_rank_noise_k1_w*_e*_m*_tbl.npz",
     {"probe_rule": "lower_cut", "termination": "nan"}, (0.00, 0.05, 0.10)),
    ("couce_0K_15K",
     "poster_fig5_couce_0K_to_15K_beta_prime_observed_window_rank_noise_k1_w*_e*_m*_tbl.npz",
     {"probe_rule": "observed_window", "termination": "hold"}, (0.00, 0.02, 0.05, 0.10)),
    ("couce_2K_15K",
     "poster_fig5_couce_2K_to_15K_beta_prime_observed_window_rank_noise_k1_w*_e*_m*_tbl.npz",
     {"probe_rule": "observed_window", "termination": "hold"}, (0.00, 0.02, 0.05, 0.10)),
)


def summary_times(transition):
    """``SUMMARY_TIMES`` plus the step this transition is actually read at.

    Without this a transition whose substitution count is not already in the fixed list --
    8 and 30 for the two Couce intervals -- would silently fall back to reporting its plateau.
    """
    times = set(SUMMARY_TIMES)
    if transition.read_at == "substitutions" and transition.substitutions is not None:
        times.add(int(transition.substitutions))
    return tuple(sorted(times))


def medians(arrays, exclusions, times, read_at):
    """Median latent and pooled-noisy ladders at each requested time, plus the plateau.

    The plateau reads every walk at its OWN last step, which is the only summary that uses all
    of them once a walk has run out of beneficial mutations.
    """
    latent = arrays["latent_correlations"]
    observed = arrays["observed_correlations"]
    lengths = np.asarray(arrays["walk_lengths"], dtype=int)
    keys = [walksim.cut_key(excluded) for excluded in exclusions]

    def pack(latent_values, observed_values):
        observed_values = observed_values.reshape(-1, observed_values.shape[-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return {
                "walks": int(np.isfinite(latent_values[:, 0]).sum()),
                "latent": dict(zip(keys, np.nanmedian(latent_values, axis=0).tolist())),
                "noisy": dict(zip(keys, np.nanmedian(observed_values, axis=0).tolist())),
            }

    out = {str(t): pack(latent[:, t, :], observed[:, :, t, :])
           for t in times if t < latent.shape[1]}
    rows = np.arange(latent.shape[0])
    out["plateau"] = pack(latent[rows, lengths, :], observed[rows, :, lengths, :])
    out["read_at"] = read_at
    return out


def write_summary(path, transition, metadata, arrays, profile, exclusions):
    """Plotted values, measured values and walk lengths, beside the cache they describe."""
    summary_path = path.replace(".npz", "_summary.json")
    lengths = np.asarray(arrays["walk_lengths"], dtype=int)
    summary = {
        "metadata": metadata,
        "measured": walksim.empirical_ladder(profile, exclusions),
        "walk_length_quantiles": np.quantile(
            lengths, (0.0, 0.16, 0.5, 0.84, 1.0)).tolist(),
        "walks_alive": {str(t): int(np.isfinite(
            arrays["latent_correlations"][:, t, 0]).sum())
            for t in summary_times(transition)
            if t < arrays["latent_correlations"].shape[1]},
        "curves": medians(arrays, exclusions, summary_times(transition), transition.read_at),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def report(transition, summary, exclusions):
    """Measured against simulated at the point this transition is meant to be read."""
    curves = summary["curves"]
    where = "plateau" if transition.read_at == "plateau" else str(transition.substitutions)
    if where not in curves:
        where = "plateau"
    measured, simulated = summary["measured"], curves[where]
    keys = [walksim.cut_key(excluded) for excluded in exclusions]
    print(f"  read at {where:8s}  "
          + "  ".join(f"{key}: meas {measured[key]:+.3f} / sim {simulated['latent'][key]:+.3f}"
                      f" / noisy {simulated['noisy'][key]:+.3f}" for key in keys), flush=True)
    print(f"  walk length median {summary['walk_length_quantiles'][2]:.0f} steps, "
          f"{simulated['walks']}/{summary['metadata']['replicates']} walks contribute",
          flush=True)


def verify_legacy(directory):
    """Re-run a few walks under the pre-merge settings and demand the old arrays back.

    Compares the engine alone: the model comes out of each legacy cache's own metadata, so a
    mismatch here means the merged simulation diverged, not that the fit source changed.
    """
    failures = 0
    for key, pattern, overrides, exclusions in LEGACY_CASES:
        matches = sorted(glob.glob(os.path.join(directory, pattern)))
        if len(matches) != 1:
            print(f"  {key:22s} SKIP ({len(matches)} legacy caches match)")
            continue
        path = matches[0]
        with np.load(path, allow_pickle=False) as cached:
            metadata = json.loads(str(cached["metadata"].item()))
            latent = np.asarray(cached["latent_correlations"], dtype=float)
            observed = np.asarray(cached["observed_correlations"], dtype=float)
        if latent.ndim == 4:
            latent, observed = latent[0], observed[0]
        stored = (metadata.get("models") or [metadata["model"]])[0]
        model = walksim.ModelConfig(
            "heavy_tailed", "Heavy-tailed", float(stored["n_fit"]),
            int(stored["n_simulation"]), float(stored["radius"]),
            float(stored["sigma"]), float(stored["mu"]))
        transition = dataclasses.replace(walksim.TRANSITIONS[key], **overrides)
        profile = walksim.noise_profile(transition)
        worst_latent = worst_observed = 0.0
        walks = 3
        for index, (biological, observation) in enumerate(itertools.islice(
                walksim.seed_pairs(transition, int(metadata["replicates"])), walks)):
            this_latent, this_observed, _, _ = walksim.simulate_replicate(
                model, profile, biological, observation,
                probe_rule=transition.probe_rule, termination=transition.termination,
                exclusions=exclusions,
                noise_replicates=int(metadata["noise_replicates_per_walk"]),
                max_walk_steps=int(metadata["max_walk_steps"]))
            worst_latent = max(worst_latent, np.nanmax(np.abs(this_latent - latent[index])))
            worst_observed = max(
                worst_observed, np.nanmax(np.abs(this_observed - observed[index])))
            if not np.array_equal(np.isnan(this_latent), np.isnan(latent[index])):
                worst_latent = np.inf
        ok = worst_latent == 0.0 and worst_observed == 0.0
        failures += 0 if ok else 1
        print(f"  {key:22s} {walks} walks vs {os.path.basename(path)[:52]}: "
              f"max|dlatent|={worst_latent:.1e} max|dobserved|={worst_observed:.1e} "
              f"{'OK' if ok else 'MISMATCH'}")
    return failures


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--select", default=None,
                        help="registry key, field=value (e.g. medium=DM25), or all")
    parser.add_argument("--replicates", type=int, default=walksim.DEFAULT_REPLICATES)
    parser.add_argument("--noise-replicates", type=int,
                        default=walksim.DEFAULT_NOISE_REPLICATES)
    parser.add_argument("--max-walk-steps", type=int,
                        default=walksim.DEFAULT_MAX_WALK_STEPS)
    parser.add_argument("--force", action="store_true",
                        help="recompute even when a matching cache exists")
    parser.add_argument("--list", action="store_true", help="print the registry and stop")
    parser.add_argument("--verify-legacy", action="store_true",
                        help="audit the engine against the pre-merge caches and stop")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        print(f"{'key':24s}{'medium':8s}{'fit':14s}{'probes':17s}"
              f"{'end':6s}{'read at':13s}{'label'}")
        for transition in walksim.TRANSITIONS.values():
            read_at = (f"t={transition.substitutions}"
                       if transition.read_at == "substitutions" else "plateau")
            print(f"{transition.key:24s}{transition.medium:8s}{transition.fit_key:14s}"
                  f"{transition.probe_rule:17s}{transition.termination:6s}"
                  f"{read_at:13s}{transition.label}")
        return
    if args.verify_legacy:
        print("Auditing the merged engine against the pre-merge caches:")
        failures = verify_legacy(walksim.WALK_DIR)
        raise SystemExit(1 if failures else 0)

    selected = walksim.select(args.select)
    print(f"{len(selected)} transition(s); ladder "
          f"{[walksim.cut_key(x) for x in walksim.TAIL_EXCLUSIONS]} "
          f"({walksim.EXCLUSION_MODE} rule), fits from "
          f"{os.path.relpath(walksim.FIT_JSON, _REPO_ROOT)}\n")
    started = time.perf_counter()
    for transition in selected:
        print(f"{transition.key}  [{transition.medium}]  {transition.label}", flush=True)
        path, metadata, arrays = walksim.load_or_run(
            transition, replicates=args.replicates,
            noise_replicates=args.noise_replicates,
            max_walk_steps=args.max_walk_steps, force=args.force)
        profile = walksim.noise_profile(transition)
        summary = write_summary(path, transition, metadata, arrays, profile,
                                walksim.TAIL_EXCLUSIONS)
        report(transition, summary, walksim.TAIL_EXCLUSIONS)
        print(flush=True)
    print(f"Done in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
