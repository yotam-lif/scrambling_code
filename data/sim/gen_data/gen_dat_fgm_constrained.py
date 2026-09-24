"""Constrained FGM adaptive walks: the datasets behind the radial- and
angular-scrambling supplementary figures.

Two ways of freezing one half of the FGM geometry so the other half can be timed
on its own:

  radial   -- SSWM walk with the direction to the optimum held fixed
              (FisherFrozenAxis). Orientation cannot wander at all, so all
              progress is radial and the walk descends toward the peak on the
              far-field law R~(t) = R~(0) - t sqrt(pi/2). The scrambling time is
              the collapse time of that descent, tau_rs = sqrt(2/pi) R~(0).

  angular  -- unbiased walk confined to a thin spherical shell of radius R~(0)
              (FisherConstantRadius). The radius cannot change, so the direction
              diffuses on the sphere at fixed R and the scrambling time is the
              rotational one, tau_as = 2 R~(0)^2 / (n - 1).

Both modes record the same observable: the Pearson autocorrelation rho(t) of the
fitness effects of one fixed pool of m mutations, sampled at t = 0. The figures
read the pickles and never simulate.

Usage (from this directory):
    python gen_dat_fgm_constrained.py --mode radial  --reps 500
    python gen_dat_fgm_constrained.py --mode angular --reps 1000
"""

import argparse
import multiprocessing
import os
import pickle

import numpy as np

SIGMA = 0.05
# The frozen walk carries two scalars per mutation instead of an n-vector, so it
# is cheap enough to run the R~(0) grid out to 256 -- far enough that the
# finite-radius excess, which dies as n / R~(0), is visibly gone at every n.
RADIAL_M = 5 * 10 ** 4
RADIAL_R0_VALUES = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
RADIAL_SEED = 20260908

# The angular walk needs ~tau_as = 2 R~(0)^2 / (n-1) steps to decorrelate, which
# blows up at large R~(0) and small n, so the shell sweep stops at R~(0) = 48 and
# runs a smaller mutation pool: m only sets the sampling noise of rho, not its
# expectation, and 10^4 mutations already pin rho to ~1%.
ANGULAR_M = 10 ** 4
ANGULAR_R0_VALUES = [8, 12, 16, 24, 32, 48]
ANGULAR_SEED = 20260909


# ----------------------------------------------------------------
# MODEL CLASSES
# ----------------------------------------------------------------
class FisherModel:
    def __init__(self, n, sigma, m, R0, seed=None):
        self.n = int(n)
        self.sigma = float(sigma)
        self.m = int(m)
        self.rng = np.random.default_rng(seed)
        self.deltas = self.rng.normal(loc=0.0, scale=self.sigma, size=(self.m, self.n))
        self.R0 = float(R0)
        self.r = np.zeros(self.n)
        self.r[0] = self.R0

    def compute_fitness(self, r):
        return np.exp(-0.5 * np.dot(r, r))

    def compute_dfe(self, r):
        w0 = self.compute_fitness(r)
        r_new = r + self.deltas
        r2_new = np.einsum("ij,ij->i", r_new, r_new)
        w_new = np.exp(-0.5 * r2_new)
        return w_new - w0

    @staticmethod
    def normalize(vec):
        norm = np.linalg.norm(vec)
        if norm <= 0:
            return np.zeros_like(vec)
        return vec / norm

    # The two things the recorder asks of a model: a static copy of the mutation
    # pool taken at t = 0, and the state of that pool at the current position.
    def freeze_pool(self):
        self.pool = self.deltas.copy()
        self.pool_norm_sq = np.sum(self.pool ** 2, axis=1)
        self.rhat0 = FisherModel.normalize(self.r)

    def observe(self):
        """(radius, cos to rhat0, effects of the frozen pool) right now.

        The effect is the selection coefficient s = w(r+delta)/w(r) - 1 =
        expm1(malthusian). Scale-free: unlike the absolute difference
        w(r+delta) - w(r), this drops the global exp(-||r||^2/2) prefactor, so the
        effects are comparable across radii rather than being swamped by the
        exponential scale.
        """
        dfe = np.expm1(-0.5 * self.pool_norm_sq - np.dot(self.pool, self.r))
        cos_t = float(np.dot(self.rhat0, FisherModel.normalize(self.r)))
        return float(np.linalg.norm(self.r)), np.clip(cos_t, -1.0, 1.0), dfe


class FisherFrozenAxis:
    """SSWM adaptive walk with the direction to the optimum held fixed,
    isolating *radial* scrambling.

    A mutation is carried as two numbers rather than an n-vector: its component
    along the fixed axis, delta_par ~ N(0, sigma^2), and the squared norm of
    everything orthogonal to it, ||delta_perp||^2 ~ sigma^2 chi^2_{n-1}. The pair
    is a sufficient statistic for the effect: landing at r + delta from radius R
    puts you at ||r + delta||^2 = (R + delta_par)^2 + ||delta_perp||^2, so the
    Malthusian effect is exactly -R delta_par - ||delta||^2 / 2, the same
    expression the vector models evaluate. Dimensionality enters only as the
    chi-square degrees of freedom. The step is the usual SSWM rule -- weight
    proportional to s among the beneficial mutations -- the radius updates
    exactly, and the fixed mutation flips sign, which is what delta -> -delta does
    to delta_par while leaving ||delta_perp||^2 alone.

    What is idealized, stated plainly: in a real walk a fixation moves the
    position off-axis by delta_perp, and the other mutations' parallel components
    should be re-projected onto the new direction. This model lets delta_perp move
    the radius but never the axis. That is exactly the assumption under which the
    radial Pearson law of figS7_fgm_scrambling.py is derived, so this is the
    theory's own simulation rather than an approximation to some constrained walk.
    It replaces a walk confined to a wedge |x_perp|^2 <= sin^2(phi) ||r0||^2, which
    is a fixed tube around rhat0 rather than a cone: its half-angle opens up as
    the walk descends, so near the peak the direction was free to rotate and the
    measurement was of rotation rather than of descent.

    The class does not subclass FisherModel: the state is a scalar radius and the
    pool is two m-vectors, so there is no (m, n) bank to inherit.
    """

    def __init__(self, n, sigma, m, R0, seed=None):
        self.n = int(n)
        self.sigma = float(sigma)
        self.m = int(m)
        self.rng = np.random.default_rng(seed)
        self.par = self.rng.normal(loc=0.0, scale=self.sigma, size=self.m)
        self.perp_sq = self.sigma ** 2 * self.rng.chisquare(self.n - 1, size=self.m)
        self.norm_sq = self.par ** 2 + self.perp_sq   # ||delta||^2, invariant under the sign flip
        self.R0 = float(R0)
        self.R = float(R0)

    @staticmethod
    def effects(R, par, norm_sq):
        return np.expm1(-0.5 * norm_sq - R * par)

    def freeze_pool(self):
        self.pool_par = self.par.copy()
        self.pool_norm_sq = self.norm_sq.copy()

    def observe(self):
        # The cosine to rhat0 is 1 by construction -- the axis is the thing this
        # model holds -- so the recorded cosine track is flat, not informative.
        return self.R, 1.0, self.effects(self.R, self.pool_par, self.pool_norm_sq)

    def step(self):
        dfe = self.effects(self.R, self.par, self.norm_sq)
        valid_indices = np.nonzero(dfe > 0)[0]
        if len(valid_indices) == 0:
            return False

        effects = dfe[valid_indices]
        probs = effects / np.sum(effects)
        choice = self.rng.choice(valid_indices, p=probs)
        self.R = float(np.sqrt((self.R + self.par[choice]) ** 2 + self.perp_sq[choice]))
        # In FGM, if a mutation fixes, the forward mutation flips sign.
        self.par[choice] *= -1.0
        return True


class FisherConstantRadius(FisherModel):
    """Walk confined to the thin shell ||r|| in [R0 - epsilon, R0 + epsilon],
    isolating *angular* scrambling.

    Any mutation that keeps the landing point in the shell is admissible and one
    is drawn uniformly among them -- no selection, since at fixed radius every
    admissible move has (nearly) the same fitness. The walk is therefore an
    unbiased diffusion of the direction rhat on the sphere of radius R0, with no
    net radial drift: the shell is defined by the *initial* R0, so it cannot
    creep. Mutations are not sign-flipped on fixation, since nothing here plays
    the role of the SSWM irreversibility.
    """

    def __init__(self, n, sigma, m, R0, epsilon, seed=None):
        super().__init__(n, sigma, m, R0, seed=seed)
        self.epsilon = float(epsilon)

    def step(self):
        r_candidates = self.r + self.deltas
        dists = np.linalg.norm(r_candidates, axis=1)
        valid_indices = np.nonzero((dists >= self.R0 - self.epsilon)
                                   & (dists <= self.R0 + self.epsilon))[0]
        if len(valid_indices) == 0:
            return False
        choice = self.rng.choice(valid_indices)
        self.r += self.deltas[choice]
        return True


# ----------------------------------------------------------------
# SIMULATION WORKER
# ----------------------------------------------------------------
def record_walk(model, max_t, time_points, hold):
    """(pearson, cosine, radii) for one walk, sampled on time_points.

    The observable is the Pearson autocorrelation of the effects of one fixed pool
    of mutations, copied at t = 0, so what is measured is the decorrelation of a
    fixed set of effects rather than of whichever mutations the walk happens to use.

    hold says what to do once the walk can no longer step. hold=False pads the
    rest of the window with NaN, so the figure can blank the tail where only a
    handful of walks survive -- the right choice for the shell walk, which never
    starves for a geometric reason. hold=True instead repeats the values it
    stopped at: an SSWM walk that has run out of beneficial mutations sits at a
    local optimum and freezes there, so holding it is exact, and it is what makes
    the end-of-run correlation a property of where the descent stopped rather than
    of when the recording did.
    """
    model.freeze_pool()
    dfe0 = model.observe()[2]
    std_dfe0 = np.std(dfe0)

    def measure():
        radius, cos_t, dfe_t = model.observe()
        rho = np.nan
        if std_dfe0 > 1e-12 and np.std(dfe_t) > 1e-12:
            rho = float(np.corrcoef(dfe0, dfe_t)[0, 1])
        return rho, cos_t, radius

    pearsons = np.full(len(time_points), np.nan)
    cosines = np.full(len(time_points), np.nan)
    radii = np.full(len(time_points), np.nan)

    pearsons[0] = 1.0
    cosines[0] = 1.0
    radii[0] = model.observe()[0]

    current_t_idx = 0
    time_points_set = set(int(t) for t in time_points)
    stalled = None                      # the values the walk froze at, once it has

    for t in range(1, max_t + 1):
        if stalled is None and not model.step():
            if not hold:
                break
            stalled = measure()

        if t in time_points_set:
            current_t_idx += 1
            pearsons[current_t_idx], cosines[current_t_idx], radii[current_t_idx] = (
                stalled if stalled is not None else measure())

    return pearsons, cosines, radii


def run_single_replicate(args):
    """One constrained walk, dispatched on the mode."""
    mode, seed, n, sigma, m, R0, params, max_t, time_points = args

    if mode == "radial":
        model = FisherFrozenAxis(n, sigma, m, R0, seed=seed)
    elif mode == "angular":
        model = FisherConstantRadius(n, sigma, m, R0, params["epsilon"], seed=seed)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return record_walk(model, max_t, time_points, hold=(mode == "radial"))


# ----------------------------------------------------------------
# SWEEP
# ----------------------------------------------------------------
MAX_RECORDED = 400   # evaluating the DFE costs about as much as taking a step


def cell_horizon(mode, R0_tilde, n):
    """(max_t, time_points) for one (R~(0), n) cell.

    The two modes decorrelate on wildly different timescales, so each gets a
    horizon scaled to its own: ~5 tau_rs for the radial descent, ~6 tau_as for the
    angular one. Both are long enough that the walk has arrived wherever it is
    going before the window ends, which is what the figure needs: it divides out
    the correlation left at the end of the run, and that is only a floor if the
    run actually reached it. The radial walk starves at a local optimum after
    about one tau_rs and then holds, so the last fifth of its window is flat by
    construction; the angular walk keeps moving, but at 6 tau_as its leftover
    exponential is e^-6 < 0.3% of the decaying amplitude.

    Long windows are subsampled to at most MAX_RECORDED times, which still
    resolves the crossing of 1/e to well inside the replicate noise.
    """
    if mode == "radial":
        tau_rs = R0_tilde / np.sqrt(np.pi / 2.0)
        max_t = int(np.ceil(5.0 * tau_rs)) + 20
    else:
        tau_as = 2.0 * R0_tilde ** 2 / (n - 1.0)
        max_t = int(np.ceil(6.0 * tau_as)) + 20

    if max_t <= MAX_RECORDED:
        return max_t, np.arange(0, max_t + 1)
    return max_t, np.unique(np.rint(np.linspace(0, max_t, MAX_RECORDED)).astype(int))


def simulate_sweep(mode, n_values, r0_tilde_values, reps, params, sigma, m, seed):
    """rho(t) for every (R~(0), n) cell of the sweep, run in parallel over
    replicates. The task seed is deterministic in (R~(0) index, n index,
    replicate), so a given (seed, grid) reproduces the pickle exactly."""
    tasks, spans = [], {}
    for p, R0_tilde in enumerate(r0_tilde_values):
        for k, n in enumerate(n_values):
            max_t, time_points = cell_horizon(mode, R0_tilde, n)
            start = len(tasks)
            for i in range(reps):
                tasks.append((
                    mode,
                    int(seed) + 100_000 * (p + 1) + 10_000 * (k + 1) + i,
                    int(n), sigma, m, R0_tilde * sigma,
                    params, max_t, time_points,
                ))
            spans[(float(R0_tilde), int(n))] = (start, len(tasks), time_points)

    num_proc = min(multiprocessing.cpu_count(), len(tasks))
    print(f"Simulating {len(tasks)} {mode} replicates on {num_proc} processes ...")
    with multiprocessing.Pool(processes=num_proc) as pool:
        results = pool.map(run_single_replicate, tasks, chunksize=4)

    cells = {}
    for key, (start, end, time_points) in spans.items():
        block = results[start:end]
        cells[key] = {
            "time_points": time_points,
            "pearson": np.array([res[0] for res in block], dtype=float),
            "cosine": np.array([res[1] for res in block], dtype=float),
            "radii": np.array([res[2] for res in block], dtype=float),
        }

    return {"mode": mode, "sigma": sigma, "m": m, "reps": reps,
            "base_seed": int(seed), **params,
            "n_values": [int(n) for n in n_values],
            "r0_tilde_values": [float(r) for r in r0_tilde_values],
            "cells": cells}


def output_name(mode, reps, sigma, params):
    if mode == "radial":
        return f"fgm_frozen_sweep_rps{reps}_sig{sigma:g}.pkl"
    return f"fgm_shell_sweep_rps{reps}_sig{sigma:g}_eps{params['epsilon']:g}.pkl"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the constrained-FGM walk sweeps used by "
                    "figS7_fgm_scrambling.py: a frozen-axis SSWM descent "
                    "(radial scrambling) and a constant-radius diffusion (angular "
                    "scrambling), swept over the starting radius R~(0) and n.")
    parser.add_argument("--mode", choices=("radial", "angular"), required=True)
    parser.add_argument("--reps", type=int, default=None,
                        help="Replicates per (R~(0), n) cell. Default: 500 radial, "
                             "1000 angular.")
    parser.add_argument("--n-values", default="4,8,16,32",
                        help="Comma-separated dimensionalities n.")
    parser.add_argument("--r0-values", default=None,
                        help="Comma-separated starting radii R~(0) = R(0)/sigma. "
                             "Default: the sweep grid of the chosen mode.")
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--m", type=int, default=None,
                        help=f"Mutations in the tracked pool. Default: {RADIAL_M} "
                             f"radial, {ANGULAR_M} angular.")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="angular: shell half-width, in units of sigma. "
                             "Default 1.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="../fgm")
    args = parser.parse_args()

    n_values = [int(x) for x in args.n_values.split(",") if x.strip()]
    if args.mode == "radial":
        reps = 500 if args.reps is None else args.reps
        m = RADIAL_M if args.m is None else args.m
        seed = RADIAL_SEED if args.seed is None else args.seed
        r0_values = RADIAL_R0_VALUES
        params = {}
    else:
        reps = 1000 if args.reps is None else args.reps
        m = ANGULAR_M if args.m is None else args.m
        seed = ANGULAR_SEED if args.seed is None else args.seed
        r0_values = ANGULAR_R0_VALUES
        # The shell half-width is one mutation scale, so a typical mutation has a
        # decent chance of landing inside it.
        params = {"epsilon": (1.0 if args.epsilon is None else args.epsilon) * args.sigma}
    if args.r0_values is not None:
        r0_values = [float(x) for x in args.r0_values.split(",") if x.strip()]

    print(f"--- {args.mode} sweep: sigma={args.sigma}, m={m}, reps={reps} ---")
    print(f"R~(0): {r0_values}")
    print(f"n    : {n_values}")

    cache = simulate_sweep(args.mode, n_values, r0_values, reps, params,
                           args.sigma, m, seed)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir,
                            output_name(args.mode, reps, args.sigma, params))
    with open(out_path, "wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Data saved to {out_path}")
