import argparse
import os
import math
import warnings
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import cramervonmises_2samp, ks_2samp, wasserstein_distance

# ----------------------------------------------------------------
# 1. VISUAL STYLE CONFIGURATION
# ----------------------------------------------------------------
plt.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
})

CMR_COLORS = sns.color_palette("CMRmap", 4)
DEFAULT_SUBSET_DISTANCE_METRIC = "emd"


def apply_axis_style(ax, label):
    ax.text(
        -0.08, 1.04, label,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    ax.tick_params(width=1.4, length=5, which="major")
    ax.tick_params(width=1.2, length=3, which="minor")
    ax.grid(False)


def normalize_distance_metric(metric):
    metric_key = metric.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cvm": "cvm",
        "cramervonmises": "cvm",
        "cramer_von_mises": "cvm",
        "emd": "emd",
        "earth_movers_distance": "emd",
        "earth_mover_distance": "emd",
        "earthmovers": "emd",
        "wasserstein": "emd",
        "ks": "ks",
        "ks_2samp": "ks",
        "kolmogorov_smirnov": "ks",
    }
    if metric_key not in aliases:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from cvm, emd, or ks.")
    return aliases[metric_key]


def distance_metric_label(metric):
    metric_key = normalize_distance_metric(metric)
    labels = {
        "cvm": "CvM distance",
        "emd": "Earth mover's distance",
        "ks": "KS statistic",
    }
    return labels[metric_key]


def compute_distance_metric(values_a, values_b, metric):
    metric_key = normalize_distance_metric(metric)
    if metric_key == "cvm":
        return cramervonmises_2samp(values_a, values_b).statistic
    if metric_key == "emd":
        return wasserstein_distance(values_a, values_b)
    if metric_key == "ks":
        return ks_2samp(values_a, values_b).statistic
    raise ValueError(f"Unsupported metric '{metric}'")


# ----------------------------------------------------------------
# 2. MODEL CLASSES
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


class FisherRadialWedge(FisherModel):
    """SSWM adaptive walk confined to a fixed angular wedge around the initial
    direction, isolating *radial* scrambling.

    The wedge is defined once, by the starting position r0: it is the tube of
    points x whose squared perpendicular distance from the fixed axis
    rhat0 = r0 / ||r0|| satisfies
        || x_perp ||^2  <=  epsilon * ||r0||^2.
    A mutation is admissible iff the landing point r + delta stays inside this
    region. Among the admissible AND beneficial candidates, one is chosen with
    probability proportional to its fitness effect (the FGM SSWM rule). Because
    admissible moves cannot wander in orientation, all progress is radial, and
    the walk descends along rhat0 toward the fitness peak.
    """

    def __init__(self, n, sigma, m, R0, epsilon, seed=None):
        super().__init__(n, sigma, m, R0, seed=seed)
        self.axis = FisherModel.normalize(self.r)        # rhat0 (fixed for all t)
        self.perp_threshold = float(epsilon) * R0 ** 2   # squared-perpendicular area

    def step(self):
        r_candidates = self.r + self.deltas
        proj = r_candidates @ self.axis                  # parallel component along rhat0
        perp_sq = np.einsum("ij,ij->i", r_candidates, r_candidates) - proj ** 2
        in_wedge = perp_sq <= self.perp_threshold

        dfe = self.compute_dfe(self.r)
        beneficial = dfe > 0

        valid_indices = np.nonzero(in_wedge & beneficial)[0]
        if len(valid_indices) == 0:
            return False

        effects = dfe[valid_indices]
        probs = effects / np.sum(effects)
        choice = self.rng.choice(valid_indices, p=probs)
        self.r += self.deltas[choice]
        # In FGM, if a mutation fixes, the forward mutation flips sign.
        self.deltas[choice] *= -1
        return True


# ----------------------------------------------------------------
# 3. SIMULATION WORKER
# ----------------------------------------------------------------
def run_single_replicate(args):
    seed, n, sigma, m, R0, params, max_t, time_points = args

    epsilon = params["epsilon"]
    subset_metric = normalize_distance_metric(
        params.get("subset_metric", DEFAULT_SUBSET_DISTANCE_METRIC)
    )

    model = FisherRadialWedge(n, sigma, m, R0, epsilon, seed=seed)

    # Static copy of the exact initial mutations to track the decorrelation of
    # those same mutational fitness effects in time (independent of stepping).
    initial_deltas = model.deltas.copy()
    delta_norms_sq = np.sum(initial_deltas ** 2, axis=1)

    def get_malthusian(r):
        return -0.5 * delta_norms_sq - np.dot(initial_deltas, r)

    def get_fitness_effects(r):
        # Selection coefficient s = w(r+delta)/w(r) - 1 = expm1(malthusian).
        # Scale-free: unlike the absolute difference w(r+delta) - w(r), this drops
        # the global exp(-||r||^2/2) prefactor, so EMD compares DFE *shapes* across
        # radii rather than being swamped by the exponentially varying scale.
        return np.expm1(get_malthusian(r))

    dfe0 = get_fitness_effects(model.r)
    # The tracked subset M is fixed at t=0: the mutations initially beneficial.
    initial_beneficial_mask = dfe0 > 0

    def get_subset_distance(dfe_t):
        if not np.any(initial_beneficial_mask):
            return np.nan
        tracked_dfe_t = dfe_t[initial_beneficial_mask]
        return compute_distance_metric(tracked_dfe_t, dfe_t, subset_metric)

    subset_distance0 = get_subset_distance(dfe0)
    std_dfe0 = np.std(dfe0)

    radii = np.full(len(time_points), np.nan)
    pearsons = np.full(len(time_points), np.nan)
    subset_distances = np.full(len(time_points), np.nan)

    radii[0] = np.linalg.norm(model.r)
    pearsons[0] = 1.0
    if np.isfinite(subset_distance0):
        subset_distances[0] = 1.0

    current_t_idx = 0
    time_points_set = set(int(t) for t in time_points)

    for t in range(1, max_t + 1):
        success = model.step()
        if not success:
            break

        if t in time_points_set:
            current_t_idx += 1
            radii[current_t_idx] = np.linalg.norm(model.r)

            dfe_t = get_fitness_effects(model.r)
            if std_dfe0 > 1e-12 and np.std(dfe_t) > 1e-12:
                pearsons[current_t_idx] = np.corrcoef(dfe0, dfe_t)[0, 1]
            if np.isfinite(subset_distance0) and subset_distance0 > 1e-12:
                subset_distance_t = get_subset_distance(dfe_t)
                if np.isfinite(subset_distance_t):
                    subset_distances[current_t_idx] = subset_distance_t / subset_distance0

    return pearsons, subset_distances, radii


# ----------------------------------------------------------------
# 4. HELPERS
# ----------------------------------------------------------------
def stack_results(res_list):
    pearsons = np.array([res[0] for res in res_list], dtype=float)
    subset_distances = np.array([res[1] for res in res_list], dtype=float)
    radii = np.array([res[2] for res in res_list], dtype=float)
    return pearsons, subset_distances, radii


# ----------------------------------------------------------------
# 5. MAIN EXPERIMENT
# ----------------------------------------------------------------
def run_experiment(n_values, r0_tilde_values, epsilon,
                   subset_metric=DEFAULT_SUBSET_DISTANCE_METRIC):
    subset_metric = normalize_distance_metric(subset_metric)
    subset_metric_label = distance_metric_label(subset_metric)

    sigma = 0.05
    reps = 80
    m = 2 * 10 ** 4

    print("--- Configuration (radial scrambling, wedge-constrained SSWM) ---")
    print(f"Subset distance metric: {subset_metric_label} ({subset_metric})")
    print(f"sigma={sigma}, m={m}, reps={reps}, epsilon={epsilon}")
    print(f"Panels (R0_tilde): {r0_tilde_values}")
    print(f"Curves (n): {n_values}")

    base_seed = np.random.randint(0, 1_000_000)

    # Each (panel R0_tilde, curve n) gets `reps` replicates.
    tasks = []
    spans = {}
    for p, R0_tilde in enumerate(r0_tilde_values):
        R0 = R0_tilde * sigma
        # Larger starting radius => longer radial descent; scale the horizon.
        max_t = int(4 * R0_tilde) + 20
        time_points = np.arange(0, max_t + 1)
        for k, n in enumerate(n_values):
            start = len(tasks)
            for i in range(reps):
                tasks.append((
                    base_seed + 100_000 * (p + 1) + 10_000 * (k + 1) + i,
                    int(n),
                    sigma,
                    m,
                    R0,
                    {"epsilon": epsilon, "subset_metric": subset_metric},
                    max_t,
                    time_points,
                ))
            spans[(p, k)] = (start, len(tasks), time_points)

    num_proc = min(multiprocessing.cpu_count(), len(tasks))
    with multiprocessing.Pool(processes=num_proc) as pool:
        results = pool.map(run_single_replicate, tasks)

    # ------------------------------
    # Plotting: one panel per R0_tilde, one EMD curve per n
    # ------------------------------
    n_panels = len(r0_tilde_values)
    ncols = 2 if n_panels > 1 else 1
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.6 * nrows), squeeze=False)
    fig.subplots_adjust(wspace=0.28, hspace=0.38)

    panel_labels = [chr(ord("A") + i) for i in range(n_panels)]
    n_colors = sns.color_palette("viridis", len(n_values))

    for p, R0_tilde in enumerate(r0_tilde_values):
        ax = axes[p // ncols][p % ncols]
        apply_axis_style(ax, panel_labels[p])
        max_reached = 1

        for k, n in enumerate(n_values):
            start, end, time_points = spans[(p, k)]
            _pear, subset, _radii = stack_results(results[start:end])

            # Time points past the longest walk are all-NaN columns; ignore the
            # resulting "empty slice" warnings (those points are trimmed by xlim).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_subset = np.nanmean(subset, axis=0)
                std_subset = np.nanstd(subset, axis=0)

            finite_t = np.where(np.isfinite(mean_subset))[0]
            last = int(finite_t[-1]) if len(finite_t) else 0
            if len(finite_t):
                max_reached = max(max_reached, int(time_points[last]))

            color = n_colors[k]
            # Mean EMD trace plus error bars (std across replicates).
            ax.plot(time_points, mean_subset, color=color, lw=2.0)
            # Subsample markers across the *populated* range, not the full horizon,
            # so the error bars span the visible curve rather than collapsing to a
            # few points near t=0.
            step = max(1, (last + 1) // 10)
            marker_idx = np.arange(0, last + 1, step)
            ax.errorbar(
                time_points[marker_idx],
                mean_subset[marker_idx],
                yerr=std_subset[marker_idx],
                fmt="o",
                color=color,
                markersize=4,
                capsize=3,
                label=fr"$n = {int(n)}$",
            )

        # Far-field theory: the radial descent is linear at the SSWM speed
        # d<R~>/dt = -sqrt(pi/2), and the normalized scrambling tracks the
        # normalized radius, EMD(t) ~ R~(t)/R~0 = 1 - sqrt(pi/2) * t / R~0.
        tp_panel = spans[(p, 0)][2]
        theory = np.clip(1.0 - (np.sqrt(np.pi / 2) / R0_tilde) * tp_panel, 0.0, None)
        ax.plot(tp_panel, theory, color="black", lw=2.0, ls="--",
                label=r"$1-\sqrt{\pi/2}\,t/\tilde{R}_0$")

        ax.set_xlim(0, max_reached)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Time (steps)")
        if p % ncols == 0:
            ax.set_ylabel(f"{subset_metric.upper()} (norm.)")
        ax.set_title(rf"$\tilde{{R}}_0 = {R0_tilde:g}$,  $\epsilon = {epsilon:g}$")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if p == 0:
            ax.legend(frameon=False, loc="best", title=None)

    # Hide any unused panels
    for j in range(n_panels, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    out_dir = "../figs_paper"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "figA4_radial_timescale.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Radial scrambling: SSWM adaptive walk confined to a fixed angular "
                    "wedge around the initial direction, measuring convergence of the "
                    "initially-beneficial subset DFE to the overall DFE across n values."
    )
    parser.add_argument(
        "--subset-metric",
        default=DEFAULT_SUBSET_DISTANCE_METRIC,
        help="Distance metric for comparing the t=0 beneficial subset to the full DFE: cvm, emd, or ks.",
    )
    parser.add_argument(
        "--n-values",
        default="5,10,20,40",
        help="Comma-separated dimensionalities n, overlaid as EMD curves in every panel.",
    )
    parser.add_argument(
        "--r0-values",
        default="20,40,80,160",
        help="Comma-separated initial R0_tilde values, one panel each.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.15,
        help="Wedge size: admissible iff ||x_perp||^2 <= epsilon * ||r0||^2. "
             "Must exceed ~n*sigma^2/R0^2 for the largest n at the smallest R0, "
             "else those walks cannot fit any mutation in the wedge and get stuck.",
    )
    args = parser.parse_args()
    n_values = [int(x) for x in args.n_values.split(",") if x.strip()]
    r0_tilde_values = [float(x) for x in args.r0_values.split(",") if x.strip()]
    run_experiment(n_values, r0_tilde_values, args.epsilon, subset_metric=args.subset_metric)
