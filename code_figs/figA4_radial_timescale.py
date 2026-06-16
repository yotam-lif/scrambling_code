import argparse
import os
import warnings
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
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

    def get_abs_fitness_effects(r):
        # True (absolute) fitness effect w(r+delta) - w(r) = w(r) * expm1(malthusian).
        # Used for the Pearson autocorrelation only; the EMD subset distance keeps
        # the scale-free selection coefficient above.
        return model.compute_fitness(r) * np.expm1(get_malthusian(r))

    # EMD subset distance is built from the selection-coefficient DFE.
    dfe0 = get_fitness_effects(model.r)
    # The tracked subset M is fixed at t=0: the mutations initially beneficial.
    initial_beneficial_mask = dfe0 > 0

    def get_subset_distance(dfe_t):
        if not np.any(initial_beneficial_mask):
            return np.nan
        tracked_dfe_t = dfe_t[initial_beneficial_mask]
        return compute_distance_metric(tracked_dfe_t, dfe_t, subset_metric)

    subset_distance0 = get_subset_distance(dfe0)

    # Pearson autocorrelation is built from the absolute fitness effects.
    abs_dfe0 = get_abs_fitness_effects(model.r)
    std_abs_dfe0 = np.std(abs_dfe0)

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

            # Pearson on absolute fitness effects.
            abs_dfe_t = get_abs_fitness_effects(model.r)
            if std_abs_dfe0 > 1e-12 and np.std(abs_dfe_t) > 1e-12:
                pearsons[current_t_idx] = np.corrcoef(abs_dfe0, abs_dfe_t)[0, 1]
            # EMD on the selection-coefficient subset DFE.
            if np.isfinite(subset_distance0) and subset_distance0 > 1e-12:
                dfe_t = get_fitness_effects(model.r)
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
    print(f"Rows (R0_tilde): {r0_tilde_values}")
    print(f"Curves (n): {n_values}")

    base_seed = np.random.randint(0, 1_000_000)

    # One row per R0_tilde; within a row, each curve n gets `reps` replicates.
    tasks = []
    spans = {}                 # (row p, curve k) -> (start, end)
    row_time_points = {}       # row p -> time_points
    for p, r0_tilde in enumerate(r0_tilde_values):
        R0 = r0_tilde * sigma
        # Larger starting radius => longer radial descent; scale the horizon.
        max_t = int(4 * r0_tilde) + 20
        time_points = np.arange(0, max_t + 1)
        row_time_points[p] = time_points
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
            spans[(p, k)] = (start, len(tasks))

    num_proc = min(multiprocessing.cpu_count(), len(tasks))
    with multiprocessing.Pool(processes=num_proc) as pool:
        results = pool.map(run_single_replicate, tasks)

    # ------------------------------
    # Plotting: one row per R0_tilde, two panels per row, one curve per n.
    #   left : normalized subset->full DFE distance (EMD, selection coeffs).
    #   right: Pearson correlation of the t=0 DFE with the DFE at time t (abs fitness).
    # ------------------------------
    nrows = len(r0_tilde_values)
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4.8 * nrows), squeeze=False)
    fig.subplots_adjust(wspace=0.26, hspace=0.42)

    n_colors = sns.color_palette("viridis", len(n_values))

    for p, r0_tilde in enumerate(r0_tilde_values):
        time_points = row_time_points[p]
        ax_emd, ax_pear = axes[p][0], axes[p][1]
        apply_axis_style(ax_emd, chr(ord("A") + 2 * p))
        apply_axis_style(ax_pear, chr(ord("A") + 2 * p + 1))

        for k, n in enumerate(n_values):
            start, end = spans[(p, k)]
            pear, subset, radii = stack_results(results[start:end])

            # Time points past the longest walk are all-NaN columns; ignore the
            # resulting "empty slice" warnings (those points are trimmed by xlim).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_subset = np.nanmean(subset, axis=0)
                std_subset = np.nanstd(subset, axis=0)
                mean_pear = np.nanmean(pear, axis=0)
                std_pear = np.nanstd(pear, axis=0)
                # x-axis: current distance from the peak (radius), in sigma units
                # (R~ = R/sigma), averaged across replicates at each time point.
                # Monotonically decreasing from R~0 toward ~0 as the walk descends.
                mean_radii = np.nanmean(radii, axis=0) / sigma

            finite_t = np.where(np.isfinite(mean_subset))[0]
            last = int(finite_t[-1]) if len(finite_t) else 0

            color = n_colors[k]
            # Subsample markers across the *populated* range, not the full horizon,
            # so the error bars span the visible curve rather than collapsing to a
            # few points near t=0.
            step = max(1, (last + 1) // 10)
            marker_idx = np.arange(0, last + 1, step)

            # Left panel: EMD vs distance from peak (mean trace + std error bars).
            ax_emd.plot(mean_radii, mean_subset, color=color, lw=2.0)
            ax_emd.errorbar(
                mean_radii[marker_idx],
                mean_subset[marker_idx],
                yerr=std_subset[marker_idx],
                fmt="o",
                color=color,
                markersize=4,
                capsize=3,
                label=fr"$n = {int(n)}$",
            )

            # Right panel: Pearson correlation of the DFE (linear scale).
            ax_pear.plot(mean_radii, mean_pear, color=color, lw=2.0)
            ax_pear.errorbar(
                mean_radii[marker_idx],
                mean_pear[marker_idx],
                yerr=std_pear[marker_idx],
                fmt="o",
                color=color,
                markersize=4,
                capsize=3,
                label=fr"$n = {int(n)}$",
            )

        # Far-field theory for the EMD panel: the radial descent is linear at the
        # SSWM speed d<R~>/dt = -sqrt(pi/2), so the normalized scrambling tracks the
        # normalized radius. In radius coordinates this is simply the identity line
        # EMD ~ R~/R~0, a diagonal from (0, 0) to (R~0, 1).
        radius_grid = np.linspace(0.0, r0_tilde, 50)
        ax_emd.plot(radius_grid, radius_grid / r0_tilde, color="black", lw=2.0,
                    ls="--", label=r"$\tilde{R}/\tilde{R}_0$")

        ax_emd.set_xlim(0, r0_tilde)
        ax_emd.set_ylim(-0.05, 1.05)
        ax_emd.set_xlabel(r"Distance from peak  $\tilde{R} = R/\sigma$")
        ax_emd.set_ylabel(f"{subset_metric.upper()} (norm.)")
        ax_emd.set_title(rf"Subset DFE distance  ($\tilde{{R}}_0 = {r0_tilde:g}$)")

        ax_pear.set_xlim(0, r0_tilde)
        ax_pear.set_ylim(-0.05, 1.05)
        ax_pear.set_xlabel(r"Distance from peak  $\tilde{R} = R/\sigma$")
        ax_pear.set_ylabel("Pearson correlation of DFE")
        ax_pear.set_title(rf"DFE autocorrelation  ($\tilde{{R}}_0 = {r0_tilde:g}$)")

        # One legend for the whole figure, on the first EMD panel only.
        if p == 0:
            ax_emd.legend(frameon=False, loc="best")

    fig.suptitle(rf"$\epsilon = {epsilon:g}$", y=1.0)

    out_dir = "../figs_paper"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "figA4_radial_timescale.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Radial scrambling: SSWM adaptive walk confined to a fixed angular "
                    "wedge around the initial direction, measuring convergence of the "
                    "initially-beneficial subset DFE to the overall DFE across n values. "
                    "Panel A shows the EMD; panel B shows the DFE Pearson correlation."
    )
    parser.add_argument(
        "--subset-metric",
        default=DEFAULT_SUBSET_DISTANCE_METRIC,
        help="Distance metric for comparing the t=0 beneficial subset to the full DFE: cvm, emd, or ks.",
    )
    parser.add_argument(
        "--n-values",
        default="5,10,20,40",
        help="Comma-separated dimensionalities n, overlaid as curves in both panels.",
    )
    parser.add_argument(
        "--r0-values",
        default="20,80",
        help="Comma-separated initial R0_tilde values, one row of (EMD, Pearson) panels each.",
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
