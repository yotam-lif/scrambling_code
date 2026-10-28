r"""Figure S9: measurement error of the three ancestral libraries, and a two-parameter fit.

Figure 4 adds noise to the simulated walks by giving each simulated mutation the published error
of the empirical gene at its own effect rank.  This figure shows that error against the effect
for the three ancestors Figure 4 starts from -- REL606 and REL607 in LB (Limdi) and REL607 in
DM25 (Couce 0K) -- and a closed form that captures its whole s-dependence in each:

    epsilon(s) = epsilon_0 [1 + ln(1 + d/b)],    d = max(-s, 0).

    A-C   epsilon against |s| on log-log axes, deleterious (s < 0) and beneficial (s > 0) apart.
          The fit is the flat floor epsilon_0 on the beneficial side and grows logarithmically
          with |s| on the deleterious side, crossing over at |s| ~ b.
          Panels A-C occupy the first three cells of a 2x2 grid; the bottom-right
          cell contains the shared legend. No relative-error panels are drawn.

The DM25 library is per insertion segment rather than per gene, and its error steps up slightly
across s = 0 (geometric mean 0.0074 just below, 0.0051 just above), which the model, continuous
at s = 0, splits between the two sides.

Why the two sides differ (Limdi).  A gene's effect is the slope of ln(reads) against time, averaged over
its insertion sites.  Beneficial sites gain reads, so their per-site slope gets more precise with
s, but the most beneficial genes are also the shortest (fewer sites to average), and the two
roughly cancel.  Deleterious sites lose reads, so their later timepoints sit in the noisy
low-count regime, and the deeper the effect the sooner; once a site hits zero within the first
6.64-generation interval its slope saturates, which is why the growth is logarithmic, not linear.

Fitting.  Around the curve the per-gene error is close to lognormal with a roughly constant width,
so the model is fitted by least squares on log epsilon, i.e. log epsilon_i = log f(s_i) + tau z_i:
the curve is the geometric-mean (= median) error at s and tau the residual log-scatter.  The best
any smooth curve can do is tau_bins, the scatter about 40 equal-count bin means.

Model choice.  Four floor-plus-growth forms were compared on all 14 libraries: s0 + a ln(1 + d/b),
this one (a tied to s0), s0 (1 + d/b)^g and s0 + A (1 - exp(-d/b)).  All four sit within 0.01 of
tau_bins in every library except Ara-2 and Ara+4 (which Limdi et al. drop on quality grounds), so
the two-parameter form is kept.  REL606: tau = 0.457 vs tau_bins = 0.451; REL607: 0.443 vs 0.435.

Run from anywhere:  python code_figs/figS9_meas_error.py
Output:             figs_paper/figS9_meas_error.pdf
"""

import colorsys
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import least_squares

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cmn import cmn_exper  # noqa: E402
from cmn.cmn_plots import FATE_COLORS  # noqa: E402


def apply_axis_style(ax, label):
    """Bold panel letter plus the spine/tick geometry shared with the other
    figures: only the left and bottom spines, both pushed outward."""
    ax.text(
        -0.08, 1.04, label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["left"].set_position(("outward", 10))
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6, which="major")
    ax.tick_params(width=1.5, length=3, which="minor")
    ax.grid(False)


# ───────────────────────────────────── Style ─────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams.update({
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

def _lighten(color, lightness):
    """``color`` with its HLS lightness set to ``lightness``, hue and saturation kept."""
    hue, _, saturation = colorsys.rgb_to_hls(*mpl.colors.to_rgb(color))
    return mpl.colors.to_hex(colorsys.hls_to_rgb(hue, lightness, saturation))


# The paper's deleterious / beneficial pair (fig2's fate bars), lifted from their ~0.3 lightness:
# at full darkness the scattered genes read as near-black clouds.
SIDE_LIGHTNESS = 0.50
SIDES = {"Deleterious": (-1, _lighten(FATE_COLORS[0], SIDE_LIGHTNESS)),
         "Beneficial": (1, _lighten(FATE_COLORS[2], SIDE_LIGHTNESS))}
REL_MARKER_SIZE = 5.5       # smaller below: equal-count bins pile up at s = 0 on a linear axis

# ───────────────────────────────────── Configuration ─────────────────────────────────────
N_QUANTILE_BINS = 40        # equal-count bins for tau_bins
DRAW_BINS = {"Deleterious": 20, "Beneficial": 10}   # equal-count bins drawn per side
ABS_XLIM = (1e-4, 1.0)
ABS_YLIM = (5e-4, 0.3)            # DM25 errors reach down to ~1e-3; below 5e-4 hides 0.1%
REL_YLIM = (1e-2, 1e3)

# (dataset, library, medium), one column each, in Figure 4's order.
LIBRARIES = (("limdi", "REL606", "LB"), ("limdi", "REL607", "LB"), ("couce", "0K", "DM25"))
# Couce 0K is the REL607 clone the Ara+2 lineage starts from, as Figure 4 titles it.
ANCESTOR_NAMES = {"REL606": "REL606", "REL607": "REL607", "0K": "REL607"}
# Bottom-row s window per medium: the DM25 effects stop near -0.23.
REL_XLIM = {"LB": (-0.62, 0.12), "DM25": (-0.25, 0.11)}

OUT_PATH = os.path.join(_REPO_ROOT, "figs_paper", "figS9_meas_error.pdf")


# ───────────────────────────────────── Model ─────────────────────────────────────

def epsilon_model(s, epsilon0, b):
    """epsilon_0 [1 + ln(1 + d/b)] with d = max(-s, 0): flat for s > 0, log growth for s < 0."""
    return epsilon0 * (1 + np.log1p(np.maximum(-s, 0.0) / b))


def library_data(dataset, pop):
    """``(s, epsilon)`` for one library, entries with a finite effect and positive error only."""
    if dataset == "limdi":
        eff, sig = cmn_exper.limdi_gene_series(pop, errors=True)
    elif dataset == "couce":
        eff = cmn_exper.load_couce_segment_series(pop)
        sig = cmn_exper.load_couce_segment_errors(pop)
        index = eff.index.intersection(sig.index)
        eff, sig = eff.loc[index], sig.loc[index]
    else:
        raise ValueError(f"no loader for dataset {dataset!r}")
    s, epsilon = eff.to_numpy(float), sig.to_numpy(float)
    keep = np.isfinite(s) & np.isfinite(epsilon) & (epsilon > 0)
    return s[keep], epsilon[keep]


def fit(s, epsilon):
    """Least squares on log epsilon, parameters fitted as logs so they stay > 0.

    Returns ``((epsilon0, b), tau)``.
    """
    log_epsilon = np.log(epsilon)
    result = least_squares(lambda p: np.log(epsilon_model(s, *np.exp(p))) - log_epsilon,
                           np.log((0.005, 0.01)))
    params = np.exp(result.x)
    return params, float(np.std(np.log(epsilon_model(s, *params)) - log_epsilon))


def tau_bins(s, epsilon):
    """Residual log-scatter about equal-count bin means: the floor any smooth model can reach."""
    order = np.argsort(s)
    resid = np.concatenate([np.log(epsilon[chunk]) - np.log(epsilon[chunk]).mean()
                            for chunk in np.array_split(order, N_QUANTILE_BINS)])
    return float(np.std(resid))


def binned_geomean(x, epsilon, n_bins):
    """Equal-count bins in x: median x and geometric-mean epsilon."""
    chunks = np.array_split(np.argsort(x), n_bins)
    return (np.array([np.median(x[c]) for c in chunks]),
            np.array([np.exp(np.log(epsilon[c]).mean()) for c in chunks]))


# ───────────────────────────────────── Plotting ─────────────────────────────────────

def plot_absolute(ax, s, epsilon, params, binned):
    """epsilon vs |s|, log-log, one colour per sign, with the fit drawn over each side's range."""
    mag = np.logspace(np.log10(ABS_XLIM[0]), np.log10(ABS_XLIM[1]), 400)
    for side, (sign, color) in SIDES.items():
        m = np.sign(s) == sign
        ax.scatter(np.abs(s[m]), epsilon[m], s=5, color=color, alpha=0.18, lw=0, zorder=1)
        top = np.abs(s[m]).max()
        ax.plot(mag[mag <= top], epsilon_model(sign * mag[mag <= top], *params),
                color=color, lw=2.5, zorder=2)
        xb, gb = binned[side]
        ax.plot(xb, gb, "o", ms=7, mfc=color, mec="k", mew=0.8, zorder=3)

    epsilon0, b = params
    ax.text(0.97, 0.04, rf"$\epsilon_0 = {epsilon0 * 1e3:.1f}\times10^{{-3}}$" + "\n"
            + rf"$b = {b:.3f}$", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*ABS_XLIM)
    ax.set_ylim(*ABS_YLIM)
    ax.set_xlabel(r"Absolute fitness effect $(|s|)$")


def plot_relative(ax, s, epsilon, params, binned, xlim):
    """epsilon/|s| vs signed s: the same genes, binned means and fit as the panel above.

    The fit is drawn only over the binned means' range: inward of the first bin it is just
    epsilon_0/|s| running off to infinity, a spike that says nothing and runs into the key.
    """
    grid = np.linspace(xlim[0], xlim[1], 1400)
    for side, (sign, color) in SIDES.items():
        m = np.sign(s) == sign
        ax.scatter(s[m], epsilon[m] / np.abs(s[m]), s=5, color=color, alpha=0.18, lw=0, zorder=1)
        xb, gb = binned[side]
        g = grid[(np.sign(grid) == sign) & (np.abs(grid) >= xb.min())
                 & (np.abs(grid) <= np.abs(s[m]).max())]
        ax.plot(g, epsilon_model(g, *params) / np.abs(g), color=color, lw=2.5, zorder=2)
        ax.plot(sign * xb, gb / xb, "o", ms=REL_MARKER_SIZE, mfc=color, mec="k", mew=0.8, zorder=3)

    ax.axhline(1, color="k", lw=1.2, ls=":", zorder=0)
    ax.text(xlim[0], 0.85, r"$\epsilon = |s|$", ha="left", va="top", fontsize=14)
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*REL_YLIM)
    ax.set_xlabel(r"Fitness effect $(s)$")


def legend_handles(relative=False):
    """One marker and one line per sign, the line labelled with its side of the fit.

    ``relative`` gives the bottom row's key: the same entries, each divided by |s|.
    """
    per = r"$/|s|$" if relative else ""
    prefactor = r"\dfrac{\epsilon_0}{|s|}" if relative else r"\epsilon_0"
    growth = {"Deleterious": r"\,[1+\ln(1+|s|/b)]", "Beneficial": ""}
    ms = REL_MARKER_SIZE if relative else 7
    handles = []
    for side, (_, color) in SIDES.items():
        handles.append(Line2D([], [], ls="none", marker="o", ms=ms, mfc=color, mec="k", mew=0.8,
                              label=rf"Binned geometric mean{per}, {side.lower()}"))
    for side, (_, color) in SIDES.items():
        handles.append(Line2D([], [], color=color, lw=2.5,
                              label=rf"Fit, {side.lower()}: ${prefactor}{growth[side]}$"))
    return handles


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.6), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.94,
                        wspace=0.24, hspace=0.48)
    for ax, label in zip(axes.flat, "ABC"):
        apply_axis_style(ax, label)

    for ax, (dataset, pop, medium) in zip(axes.flat, LIBRARIES):
        s, epsilon = library_data(dataset, pop)
        params, tau = fit(s, epsilon)
        binned = {side: binned_geomean(np.abs(s[np.sign(s) == sign]),
                                       epsilon[np.sign(s) == sign], DRAW_BINS[side])
                  for side, (sign, _) in SIDES.items()}
        plot_absolute(ax, s, epsilon, params, binned)
        ax.set_title(f"{ANCESTOR_NAMES[pop]} ancestor ({medium})")

        rel = epsilon / np.abs(s)
        print(f"{pop} ({medium}): epsilon0 = {params[0]:.4g}, b = {params[1]:.4g}, tau = {tau:.3f} "
              f"(tau_bins {tau_bins(s, epsilon):.3f}), N = {s.size}; "
              f"epsilon/|s| > 1: {100 * np.mean(rel > 1):.1f}%, > 10: {100 * np.mean(rel > 10):.1f}%")

    for ax in axes[:, 0]:
        ax.set_ylabel(r"Measurement error $(\epsilon)$")
    axes[1, 1].set_axis_off()
    axes[1, 1].legend(handles=legend_handles(), loc="center", frameon=False,
                      handlelength=1.5, handletextpad=0.4, labelspacing=1.2)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
