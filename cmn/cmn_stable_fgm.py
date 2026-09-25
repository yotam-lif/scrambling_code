r"""Sub-Gaussian alpha-stable mutation vectors in Fisher's Geometric Model.

Same geometry as :mod:`cmn.cmn_fgm` and :mod:`cmn.cmn_cauchy_fgm` -- log-fitness on
``w(x) = exp(-|x|^2 / 2)``, so a mutation ``delta`` at radius ``r`` has effect

    s = log(w(r + delta) / w(r)) = -r . delta - |delta|^2 / 2,   s <= r^2 / 2.

Only the mutation law changes.  Both heavy-tailed families in this repo are normal
SCALE MIXTURES -- one scalar shared across all n coordinates, so the vector stays
isotropic:

    delta = sigma * sqrt(A) * Z,   Z ~ N_n(0, I),   A > 0 independent of Z.

    Student (cmn_cauchy_fgm)  A = 1 / (2 G),  G ~ Gamma(mu, 1)
                              -> delta is multivariate t with nu = 2 mu degrees of
                                 freedom; |delta|^2 / sigma^2 ~ BetaPrime(n/2, mu);
                                 P(|delta| > x) ~ x^(-2 mu).

    Stable (this module)      A ~ positive stable of index kappa = alpha / 2,
                              normalised by E[exp(-lambda A)] = exp(-lambda^kappa)
                              -> delta is sub-Gaussian alpha-stable, the isotropic
                                 (elliptically contoured) multivariate stable law;
                                 P(|delta| > x) ~ x^(-alpha).

The two families meet at exactly one point, the multivariate Cauchy: alpha = 1 is
nu = 1 is mu = 1/2.  Everywhere else they are different laws with the same tail index
once alpha = 2 mu, which is the comparison this module exists to make.

alpha is capped at 2 by the definition of stability, and alpha = 2 is the Gaussian with
no power-law tail at all, so the family cannot express a power-law tail with finite
variance.  The Student family can (any nu > 2).  Whether that matters for a given
dataset is an empirical question -- for the Limdi ancestors the fitted 2 mu is well
below 2, so both families are admissible there and the comparison is about the shape of
the BODY, not the reachability of the tail.


How the density is computed
---------------------------
The positive stable density has no closed form, so unlike cmn_cauchy_fgm there is no
exact special-function expression to evaluate.  But CONDITIONAL ON A = a the mutation is
exactly Gaussian with scale ``sigma sqrt(a)``, and the Gaussian FGM DFE is already exact
in cmn_fgm.  So the DFE and its survival are one-dimensional mixtures

    p(s)      = INT p_gauss(s | n, sigma sqrt(a), r) f_A(a) da
    P(s >= c) = INT S_gauss(c | n, sigma sqrt(a), r) f_A(a) da

evaluated as quadratures in ``y = log a``, where both integrands decay at both ends and
the trapezoid rule is spectrally accurate.

``f_A`` itself comes from Kanter's (1975) representation: with U ~ Uniform(0,1) and
E ~ Exp(1) independent, and k = kappa / (1 - kappa),

    A = (a(U) / E)^(1/k),
    log a(u) = k log sin(kappa pi u) + log sin((1-kappa) pi u)
               - (1/(1-kappa)) log sin(pi u).

Conditional on U = u, A is Frechet with shape k and scale a(u)^(1/k), whose log-density
is bounded, so the density of Y = log A is the well-behaved one-dimensional integral

    f_Y(y) = k INT_0^1 t e^(-t) du,   t = a(u) exp(-k y),

done once per alpha with Gauss-Legendre.  ``t e^(-t) <= 1/e`` everywhere, so this
integral has no singular endpoint to fight.

Both steps are checked against closed forms in ``selfcheck()``: the mixing law against
its own Laplace transform ``E[exp(-lambda A)] = exp(-lambda^kappa)`` and against the
Levy density at kappa = 1/2, and the whole mixture pipeline against the exact Student
FGM DFE of cmn_cauchy_fgm, by running it with the Student mixing law in place of the
stable one.  That last test is the important one: it exercises the quadrature, the node
placement, the survival and the pruning against an independent exact answer.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import digamma, gammaln, polygamma

from cmn import cmn_fgm

# ─────────────────────────────── Quadrature settings ──────────────────────────────
# Nodes in u for the Kanter integral.  The integrand is bounded by 1/e and smooth in the
# interior, but a(u) diverges as u -> 1 (that divergence IS the heavy tail), so the
# endpoint clustering of Gauss-Legendre is doing real work here.
_KANTER_NODES = 2000
_KANTER_X, _KANTER_W = leggauss(_KANTER_NODES)
_KANTER_U = 0.5 * (_KANTER_X + 1.0)
_KANTER_WU = 0.5 * _KANTER_W

# The mixture grid in y = log a spans the range of mutation scales that can matter, not
# the range where the mixing law has mass.  Below SIGMA_EFF_MIN every mutation is smaller
# than the finest effect any assay resolves, so such nodes contribute nothing to the
# density (they still contribute their mass to the survival, which is why they are kept
# rather than dropped).  Above SIGMA_EFF_MAX every mutation is far below any cut of
# interest, so they contribute to neither.
SIGMA_EFF_MIN = 1.0e-12
SIGMA_EFF_MAX = 1.0e4

# Step in y.  Both mixture integrands vary on a scale of order 1 in y -- sigma_eff moves
# by e^(1/2) per unit -- so this is what sets accuracy over most of the grid.
COARSE_STEP = 0.15
# ... except where the mixing law itself is narrower than that, which is the alpha -> 2
# (and mu -> infinity) corner where A collapses onto a point and the coarse step would
# step straight over it.  The step is then set by the mixing law instead.
#
# The grid stays UNIFORM either way, which matters more than it looks: on a smooth
# integrand that decays at both ends the trapezoid rule is spectrally accurate, and a
# grid that is fine in the core and coarse outside it is not uniform and falls back to
# second order.  Laying a fine core over a coarse grid cost four orders of magnitude
# here -- 1e-11 to 1e-4 against the exact Student DFE -- so the whole grid is refined
# together and the extra nodes are pruned by weight afterwards instead.
STEPS_PER_SD = 20.0
# Left truncation of the grid: below this many sds the mixing density dies like a double
# exponential, far faster than anything it multiplies.
LEFT_TAIL_SDS = 14.0

# Nodes dropped from the DENSITY sum (never from the survival, which is cheap and where
# small weights still accumulate).  A node is skipped when its weight is negligible
# against the largest, or when its noncentrality is past where scipy's ncx2.logpdf stops
# returning finite values -- which happens only for scales so far below the data that the
# node's density at every observed effect is zero to machine precision anyway.
WEIGHT_FLOOR = 1.0e-40
NC_MAX = 1.0e8

# Known limitation.  Where the mixing law is very concentrated -- mu of order 50, or
# alpha within a few percent of 2 -- the model is all but Gaussian and its deleterious
# tail collapses to densities of order 1e-70 and below.  The grid does not resolve which
# tiny number those are, and reports one that is smaller still.  It does not affect a
# fit: a likelihood only ever evaluates the density AT the observed effects, and any
# parameters that put an observed effect out at 1e-70 are already excluded by hundreds of
# nats.  ``fit_is_resolved`` is the operational check -- it re-evaluates the loglik on a
# refined grid at the fitted point, which is the only place accuracy has to hold.


# ────────────────────────────────── Mixing laws ───────────────────────────────────
def stable_log_mixing_density(y, kappa):
    """log f_Y(y) for Y = log A, A positive stable with E exp(-lambda A) = exp(-lambda^kappa).

    Kanter's representation, integrated over u with Gauss-Legendre.  Vectorised over ``y``.
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    kappa = float(kappa)
    if not 0.0 < kappa < 1.0:
        return np.full(y.shape, -np.inf)
    k = kappa / (1.0 - kappa)

    # log a(u).  Written as a single difference over (1 - kappa) rather than as two
    # separately huge terms: as kappa -> 1 both k and 1/(1-kappa) diverge while the
    # bracket goes to zero, and the grouped form keeps that cancellation exact.
    sin_k = np.sin(kappa * np.pi * _KANTER_U)
    sin_1k = np.sin((1.0 - kappa) * np.pi * _KANTER_U)
    sin_1 = np.sin(np.pi * _KANTER_U)
    log_a = ((kappa * np.log(sin_k) - np.log(sin_1)) / (1.0 - kappa)
             + np.log(sin_1k))

    log_t = log_a[None, :] - k * y[:, None]
    # t e^-t underflows to zero well before exp(log_t) overflows; short-circuiting there
    # keeps the overflow out of the arithmetic entirely.
    term = np.where(log_t > 50.0, 0.0,
                    np.exp(log_t - np.exp(np.minimum(log_t, 50.0))))
    integral = term @ _KANTER_WU
    with np.errstate(divide="ignore"):
        return np.log(k) + np.log(integral)


def stable_mixing_moments(kappa):
    """(mean, sd) of Y = log A, from E[A^p] = Gamma(1 - p/kappa) / Gamma(1 - p).

    Both vanish as kappa -> 1, where A collapses onto the point 1 and the model becomes
    the Gaussian FGM.  They only place the quadrature grid; nothing depends on them
    being anything more than roughly right.
    """
    kappa = float(kappa)
    mean = -digamma(1.0) * (1.0 / kappa - 1.0)
    variance = polygamma(1, 1) * (1.0 / (kappa * kappa) - 1.0)
    return float(mean), float(np.sqrt(max(variance, 0.0)))


def student_log_mixing_density(y, mu):
    """log f_Y(y) for the Student mixing law A = 1 / (2 G), G ~ Gamma(mu, 1).

    Present so that ``selfcheck`` can drive the same quadrature with a mixing law whose
    FGM DFE has an exact closed form in cmn_cauchy_fgm.
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    return -mu * (np.log(2.0) + y) - 0.5 * np.exp(-y) - gammaln(mu)


def student_mixing_moments(mu):
    """(mean, sd) of Y = log A for the Student mixing law."""
    return float(-np.log(2.0) - digamma(mu)), float(np.sqrt(polygamma(1, mu)))


# ──────────────────────────────── Mixture quadrature ──────────────────────────────
def _grid(sigma, moments):
    """Trapezoid nodes and weights in y = log a.

    One uniform grid over the whole range of mutation scales that can matter, stepped
    finely enough for both the FGM factor and the mixing law.  Uniform on purpose -- see
    the note on STEPS_PER_SD.
    """
    mean, sd = moments
    low = max(2.0 * np.log(SIGMA_EFF_MIN / sigma), mean - LEFT_TAIL_SDS * sd)
    high = 2.0 * np.log(SIGMA_EFF_MAX / sigma)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None, None

    step = COARSE_STEP if sd <= 0.0 else min(COARSE_STEP, sd / STEPS_PER_SD)
    counts = int(np.ceil((high - low) / step)) + 1
    y = np.linspace(low, high, counts)
    weights = np.full(counts, (y[-1] - y[0]) / (counts - 1))
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return y, weights


def _mixture_weights(sigma, log_mixing, moments):
    """``(sigma_eff, weight)`` for the quadrature, weights already carrying f_Y."""
    y, quad = _grid(sigma, moments)
    if y is None:
        return None, None
    log_weight = np.log(quad) + log_mixing(y)
    weight = np.where(np.isfinite(log_weight), np.exp(log_weight), 0.0)
    return sigma * np.exp(0.5 * y), weight


def mixture_dfe_pdf(s, n, sigma, r, log_mixing, moments):
    """Density of the log-fitness FGM DFE under a normal scale mixture on the mutation.

    Nodes whose Gaussian DFE cannot reach any of the requested ``s`` are skipped: for a
    scale far below the data the conditional DFE is a needle at s = 0 that is thousands
    of standard deviations from every observation, and scipy's ncx2 stops returning
    finite log-densities there in any case.
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    scales, weight = _mixture_weights(sigma, log_mixing, moments)
    if scales is None:
        return np.zeros(s.shape)

    keep = (weight > WEIGHT_FLOOR * weight.max()) & (r * r / scales ** 2 < NC_MAX)
    total = np.zeros(s.shape)
    for scale, node_weight in zip(scales[keep], weight[keep]):
        total += node_weight * cmn_fgm.fgm_dfe_pdf(s, n=n, sigma=scale, r=r)
    return total


def mixture_survival(cut, n, sigma, r, log_mixing, moments):
    """``P(s >= cut)`` under the same mixture.

    Every node contributes, including the ones the density sum skips: a mutation scale
    far below the data resolves no effect at all but is still a mutation that clears the
    cut, and dropping those would understate the surviving mass.
    """
    scales, weight = _mixture_weights(sigma, log_mixing, moments)
    if scales is None:
        return np.nan
    survival = cmn_fgm._ncx2_cdf(
        np.maximum((r * r - 2.0 * float(cut)) / scales ** 2, 0.0),
        np.asarray(float(n)), r * r / scales ** 2)
    return float(np.sum(weight * survival))


# ──────────────────────────────────── Public API ──────────────────────────────────
def stable_fgm_dfe_pdf(s, n, sigma, r, alpha):
    """Density of the log-fitness FGM DFE with sub-Gaussian alpha-stable mutations."""
    kappa = 0.5 * float(alpha)
    return mixture_dfe_pdf(s, n, sigma, r,
                           lambda y: stable_log_mixing_density(y, kappa),
                           stable_mixing_moments(kappa))


def stable_fgm_dfe_logpdf(s, n, sigma, r, alpha):
    """Log of :func:`stable_fgm_dfe_pdf`, with -inf where the mixture underflows."""
    with np.errstate(divide="ignore"):
        return np.log(stable_fgm_dfe_pdf(s, n, sigma, r, alpha))


def stable_fgm_survival(cut, n, sigma, r, alpha):
    """``P(s >= cut)`` for the sub-Gaussian alpha-stable FGM DFE."""
    kappa = 0.5 * float(alpha)
    return mixture_survival(cut, n, sigma, r,
                            lambda y: stable_log_mixing_density(y, kappa),
                            stable_mixing_moments(kappa))


def student_fgm_dfe_pdf(s, n, sigma, r, mu):
    """The Student FGM DFE through the SAME quadrature, for validation only.

    cmn_cauchy_fgm.cauchy_fgm_dfe_pdf is the exact closed form and is what analysis code
    should call; this exists so ``selfcheck`` can compare the two.
    """
    return mixture_dfe_pdf(s, n, sigma, r,
                           lambda y: student_log_mixing_density(y, mu),
                           student_mixing_moments(mu))


def student_fgm_survival(cut, n, sigma, r, mu):
    """The Student FGM survival through the same quadrature, for validation only."""
    return mixture_survival(cut, n, sigma, r,
                            lambda y: student_log_mixing_density(y, mu),
                            student_mixing_moments(mu))


# ───────────────────────────────────── Checks ─────────────────────────────────────
def selfcheck(verbose=True):
    """Validate the mixing law and the mixture pipeline against closed forms.

    Returns the worst relative error seen.  Run as ``python cmn/cmn_stable_fgm.py``.
    """
    from cmn.cmn_cauchy_fgm import cauchy_fgm_dfe_pdf, cauchy_fgm_survival

    worst = 0.0

    def report(name, error):
        nonlocal worst
        worst = max(worst, error)
        if verbose:
            print(f"  {name:<52s} max rel err {error:.3e}")

    if verbose:
        print("mixing law")
    # The Levy density is the one closed form the positive stable family has.
    y = np.linspace(-4.0, 14.0, 60)
    a = np.exp(y)
    levy = a * np.exp(-1.0 / (4.0 * a)) / (2.0 * np.sqrt(np.pi) * a ** 1.5)
    report("kappa=1/2 against the Levy density",
           float(np.max(np.abs(np.exp(stable_log_mixing_density(y, 0.5)) - levy) / levy)))

    # The defining Laplace transform, which probes the whole law rather than one point.
    grid = np.linspace(-70.0, 70.0, 28001)
    for kappa in (0.2, 0.375, 0.5, 0.75, 0.9, 0.98):
        density = np.exp(stable_log_mixing_density(grid, kappa))
        errors = [abs(np.trapezoid(density * np.exp(-lam * np.exp(grid)), grid)
                      - np.exp(-lam ** kappa)) / np.exp(-lam ** kappa)
                  for lam in (0.3, 1.0, 3.0)]
        report(f"kappa={kappa}: E exp(-lam A) = exp(-lam^kappa)", float(max(errors)))

    if verbose:
        print("mixture pipeline, against the exact Student FGM DFE")
    # The whole pipeline -- grid placement, trapezoid, pruning, survival -- driven with
    # the Student mixing law, whose FGM DFE cmn_cauchy_fgm evaluates in closed form.
    probe = np.array([-0.45, -0.3, -0.15, -0.05, -0.01, -1.0e-3, -1.0e-4,
                      -1.0e-5, 0.0, 0.01, 0.03, 0.06])
    for n, sigma, r, mu in ((6, 0.017, 0.379, 0.385), (7, 0.014, 0.485, 0.343),
                            (10, 0.031, 0.534, 1.298), (6, 0.020, 0.400, 0.150),
                            (8, 0.050, 0.450, 5.000), (8, 0.100, 0.450, 50.00)):
        exact = cauchy_fgm_dfe_pdf(probe, n=n, sigma=sigma, r=r, mu=mu)
        got = student_fgm_dfe_pdf(probe, n=n, sigma=sigma, r=r, mu=mu)
        # Only where the density is a number a likelihood could ever meet.  Below 1e-10
        # of the peak the two disagree wildly in relative terms and identically in
        # absolute ones; see the note on WEIGHT_FLOOR.
        live = exact > 1.0e-10 * exact.max()
        report(f"n={n} sigma={sigma} r={r} mu={mu}: density",
               float(np.max(np.abs(got[live] - exact[live]) / exact[live])))
        exact_survival = float(cauchy_fgm_survival(
            -0.5, n=n, sigma=sigma, r=r, eps=0.0, mu=mu))
        got_survival = student_fgm_survival(-0.5, n=n, sigma=sigma, r=r, mu=mu)
        # cmn_cauchy_fgm's own survival quadrature loses accuracy as mu falls -- at
        # mu = 0.15 it is the one that moves, not this one, which is stable to ten
        # digits under a 3x refinement.  Reported, not silently tolerated.
        report(f"n={n} sigma={sigma} r={r} mu={mu}: survival",
               float(abs(got_survival - exact_survival) / exact_survival))

    if verbose:
        print("stable mixture, internal convergence under grid refinement")
    # No closed form to compare against, so instead check that halving the step and
    # widening the range does not move the answer.
    global COARSE_STEP, SIGMA_EFF_MIN, SIGMA_EFF_MAX
    saved = (COARSE_STEP, SIGMA_EFF_MIN, SIGMA_EFF_MAX)
    for alpha in (0.4, 0.7, 1.0, 1.4, 1.8, 1.95):
        reference = stable_fgm_dfe_pdf(probe, 6, 0.017, 0.379, alpha)
        live = reference > 1.0e-10 * reference.max()
        reference_survival = stable_fgm_survival(-0.5, 6, 0.017, 0.379, alpha)
        COARSE_STEP, SIGMA_EFF_MIN, SIGMA_EFF_MAX = 0.08, 1.0e-15, 1.0e6
        fine = stable_fgm_dfe_pdf(probe, 6, 0.017, 0.379, alpha)
        fine_survival = stable_fgm_survival(-0.5, 6, 0.017, 0.379, alpha)
        COARSE_STEP, SIGMA_EFF_MIN, SIGMA_EFF_MAX = saved
        report(f"alpha={alpha}: density under 3x refinement",
               float(np.max(np.abs(fine[live] - reference[live]) / reference[live])))
        report(f"alpha={alpha}: survival under 3x refinement",
               float(abs(fine_survival - reference_survival) / reference_survival))

    if verbose:
        print(f"worst relative error anywhere: {worst:.3e}")
    return worst


if __name__ == "__main__":
    selfcheck()
