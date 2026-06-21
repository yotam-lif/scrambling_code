import numpy as np
from scipy.stats import ncx2

class Fisher:
    """
    Fisher Geometric Model with Gaussian mutation steps and SSWM relaxation.

    In this version, the selection matrix S is represented only by its eigenvalues
    (i.e., working in the diagonal basis). Both isotropic and anisotropic cases
    are supported through the eigenvalue spectrum.

    Attributes
    ----------
    n : int
        Dimensionality of phenotype space.
    sigma : float
        Standard deviation for Gaussian mutation steps.
    deltas : numpy.ndarray
        Array of pre-sampled mutation steps of shape (m, n).
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    """

    def __init__(self, n, sigma=0.05, m=10**3, random_state=None):
        """
        Initialize the model in the diagonal basis.

        Parameters
        ----------
        n : int
            Number of phenotypic traits (dimensions).
        sigma : float
            Scale parameter for mutations.
        m : int
            Number of mutation vectors to pre-sample.
        random_state : int or numpy.random.Generator, optional
            Seed or RNG for reproducibility.
        """
        self.n = int(n)
        self.sigma = float(sigma)
        self.m = int(m)
        # RNG setup
        if isinstance(random_state, (int, np.integer)):
            self.rng = np.random.default_rng(random_state)
        elif isinstance(random_state, np.random.Generator):
            self.rng = random_state
        else:
            self.rng = np.random.default_rng()

        # Pre-sample Gaussian mutation steps
        self.deltas = self.rng.normal(loc=0.0, scale=self.sigma, size=(self.m, self.n))
        # Initialize r_0 = (sqrt(n), 0, ..., 0), isotropic so only initial radius of n matters
        # Instead of starting at random r and taking iid gaussian entries of sigma_0 so that r_0 ~ n * sigma_0 ^2,
        # We set scale by sigma_0 = 1 and sigma becomes in units of sigma_0.
        # initial radius scales with n so that effective initial position doesn't become smaller as n increases (mutation size scales with n as well).
        self.r = np.zeros(n)
        self.r[0] = np.sqrt(n)

    def _sample_semicircle(self, n, sigma):
        """
        Sample n values from the semicircle distribution on [-2*sigma, 2*sigma]:
        density f(x) ∝ sqrt(4*sigma^2 - x^2) using rejection sampling.
        """
        radius = 2.0 * sigma
        samples = []
        while len(samples) < n:
            x = self.rng.uniform(-radius, radius)
            accept_prob = np.sqrt(radius**2 - x**2) / radius
            if self.rng.uniform(0.0, 1.0) < accept_prob:
                samples.append(x)
        return np.array(samples)

    def compute_log_fitness(self, r):
        r = np.asarray(r, dtype=float)
        return - float(np.dot(r, r))

    def compute_fitness(self, r):
        """
        Compute fitness: w(r) = exp(log_fitness(r)).
        """
        return float(np.exp(self.compute_log_fitness(r)))

    def compute_dfe(self, r):
        """
        Compute distribution of fitness effects at phenotype r.
        Returns array of w(r + delta_i) - w(r) for each pre-sampled delta.
        """
        r = np.asarray(r, dtype=float)
        w0 = self.compute_fitness(r)
        return np.array([self.compute_fitness(r + delta) - w0 for delta in self.deltas])

    def compute_bdfe(self, dfe):
        """
        Extract beneficial fitness effects and their indices from dfe array.
        """
        dfe = np.asarray(dfe, dtype=float)
        mask = dfe > 0
        return dfe[mask], np.nonzero(mask)[0]

    def sswm_choice(self, bdfe, b_ind):
        """
        Choose a substitution under SSWM: probability ∝ fitness effect.
        """
        bdfe = np.asarray(bdfe, dtype=float)
        total = bdfe.sum()
        if total > 0:
            probs = bdfe / bdfe.sum()
        else:
            probs = bdfe / len(bdfe)
        return int(self.rng.choice(b_ind, p=probs))

    def relax(self, max_steps=10 ** 4):
        """
        Perform an adaptive walk using SSWM.
        Returns list of chosen mutation indices, r history and dfe history.
        """
        traj = [self.r.copy()]
        flips = []
        dfes = []
        for _ in range(max_steps):
            dfe = self.compute_dfe(self.r)
            dfes.append(dfe.copy())
            bdfe, b_ind = self.compute_bdfe(dfe)
            if len(b_ind) == 0:
                break
            choice = self.sswm_choice(bdfe, b_ind)
            flips.append(choice)
            self.r += self.deltas[choice]
            # self.deltas[choice] = -1 * self.deltas[choice]
            traj.append(self.r.copy())
        return flips, traj, dfes


# ──────────────────────────────────────────────────────────────────────────────
# Analytic isotropic FGM distribution of fitness effects (DFE)
#
# Derivation: Theory/fgm_dfe.tex. A genotype sits at radius r in an n-dimensional
# phenotype space with Gaussian fitness w(x) = exp(-|x|^2 / 2). A mutation adds an
# isotropic step delta ~ N(0, sigma^2 I_n); the fitness effect is
#
#       Delta = exp(-U/2) - exp(-r^2/2),   U = |r + delta|^2,
#
# and U/sigma^2 ~ noncentral chi^2 with n d.o.f. and noncentrality lambda=r^2/sigma^2.
# Three parameters: n (phenotypic dimension), sigma (mutation step s.d.), r (radius).
# Support: Delta in [-exp(-r^2/2), 1 - exp(-r^2/2)].
# ──────────────────────────────────────────────────────────────────────────────
def fgm_support(r):
    """Support (delta_min, delta_max) of the FGM DFE at radius ``r``."""
    e = np.exp(-0.5 * np.square(r))
    return -e, 1.0 - e


def fgm_dfe_logpdf(delta, n, sigma, r):
    """Log density of the FGM DFE P(Delta | n, sigma, r), vectorized over ``delta``.

    ``n``, ``sigma``, ``r`` are scalars. Returns -inf outside the support.
    """
    delta = np.asarray(delta, float)
    s2 = sigma * sigma
    e = np.exp(-0.5 * r * r)
    arg = delta + e                       # = exp(-U/2); must lie in (0, 1]
    out = np.full(arg.shape, -np.inf)
    m = (arg > 0.0) & (arg <= 1.0)
    if np.any(m):
        u = -2.0 * np.log(arg[m])
        out[m] = (np.log(2.0) - np.log(arg[m])
                  + ncx2.logpdf(u / s2, df=n, nc=r * r / s2) - np.log(s2))
    return out


def fgm_dfe_pdf(delta, n, sigma, r):
    """Density of the FGM DFE (exp of :func:`fgm_dfe_logpdf`)."""
    return np.exp(fgm_dfe_logpdf(delta, n, sigma, r))


def _fgm_cdf_U_at(edges, n, sigma, r):
    """F_U(u(edge)) = P(U <= u(edge)) for each bin edge, broadcast over a grid.

    ``n``, ``sigma``, ``r`` are 1-D arrays of length G (grid points); ``edges`` is
    a 1-D array of length E. Returns an array of shape (G, E). Edges below the
    support map to 1 (all mass is above), edges at/above delta_max map to 0.
    """
    n = np.asarray(n, float)[:, None]
    sigma = np.asarray(sigma, float)[:, None]
    r = np.asarray(r, float)[:, None]
    e = np.exp(-0.5 * r * r)                      # (G, 1)
    s = edges[None, :] + e                         # (G, E) = exp(-U/2) at each edge
    below = s <= 0.0                               # delta <= delta_min  -> F_U = 1
    above = s >= 1.0                               # delta >= delta_max  -> F_U = 0
    mid = ~(below | above)
    s2 = sigma * sigma
    u = -2.0 * np.log(np.where(mid, s, 1.0))       # safe log (clamped off-support)
    F = ncx2.cdf(u / s2, df=n, nc=r * r / s2)      # broadcast to (G, E)
    F = np.where(mid, F, 0.0)
    F[below] = 1.0
    return F


def fgm_bin_loglik(counts, edges, n, sigma, r):
    """Fine-binned multinomial log-likelihood of the FGM DFE over a parameter grid.

    Parameters
    ----------
    counts : (B,) array        histogram counts of the observed effects
    edges  : (B+1,) array      monotone bin edges the counts were built on
    n, sigma, r : (G,) arrays  grid of parameter triples to evaluate

    Returns
    -------
    (G,) array of log-likelihoods. A triple is -inf when any populated bin falls
    outside the FGM support (this is what pins r to the data's extreme effects).
    """
    counts = np.asarray(counts, float)
    edges = np.asarray(edges, float)
    F = _fgm_cdf_U_at(edges, n, sigma, r)          # (G, B+1)
    P = F[:, :-1] - F[:, 1:]                        # (G, B); P(bin_j) >= 0
    P = np.clip(P, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0.0, np.log(P), -np.inf)
        contrib = np.where(counts[None, :] > 0.0, counts[None, :] * logP, 0.0)
    return contrib.sum(axis=1)
