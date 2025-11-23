"""
Sensitivity Analysis: Acquisition × Batch Selection Method × Batch Size (iterations=15)
- Allowed imports only: numpy, scipy.stats.norm, scipy.linalg, scipy.special.erfcx, random, sobol_seq, datetime, matplotlib.pyplot
"""

import numpy as np
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erfcx
import random
import sobol_seq
from datetime import datetime
import matplotlib.pyplot as plt

def compute_throughput_and_cumulative(time_log, Y_data):
    """
    Compute cumulative time, cumulative titre, and throughput using the exact same
    approach as:
        t = np.cumsum(time)
        cumulative_titre = np.cumsum(Y)
    Throughput is the final cumulative titre divided by the final cumulative time.
    """
    cum_t = np.cumsum(time_log)
    cum_y = np.cumsum(Y_data)
    thr   = float(cum_y[-1]) / float(cum_t[-1] + 1e-12)
    return thr, cum_t, cum_y

# ====================== Sobol sampling & encoding ======================
def sobol_initial_samples(n_samples):
    sobol_points = sobol_seq.i4_sobol_generate(5, n_samples)
    temp = 30 + sobol_points[:,0] * 10
    pH   = 6  + sobol_points[:,1] * 2
    f1   = sobol_points[:,2] * 50
    f2   = sobol_points[:,3] * 50
    f3   = sobol_points[:,4] * 50
    celltype = ['celltype_1','celltype_2','celltype_3']
    cells = [random.choice(celltype) for _ in range(n_samples)]
    return [[float(temp[i]), float(pH[i]), float(f1[i]), float(f2[i]), float(f3[i]), cells[i]] for i in range(n_samples)]

def encode_features_vectorized(X):
    X_arr = np.array(X, dtype=object)
    num = X_arr[:, :5].astype(np.float64)
    normed = np.column_stack([
        (num[:,0]-30)/10.0, (num[:,1]-6)/2.0, num[:,2]/50.0, num[:,3]/50.0, num[:,4]/50.0
    ])
    cell_map = {'celltype_1':np.array([1,0,0]), 'celltype_2':np.array([0,1,0]), 'celltype_3':np.array([0,0,1])}
    cells = X_arr[:,5]
    onehot = np.array([cell_map.get(c, np.array([0,0,0])) for c in cells])
    return np.hstack([normed, onehot])

# ====================== Stable Analytic LogEI ==========================
def _log1mexp(x):
    out = np.empty_like(x)
    m = x > -np.log(2.0)
    out[m]  = np.log1p(-np.exp(x[m]))
    out[~m] = np.log(-np.expm1(x[~m]))
    return out

def _log_h(z, eps=np.finfo(float).eps):
    """
    Piecewise-stable log h(z), h(z) = phi(z) + z Phi(z)
    """
    c1 = 0.5*np.log(2*np.pi); c2 = 0.5*np.log(np.pi/2.0)
    z = np.asarray(z, dtype=np.float64)
    out = np.full_like(z, -np.inf)
    thr = -1.0/np.sqrt(eps)

    # Branch 1: z > -1
    m1 = z > -1.0
    if np.any(m1):
        z1 = z[m1]
        h  = norm.pdf(z1) + z1*norm.cdf(z1)
        out[m1] = np.log(np.maximum(h, 1e-300))

    # Branch 2: thr < z <= -1
    m2 = (~m1) & (z > thr)
    if np.any(m2):
        z2 = z[m2]
        t  = np.log(erfcx(-z2/np.sqrt(2.0)) * np.abs(z2)) + c2
        out[m2] = (-0.5*z2**2) - c1 + _log1mexp(t)

    # Branch 3: z <= thr
    m3 = z <= thr
    if np.any(m3):
        z3 = z[m3]
        out[m3] = (-0.5*z3**2) - c1 - 2.0*np.log(np.abs(z3))

    return out
# ---------- MES utilities (non-myopic, information-theoretic) ----------
def sample_max_values(gp, X_search_enc, n_gamma=32, subset=500, rng=None):
    """
    Draw Monte Carlo samples of the max value γ = max_x f(x) from the GP posterior on a subset.
    - gp: fitted GaussianProcess
    - X_search_enc: encoded candidate set (N x d)
    - n_gamma: number of γ samples
    - subset: number of candidate points used per γ sample
    Returns: gamma_samples, shape (n_gamma,)
    """
    if rng is None:
        rng = np.random.default_rng()
    N = X_search_enc.shape[0]
    m = min(subset, N)
    gamma = np.empty(n_gamma, dtype=np.float64)

    # Precompute one subset of points to stabilize sampling cost
    idx = rng.choice(N, size=m, replace=False)
    X_grid = X_search_enc[idx]
    mu_grid, sigma_grid = gp.predict(X_grid)
    sigma_grid = np.maximum(sigma_grid, 1e-12)

    # Draw n_gamma posterior samples across the grid and take max
    for i in range(n_gamma):
        z = rng.standard_normal(m)
        f_samp = mu_grid + sigma_grid * z
        gamma[i] = np.max(f_samp)
    return gamma

def mes_acquisition(mu, sigma, gamma_samples):
    """
    MES q=1 approximation:
      α(x) ≈ E_γ[ -log Φ((γ - μ(x)) / σ(x)) ]
    Returns vector α of shape (len(mu),).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    gam = np.asarray(gamma_samples, dtype=np.float64).reshape(-1, 1)  # (S,1)

    # Compute normalized margins for all x and γ
    z = (gam - mu.reshape(1, -1)) / sigma.reshape(1, -1)              # (S, n)
    Phi = np.clip(norm.cdf(z), 1e-12, 1.0)
    a = -np.log(Phi)                                                  # (S, n)
    return np.mean(a, axis=0)                                         # (n,)

def logei(mu, sigma, best_y, xi=0.01):
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    mu    = np.asarray(mu, dtype=np.float64)
    z     = (mu - (best_y + xi)) / sigma
    log_ei= _log_h(z) + np.log(sigma)
    ei    = np.exp(np.clip(log_ei, -745, 700))
    return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)

# ---------- NEI (Noisy Expected Improvement, MC approximation) ----------
def nei_acquisition(mu, sigma, best_y, xi=0.01, n_mc=64, rng=None):
    """
    Monte Carlo Noisy Expected Improvement (vectorized):
      NEI(x) ≈ E[max(f - best_y - xi, 0)],  f ~ N(mu, sigma)
    Args:
      mu, sigma: arrays of shape (n,)
      best_y: scalar (current best observed value)
      xi: small offset
      n_mc: number of MC samples
    Returns:
      nei: (n,) acquisition values
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    rng = np.random.default_rng() if rng is None else rng

    # Draw z ~ N(0,1), then f = mu + sigma*z  (broadcast to (n_mc, n))
    z = rng.standard_normal((n_mc, mu.shape[0]))
    f = mu.reshape(1, -1) + sigma.reshape(1, -1) * z
    imp = f - (best_y + xi)
    return np.mean(np.maximum(imp, 0.0), axis=0)

# ---------- KG (Knowledge Gradient, fast linear one-step lookahead) ----------
def make_kg_callable(gp, X_search_enc, n_mc=32, subset=400, rng=None):
    """
    Returns a callable kg(mu, sigma) that approximates one-step Knowledge Gradient:
      KG(x) ≈ E[ max_x' mu_plus(x') ] - max_x' mu(x')
    Using a linearized posterior mean update over a subset grid:
      mu_plus(X_grid) ≈ mu_grid + w(x) * (y - mu_x), with y ~ N(mu_x, sigma_x)
      w(x) = k(X_grid, x)/sigma_x^2
    Args:
      gp: fitted GaussianProcess
      X_search_enc: (N,d) encoded candidate set
      n_mc: number of fantasy draws
      subset: size of grid subset used to evaluate KG
    Returns:
      kg(mu, sigma): function mapping vectors (mu, sigma) to KG values for all candidates
    """
    rng = np.random.default_rng() if rng is None else rng
    N = X_search_enc.shape[0]
    m = min(subset, N)
    grid_idx = rng.choice(N, size=m, replace=False)
    X_grid = X_search_enc[grid_idx]
    mu_grid, _ = gp.predict(X_grid)
    base_max = float(np.max(mu_grid))

    # Precompute kernel between grid and each evaluated candidate on the fly
    def kg(mu, sigma):
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
        n = mu.shape[0]
        out = np.zeros(n, dtype=np.float64)

        # For each candidate x_i, compute KG via MC fantasies
        for i in range(n):
            # k(X_grid, x_i): shape (m,)
            k_grid_xi = gp.rbf_kernel(X_grid, X_search_enc[i:i+1]).reshape(-1)
            w = k_grid_xi / (sigma[i]**2 + 1e-12)  # linear influence on mu_grid

            # MC fantasies: y_i ~ N(mu[i], sigma[i])
            z = rng.standard_normal(n_mc)
            y_f = mu[i] + sigma[i]*z  # (n_mc,)
            # mu_plus = mu_grid + w*(y_f - mu[i])
            mu_plus_max = np.max(mu_grid.reshape(1, -1) + (y_f - mu[i]).reshape(-1, 1)*w.reshape(1, -1), axis=1)
            out[i] = float(np.mean(mu_plus_max) - base_max)
        return out

    return kg

# ====================== Gaussian Process (RBF) ==========================
class GaussianProcess:
    def __init__(self, length_scale=0.5, sigma_f=1.0, noise=1e-6):
        self.length_scale = float(length_scale)
        self.sigma_f = float(sigma_f)
        self.noise = float(noise)
        self.X_train = None
        self.alpha = None
        self.L = None
        self.Y_mean = 0.0
        self.Y_std = 1.0
        self._K_cache = None
        self._X_train_shape = None

    @staticmethod
    def compute_squared_distances(X1, X2):
        X1 = np.atleast_2d(X1); X2 = np.atleast_2d(X2)
        X1_sq = np.sum(X1**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2**2, axis=1, keepdims=True).T
        return np.maximum(X1_sq + X2_sq - 2.0*(X1 @ X2.T), 0.0)

    def rbf_kernel(self, X1, X2):
        d2 = self.compute_squared_distances(X1, X2)
        return (self.sigma_f**2) * np.exp(-0.5 * d2 / (self.length_scale**2))

    def fit(self, X, Y):
        self.X_train = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Y = np.asarray(Y, dtype=np.float64).flatten()
        self.Y_mean = float(np.mean(Y))
        self.Y_std  = float(np.std(Y)) if np.std(Y) > 1e-12 else 1.0
        Y_norm = (Y - self.Y_mean) / self.Y_std

        shape = self.X_train.shape
        if self._X_train_shape != shape or self._K_cache is None:
            K = self.rbf_kernel(self.X_train, self.X_train)
            K[np.diag_indices_from(K)] += self.noise
            self._K_cache = K
            self._X_train_shape = shape
        else:
            K = self._K_cache

        c, lower = cho_factor(K, lower=True, check_finite=False)
        self.L = c if lower else c.T
        self.alpha = cho_solve((c, lower), Y_norm, check_finite=False)

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        K_star = self.rbf_kernel(X, self.X_train)
        mu_norm = K_star @ self.alpha
        mu = mu_norm * self.Y_std + self.Y_mean
        v = np.linalg.solve(self.L, K_star.T)
        var = (self.sigma_f**2) - np.sum(v**2, axis=0)
        var = np.maximum(var, 1e-12) * (self.Y_std**2)
        sigma = np.sqrt(var)
        return mu, sigma

# ====================== Acquisition functions ===========================
class AcquisitionFunctions:
    @staticmethod
    def expected_improvement(mu, sigma, best_y, xi=0.01):
        mu = np.asarray(mu); sigma = np.asarray(sigma)
        imp = mu - best_y - xi
        z   = np.divide(imp, sigma, out=np.zeros_like(imp), where=sigma>1e-9)
        ei  = imp*norm.cdf(z) + sigma*norm.pdf(z)
        ei[sigma < 1e-9] = 0.0
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def log_expected_improvement(mu, sigma, best_y, xi=0.01):
        return logei(mu, sigma, best_y, xi)

    @staticmethod
    def upper_confidence_bound(mu, sigma, beta=2.0):
        return mu + beta*sigma

    @staticmethod
    def probability_of_improvement(mu, sigma, best_y, xi=0.01):
        mu = np.asarray(mu); sigma = np.asarray(sigma)
        z  = np.divide(mu - best_y - xi, sigma, out=np.zeros_like(mu), where=sigma>1e-9)
        poi= norm.cdf(z)
        poi[sigma < 1e-9] = 0.0
        return np.nan_to_num(poi, nan=0.0)

def make_acq_callable(kind, best_y, stagnation_count, restart_threshold, mes_gamma=None, kg_call=None):
    xi = 0.01 + (0.1 * min(stagnation_count / restart_threshold, 1.0))
    beta = 2.0 * (1 + stagnation_count / 10.0)

    if kind == 'ei':
        return lambda mu, sigma: AcquisitionFunctions.expected_improvement(mu, sigma, best_y, xi)
    elif kind == 'logei':
        return lambda mu, sigma: AcquisitionFunctions.log_expected_improvement(mu, sigma, best_y, xi)
    elif kind == 'ucb':
        return lambda mu, sigma: AcquisitionFunctions.upper_confidence_bound(mu, sigma, beta)
    elif kind == 'mes':
        assert mes_gamma is not None and len(mes_gamma) > 0, "MES requires mes_gamma samples."
        return lambda mu, sigma: mes_acquisition(mu, sigma, mes_gamma)
    elif kind == 'nei':
        return lambda mu, sigma: nei_acquisition(mu, sigma, best_y, xi=xi, n_mc=64)
    elif kind == 'kg':
        assert kg_call is not None, "KG requires a kg_call prepared with gp and grid subset."
        return lambda mu, sigma: kg_call(mu, sigma)
    else:
        # default to LogEI
        return lambda mu, sigma: AcquisitionFunctions.log_expected_improvement(mu, sigma, best_y, xi)

# ====================== Batch selection methods =========================
def select_batch_local_pen(gp, X_search_enc, Y_train, n_batch, acq_callable, exclude_idx=None, length_scale=0.5):
    exclude_idx = set(exclude_idx or [])
    selected = []
    mu, sigma = gp.predict(X_search_enc)
    base_acq = acq_callable(mu, sigma).copy()
    base_acq[list(exclude_idx)] = -np.inf

    for _ in range(n_batch):
        acq = base_acq.copy()
        if selected:
            sel_pts = X_search_enc[selected]
            d = np.sqrt(GaussianProcess.compute_squared_distances(X_search_enc, sel_pts))
            penalties = np.exp(-d**2 / (2.0 * length_scale**2))
            total_penalty = 1 - 0.9 * np.max(penalties, axis=1)
            acq *= total_penalty
        acq[selected] = -np.inf
        if np.max(acq) <= -1e10: break
        best_idx = int(np.argmax(acq))
        selected.append(best_idx)
    return selected

def select_batch_constant_liar(gp, X_search_enc, Y_train, n_batch, acq_callable, exclude_idx=None):
    exclude_idx = list(exclude_idx or [])
    selected = []
    temp_X = gp.X_train.copy()
    temp_Y = Y_train.copy()
    lie_value = float(np.mean(temp_Y))
    for k in range(n_batch):
        mu, sigma = gp.predict(X_search_enc)
        acq = acq_callable(mu, sigma)
        acq[exclude_idx + selected] = -np.inf
        if np.max(acq) <= -1e10: break
        best = int(np.argmax(acq))
        selected.append(best)
        if k < n_batch-1:
            temp_X = np.vstack([temp_X, X_search_enc[best]])
            temp_Y = np.append(temp_Y, lie_value)
            gp.fit(temp_X, temp_Y)
    return selected

def select_batch_kriging_believer(gp, X_search_enc, Y_train, n_batch, acq_callable, exclude_idx=None):
    exclude_idx = list(exclude_idx or [])
    selected = []
    temp_X = gp.X_train.copy()
    temp_Y = Y_train.copy()
    for k in range(n_batch):
        mu, sigma = gp.predict(X_search_enc)
        acq = acq_callable(mu, sigma)
        acq[exclude_idx + selected] = -np.inf
        if np.max(acq) <= -1e10: break
        best = int(np.argmax(acq))
        selected.append(best)
        if k < n_batch-1:
            temp_X = np.vstack([temp_X, X_search_enc[best]])
            temp_Y = np.append(temp_Y, mu[best])
            gp.fit(temp_X, temp_Y)
    return selected

def select_batch_pess_believer(gp, X_search_enc, Y_train, n_batch, acq_callable, exclude_idx=None):
    exclude_idx = list(exclude_idx or [])
    selected = []
    temp_X = gp.X_train.copy()
    temp_Y = Y_train.copy()
    for k in range(n_batch):
        mu, sigma = gp.predict(X_search_enc)
        acq = acq_callable(mu, sigma)
        acq[exclude_idx + selected] = -np.inf
        if np.max(acq) <= -1e10: break
        best = int(np.argmax(acq))
        selected.append(best)
        if k < n_batch-1:
            y_fake = mu[best] - 2.0*sigma[best]
            temp_X = np.vstack([temp_X, X_search_enc[best]])
            temp_Y = np.append(temp_Y, y_fake)
            gp.fit(temp_X, temp_Y)
    return selected

def select_batch_thompson_sampling(gp, X_search_enc, Y_train, n_batch, exclude_idx=None):
    exclude_idx = list(exclude_idx or [])
    selected = []
    mu, sigma = gp.predict(X_search_enc)
    for _ in range(n_batch):
        sample = mu + sigma * np.random.randn(len(mu))
        sample[exclude_idx + selected] = -np.inf
        if np.max(sample) <= -1e10: break
        selected.append(int(np.argmax(sample)))
    return selected

def select_batch_fantasized_qei(gp, X_search_enc, Y_train, n_batch, exclude_idx=None):
    """
    SOTA-like qEI via fantasization:
    - Sequentially pick points by single-point EI
    - Fantasize labels by sampling from GP posterior at selected points
    """
    exclude_idx = list(exclude_idx or [])
    selected = []
    temp_X = gp.X_train.copy()
    temp_Y = Y_train.copy()
    for k in range(n_batch):
        mu, sigma = gp.predict(X_search_enc)
        best_y = float(np.max(temp_Y))
        # Single-point EI
        imp = mu - best_y - 0.01
        z = np.divide(imp, sigma, out=np.zeros_like(imp), where=sigma>1e-9)
        acq = imp*norm.cdf(z) + sigma*norm.pdf(z)
        acq[exclude_idx + selected] = -np.inf
        if np.max(acq) <= -1e10: break
        best = int(np.argmax(acq))
        selected.append(best)
        if k < n_batch-1:
            # Fantasize: sample from posterior (noisy observation optional)
            y_fake = float(mu[best] + sigma[best]*np.random.randn())
            temp_X = np.vstack([temp_X, X_search_enc[best]])
            temp_Y = np.append(temp_Y, y_fake)
            gp.fit(temp_X, temp_Y)
    return selected

def select_batch_fantasized_qnei(gp, X_search_enc, Y_train, n_batch, exclude_idx=None, obs_noise_std=None):
    """
    SOTA-like qNEI via fantasization with observation noise:
    - Same as qEI, but add observation noise to the fantasized labels
    """
    exclude_idx = list(exclude_idx or [])
    selected = []
    temp_X = gp.X_train.copy()
    temp_Y = Y_train.copy()
    noise_std = float(obs_noise_std) if obs_noise_std is not None else 0.02
    for k in range(n_batch):
        mu, sigma = gp.predict(X_search_enc)
        best_y = float(np.max(temp_Y))
        imp = mu - best_y - 0.01
        z = np.divide(imp, sigma, out=np.zeros_like(imp), where=sigma>1e-9)
        acq = imp*norm.cdf(z) + sigma*norm.pdf(z)
        acq[exclude_idx + selected] = -np.inf
        if np.max(acq) <= -1e10: break
        best = int(np.argmax(acq))
        selected.append(best)
        if k < n_batch-1:
            y_fake = float(mu[best] + sigma[best]*np.random.randn() + noise_std*np.random.randn())
            temp_X = np.vstack([temp_X, X_search_enc[best]])
            temp_Y = np.append(temp_Y, y_fake)
            gp.fit(temp_X, temp_Y)
    return selected

#========= Printing best combinations
def print_top5_combos_by_throughput(results, top_k=5):
    """
    Prints the top-K (acquisition, batch_method, batch_size) combos
    ranked by average throughput across repeats.
    """
    if results is None or len(results) == 0:
        print("No results to summarize!")
        return

    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        key = (r['acquisition'], r['batch_method'], int(r['batch_size']))
        agg[key].append(float(r['throughput']))

    rows = []
    for (acq, method, bs), vals in agg.items():
        rows.append((acq, method, bs, float(np.mean(vals))))
    rows.sort(key=lambda t: t[3], reverse=True)

    k = min(top_k, len(rows))
    print("\n" + "="*60)
    print(f"TOP {k} COMBINATIONS BY THROUGHPUT (avg over repeats)")
    print("="*60)
    for i in range(k):
        acq, method, bs, thr = rows[i]
        print(f"{i+1:>2}. acquisition={acq:>6}, method={method:>22}, batch={bs:>2}  →  throughput={thr:.3f} g/L per s")

# ====================== Single BO run (iterations=15) ===================
def run_single_bo(
    X_initial, X_search_space, objective_func,
    acquisition='logei', batch_method='local_penalization', batch_size=5, iterations=15,
    random_ratio=0.3, restart_threshold=4, length_scale=0.5
):
    # Pre-encode
    X_search_enc = encode_features_vectorized(X_search_space)
    X_init_enc   = encode_features_vectorized(X_initial)

    # Evaluate initial
    t0 = datetime.now()
    Y_init = objective_func(X_initial)
    time_log = [ (datetime.now() - t0).total_seconds() ] + [0]*(len(Y_init)-1)

    X_data = X_init_enc.copy()
    Y_data = np.asarray(Y_init, dtype=np.float64)

    best_yield = float(np.max(Y_data))
    stagnation = 0
    restart_count = 0

    for it in range(iterations):
        gp = GaussianProcess(length_scale=length_scale*(1.0 + 0.8*min(stagnation/5,1.0)))
        gp.fit(X_data, Y_data)
        # After gp.fit(X_data, Y_data):
# Prepare MES γ samples (reused for this iteration)
        # Prepare MES γ samples if needed
        if acquisition == 'mes':
            mes_gamma = sample_max_values(gp, X_search_enc, n_gamma=32, subset=500)
        else:
            mes_gamma = None

        # Prepare KG callable if needed (uses current gp and a grid subset)
        if acquisition == 'kg':
            kg_call = make_kg_callable(gp, X_search_enc, n_mc=32, subset=400)
        else:
            kg_call = None

        acq_call = make_acq_callable(
            acquisition, float(np.max(Y_data)), stagnation, restart_threshold,
            mes_gamma=mes_gamma, kg_call=kg_call
        )
                

        # Mix random + BO (kept consistent)
        n_random = max(1, int(batch_size*random_ratio))
        n_bo     = batch_size - n_random
        selected_indices = []
        methods = []

        # Random picks
        if n_random > 0:
            idx_rand = random.sample(range(len(X_search_space)), n_random)
            selected_indices += idx_rand
            methods += ['Random']*n_random

        # BO picks via chosen batch method
        if n_bo > 0 and len(Y_data) >= 3:
            if batch_method == 'local_penalization':
                idx_bo = select_batch_local_pen(gp, X_search_enc, Y_data, n_bo, acq_call, exclude_idx=selected_indices, length_scale=length_scale)
            elif batch_method == 'constant_liar':
                idx_bo = select_batch_constant_liar(gp, X_search_enc, Y_data, n_bo, acq_call, exclude_idx=selected_indices)
            elif batch_method == 'kriging_believer':
                idx_bo = select_batch_kriging_believer(gp, X_search_enc, Y_data, n_bo, acq_call, exclude_idx=selected_indices)
            elif batch_method == 'pessimistic_believer':
                idx_bo = select_batch_pess_believer(gp, X_search_enc, Y_data, n_bo, acq_call, exclude_idx=selected_indices)
            elif batch_method == 'thompson_sampling':
                idx_bo = select_batch_thompson_sampling(gp, X_search_enc, Y_data, n_bo, exclude_idx=selected_indices)
            elif batch_method == 'fantasized_qei':
                idx_bo = select_batch_fantasized_qei(gp, X_search_enc, Y_data, n_bo, exclude_idx=selected_indices)
            elif batch_method == 'fantasized_qnei':
                idx_bo = select_batch_fantasized_qnei(gp, X_search_enc, Y_data, n_bo, exclude_idx=selected_indices, obs_noise_std=0.02)
            else:
                idx_bo = []
            selected_indices += idx_bo
            methods += ['BO']*len(idx_bo)

        # Evaluate batch; record time as [elapsed] + zeros
        X_batch = [X_search_space[i] for i in selected_indices]
        t_eval = datetime.now()
        Y_batch = objective_func(X_batch)
        elapsed = (datetime.now() - t_eval).total_seconds()

        Y_batch = np.asarray(Y_batch, dtype=np.float64)
        X_data = np.vstack([X_data, X_search_enc[selected_indices]])
        Y_data = np.concatenate([Y_data, Y_batch])
        time_log += [elapsed] + [0]*(len(Y_batch)-1)

        # Improvement tracking
        y_max = float(np.max(Y_batch))
        if y_max > best_yield:
            best_yield = y_max
            stagnation = 0
        else:
            stagnation += 1
        if stagnation >= restart_threshold:
            restart_count += 1
            stagnation = 0

    #... end of run_single_bo loop...

    # Final cumulative stats (match your example plot semantics)
    # Final cumulative stats
    final_cum_titre = np.cumsum(Y_data)
    final_cum_time  = np.cumsum(time_log)
    throughput = final_cum_titre[-1] / (final_cum_time[-1])

    # Final cumulative stats via the unified helper (same as your example function)
        #... end of loop

    # Final cumulative stats via the unified helper (same as your example)
    throughput, cum_t, cum_y = compute_throughput_and_cumulative(time_log, Y_data)

    return {
        'acquisition': acquisition,
        'batch_method': batch_method,
        'batch_size': batch_size,
        'repeat': 0,  # set by caller
        'throughput': throughput
    }

# ====================== Sensitivity runner (iterations=15) ==============
def run_sensitivity_acq_vs_method_vs_batch(
    objective_func,
    acquisitions=('ei','logei','ucb','mes','nei','kg'),
    batch_methods=('local_penalization','constant_liar','kriging_believer','pessimistic_believer','thompson_sampling','fantasized_qei','fantasized_qnei'),
    batch_sizes=(1,2,3,4,5),
    iterations=15,
    n_repeats=3,
    random_ratio=0.3,
    restart_threshold=4,
    length_scale=0.5
):
    print("\n" + "="*60)
    print("SENSITIVITY: Acquisition × Batch Method × Batch Size (iterations=15)")
    print("="*60)
    results = []
    total = len(acquisitions)*len(batch_methods)*len(batch_sizes)*n_repeats
    count = 0
    for acq in acquisitions:
        for bm in batch_methods:
            print(f"\n--- Acquisition={acq}, Batch Method={bm} ---")
            for bs in batch_sizes:
                print(f"   Batch Size={bs}")
                for r in range(n_repeats):
                    count += 1
                    print(f"     [{count}/{total}] Repeat {r+1}/{n_repeats}...")
                    X_init = sobol_initial_samples(6)
                    X_search = sobol_initial_samples(800)
                    try:
                        res = run_single_bo(
                            X_initial=X_init,
                            X_search_space=X_search,
                            objective_func=objective_func,
                            acquisition=acq,
                            batch_method=bm,
                            batch_size=bs,
                            iterations=iterations,
                            random_ratio=random_ratio,
                            restart_threshold=restart_threshold,
                            length_scale=length_scale
                        )
                        res['repeat'] = r+1
                        results.append(res)
                    except Exception as e:
                        print(f"        Error: {e}")
    return results

# ====================== Plotting: heatmaps per acquisition ==============
def plot_acq_method_batch_throughput(results):
    """
    Plots throughput heatmaps per acquisition:
    - One column per acquisition
    - Heatmap values: average throughput over repeats
    - Axes: methods × batch sizes
    """
    if results is None or len(results) == 0:
        print("No results to plot!")
        return

    # Convert list of dicts → structured array
    rows = []
    for r in results:
        rows.append((
            r['acquisition'], r['batch_method'], int(r['batch_size']),
            int(r.get('repeat', 1)), float(r['throughput'])
        ))
    dtype = [
        ('acquisition','U10'), ('batch_method','U25'), ('batch_size','i4'),
        ('repeat','i4'), ('throughput','f8')
    ]
    res = np.array(rows, dtype=dtype)

    acquisitions = np.unique(res['acquisition'])
    methods      = np.unique(res['batch_method'])
    batch_sizes  = np.unique(res['batch_size'])
    m_index = {m:i for i,m in enumerate(methods)}
    b_index = {b:j for j,b in enumerate(batch_sizes)}

    # Build per-acquisition throughput matrices
    mats_thr = {}
    for acq in acquisitions:
        mat_thr = np.full((len(methods), len(batch_sizes)), np.nan)
        mask_acq = (res['acquisition'] == acq)
        for m in methods:
            for b in batch_sizes:
                mask = mask_acq & (res['batch_method'] == m) & (res['batch_size'] == b)
                if np.any(mask):
                    mat_thr[m_index[m], b_index[b]] = np.mean(res['throughput'][mask])
        mats_thr[acq] = mat_thr

    # Plot one heatmap per acquisition
    fig, axes = plt.subplots(1, len(acquisitions), figsize=(6*len(acquisitions), 6))
    if len(acquisitions) == 1:
        axes = [axes]
    fig.suptitle('Sensitivity: Throughput (final cumulative titre / time)', fontsize=16, fontweight='bold')

    for j, acq in enumerate(acquisitions):
        ax = axes[j]
        im = ax.imshow(mats_thr[acq], cmap='viridis', aspect='auto')
        ax.set_title(f'Throughput — {acq}')
        ax.set_xlabel('Batch Size'); ax.set_ylabel('Batch Method')
        ax.set_xticks(range(len(batch_sizes))); ax.set_xticklabels(batch_sizes)
        ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
        plt.colorbar(im, ax=ax, label='Throughput (g/L per s)')
        for i in range(len(methods)):
            for k in range(len(batch_sizes)):
                v = mats_thr[acq][i, k]
                if not np.isnan(v):
                    ax.text(k, i, f'{v:.3f}', ha='center', va='center', color='white', fontsize=9)

    plt.tight_layout()
    plt.show()

    # Text summary
    print("\n" + "="*60)
    print("SENSITIVITY SUMMARY (per acquisition):")
    print("="*60)
    
def plot_best_combo_cumulative_curve(
    results,
    objective_func,
    iterations=15,
    random_ratio=0.3,
    restart_threshold=4,
    length_scale=0.5
):
    """
    Plot cumulative titre concentration vs cumulative time for the best combo
    by average throughput (final_cum_titre / final_cum_time) across repeats.
    """
    if results is None or len(results) == 0:
        print("No results to plot cumulative curve!")
        return

    # Pick best combo by average throughput across repeats
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        key = (r['acquisition'], r['batch_method'], int(r['batch_size']))
        agg[key].append(float(r['throughput']))

    best_key, best_thr = None, -np.inf
    for k, vals in agg.items():
        m = float(np.mean(vals))
        if m > best_thr:
            best_thr = m
            best_key = k

    acq, method, bs = best_key
    print(f"\nSelected best combo for cumulative plot: acquisition={acq}, method={method}, batch={bs} (avg throughput={best_thr:.3f} g/L per s)")

    # Re-run once to get per-run arrays (time_log and Y_data)
    X_init = sobol_initial_samples(6)
    X_search = sobol_initial_samples(1000)
    X_search_enc = encode_features_vectorized(X_search)
    X_init_enc   = encode_features_vectorized(X_init)

    # Evaluate initial
    t0 = datetime.now()
    Y_init = objective_func(X_init)
    time_log = [ (datetime.now() - t0).total_seconds() ] + [0]*(len(Y_init)-1)

    X_data = X_init_enc.copy()
    Y_data = np.asarray(Y_init, dtype=np.float64)
    stagnation = 0

    for it in range(iterations):
        gp = GaussianProcess(length_scale=length_scale*(1.0 + 0.8*min(stagnation/5,1.0)))
        gp.fit(X_data, Y_data)

        # Prepare MES/KG contexts if needed
        mes_gamma = sample_max_values(gp, X_search_enc, n_gamma=32, subset=500) if acq == 'mes' else None
        kg_call   = make_kg_callable(gp, X_search_enc, n_mc=32, subset=400)     if acq == 'kg'  else None

        acq_call = make_acq_callable(
            acq, float(np.max(Y_data)), stagnation, restart_threshold,
            mes_gamma=mes_gamma, kg_call=kg_call
        )

        # Random + BO selection
        n_random = max(1, int(bs*random_ratio))
        n_bo     = bs - n_random
        selected_indices = []

        if n_random > 0:
            selected_indices += random.sample(range(len(X_search)), n_random)

        if n_bo > 0 and len(Y_data) >= 3:
            if method == 'local_penalization':
                idx_bo = select_batch_local_pen(
                    gp, X_search_enc, Y_data, n_bo, acq_call,
                    exclude_idx=selected_indices, length_scale=length_scale
                )
            elif method == 'constant_liar':
                idx_bo = select_batch_constant_liar(
                    gp, X_search_enc, Y_data, n_bo, acq_call,
                    exclude_idx=selected_indices
                )
            elif method == 'kriging_believer':
                idx_bo = select_batch_kriging_believer(
                    gp, X_search_enc, Y_data, n_bo, acq_call,
                    exclude_idx=selected_indices
                )
            elif method == 'pessimistic_believer':
                idx_bo = select_batch_pess_believer(
                    gp, X_search_enc, Y_data, n_bo, acq_call,
                    exclude_idx=selected_indices
                )
            elif method == 'thompson_sampling':
                idx_bo = select_batch_thompson_sampling(
                    gp, X_search_enc, Y_data, n_bo, exclude_idx=selected_indices
                )
            elif method == 'fantasized_qei':
                idx_bo = select_batch_fantasized_qei(
                    gp, X_search_enc, Y_data, n_bo, exclude_idx=selected_indices
                )
            elif method == 'fantasized_qnei':
                idx_bo = select_batch_fantasized_qnei(
                    gp, X_search_enc, Y_data, n_bo, exclude_idx=selected_indices, obs_noise_std=0.02
                )
            else:
                idx_bo = []
            selected_indices += idx_bo

        # Evaluate batch and record time
        X_batch = [X_search[i] for i in selected_indices]
        t_eval = datetime.now()
        Y_batch = objective_func(X_batch)
        elapsed = (datetime.now() - t_eval).total_seconds()

        Y_batch = np.asarray(Y_batch, dtype=np.float64)
        X_data = np.vstack([X_data, X_search_enc[selected_indices]])
        Y_data = np.concatenate([Y_data, Y_batch])
        time_log += [elapsed] + [0]*(len(Y_batch)-1)

        # Simple stagnation update
        y_max = float(np.max(Y_batch))
        if y_max > float(np.max(Y_data[:-len(Y_batch)])) if len(Y_batch)>0 else False:
            stagnation = 0
        else:
            stagnation += 1

    # Plot cumulative titre vs cumulative time (like your example)
    cum_t = np.cumsum(time_log)
    cum_y = np.cumsum(Y_data)
    # Compute cumulative arrays and throughput using the unified helper
        # Compute cumulative arrays and throughput using the unified helper
    thr, cum_t, cum_y = compute_throughput_and_cumulative(time_log, Y_data)

    # Keep only points where cumulative time increases (one per iteration + initial)
    step_mask = np.r_[True, np.diff(cum_t) > 0]  # True at indices where time advanced
    cum_t_step = cum_t[step_mask]
    cum_y_step = cum_y[step_mask]

    # Optional: sanity check expected number of visible increments = 1 + iterations
    # print(len(cum_t_step))  # should be 1 (initial) + iterations

    plt.figure(figsize=(8, 6))
    plt.plot(cum_t, cum_y, '-o', color='red', label='Cumulative Titre Conc.', markersize=4)
    plt.xlabel('Cumulative Time [s]')
    plt.ylabel('Cumulative Titre Conc. [g/L]')
    plt.title(f'Cumulative Titre vs Time — best combo ({acq}, {method}, batch={bs})')
    plt.legend(); plt.grid(True); plt.show()

    print(f"Final cumulative time: {cum_t[-1]:.3f} s")
    print(f"Final cumulative titre: {cum_y[-1]:.3f} g/L")
    print(f"Throughput: {thr:.3f} g/L per s")

# ====================== Main: run sensitivity only ======================
if __name__ == "__main__":
    # Objective: virtual lab or mock
    try:
        import MLCE_CWBO2025.virtual_lab as virtual_lab
        print("✓ Using virtual_lab objective")
        def objective_func(X): return np.array(virtual_lab.conduct_experiment(X))
    except Exception:
        print("⚠ Using mock objective")
        def objective_func(X):
            out = []
            for temp, pH, f1, f2, f3, cell in X:
                val = (100 - (temp-35)**2 - 2*(pH-7)**2 - 0.1*(f1-25)**2 - 0.1*(f2-30)**2
                       + np.random.randn()*3.0)
                out.append(max(0.0, val))
            return np.array(out, dtype=float)

    print("\n" + "="*60)
    print("Sensitivity Analysis: Acquisition × Batch Method × Batch Size (iterations=15)")
    print("="*60)

    # Optional reproducibility
    # random.seed(42); np.random.seed(42)

    # Updated acquisitions list: removed 'poi'
    acquisitions = ('ei', 'logei', 'ucb', 'mes', 'nei', 'kg')
    batch_methods = ('local_penalization','constant_liar','kriging_believer','pessimistic_believer')#,'thompson_sampling','fantasized_qei','fantasized_qnei')
    batch_sizes  = (2,3,4,5)
    iterations   = 15
    n_repeats    = 1

    results = run_sensitivity_acq_vs_method_vs_batch(
    objective_func,
    acquisitions=acquisitions,
    batch_methods=batch_methods,
    batch_sizes=batch_sizes,
    iterations=iterations,
    n_repeats=n_repeats,
    random_ratio=0.3,
    restart_threshold=4,
    length_scale=0.5
)

# Plot throughput heatmaps (per acquisition)
plot_acq_method_batch_throughput(results)

# Print top-5 combinations by throughput
print_top5_combos_by_throughput(results, top_k=5)

# Plot the cumulative titre vs time for the best throughput combo
plot_best_combo_cumulative_curve(
    results,
    objective_func,
    iterations=15,
    random_ratio=0.3,
    restart_threshold=4,
    length_scale=0.5
)