# ====================== Imports & Group information ======================
import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
import random
import sobol_seq
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve

group_names     = ['Name 1','Name 2']
cid_numbers     = ['000000','111111']
oral_assessment = [0, 1]

# ====================== Objective (black-box) ============================
def objective_func(X):
    return np.array(virtual_lab.conduct_experiment(X))

# ====================== Sampling (raw) ==================================
def sobol_initial_samples(n_samples):
    # 5 continuous variables → dimensionality = 5
    sobol_points = sobol_seq.i4_sobol_generate(5, n_samples)

    temp_range = [30, 40]
    pH_range   = [6, 8]
    f1_range   = [0, 50]
    f2_range   = [0, 50]
    f3_range   = [0, 50]
    celltype = ['celltype_1','celltype_2','celltype_3']

    # Scale each dimension to its physical range
    temp = temp_range[0] + sobol_points[:,0] * (temp_range[1] - temp_range[0])
    pH   = pH_range[0]   + sobol_points[:,1] * (pH_range[1]   - pH_range[0])
    f1   = f1_range[0]   + sobol_points[:,2] * (f1_range[1]   - f1_range[0])
    f2   = f2_range[0]   + sobol_points[:,3] * (f2_range[1]   - f2_range[0])
    f3   = f3_range[0]   + sobol_points[:,4] * (f3_range[1]   - f3_range[0])

    # Randomly assign a categorical cell type
    celltype_list = [random.choice(celltype) for _ in range(n_samples)]

    # Combine into list of lists (raw)
    X_init = [[float(temp[i]), float(pH[i]), float(f1[i]), float(f2[i]), float(f3[i]), celltype_list[i]] for i in range(n_samples)]
    return X_init

# ====================== Normalized encoding ==============================
RANGES = {
    'temp': (30.0, 40.0),
    'pH':   (6.0,  8.0),
    'f1':   (0.0, 50.0),
    'f2':   (0.0, 50.0),
    'f3':   (0.0, 50.0)
}

def encode_celltype_normalized_onehot(X):
    """
    Returns 8-D normalized features:
      [temp_norm, pH_norm, f1_norm, f2_norm, f3_norm, ct1, ct2, ct3]
    where continuous features are scaled to [0,1] and category is one-hot.
    """
    cats = {'celltype_1':[1,0,0], 'celltype_2':[0,1,0], 'celltype_3':[0,0,1]}
    def norm(v, lo, hi): return (v - lo) / (hi - lo + 1e-12)
    X_encoded = []
    for row in X:
        t,p,f1,f2,f3,ct = row
        x_cont = [
            norm(t, *RANGES['temp']),
            norm(p, *RANGES['pH']),
            norm(f1, *RANGES['f1']),
            norm(f2, *RANGES['f2']),
            norm(f3, *RANGES['f3'])
        ]
        x_cat = cats[ct]
        X_encoded.append(x_cont + x_cat)
    return np.array(X_encoded, dtype=float)

# ====================== Gaussian Process (stable Cholesky) ==============
def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    # RBF kernel with normalized inputs
    X1 = np.array(X1); X2 = np.array(X2)
    sqdist = (
        np.sum(X1**2, axis=1).reshape(-1, 1)
        + np.sum(X2**2, axis=1)
        - 2 * np.dot(X1, X2.T)
    )
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

# ====================== Mean functions ==================================
def mean_zero(X):
    X = np.asarray(X)
    return np.zeros(X.shape[0], dtype=float)

def make_constant_mean(c=0.0):
    def m(X):
        X = np.asarray(X)
        return np.full(X.shape[0], float(c))
    return m

def make_linear_mean(w, b=0.0):
    """
    m(x) = b + w^T x
    w length must match encoded feature dimension (8 with normalized one-hot).
    """
    w = np.asarray(w, dtype=float).reshape(-1)
    b = float(b)
    def m(X):
        X = np.asarray(X, dtype=float)
        assert X.shape[1] == w.shape[0], f"w length {w.shape[0]} != X dim {X.shape[1]}"
        return X.dot(w) + b
    return m

def make_quadratic_mean(q_diag, b=0.0):
    """
    Diagonal quadratic: m(x) = b + sum_i q_i x_i^2
    q_diag length must match encoded feature dimension (8 with normalized one-hot).
    """
    q = np.asarray(q_diag, dtype=float).reshape(-1)
    b = float(b)
    def m(X):
        X = np.asarray(X, dtype=float)
        assert X.shape[1] == q.shape[0], f"q length {q.shape[0]} != X dim {X.shape[1]}"
        return (X**2).dot(q) + b
    return m

class GaussianProcess:
    def __init__(self, length_scale=0.3, sigma_f=1.0, noise=1e-5, mean_func=None):
        self.l = length_scale
        self.sigma_f = sigma_f
        self.noise = noise
        self.mean_func = mean_func if mean_func is not None else mean_zero
        self._fitted = False

    def fit(self, X_train, Y_train):
        self.X_train = np.array(X_train, dtype=float)
        Y = np.array(Y_train, dtype=float).reshape(-1)

        # Residuals r = y - m(X)
        m_train = self.mean_func(self.X_train)
        R = Y - m_train

        # Standardize residuals
        self.r_mean = float(np.mean(R))
        self.r_std  = float(np.std(R) + 1e-12)
        R_norm = (R - self.r_mean) / self.r_std

        # Kernel + jitter
        K = rbf_kernel(self.X_train, self.X_train, self.l, self.sigma_f)
        K[np.diag_indices_from(K)] += self.noise

        # Cholesky solves
        self.cF, self.lower = cho_factor(K, lower=True, check_finite=False)
        self.alpha = cho_solve((self.cF, self.lower), R_norm, check_finite=False)
        self._fitted = True

    def predict(self, X_s):
        assert self._fitted, "Call fit() first."
        X_s = np.array(X_s, dtype=float)
        K_s = rbf_kernel(X_s, self.X_train, self.l, self.sigma_f)

        # Residual predictive mean (normalized)
        mu_resid_norm = K_s @ self.alpha
        mu_resid = mu_resid_norm * self.r_std + self.r_mean

        # Add prior mean
        m_s = self.mean_func(X_s)
        mu = m_s + mu_resid

        # Diagonal variance: k_ss - v^T v
        v = cho_solve((self.cF, self.lower), K_s.T, check_finite=False)
        K_ss_diag = np.full(X_s.shape[0], self.sigma_f**2)
        var = K_ss_diag - np.sum(K_s * v.T, axis=1)
        var = np.maximum(var, 1e-12)
        return mu, var
    def posterior_cov(self, X_s):
        """
        Full posterior covariance over X_s:
        Cov[f(X_s)] = K_ss - K_s K^{-1} K_s^T
        Note: builds an (n x n) dense matrix; use a subset for large candidate pools.
        """
        X_s = np.array(X_s, dtype=float)
        K_s = rbf_kernel(X_s, self.X_train, self.l, self.sigma_f)           # (n, n_train)
        K_ss = rbf_kernel(X_s, X_s, self.l, self.sigma_f)                    # (n, n)
        # Solve K^{-1} K_s^T via Cholesky
        V = cho_solve((self.cF, self.lower), K_s.T, check_finite=False)      # (n_train, n)
        C = K_ss - K_s @ V                                                   # (n, n)
        # Numerical stabilizer
        n = C.shape[0]
        C[np.diag_indices_from(C)] = np.maximum(np.diag(C), 1e-12)
        return C

# ====================== Acquisitions: EI / PI / UCB / LCB ===============
def expected_improvement(X, gp, Y_best, xi=0.0, maximize=True):
    mu, var = gp.predict(X)
    std = np.sqrt(np.maximum(var, 1e-12))
    if maximize:
        delta = mu - (Y_best + xi)
    else:
        delta = (Y_best - xi) - mu
    Z = delta / std
    return delta * norm.cdf(Z) + std * norm.pdf(Z)

def probability_of_improvement(X, gp, Y_best, xi=0.01, maximize=True):
    mu, var = gp.predict(X)
    std = np.sqrt(np.maximum(var, 1e-12))
    if maximize:
        Z = (mu - (Y_best + xi)) / std
    else:
        Z = ((Y_best - xi) - mu) / std
    return norm.cdf(Z)

def upper_confidence_bound(X, gp, beta=2.0):
    mu, var = gp.predict(X)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mu + beta * std

def lower_confidence_bound(X, gp, beta=2.0):
    mu, var = gp.predict(X)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mu - beta * std

def compute_acquisition(X, gp, Y_best=None, kind='ei', maximize=True, beta=2.0, xi=0.01):
    if kind == 'ei':
        return expected_improvement(X, gp, Y_best, xi=xi, maximize=maximize), 'max'
    elif kind == 'pi':
        return probability_of_improvement(X, gp, Y_best, xi=xi, maximize=maximize), 'max'
    elif kind == 'ucb':
        return upper_confidence_bound(X, gp, beta=beta), 'max'
    elif kind == 'lcb':
        return lower_confidence_bound(X, gp, beta=beta), 'min'
    else:
        raise ValueError(f"Unknown acquisition kind: {kind}")

# ====================== Batch BO (GP + CL/KB) ===========================
class BO:
    def __init__(
        self,
        X_initial,
        X_searchspace,
        iterations,
        batch,
        objective_func,
        strategy='constant_liar',   # 'constant_liar' | 'kriging_believer' | 'pessimistic_believer' | 'local_penalization' | 'thompson_sampling'
        liar='worst',               # for constant liar: 'mean' | 'best' | 'worst'
        gp_params=None,
        acquisition='ei',           # 'ei' | 'pi' | 'ucb' | 'lcb'
        maximize=True,
        beta=2.0,
        xi=0.01,
        # NEW knobs
        pess_kappa=1.0,             # PB: strength of pessimism
        lp_lambda=1.0,              # LP: penalty height
        lp_radius_scale=0.5,        # LP: radius = lp_radius_scale * (ell + sigma_sel)
        ts_subset=500               # TS: compute posterior sample only on a subset of pool
    ):
        # Config
        self.X_initial_raw = X_initial
        self.X_search_raw  = X_searchspace
        self.iterations    = iterations
        self.batch         = batch
        self.objective_func= objective_func
        self.strategy      = strategy
        self.liar          = liar

        # Acquisition config
        self.acquisition = acquisition
        self.maximize    = maximize
        self.beta        = beta
        self.xi          = xi

        # Strategy knobs
        self.pess_kappa  = float(pess_kappa)
        self.lp_lambda   = float(lp_lambda)
        self.lp_radius_scale = float(lp_radius_scale)
        self.ts_subset   = int(ts_subset)

        # Encode with normalized one-hot
        self.X_initial_enc = encode_celltype_normalized_onehot(self.X_initial_raw)
        self.X_search_enc  = encode_celltype_normalized_onehot(self.X_search_raw)

        # Availability mask
        self.available = np.ones(len(self.X_search_raw), dtype=bool)

        # Evaluate initial
        start_time = datetime.timestamp(datetime.now())
        self.Y = objective_func(self.X_initial_raw)
        self.X_enc = np.array(self.X_initial_enc, dtype=float)
        self.time  = [datetime.timestamp(datetime.now()) - start_time] + [0]*(len(self.X_initial_raw)-1)

        # Initialize GP
        if gp_params is None:
            gp_params = {'length_scale':0.3,'sigma_f':1.0,'noise':1e-5}
        self.gp = GaussianProcess(
            length_scale=gp_params.get('length_scale', 0.3),
            sigma_f=gp_params.get('sigma_f', 1.0),
            noise=gp_params.get('noise', 1e-5),
            mean_func=gp_params.get('mean_func', mean_zero)
        )
        self.gp.fit(self.X_enc, self.Y)

        # Iterations
        for _ in range(self.iterations):
            selected_raw, selected_enc, selected_indices = self._select_batch(self.batch)
            if len(selected_raw) == 0:
                break

            start_time = datetime.timestamp(datetime.now())
            Y_batch = objective_func(selected_raw)
            elapsed = datetime.timestamp(datetime.now()) - start_time

            self.X_enc = np.vstack([self.X_enc, selected_enc])
            self.Y     = np.concatenate([self.Y, Y_batch])
            self.gp.fit(self.X_enc, self.Y)

            for idx in selected_indices:
                self.available[idx] = False
            self.time += [elapsed] + [0]*(len(Y_batch)-1)

    def _select_batch(self, batch_size):
        pool_indices = [i for i, avail in enumerate(self.available) if avail]
        if len(pool_indices) == 0:
            return [], np.empty((0, self.X_enc.shape[1])), []

        X_pool_enc = self.X_search_enc[pool_indices]
        X_tmp = np.array(self.X_enc)
        Y_tmp = np.array(self.Y)

        # Dispatch by strategy
        if self.strategy == 'kriging_believer':
            return self._select_batch_believer(
                X_pool_enc, pool_indices, X_tmp, Y_tmp, mode='kb'
            )
        elif self.strategy == 'pessimistic_believer':
            return self._select_batch_believer(
                X_pool_enc, pool_indices, X_tmp, Y_tmp, mode='pb'
            )
        elif self.strategy == 'constant_liar':
            return self._select_batch_constant_liar(
                X_pool_enc, pool_indices, X_tmp, Y_tmp
            )
        elif self.strategy == 'local_penalization':
            return self._select_batch_local_penalization(
                X_pool_enc, pool_indices, X_tmp, Y_tmp
            )
        elif self.strategy == 'thompson_sampling':
            return self._select_batch_thompson_sampling(
                X_pool_enc, pool_indices, X_tmp, Y_tmp
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    # ---------- Helper strategies ----------
    def _select_batch_believer(self, X_pool_enc, pool_idx, X_tmp, Y_tmp, mode='kb'):
        # mode: 'kb' (mean) or 'pb' (pessimistic)
        selected_raw = []; selected_enc = []; selected_indices_global = []
        for _ in range(min(self.batch, len(pool_idx))):
            gp_tmp = GaussianProcess(length_scale=self.gp.l, sigma_f=self.gp.sigma_f, noise=self.gp.noise, mean_func=self.gp.mean_func)
            gp_tmp.fit(X_tmp, Y_tmp)

            Y_best_tmp = np.max(Y_tmp) if self.maximize else np.min(Y_tmp)
            acq_vals, goal = compute_acquisition(
                np.array([X_pool_enc[i] for i in pool_idx]), gp_tmp,
                Y_best=Y_best_tmp, kind=self.acquisition,
                maximize=self.maximize, beta=self.beta, xi=self.xi
            )
            loc = int(np.argmax(acq_vals)) if goal == 'max' else int(np.argmin(acq_vals))
            choice = pool_idx[loc]

            x_best_enc = self.X_search_enc[choice]
            x_best_raw = self.X_search_raw[choice]
            mu_sel, var_sel = gp_tmp.predict([x_best_enc])
            mu_sel = float(mu_sel[0]); std_sel = float(np.sqrt(max(var_sel[0], 1e-12)))

            # Fantasy label
            if mode == 'kb':
                y_fake = mu_sel
            else:  # pessimistic believer
                # For maximization, pessimistic = LCB(mu - kappa*std); for minimization, UCB
                y_fake = mu_sel - self.pess_kappa*std_sel if self.maximize else mu_sel + self.pess_kappa*std_sel

            # Augment fantasy data
            X_tmp = np.vstack([X_tmp, x_best_enc])
            Y_tmp = np.concatenate([Y_tmp, [y_fake]])

            selected_raw.append(x_best_raw)
            selected_enc.append(x_best_enc)
            selected_indices_global.append(choice)
            pool_idx.pop(loc)

        return selected_raw, np.array(selected_enc, dtype=float), selected_indices_global

    def _select_batch_constant_liar(self, X_pool_enc, pool_idx, X_tmp, Y_tmp):
        selected_raw = []; selected_enc = []; selected_indices_global = []
        for _ in range(min(self.batch, len(pool_idx))):
            gp_tmp = GaussianProcess(length_scale=self.gp.l, sigma_f=self.gp.sigma_f, noise=self.gp.noise, mean_func=self.gp.mean_func)
            gp_tmp.fit(X_tmp, Y_tmp)

            Y_best_tmp = np.max(Y_tmp) if self.maximize else np.min(Y_tmp)
            acq_vals, goal = compute_acquisition(
                np.array([X_pool_enc[i] for i in pool_idx]), gp_tmp,
                Y_best=Y_best_tmp, kind=self.acquisition,
                maximize=self.maximize, beta=self.beta, xi=self.xi
            )
            loc = int(np.argmax(acq_vals)) if goal == 'max' else int(np.argmin(acq_vals))
            choice = pool_idx[loc]

            x_best_enc = self.X_search_enc[choice]
            x_best_raw = self.X_search_raw[choice]

            # Fantasy label by liar policy
            if self.liar == 'mean':
                y_fake = float(np.mean(Y_tmp))
            elif self.liar == 'best':
                y_fake = float(np.max(Y_tmp)) if self.maximize else float(np.min(Y_tmp))
            elif self.liar == 'worst':
                y_fake = float(np.min(Y_tmp)) if self.maximize else float(np.max(Y_tmp))
            else:
                y_fake = float(np.mean(Y_tmp))

            X_tmp = np.vstack([X_tmp, x_best_enc])
            Y_tmp = np.concatenate([Y_tmp, [y_fake]])

            selected_raw.append(x_best_raw)
            selected_enc.append(x_best_enc)
            selected_indices_global.append(choice)
            pool_idx.pop(loc)

        return selected_raw, np.array(selected_enc, dtype=float), selected_indices_global

    def _select_batch_local_penalization(self, X_pool_enc, pool_idx, X_tmp, Y_tmp):
        selected_raw = []; selected_enc = []; selected_indices_global = []
        centers = []; radii = []

        for _ in range(min(self.batch, len(pool_idx))):
            gp_tmp = GaussianProcess(length_scale=self.gp.l, sigma_f=self.gp.sigma_f, noise=self.gp.noise, mean_func=self.gp.mean_func)
            gp_tmp.fit(X_tmp, Y_tmp)

            Y_best_tmp = np.max(Y_tmp) if self.maximize else np.min(Y_tmp)
            X_eval = np.array([X_pool_enc[i] for i in pool_idx])
            acq_vals, goal = compute_acquisition(
                X_eval, gp_tmp, Y_best=Y_best_tmp, kind=self.acquisition,
                maximize=self.maximize, beta=self.beta, xi=self.xi
            )

            # Apply penalties
            if len(centers) > 0:
                # distance matrix between X_eval and centers
                D2 = np.array([np.sum((X_eval - c)**2, axis=1) for c in centers])  # (n_sel, n_eval)
                # Gaussian penalty per center
                pen = np.zeros_like(acq_vals)
                for j, r in enumerate(radii):
                    pen += self.lp_lambda * np.exp(-D2[j] / (2.0 * r*r))
                acq_vals = acq_vals - pen

            loc = int(np.argmax(acq_vals)) if goal == 'max' else int(np.argmin(acq_vals))
            choice = pool_idx[loc]

            x_best_enc = self.X_search_enc[choice]
            x_best_raw = self.X_search_raw[choice]
            mu_sel, var_sel = gp_tmp.predict([x_best_enc])
            std_sel = float(np.sqrt(max(var_sel[0], 1e-12)))

            # Heuristic radius: combine length-scale and local std
            r = self.lp_radius_scale * (float(self.gp.l) + std_sel)
            centers.append(x_best_enc)
            radii.append(r)

            # Fantasy label: use KB (mean) by default
            y_fake = float(mu_sel[0])
            X_tmp = np.vstack([X_tmp, x_best_enc])
            Y_tmp = np.concatenate([Y_tmp, [y_fake]])

            selected_raw.append(x_best_raw)
            selected_enc.append(x_best_enc)
            selected_indices_global.append(choice)
            pool_idx.pop(loc)

        return selected_raw, np.array(selected_enc, dtype=float), selected_indices_global

    def _select_batch_thompson_sampling(self, X_pool_enc, pool_idx, X_tmp, Y_tmp):
        selected_raw = []; selected_enc = []; selected_indices_global = []

        for _ in range(min(self.batch, len(pool_idx))):
            # Fit temp model
            gp_tmp = GaussianProcess(length_scale=self.gp.l, sigma_f=self.gp.sigma_f, noise=self.gp.noise, mean_func=self.gp.mean_func)
            gp_tmp.fit(X_tmp, Y_tmp)

            # Subset of pool to keep covariance manageable
            if len(pool_idx) > self.ts_subset:
                sub_idx_local = np.random.choice(len(pool_idx), size=self.ts_subset, replace=False)
            else:
                sub_idx_local = np.arange(len(pool_idx))
            sub_global = [pool_idx[i] for i in sub_idx_local]
            X_eval = np.array([self.X_search_enc[i] for i in sub_global])

            # Posterior mean and covariance over subset
            mu_eval, _ = gp_tmp.predict(X_eval)
            C_eval = gp_tmp.posterior_cov(X_eval)  # (m, m)

            # Sample from multivariate normal: f ~ N(mu, C)
            # Use Cholesky of C_eval (add tiny jitter for safety)
            jitter = 1e-10
            try:
                Lc = np.linalg.cholesky(C_eval + jitter*np.eye(C_eval.shape[0]))
            except np.linalg.LinAlgError:
                # fallback: diagonal only sampling
                diag = np.maximum(np.diag(C_eval), 1e-12)
                f_samp = mu_eval + np.random.randn(len(diag)) * np.sqrt(diag)
            else:
                z = np.random.randn(C_eval.shape[0])
                f_samp = mu_eval + Lc @ z

            # Choose best by sampled function
            loc_sub = int(np.argmax(f_samp)) if self.maximize else int(np.argmin(f_samp))
            choice = sub_global[loc_sub]

            x_best_enc = self.X_search_enc[choice]
            x_best_raw = self.X_search_raw[choice]

            # Fantasy label: use sampled value (TS consistent)
            y_fake = float(f_samp[loc_sub])

            X_tmp = np.vstack([X_tmp, x_best_enc])
            Y_tmp = np.concatenate([Y_tmp, [y_fake]])

            selected_raw.append(x_best_raw)
            selected_enc.append(x_best_enc)
            selected_indices_global.append(choice)
            # Remove from pool
            pool_idx.remove(choice)

        return selected_raw, np.array(selected_enc, dtype=float), selected_indices_global

# ====================== Execution & Plotting =============================
# Initial designs (raw)
X_initial      = sobol_initial_samples(6)
X_searchspace  = sobol_initial_samples(1000)

# Example mean function on normalized one-hot (8D)
# Quadratic on continuous only (first 5 dims); zero on one-hot dims
q = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 0.0, 0.0, 0.0], dtype=float)
quad_mean = make_quadratic_mean(q_diag=q, b=0.0)

# bo_run = BO(
#     X_initial=X_initial,
#     X_searchspace=X_searchspace,
#     iterations=15, batch=5,
#     objective_func=objective_func,
#     strategy='kriging_believer',
#     #liar = 'worst',
#     acquisition='ucb', maximize=True, beta=2.0,
#     #acquisition='ei', maximize=True, xi=0.01,
#     gp_params={'length_scale':0.3,'sigma_f':0.5,'noise':1e-5, 'mean_func': quad_mean}
# )

# bo_run = BO(
#     X_initial=X_initial,
#     X_searchspace=X_searchspace,
#     iterations=15, batch=4,
#     objective_func=objective_func,
#     strategy='local_penalization',
#     acquisition='ucb', maximize=True, beta=2.0,
#     gp_params={'length_scale':0.3,'sigma_f':1.0,'noise':1e-5},
#     lp_lambda=1.0,
#     lp_radius_scale=0.5
# )

# bo_run = BO(
#     X_initial=X_initial,
#     X_searchspace=X_searchspace,
#     iterations=15, batch=4,
#     objective_func=objective_func,
#     strategy='pessimistic_believer',
#     acquisition='ei', maximize=True, xi=0.01,
#     gp_params={'length_scale':0.3,'sigma_f':1.0,'noise':1e-5},
#     pess_kappa=1.0
# )

bo_run = BO(
    X_initial=X_initial,
    X_searchspace=X_searchspace,
    iterations=15, batch=5,
    objective_func=objective_func,
    strategy='thompson_sampling',
    acquisition='ei', maximize=True, xi=0.01,  # acquisition not used directly by TS
    gp_params={'length_scale':0.3,'sigma_f':1.0,'noise':1e-5},
    ts_subset=400
)

# Plot: cumulative time vs cumulative titre concentration
t = np.cumsum(bo_run.time)
cumulative_titre = np.cumsum(bo_run.Y)

plt.figure(figsize=(8, 6))
plt.plot(t, cumulative_titre, color='red', label='Cumulative Titre Conc.')
plt.xlabel('Cumulative Time [ms]')
plt.ylabel('Cumulative Titre Conc. [g/L]')
plt.title('Cumulative Titre Concentration vs. Cumulative Time (GP BO, normalized inputs)')
plt.legend()
plt.grid(True)
plt.show()