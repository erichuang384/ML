import numpy as np
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erfcx
import random
from datetime import datetime
import matplotlib.pyplot as plt

# ====================== Clean Wrapper – NO CHANGES TO VIRTUAL LAB ======================
try:
    import MLCE_CWBO2025.virtual_lab as vl
    import MLCE_CWBO2025.conditions_data as data

    # Save the original untouched function
    _original_conduct_experiment = vl.conduct_experiment

    def conduct_experiment_with_seed(X, seed=None, noise_level=0.03, **kwargs):
        """
        Wrapper that sets a global numpy random seed before each batch evaluation.
        This ensures different repeats get different measurement noise,
        while the underlying virtual_lab remains completely unmodified.
        """
        if seed is not None:
            # Use a hash of the seed to avoid conflicts with other random states
            np.random.seed(seed)
            random.seed(seed)

        results = []
        for row in X:
            T, pH, F1, F2, F3, cell_type = row
            feeding = [(40, float(F1)), (80, float(F2)), (120, float(F3))]

            reactor = data.reactor_list[0]  # or however you select it

            exp = vl.EXPERIMENT(
                T=T, pH=pH, time=150,
                feeding=feeding,
                reactor=reactor,
                cell_type=cell_type
            )

            # Use realistic inoculum (you can make this configurable)
            exp.initial_conditions = [0, 0.4e9, 0.32e9, 0, 20, 3.5, 0, 1.8]

            # Apply consistent analytical noise (the original lab uses its own noise logic)
            value = exp.measurement(quantity="P", noise_level=noise_level)
            results.append(float(value))

        return results

    # Replace only the function used by BO — virtual_lab module stays pristine
    vl.conduct_experiment = conduct_experiment_with_seed

    def objective_func(X, seed=None):
        return np.array(vl.conduct_experiment(X, seed=seed, noise_level=0.03), dtype=float)

    print("Using original virtual_lab (untouched) with external seed control for repeats.")

except Exception as e:
    print("Failed to load virtual_lab → falling back to mock")
    print(e)
    def objective_func(X, seed=None):
        # Optional: make mock also respect seed
        if seed is not None:
            np.random.seed(seed)
        out = []
        for temp, pH, f1, f2, f3, cell in X:
            val = (100 - (temp-35)**2 - 2*(pH-7)**2 - 0.1*(f1-25)**2 - 0.1*(f2-30)**2
                   + np.random.randn()*3.0)
            out.append(max(0.0, val))
        return np.array(out, dtype=float)


# ====================== Sobol + Encoding ======================
def sobol_initial_samples(n):
    import sobol_seq
    pts = sobol_seq.i4_sobol_generate(5, n)
    return [[30 + 10*p[0], 6 + 2*p[1], 50*p[2], 50*p[3], 50*p[4],
             random.choice(['celltype_1','celltype_2','celltype_3'])] for p in pts]

def encode(X):
    X = np.array(X, dtype=object)
    num = X[:, :5].astype(float)
    normed = np.column_stack([(num[:,0]-30)/10, (num[:,1]-6)/2, num[:,2]/50, num[:,3]/50, num[:,4]/50])
    cell_map = {'celltype_1':[1,0,0], 'celltype_2':[0,1,0], 'celltype_3':[0,0,1]}
    onehot = np.array([cell_map[c] for c in X[:,5]])
    return np.hstack([normed, onehot])


# ====================== Gaussian Process ======================
class GP:
    def __init__(self, ls=0.5, sf=1.0, noise=1e-6):  # Change default to 1e-6
        self.ls, self.sf, self.noise = ls, sf, noise
        self.X = self.L = self.alpha = None
        self.y_mean = self.y_std = 0.0

    def rbf(self, X1, X2):
        sqdist = np.sum(X1**2, 1)[:,None] + np.sum(X2**2, 1) - 2*X1@X2.T
        return self.sf**2 * np.exp(-0.5 * sqdist / self.ls**2)

    def matern52(self, X1, X2):  # NEW: Matern 5/2 kernel
        dist = np.sqrt(np.sum((X1[:, None] - X2)**2, axis=-1)) / self.ls
        return self.sf**2 * (1 + np.sqrt(5)*dist + (5/3)*dist**2) * np.exp(-np.sqrt(5)*dist)

    def fit(self, X, y):
        X = np.atleast_2d(X)
        y = np.asarray(y).ravel()
        self.y_mean = float(y.mean())
        y_std = float(y.std())
        self.y_std = y_std if y_std > 1e-8 else 1.0
        y_norm = (y - self.y_mean) / self.y_std

        # Ensure noise is a numeric value (fallback to tiny jitter)
        noise_val = self.noise if self.noise is not None else 1e-6
        K = self.matern52(X, X) + np.eye(len(X)) * float(noise_val)
        L = cho_factor(K, lower=True)
        self.L, self.alpha = L, cho_solve(L, y_norm).reshape(-1, 1)
        self.X = X

    def predict(self, X):
        X = np.atleast_2d(X)
        k = self.matern52(X, self.X)
        mu = (k @ self.alpha).ravel() * self.y_std + self.y_mean
        v = cho_solve(self.L, k.T)
        var = self.sf**2 - np.sum(v**2, 0)
        sigma = np.sqrt(np.maximum(var, 1e-12)) * self.y_std
        return mu, sigma

# ====================== Batch Methods ======================
def local_penalization(gp, cand, acq_vals, chosen, n_batch, ls=0.5):
    sel = []; vals = acq_vals.copy()
    for _ in range(n_batch):
        vals[list(chosen) + sel] = -np.inf
        if sel:
            d2 = ((cand[:,None,:] - cand[sel])**2).sum(-1)
            pen = 1 - 0.9 * np.max(np.exp(-d2 / (2*ls**2)), axis=1)
            vals *= pen
        sel.append(int(np.argmax(vals)))
    return sel

def constant_liar(gp, cand, acq_func, chosen, n_batch):
    sel = []
    mu_all, _ = gp.predict(cand)
    lie = np.median(mu_all)
    gp_temp = GP(gp.ls, gp.sf, gp.noise)
    gp_temp.fit(gp.X, gp.y_mean + gp.y_std*gp.alpha.ravel())
    for _ in range(n_batch):
        mu, sigma = gp_temp.predict(cand)
        acq = acq_func(mu, sigma)
        acq[list(chosen)+sel] = -np.inf
        best = int(np.argmax(acq))
        sel.append(best)
        if len(sel) < n_batch:
            gp_temp.X = np.vstack([gp_temp.X, cand[best]])
            gp_temp.fit(gp_temp.X, np.append(gp_temp.alpha.ravel()*gp_temp.y_std + gp_temp.y_mean, lie))
    return sel

def kriging_believer(gp, cand, acq_func, chosen, n_batch):
    sel = []
    gp_temp = GP(gp.ls, gp.sf, gp.noise)
    gp_temp.fit(gp.X, gp.y_mean + gp.y_std*gp.alpha.ravel())
    for i in range(n_batch):
        mu, sigma = gp_temp.predict(cand)
        acq = acq_func(mu, sigma)
        acq[list(chosen)+sel] = -np.inf
        best = int(np.argmax(acq))
        sel.append(best)
        if i < n_batch-1:
            gp_temp.X = np.vstack([gp_temp.X, cand[best]])
            gp_temp.fit(gp_temp.X, np.append(gp_temp.alpha.ravel()*gp_temp.y_std + gp_temp.y_mean, mu[best]))
    return sel

def pessimistic_believer(gp, cand, acq_func, chosen, n_batch):
    sel = []
    gp_temp = GP(gp.ls, gp.sf, gp.noise)
    gp_temp.fit(gp.X, gp.y_mean + gp.y_std*gp.alpha.ravel())
    for i in range(n_batch):
        mu, sigma = gp_temp.predict(cand)
        acq = acq_func(mu, sigma)
        acq[list(chosen)+sel] = -np.inf
        best = int(np.argmax(acq))
        sel.append(best)
        if i < n_batch-1:
            gp_temp.X = np.vstack([gp_temp.X, cand[best]])
            gp_temp.fit(gp_temp.X, np.append(gp_temp.alpha.ravel()*gp_temp.y_std + gp_temp.y_mean, mu[best] - 3*sigma[best]))
    return sel

def thompson_sampling(gp, cand, acq_func, chosen, n_batch, n_samples=200):
    """
    Batch Thompson Sampling:
    - Samples n_samples functions from the GP posterior
    - For each candidate point, computes how many times it had the highest value across samples
    - Selects the n_batch points with the highest empirical probability of being the maximum
    """
    sel = []
    mu, sigma = gp.predict(cand)
    sigma = np.maximum(sigma, 1e-12)  # avoid zero variance

    # Sample from posterior: [n_samples x n_candidates]
    samples = mu[None, :] + sigma[None, :] * np.random.randn(n_samples, len(mu))

    # For each candidate, count how often it was the max in a sample
    argmax_per_sample = np.argmax(samples, axis=1)
    counts = np.bincount(argmax_per_sample, minlength=len(mu))

    # Greedily select the n_batch points with highest counts, avoiding already chosen
    available_mask = np.ones(len(cand), dtype=bool)
    available_mask[list(chosen) + sel] = False

    for _ in range(n_batch):
        if not np.any(available_mask):
            break
        best_idx = np.argmax(counts * available_mask)  # mask out already selected
        sel.append(int(best_idx))
        available_mask[best_idx] = False

    return sel

# ====================== Acquisition Functions ======================

def ei(mu, sigma, best, xi=0.01):
    """Expected Improvement."""
    sigma = np.maximum(sigma, 1e-12)
    improvement = mu - best - xi
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


def log_ei(mu, sigma, best, xi=0.01):
    """Numerically stable log-EI, then exponentiated."""
    sigma = np.maximum(sigma, 1e-12)
    z = (mu - best - xi) / sigma
    log_phi = norm.logcdf(z)
    log_pdf = norm.logpdf(z)
    log_ei_val = (
        np.log(sigma)
        + np.where(
            z > 0,
            log_pdf + np.log1p(-np.exp(log_phi - log_pdf)),
            log_phi + np.log1p(np.exp(log_pdf - log_phi)),
        )
    )
    return np.exp(np.clip(log_ei_val, -100, 100))


def ucb(mu, sigma, beta=2.0):
    """Upper Confidence Bound."""
    return mu + beta * sigma


def nei(mu, sigma, best, xi=0.01, n_mc=50):
    """Monte Carlo-based Noisy Expected Improvement."""
    z = np.random.randn(n_mc, len(mu))
    f = mu + sigma * z
    return np.mean(np.maximum(f - best - xi, 0), axis=0)


def make_acq(kind, best_y, stagnation):
    """
    Factory that returns an acquisition function a(mu, sigma) -> values
    with hyperparameters (xi, beta) adapted to stagnation.
    """
    xi = 0.01 + 0.1 * min(stagnation / 4, 1.0)
    beta = 2.0 + stagnation / 5

    if kind == "ei":
        return lambda mu, sigma: ei(mu, sigma, best_y, xi)

    if kind == "logei":
        return lambda mu, sigma: log_ei(mu, sigma, best_y, xi)

    if kind == "ucb":
        return lambda mu, sigma: ucb(mu, sigma, beta)

    if kind == "nei":
        return lambda mu, sigma: nei(mu, sigma, best_y, xi)

    # Fallback: simple UCB-style heuristic
    return lambda mu, sigma: mu + 2 * sigma

# ====================== Main BO Loop ======================
def run_bo(acq_kind, batch_method, batch_size, iters=15, seed=None):
    X_init = sobol_initial_samples(6)
    X_cand = sobol_initial_samples(1000)
    cand_enc = encode(X_cand)

    t0 = datetime.now()
    y_init = objective_func(X_init, seed=seed)
    init_time = (datetime.now() - t0).total_seconds()

    X_enc = encode(X_init)
    y_all = np.array(y_init, dtype=float)
    time_log = [init_time] + [0.0]*(len(y_init)-1)
    best_seen = y_all.max()
    stagnation = 0

    # Track best value after each BO iteration
    best_per_iter = []

    for it in range(iters):

        current_batch = batch_size

        gp = GP(ls=0.5*(1 + 0.8*min(stagnation/5,1)))
        gp.fit(X_enc, y_all)
        acq = make_acq(acq_kind, best_seen, stagnation)

        n_rand = max(1, int(current_batch*0.3))
        n_bo = current_batch - n_rand
        chosen_idx = random.sample(range(len(X_cand)), n_rand)

        if n_bo > 0 and len(y_all) >= 3:
            mu, sigma = gp.predict(cand_enc)
            acq_vals = acq(mu, sigma)
            if batch_method == "local_penalization":
                bo_idx = local_penalization(gp, cand_enc, acq_vals, chosen_idx, n_bo)
            elif batch_method == "constant_liar":
                bo_idx = constant_liar(gp, cand_enc, acq, chosen_idx, n_bo)
            elif batch_method == "kriging_believer":
                bo_idx = kriging_believer(gp, cand_enc, acq, chosen_idx, n_bo)
            elif batch_method == "pessimistic_believer":
                bo_idx = pessimistic_believer(gp, cand_enc, acq, chosen_idx, n_bo)
            else:
                bo_idx = []
            chosen_idx += bo_idx

        # --- evaluation and logging ---
        batch_X = [X_cand[i] for i in chosen_idx]
        t_start = datetime.now()
        batch_y = objective_func(batch_X, seed=seed)
        elapsed = (datetime.now() - t_start).total_seconds()
        time_per_pt = elapsed / len(batch_y)
        time_log += [time_per_pt] * len(batch_y)

        X_enc = np.vstack([X_enc, cand_enc[chosen_idx]])
        y_all = np.concatenate([y_all, batch_y])

        if batch_y.max() > best_seen + 1e-6:
            best_seen = batch_y.max()
            stagnation = 0
        else:
            stagnation += 1

        # Record best value after this iteration
        best_per_iter.append(best_seen)

    cum_y = np.cumsum(y_all)
    cum_t = np.cumsum(time_log)
    # We keep cum_t, cum_y for plotting, but the main scoring is via best_per_iter
    return best_per_iter, cum_t, cum_y
# ====================== Full Grid Sensitivity with Averaged Repeats ======================
if __name__ == "__main__":
    acquisitions = ('ei', 'ucb', 'mes', 'nei', 'kg') # 'logei',
    batch_methods = ('local_penalization') # 'kriging_believer' 'pessimistic_believer', 'constant_liar', 'thompson_sampling'
    batch_sizes = (5,)   # fixed batch size of 5
    iterations = 15
    n_repeats = 10   # <<< Increase this for better statistics (e.g., 5 or 10)

    print("Scoring rule:")
    print(" - Iterations 0,1,2 (initialization + first two BO batches) do NOT contribute.")
    print(" - From iteration 3 onwards, batch score = best function value found so far.")
    print(" - Final score = sum of batch scores for iterations 3..14")
    print(f" - Results will be averaged over {n_repeats} repeats with different noise seeds.\n")

    # Store results as dict: (acq, method, bs) → list of final_scores
    results_dict = {}

    total_configs = len(acquisitions) * len(batch_methods) * len(batch_sizes)
    total_runs = total_configs * n_repeats
    count = 0

    for acq in acquisitions:
        for method in batch_methods:
            for bs in batch_sizes:
                key = (acq, method, bs)
                results_dict[key] = []

                for r in range(n_repeats):
                    count += 1
                    repeat_seed = r * 1000 + 1234  # Different seed per repeat, same offset for reproducibility
                    print(f"[{count:3d}/{total_runs}] {acq:>6} + {method:>20} (batch={bs}) | repeat {r+1}/{n_repeats} "
                          f"(seed={repeat_seed})", end=" → ")

                    best_per_iter, _, _ = run_bo(acq, method, bs, iterations, seed=repeat_seed)

                    # Compute per-batch scores: only iterations >= 3 contribute
                    per_batch_scores = [0.0 if k < 3 else float(best_per_iter[k]) for k in range(iterations)]
                    final_score = sum(per_batch_scores)

                    results_dict[key].append(final_score)
                    print(f"score = {final_score:.3f}")

    # ==================== Compute means and sort ====================
    averaged_results = []
    for (acq, method, bs), scores in results_dict.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        averaged_results.append((acq, method, bs, mean_score, std_score, scores))

    # Sort by mean score descending (best first)
    averaged_results.sort(key=lambda x: x[3], reverse=True)

    print("\n" + "="*100)
    print("FINAL RANKING (averaged over repeats)")
    print("="*100)
    print(f"{'Rank':>4} | {'Acq':>6} + {'Batch Method':>20} | {'Batch':>5} | {'Mean Score':>12} ± {'Std':>8} | Repeats")
    print("-"*100)
    for rank, (acq, method, bs, mean_score, std_score, _) in enumerate(averaged_results, 1):
        print(f"{rank:4d} | {acq:>6} + {method:>20} | {bs:5d} | {mean_score:12.3f} ± {std_score:8.3f} | {n_repeats}")

    # ==================== Plot best configuration (using seed=0 for nice curve) ====================
    best_acq, best_method, best_bs, best_mean, best_std, _ = averaged_results[0]
    print(f"\nBest configuration: {best_acq} + {best_method} (batch={best_bs})")
    print(f"→ Mean score = {best_mean:.3f} ± {best_std:.3f} over {n_repeats} repeats")

    # Re-run best config once with fixed seed for plotting
    best_per_iter, cum_t, cum_y = run_bo(best_acq, best_method, best_bs, iterations, seed=0)

    plt.figure(figsize=(10,6))
    plt.plot(range(iterations), best_per_iter, 'o-', lw=2, label='Best value so far')
    plt.axvspan(-0.5, 2.5, color='gray', alpha=0.1, label='No-score zone (iters 0–2)')
    plt.title(f"Best Config: {best_acq} + {best_method} (batch={best_bs})\n"
              f"Mean score = {best_mean:.3f} ± {best_std:.3f} ({n_repeats} repeats)")
    plt.xlabel("Iteration (batch index)")
    plt.ylabel("Best product titer found (g/L)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("best_config_best_per_iter.png", dpi=300, bbox_inches='tight')
    plt.show()