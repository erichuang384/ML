import numpy as np
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erfcx
import random
from datetime import datetime
import matplotlib.pyplot as plt

# ====================== Objective with Biological Jitter ======================
try:
    import MLCE_CWBO2025.virtual_lab as vl
    import MLCE_CWBO2025.conditions_data as data

    # Save original
    _original = vl.conduct_experiment

    def conduct_experiment_with_jitter(X, jitter_level=0.13, seed=None, **kwargs):
        if seed == 0:  # For first repeat, use original (no jitter)
            jitter_level = 0.0

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        results = []
        for row in X:
            T, pH, F1, F2, F3, cell_type = row
            feeding = [(40, float(F1)), (80, float(F2)), (120, float(F3))]

            # Use the correct reactor from data
            reactor = data.reactor_list[0]

            # Create real experiment
            exp = vl.EXPERIMENT(
                T=T, pH=pH, time=150,
                feeding=feeding,
                reactor=reactor,
                cell_type=cell_type
            )

            # === Apply biological jitter ===
            if jitter_level > 0.0:
                # Use legacy np.random (seeded globally above)
                exp.my_max     *= np.random.lognormal(0, jitter_level * 0.25)
                exp.K_G        *= np.random.lognormal(0, jitter_level * 0.20)
                exp.K_Q        *= np.random.lognormal(0, jitter_level * 0.20)

                # Convert tuples → lists for mutability
                Y = list(exp.Y)
                m = list(exp.m)

                Y[0] *= np.random.lognormal(0, jitter_level * 0.18)   # Y_X/G
                Y[4] *= np.random.lognormal(0, jitter_level * 0.35)   # Y_P/X — most important!
                m[0] *= np.random.lognormal(0, jitter_level * 0.18)

                exp.Y = tuple(Y)
                exp.m = tuple(m)

                exp.k_d_max *= np.random.lognormal(0, jitter_level * 0.40)
                exp.k_d_Q   *= np.random.lognormal(0, jitter_level * 0.30)
                exp.E_a     += np.random.normal(0, 1.2)
                exp.pH_opt  += np.random.normal(0, 0.12)

                # Hard physical bounds
                exp.my_max = np.clip(exp.my_max, 0.025, 0.11)
                exp.K_G = np.clip(exp.K_G, 0.2, 8.0)
                exp.K_Q = np.clip(exp.K_Q, 0.02, 4.0)
                exp.Y = tuple(np.clip(Y, [1e7, 1e7, 0.4, 0.05, 1e-11, 1e-12],
                                          [5e8, 5e8, 2.5, 1.2, 5e-10, 1e-10]))
                exp.k_d_max = np.clip(exp.k_d_max, 0.002, 0.04)
                exp.pH_opt = np.clip(exp.pH_opt, 6.6, 7.5)

            # Realistic inoculum + small analytical noise
            exp.initial_conditions = [0, 0.4e9, 0.32e9, 0, 20, 3.5, 0, 1.8]
            value = exp.measurement(quantity="P", noise_level=0.03)
            results.append(float(value))

        return results

    # === MONKEY PATCH ===
    vl.conduct_experiment = conduct_experiment_with_jitter

    # objective_func now accepts seed
    def objective_func(X, seed=None):
        return np.array(vl.conduct_experiment(X, jitter_level=0.13, seed=seed), dtype=float)

    print("Using real virtual_lab with biological jitter (jitter_level=0.13)")

except Exception as e:
    print("Failed to load virtual_lab → using mock")
    print(e)
    # Mock also accepts seed for consistency (but ignores it)
    def objective_func(X, seed=None):
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



# ====================== Stable LogEI ======================
def logei(mu, sigma, best, xi=0.01):
    sigma = np.maximum(sigma, 1e-12)
    z = (mu - best - xi) / sigma
    # stable log(EI) then exp
    log_phi = norm.logcdf(z)
    log_pdf = norm.logpdf(z)
    log_ei = np.log(sigma) + np.where(z > 0, log_pdf + np.log1p(-np.exp(log_phi - log_pdf)), 
                                       log_phi + np.log1p(np.exp(log_pdf - log_phi)))
    return np.exp(np.clip(log_ei, -100, 100))

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

# ====================== Acquisition Functions ======================
def make_acq(kind, best_y, stagnation):
    xi = 0.01 + 0.1 * min(stagnation/4, 1.0)
    beta = 2.0 + stagnation/5
    if kind == "ei":
        return lambda mu, sigma: (mu - best_y - xi)*norm.cdf((mu - best_y - xi)/sigma) + sigma*norm.pdf((mu - best_y - xi)/sigma)
    if kind == "logei":
        return lambda mu, sigma: logei(mu, sigma, best_y, xi)
    if kind == "ucb":
        return lambda mu, sigma: mu + beta*sigma
    if kind == "nei":
        return lambda mu, sigma: nei(mu, sigma, best_y, xi)
    return lambda mu, sigma: mu + 2*sigma  # fallback

def nei(mu, sigma, best, xi=0.01, n_mc=50):
    z = np.random.randn(n_mc, len(mu))
    f = mu + sigma*z
    return np.mean(np.maximum(f - best - xi, 0), axis=0)

# ====================== Main BO Loop ======================
def run_bo(acq_kind, batch_method, batch_size, iters=15, seed=None):
    X_init = sobol_initial_samples(6)
    X_cand = sobol_initial_samples(500)
    cand_enc = encode(X_cand)

    t0 = datetime.now()
    y_init = objective_func(X_init, seed=seed)
    init_time = (datetime.now() - t0).total_seconds()

    X_enc = encode(X_init)
    y_all = np.array(y_init, dtype=float)
    time_log = [init_time] + [0.0]*(len(y_init)-1)
    best_seen = y_all.max()
    stagnation = 0

    for it in range(iters):
        if batch_size == "adaptive":
            current_batch = max(2, 5 - (it // 3))  # Start at 5, decay to 2 every 3 iters
        else:
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

    cum_y = np.cumsum(y_all)
    cum_t = np.cumsum(time_log)
    return cum_y[-1] / cum_t[-1], cum_t, cum_y

# ====================== Full Grid Sensitivity ======================
if __name__ == "__main__":
    acquisitions = ('ei', 'logei', 'ucb', 'mes', 'nei', 'kg')
    batch_methods = ('local_penalization', 'kriging_believer', 'pessimistic_believer')#'constant_liar'
    # ran 40 repeats with lp and cl and cl clearly worse
    batch_sizes = (2, 3, 4, 5, 'adaptive')
    iterations = 15
    n_repeats = 40   # Suggest at least 3-5 for good stats

    print(f"Starting full sensitivity: {len(acquisitions)} acq × {len(batch_methods)} methods × {len(batch_sizes)} sizes × {n_repeats} repeats\n")

    # Store all results per repeat for normalization and per-repeat best plotting
    repeat_thrs = [{} for _ in range(n_repeats)]  # repeat -> (acq, method, bs) -> thr
    repeat_cum = [{} for _ in range(n_repeats)]   # repeat -> (acq, method, bs) -> (cum_t, cum_y)
    results = []  # For absolute averages

    total = len(acquisitions) * len(batch_methods) * len(batch_sizes) * n_repeats
    count = 0

    for acq in acquisitions:
        for method in batch_methods:
            for bs in batch_sizes:
                key = (acq, method, bs)
                thr_list = []
                for r in range(n_repeats):
                    count += 1
                    repeat_seed = r * 1000
                    print(f"[{count:3d}/{total}] {acq:>6} | {method:>20} | batch={bs} | repeat {r+1}/{n_repeats} (seed={repeat_seed})", end=" → ")
                    thr, cum_t, cum_y = run_bo(acq, method, bs, iterations, seed=repeat_seed)
                    thr_list.append(thr)
                    repeat_thrs[r][key] = thr
                    repeat_cum[r][key] = (cum_t, cum_y)  # Store for plotting
                    print(f"{thr:.3f} g/L/s")
                mean_thr = np.mean(thr_list)
                std_thr = np.std(thr_list)
                results.append((acq, method, bs, mean_thr, std_thr, thr_list))
                print(f"    → avg {mean_thr:.3f} ± {std_thr:.3f} g/L/s\n")

    # Compute normalized means
    normalized_results = []
    for acq in acquisitions:
        for method in batch_methods:
            for bs in batch_sizes:
                key = (acq, method, bs)
                rel_thrs = []
                for r in range(1, n_repeats):  # Skip r=0
                    if key in repeat_thrs[r]:
                        max_in_repeat = max(repeat_thrs[r].values())  # Best thr in this repeat's world
                        rel = repeat_thrs[r][key] / max_in_repeat if max_in_repeat > 0 else 0
                        rel_thrs.append(rel)
                mean_rel = np.mean(rel_thrs) if rel_thrs else 0
                std_rel = np.std(rel_thrs) if rel_thrs else 0
                normalized_results.append((acq, method, bs, mean_rel, std_rel))

    # Sort absolute results
    results.sort(key=lambda x: x[3], reverse=True)

    print("="*90)
    print("ABSOLUTE RANKING (best → worst) - Averages raw throughput")
    print("="*90)
    for i, (acq, method, bs, mean, std, _) in enumerate(results[:20], 1):
        print(f"{i:2d}. {acq:>6} + {method:>20} (batch={bs}) → {mean:.3f} ± {std:.3f} g/L/s")

    # Sort normalized results
    normalized_results.sort(key=lambda x: x[3], reverse=True)

    print("\n"+"="*90)
    print("NORMALIZED RANKING (best → worst) - Relative to best in each kinetic world (excluding no-jitter repeat)")
    print("="*90)
    for i, (acq, method, bs, mean_rel, std_rel) in enumerate(normalized_results[:20], 1):
        print(f"{i:2d}. {acq:>6} + {method:>20} (batch={bs}) → {mean_rel:.3f} ± {std_rel:.3f} (normalized)")

    # Plot the best config of each repeat (absolute, non-normalized)
    print("\n"+"="*90)
    print("Plotting the best config for each repeat (absolute throughput)")
    print("="*90)

    for r in range(1):
        if repeat_thrs[r]:
            best_key = max(repeat_thrs[r], key=repeat_thrs[r].get)
            best_thr = repeat_thrs[r][best_key]
            cum_t, cum_y = repeat_cum[r][best_key]
            acq, method, bs = best_key

            plt.figure(figsize=(10,6))
            plt.plot(cum_t, cum_y, 'b-', lw=2.5, label='Cumulative Titre')
            plt.title(f"Repeat {r+1} Best: {acq} + {method} (batch={bs})\nThroughput = {best_thr:.3f} g/L/s")
            plt.xlabel("Cumulative Time [s]")
            plt.ylabel("Cumulative Titre Conc. [g/L]")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"repeat_{r+1}_best_cumulative.png", dpi=300, bbox_inches='tight')
            plt.show()

    # Plot the overall absolute best (unchanged)
    # best = results[0]
    # acq, method, bs, _, _, _ = best
    # # Note: For overall best, we need to choose one cum_t, cum_y; here we take the last one from the list
    # plt.figure(figsize=(10,6))
    # plt.plot(cum_t, cum_y, 'r-', lw=2.5, label='Cumulative Titre')  # Using last cum_t, cum_y; could average if needed
    # plt.title(f"Overall Best combo: {acq} + {method} (batch={bs})\nThroughput = {best[3]:.3f} g/L/s")
    # plt.xlabel("Cumulative Time [s]")
    # plt.ylabel("Cumulative Titre Conc. [g/L]")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("overall_best_combo_cumulative.png", dpi=300, bbox_inches='tight')
    # plt.show()