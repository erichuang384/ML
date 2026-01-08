import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
from datetime import datetime
import random
import matplotlib.pyplot as plt

# =============== Objective ===============
def objective_func(X: list):
    # Each row: [T, pH, F1, F2, F3, cell_type]
    return np.array(virtual_lab.conduct_experiment(X), dtype=float)

# =============== Encoding ===============
def encode(X):
    X = np.array(X, dtype=object)
    num = X[:, :5].astype(float)
    normed = np.column_stack([
        (num[:,0]-30)/10, (num[:,1]-6)/2, num[:,2]/50, num[:,3]/50, num[:,4]/50
    ])
    cmap = {'celltype_1':[1,0,0], 'celltype_2':[0,1,0], 'celltype_3':[0,0,1]}
    onehot = np.array([cmap[c] for c in X[:,5]])
    return np.hstack([normed, onehot])

def sobol_initial_samples(n):
    import sobol_seq
    pts = sobol_seq.i4_sobol_generate(5, n)
    return [[30 + 10*p[0], 6 + 2*p[1], 50*p[2], 50*p[3], 50*p[4],
             random.choice(['celltype_1','celltype_2','celltype_3'])] for p in pts]

# =============== GP (Matérn 5/2, scalar length-scale) ===============
class GP:
    def __init__(self, ls=0.5, sf=1.0, noise=1e-6):
        self.ls = float(ls)
        self.sf = float(sf)
        self.noise = float(noise)
        self.X = None
        self.L = None
        self.alpha = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def matern52(self, X1, X2):
        X1 = np.atleast_2d(X1); X2 = np.atleast_2d(X2)
        dist = np.linalg.norm(X1[:, None, :] - X2[None, :, :], axis=2) / self.ls
        return (self.sf**2) * (1 + np.sqrt(5)*dist + (5/3)*(dist**2)) * np.exp(-np.sqrt(5)*dist)

    def fit(self, X, y):
        X = np.atleast_2d(X)
        y = np.asarray(y).ravel()
        self.X = X
        self.y_mean = float(np.mean(y))
        y_std = float(np.std(y))
        self.y_std = y_std if y_std > 1e-8 else 1.0
        y_norm = (y - self.y_mean) / self.y_std
        K = self.matern52(X, X) + np.eye(len(X)) * self.noise
        L = cho_factor(K, lower=True, check_finite=False)
        self.L = L
        self.alpha = cho_solve(L, y_norm).reshape(-1, 1)

    def predict(self, X):
        X = np.atleast_2d(X)
        k = self.matern52(X, self.X)
        mu_norm = (k @ self.alpha).ravel()
        mu = mu_norm * self.y_std + self.y_mean
        v = cho_solve(self.L, k.T)
        kxx = np.full(X.shape[0], self.sf**2)
        var = kxx - np.sum(k * v.T, axis=1)
        var = np.maximum(var, 1e-12)
        sigma = np.sqrt(var) * self.y_std
        return mu, sigma

# =============== Acquisition functions ===============
class Acquisition:
    @staticmethod
    def ei(mu, sigma, best, xi=0.01):
        sigma = np.maximum(sigma, 1e-12)
        imp = mu - best - xi
        z = imp / sigma
        return imp * norm.cdf(z) + sigma * norm.pdf(z)

    @staticmethod
    def ucb(mu, sigma, beta=2.0):
        return mu + beta * sigma

    @staticmethod
    def nei(mu, sigma, best, xi=0.01, n_mc=50):
        z = np.random.randn(n_mc, len(mu))
        f = mu + sigma * z
        return np.mean(np.maximum(f - best - xi, 0), axis=0)

    @staticmethod
    def make(kind, best_y, it):
        # Phase-aware acquisition
        if it < 3:          # exploration
            xi, beta = 0.1, 2.5
        elif it < 6:        # early scoring
            xi, beta = 0.02, 1.0
        else:               # exploitation
            xi, beta = 1e-4, 0.2

        if kind == "ei":
            return lambda mu, sig: Acquisition.ei(mu, sig, best_y, xi=xi)
        if kind == "ucb":
            return lambda mu, sig: Acquisition.ucb(mu, sig, beta=beta)
        if kind == "nei":
            return lambda mu, sig: Acquisition.nei(mu, sig, best_y, xi=xi)

        return lambda mu, sig: mu + beta * sig

# =============== Batch methods ===============
class BatchMethods:
    @staticmethod
    def local_penalization(cand_enc, acq_vals, chosen_external, n_batch, radius=0.25):
        sel = []
        vals = acq_vals.copy()
        vals[list(chosen_external)] = -np.inf

        for _ in range(n_batch):
            if sel:
                cont = cand_enc[:, :5]
                chosen_cont = cont[sel]
                d = np.linalg.norm(cont[:, None, :] - chosen_cont[None, :, :], axis=2)
                penalties = 0.9 * np.exp(-0.5 * (np.min(d, axis=1) / radius)**2)
                vals = vals * (1.0 - penalties)
            vals[sel] = -np.inf
            pick = int(np.argmax(vals))
            sel.append(pick)
        return sel

    @staticmethod
    def thompson_sampling(gp, cand_enc, chosen_external, n_batch, n_samples=200):
        sel = []
        mu, sigma = gp.predict(cand_enc)
        sigma = np.maximum(sigma, 1e-12)
        samples = mu[None, :] + sigma[None, :] * np.random.randn(n_samples, len(mu))
        argmax_per_sample = np.argmax(samples, axis=1)
        counts = np.bincount(argmax_per_sample, minlength=len(mu))
        mask = np.ones(len(cand_enc), dtype=bool)
        mask[list(chosen_external)] = False

        for _ in range(n_batch):
            counts_masked = np.where(mask, counts, -np.inf)
            idx = int(np.argmax(counts_masked))
            if not mask[idx]:
                break
            sel.append(idx)
            mask[idx] = False
        return sel

    @staticmethod
    def kriging_believer(gp, cand_enc, chosen_external, n_batch, acq_func):
        sel = []
        X_train = gp.X.copy()
        y_train = (gp.alpha.ravel() * gp.y_std) + gp.y_mean
        gp_temp = GP(ls=gp.ls, sf=gp.sf, noise=gp.noise)
        gp_temp.fit(X_train, y_train)

        mask = np.ones(len(cand_enc), dtype=bool)
        mask[list(chosen_external)] = False

        for i in range(n_batch):
            mu, sigma = gp_temp.predict(cand_enc)
            acq_vals = acq_func(mu, sigma)
            acq_vals[~mask] = -np.inf
            acq_vals[sel] = -np.inf
            idx = int(np.argmax(acq_vals))
            sel.append(idx)

            if i < n_batch - 1:
                X_train = np.vstack([X_train, cand_enc[idx]])
                y_train = np.append(y_train, mu[idx])
                gp_temp.fit(X_train, y_train)

        return sel

    @staticmethod
    def adaptive_batch(gp, cand_enc, chosen_external, n_batch, acq_func, it=15, radius=0.25):
        """
        Adaptive batch method:
        - Early iterations: Thompson Sampling
        - Later iterations: Use normal batch method (e.g., local penalization)
        """
        if it < 5:
            # Early exploration: Thompson Sampling
            return BatchMethods.thompson_sampling(gp, cand_enc, chosen_external, n_batch)
        else:
            # Exploitation / standard batch method
            mu, sigma = gp.predict(cand_enc)      # <-- evaluate GP
            acq_vals = acq_func(mu, sigma)        # <-- compute acquisition values
            return BatchMethods.kriging_believer(gp, cand_enc, chosen_external, n_batch, acq_func)
            return BatchMethods.local_penalization(cand_enc, acq_vals, chosen_external, n_batch, radius)



# =============== BO Orchestrator ===============
class BO:
    def __init__(self, X_initial, X_searchspace, iterations, batch, objective_func,
                 acq_kind="ei", batch_method="local_penalization", random_fraction=0.3):
        
        self.X_initial = X_initial
        self.X_searchspace = X_searchspace
        self.iterations = iterations
        self.batch = batch
        self.acq_kind = acq_kind
        self.batch_method = batch_method
        self.random_fraction = float(random_fraction)

        self.best_per_iter = []
        self.batch_max_per_iter = []
        self.batch_mean_per_iter = []

        start_time = datetime.timestamp(datetime.now())
        self.Y = objective_func(self.X_initial)
        self.time = [datetime.timestamp(datetime.now()) - start_time] * len(self.Y)
        self.X_enc = encode(self.X_initial)

        def as_key(x): return (float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), str(x[5]))
        self.seen = {as_key(x) for x in self.X_initial}

        self.cand = self.X_searchspace
        self.cand_enc = encode(self.cand)

        # Main BO loop
        for it in range(iterations):
            gp = GP(ls=0.5, sf=1.0, noise=1e-6)
            gp.fit(self.X_enc, self.Y)

            best_so_far = float(np.max(self.Y))
            acq_fn = Acquisition.make(self.acq_kind, best_so_far, it)

            mu, sigma = gp.predict(self.cand_enc)
            acq_vals = acq_fn(mu, sigma)

            for i, x in enumerate(self.cand):
                if as_key(x) in self.seen:
                    acq_vals[i] = -np.inf

            # Dynamic exploration fraction
            if it < 3: rand_frac = 0.4
            elif it < 6: rand_frac = 0.1
            else: rand_frac = 0.0
            n_rand = int(self.batch * rand_frac)
            n_bo = self.batch - n_rand

            all_indices = list(range(len(self.cand)))
            available = [i for i in all_indices if np.isfinite(acq_vals[i])]
            rand_idx = random.sample(available, k=min(n_rand, len(available)))
            chosen_external = rand_idx.copy()

            # Dynamic radius for local penalization
            if self.batch_method == "local_penalization" and n_bo > 0:
                if it < 3: radius = 0.4
                elif it < 6: radius = 0.25
                else: radius = 0.1
                bo_idx = BatchMethods.local_penalization(self.cand_enc, acq_vals, chosen_external, n_bo, radius)
            elif self.batch_method == "thompson_sampling" and n_bo > 0:
                bo_idx = BatchMethods.thompson_sampling(gp, self.cand_enc, chosen_external, n_bo)
            elif self.batch_method == 'kriging_believer' and n_bo > 0:
                bo_idx = BatchMethods.kriging_believer(gp, self.cand_enc, chosen_external, n_bo, acq_fn)
            elif self.batch_method == 'adaptive' and n_bo > 0:
                bo_idx = BatchMethods.adaptive_batch(gp, self.cand_enc, chosen_external, n_bo, acq_fn)
            else:
                bo_idx = []

            chosen = rand_idx + bo_idx if n_bo > 0 else rand_idx

            X_batch = [self.cand[i] for i in chosen]
            batch_Y = objective_func(X_batch)

            prev_best = float(np.max(self.Y))  # Track previous best before update

            self.Y = np.concatenate([self.Y, batch_Y])
            self.time += [datetime.timestamp(datetime.now()) - start_time] * len(batch_Y)
            self.X_enc = np.vstack([self.X_enc, self.cand_enc[chosen]])
            for x in X_batch: self.seen.add(as_key(x))

            self.batch_max_per_iter.append(float(np.max(batch_Y)))
            self.batch_mean_per_iter.append(float(np.mean(batch_Y)))
            current_best = float(np.max(self.Y))
            self.best_per_iter.append(current_best)

            # ===== Logging for best titre =====
            if current_best > prev_best:
                print(f"Iteration {it}: Best Titre: {current_best:.4f} → Updated!")
            else:
                print(f"Iteration {it}: Best Titre: {current_best:.4f}")
            

class Plotter:
    @staticmethod
    def plot_solo_run(best_per_iter, final_score, filename="solo_run_best_per_iter.png"):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,6))
        plt.plot(best_per_iter, 'o-', lw=2, label='Best titre so far')
        plt.axvspan(-0.5, 2.5, color='gray', alpha=0.15, label='No-score zone (iters 0–2)')
        plt.xlabel("Iteration")
        plt.ylabel("Titre (g/L)")
        plt.title(f"Solo BO run: Best titre per iteration\nFinal score = {final_score:.2f}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()

    @staticmethod
    def plot_best_config_heatmap(averaged_results, filename="config_heatmap.png"):
        """
        averaged_results: list of tuples
        (acq, method, bs, mean_score, std_score, scores)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Extract unique acquisitions and batch methods
        acqs = sorted(list({r[0] for r in averaged_results}))
        methods = sorted(list({r[1] for r in averaged_results}))

        heat = np.zeros((len(methods), len(acqs)))

        # Fill heatmap with mean scores
        for r in averaged_results:
            acq_idx = acqs.index(r[0])
            method_idx = methods.index(r[1])
            heat[method_idx, acq_idx] = r[3]  # mean_score

        plt.figure(figsize=(8,6))
        im = plt.imshow(heat, cmap='viridis', origin='lower')

        # Axis ticks
        plt.xticks(np.arange(len(acqs)), acqs)
        plt.yticks(np.arange(len(methods)), methods)

        # Annotate heatmap with values
        for i in range(len(methods)):
            for j in range(len(acqs)):
                plt.text(j, i, f"{heat[i,j]:.2f}", ha='center', va='center', color='w', fontsize=10)

        plt.colorbar(im, label="Mean score")
        plt.xlabel("Acquisition function")
        plt.ylabel("Batch method")
        plt.title("BO sensitivity: Mean score heatmap")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()


# =============== Runner ===============
def run_bo(acq, batch_method, batch_size, iterations):
    X_initial = sobol_initial_samples(6)
    X_searchspace = sobol_initial_samples(10000)
    bo = BO(
        X_initial=X_initial,
        X_searchspace=X_searchspace,
        iterations=iterations,
        batch=batch_size,
        objective_func=objective_func,
        acq_kind=acq,
        batch_method=batch_method,
        random_fraction=0.3,
    )
    return np.array(bo.best_per_iter), np.array(bo.time), np.array(bo.Y)

# ====================== Main ======================
if __name__ == "__main__":

    solo = True   # Set True for a single run, False for full sensitivity
    iterations = 15
    batch_size = 5
    n_repeats = 10   # only used if solo=False

    if solo:
        # --------- Solo run ---------
        X_initial = sobol_initial_samples(6)
        X_searchspace = sobol_initial_samples(10000)

        bo = BO(
            X_initial=X_initial,
            X_searchspace=X_searchspace,
            iterations=iterations,
            batch=batch_size,
            objective_func=objective_func,
            acq_kind="ei",                      # 'ei', 'nei', 'ucb'
            batch_method="thompson_sampling",  # 'thompson_sampling', 'local_penalization', 'kriging_believer', 'adaptive'
            random_fraction=0.3,
        )

        # Compute batch-wise scores according to scheme
        per_batch_scores = [0.0 if it < 3 else best for it, best in enumerate(bo.best_per_iter)]
        final_score = sum(per_batch_scores)
        print(f"\nFinal score for this solo run = {final_score:.3f}")

        # Plot solo run
        Plotter.plot_solo_run(bo.best_per_iter, final_score, filename="solo_run_best_per_iter.png")

    else:
        # --------- Full sensitivity / grid search ---------
        acquisitions = ('ei', 'ucb', 'nei')
        batch_methods = ('local_penalization', 'thompson_sampling', 'kriging_believer', 'adaptive')
        batch_sizes = (batch_size,)

        results_dict = {}
        total_configs = len(acquisitions) * len(batch_methods) * len(batch_sizes)
        count = 0

        for acq in acquisitions:
            for method in batch_methods:
                for bs in batch_sizes:
                    key = (acq, method, bs)
                    results_dict[key] = []
                    for r in range(n_repeats):
                        count += 1
                        print(f"[{count}/{total_configs*n_repeats}] {acq} + {method} batch={bs} repeat {r+1}", end=" → ")
                        best_per_iter, _, _ = run_bo(acq, method, bs, iterations)
                        per_batch_scores = [0.0 if it < 3 else float(best_per_iter[it]) for it in range(iterations)]
                        final_score = sum(per_batch_scores)
                        results_dict[key].append(final_score)
                        print(f"score = {final_score:.3f}")

        # Average and sort results
        averaged_results = []
        for (acq, method, bs), scores in results_dict.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            averaged_results.append((acq, method, bs, mean_score, std_score, scores))
        averaged_results.sort(key=lambda x: x[3], reverse=True)

        print("\nFINAL RANKING")
        for rank, (acq, method, bs, mean_score, std_score, _) in enumerate(averaged_results, 1):
            print(f"{rank:2d} | {acq:>6} + {method:>20} | {bs:5d} | {mean_score:8.3f} ± {std_score:6.3f}")

        # Heatmap of mean scores across acquisition functions and batch methods
        Plotter.plot_best_config_heatmap(averaged_results, filename="config_heatmap.png")
