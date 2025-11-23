import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm
import random
import sobol_seq
import matplotlib.pyplot as plt
from datetime import datetime

# ====================== YOUR EXISTING HELPER FUNCTIONS ======================

def objective_func(X):
    return np.array(virtual_lab.conduct_experiment(X))

def sobol_initial_samples(n_samples):
    sobol_points = sobol_seq.i4_sobol_generate(5, n_samples)
    
    temp_range = [30, 40]
    pH_range   = [6, 8]
    f1_range   = [0, 50]
    f2_range   = [0, 50]
    f3_range   = [0, 50]
    celltype = ['celltype_1','celltype_2','celltype_3']

    temp = temp_range[0] + sobol_points[:,0] * (temp_range[1] - temp_range[0])
    pH   = pH_range[0]   + sobol_points[:,1] * (pH_range[1]   - pH_range[0])
    f1   = f1_range[0]   + sobol_points[:,2] * (f1_range[1]   - f1_range[0])
    f2   = f2_range[0]   + sobol_points[:,3] * (f2_range[1]   - f2_range[0])
    f3   = f3_range[0]   + sobol_points[:,4] * (f3_range[1]   - f3_range[0])
    celltype_list = [random.choice(celltype) for _ in range(n_samples)]

    return [[temp[i], pH[i], f1[i], f2[i], f3[i], celltype_list[i]] for i in range(n_samples)]

def encode_features(X):
    """Encode features with one-hot encoding for categorical variables"""
    encoded = []
    for x in X:
        temp, pH, f1, f2, f3, cell_type = x
        
        # Normalize continuous variables to [0,1]
        temp_norm = (temp - 30) / 10.0
        pH_norm = (pH - 6) / 2.0
        f1_norm = f1 / 50.0
        f2_norm = f2 / 50.0
        f3_norm = f3 / 50.0
        
        # One-hot encode cell type
        if cell_type == 'celltype_1':
            cell_encoded = [1, 0, 0]
        elif cell_type == 'celltype_2':
            cell_encoded = [0, 1, 0]
        else:  # celltype_3
            cell_encoded = [0, 0, 1]
        
        encoded.append([temp_norm, pH_norm, f1_norm, f2_norm, f3_norm] + cell_encoded)
    
    return np.array(encoded)

# ====================== IMPLEMENT ALL 5 METHODS ======================

# 1. Gaussian Process with Advanced Kernels
class AdvancedGP:
    def __init__(self, length_scales_cont=None, sigma_cat=1.0):
        self.length_scales_cont = length_scales_cont if length_scales_cont is not None else np.ones(5)
        self.sigma_cat = sigma_cat
        self.X_train = None
        self.y_train = None
        self.L = None
        self.alpha = None
            
    def mixed_kernel(self, X1, X2):
        # Ensure 2D arrays
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        
        # Continuous part (RBF) - proper broadcasting
        # X1 shape: (n1, 8), X2 shape: (n2, 8)
        # We want output shape: (n1, n2)
        
        # Reshape for broadcasting: (n1, 5, 1) - (1, 5, n2) = (n1, 5, n2)
        cont_diff = X1[:, :5, np.newaxis] - X2[:, :5].T[np.newaxis, :, :]
        cont_dist = np.sum((cont_diff / self.length_scales_cont[:, np.newaxis])**2, axis=1)
        k_cont = np.exp(-0.5 * cont_dist)
        
        # Categorical part (overlap kernel)
        # Compare each row of X1 with each row of X2
        cat_overlap = np.array([[np.all(x1[5:] == x2[5:]) for x2 in X2] for x1 in X1]).astype(float)
        k_cat = self.sigma_cat * cat_overlap
        
        return k_cont + k_cat
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        K = self.mixed_kernel(X, X)
        K += np.eye(len(K)) * 1e-8
        self.L = cho_factor(K, lower=True)
        self.alpha = cho_solve(self.L, y)
    
    def predict(self, X):
        K_s = self.mixed_kernel(X, self.X_train)
        mu = K_s @ self.alpha
        
        # Simple variance approximation
        if len(X) == len(self.X_train) and np.allclose(X, self.X_train):
            var = np.zeros(len(X))
        else:
            var = np.ones(len(X)) * 0.1
            
        return mu, np.sqrt(var)

# 2. Simple Random Forest
class SimpleRandomForest:
    def __init__(self, n_trees=20, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        
    def _find_best_split(self, X, y):
        best_gain = -np.inf
        best_feature = 0
        best_threshold = 0
        
        parent_variance = np.var(y)
        
        for feature in range(X.shape[1]):
            unique_vals = np.unique(X[:, feature])
            for threshold in unique_vals:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                    
                left_var = np.var(y[left_mask])
                right_var = np.var(y[right_mask])
                
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                total = n_left + n_right
                
                gain = parent_variance - (n_left/total * left_var + n_right/total * right_var)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold, best_gain
    
    def _grow_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) <= 2:
            return {'leaf': True, 'value': np.mean(y)}
            
        feature, threshold, gain = self._find_best_split(X, y)
        
        if gain <= 0:
            return {'leaf': True, 'value': np.mean(y)}
            
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': self._grow_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        }
    
    def _predict_tree(self, tree, x):
        if tree['leaf']:
            return tree['value']
            
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_tree(tree['left'], x)
        else:
            return self._predict_tree(tree['right'], x)
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            # Bootstrap
            idx = np.random.choice(len(X), len(X), replace=True)
            tree = self._grow_tree(X[idx], y[idx], 0)
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = []
        for x in X:
            tree_preds = [self._predict_tree(tree, x) for tree in self.trees]
            predictions.append(np.mean(tree_preds))
        return np.array(predictions), np.ones(len(X)) * 0.1  # Dummy uncertainty

# 3. Simple Neural Network
class SimpleNeuralNetwork:
    def __init__(self, hidden_size=10, learning_rate=0.01):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weights1 = None
        self.weights2 = None
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def fit(self, X, y, epochs=100):
        n_features = X.shape[1]
        
        # Initialize weights
        self.weights1 = np.random.randn(n_features, self.hidden_size) * 0.1
        self.weights2 = np.random.randn(self.hidden_size, 1) * 0.1
        
        for epoch in range(epochs):
            # Forward pass
            hidden = self._sigmoid(X @ self.weights1)
            output = hidden @ self.weights2
            
            # Backward pass (simplified)
            error = output.flatten() - y
            
            # Update weights
            grad_output = error.reshape(-1, 1)
            grad_weights2 = hidden.T @ grad_output
            
            grad_hidden = grad_output @ self.weights2.T * hidden * (1 - hidden)
            grad_weights1 = X.T @ grad_hidden
            
            self.weights2 -= self.learning_rate * grad_weights2 / len(X)
            self.weights1 -= self.learning_rate * grad_weights1 / len(X)
    
    def predict(self, X):
        hidden = self._sigmoid(X @ self.weights1)
        output = hidden @ self.weights2
        return output.flatten(), np.ones(len(X)) * 0.1  # Dummy uncertainty

# 4. Ensemble of Surrogates
class EnsembleSurrogate:
    def __init__(self):
        self.models = [
            AdvancedGP(length_scales_cont=np.ones(5)*0.5),
            SimpleRandomForest(n_trees=10, max_depth=4),
            SimpleNeuralNetwork(hidden_size=8)
        ]
        self.weights = None
        
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        
        # Simple equal weighting
        self.weights = np.ones(len(self.models)) / len(self.models)
    
    def predict(self, X):
        predictions = []
        uncertainties = []
        
        for model in self.models:
            mu, std = model.predict(X)
            predictions.append(mu)
            uncertainties.append(std)
        
        # Weighted average
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        # Average uncertainty
        weighted_std = np.average(uncertainties, axis=0, weights=self.weights)
        
        return weighted_pred, weighted_std

# 5. Knowledge-Enhanced BO function
def domain_informed_ei(X, gp, best_y, xi=0.01):
    """Expected Improvement with domain knowledge"""
    mu, std = gp.predict(X)
    
    # Standard EI calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma = (mu - best_y - xi) / std
        ei = (mu - best_y - xi) * norm.cdf(gamma) + std * norm.pdf(gamma)
        ei = np.nan_to_num(ei, nan=0.0)
    
    # Domain knowledge penalties/rewards
    penalties = np.ones(len(X))
    
    # Extract normalized features
    if len(X.shape) == 2 and X.shape[1] == 8:  # Assuming 5 cont + 3 one-hot
        # pH preference (around neutral)
        pH_values = X[:,1] * 2 + 6  # Denormalize
        penalties *= 1.0 - 0.3 * np.exp(-((pH_values - 7.0) / 0.8)**2)
        
        # Temperature preference (around 36°C)
        temp_values = X[:,0] * 10 + 30
        penalties *= 1.0 - 0.2 * np.exp(-((temp_values - 36.0) / 2.0)**2)
    
    return ei * penalties

def expected_improvement(mu, sigma, best_y, xi=0.01):
    """Standard Expected Improvement"""
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma = (mu - best_y - xi) / sigma
        ei = (mu - best_y - xi) * norm.cdf(gamma) + sigma * norm.pdf(gamma)
        ei = np.nan_to_num(ei, nan=0.0)
    return ei

# ====================== COMPARISON FRAMEWORK ======================

def compare_methods(X_initial, X_searchspace, iterations=10, batch_size=3):
    """Compare all 5 methods on the same problem - FIXED VERSION"""
    
    methods = {
        'Advanced GP': AdvancedGP(length_scales_cont=np.ones(5)*0.5),
        'Random Forest': SimpleRandomForest(n_trees=15, max_depth=4),
        'Neural Network': SimpleNeuralNetwork(hidden_size=8),
        'Ensemble': EnsembleSurrogate(),
        'Domain GP': AdvancedGP(length_scales_cont=np.ones(5)*0.5)
    }
    
    results = {name: {'best_yields': [], 'cumulative_yields': [], 'time': []} for name in methods.keys()}
    
    for method_name, model in methods.items():
        print(f"Running {method_name}...")
        
        # Each method gets its OWN independent search process
        X_data = X_initial.copy()
        y_data = objective_func(X_initial).copy()
        
        best_yield = np.max(y_data)
        cumulative_yield = np.sum(y_data)
        
        results[method_name]['best_yields'].append(best_yield)
        results[method_name]['cumulative_yields'].append(cumulative_yield)
        results[method_name]['time'].append(0)  # Start time
        
        # Create a copy of search space that we can modify for this method
        available_indices = list(range(len(X_searchspace)))
        
        for iteration in range(iterations):
            start_time = datetime.now()
            
            # Fit the model on current data
            X_encoded = encode_features(X_data)
            model.fit(X_encoded, y_data)
            
            # Encode available search points
            X_available_encoded = encode_features([X_searchspace[i] for i in available_indices])
            
            if method_name == 'Domain GP':
                acquisition = domain_informed_ei(X_available_encoded, model, best_yield)
            else:
                mu, std = model.predict(X_available_encoded)
                acquisition = expected_improvement(mu, std, best_yield)
            
            # Select top points from available pool
            selected_local_indices = np.argsort(acquisition)[-batch_size:]
            selected_global_indices = [available_indices[i] for i in selected_local_indices]
            next_batch = [X_searchspace[i] for i in selected_global_indices]
            
            # Remove selected points from available pool
            for idx in selected_global_indices:
                available_indices.remove(idx)
            
            # Evaluate batch
            y_batch = objective_func(next_batch)
            
            # Update data
            X_data.extend(next_batch)
            y_data = np.concatenate([y_data, y_batch])
            
            # Track results
            best_yield = max(best_yield, np.max(y_batch))
            cumulative_yield = np.sum(y_data)
            
            # Calculate time for this iteration
            iteration_time = (datetime.now() - start_time).total_seconds() * 1000  # Convert to milliseconds
            results[method_name]['time'].append(iteration_time)
            
            results[method_name]['best_yields'].append(best_yield)
            results[method_name]['cumulative_yields'].append(cumulative_yield)
            
            print(f"  {method_name} - Iter {iteration+1}: Best = {best_yield:.1f} g/L")
    
    return results

# ====================== PLOTTING COMPARISON ======================

def plot_comparison(results):
    """Plot all methods on the same graph with different colors"""
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    methods = list(results.keys())
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Best yield progression
    plt.subplot(2, 2, 1)
    for i, method in enumerate(methods):
        yields = results[method]['best_yields']
        plt.plot(range(len(yields)), yields, color=colors[i], linewidth=2, label=method, marker='o', markersize=3)
    
    plt.xlabel('Experiment Number')
    plt.ylabel('Best Yield (g/L)')
    plt.title('Best Yield Progression - Method Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative yield progression
    plt.subplot(2, 2, 2)
    for i, method in enumerate(methods):
        yields = results[method]['cumulative_yields']
        plt.plot(range(len(yields)), yields, color=colors[i], linewidth=2, label=method, marker='s', markersize=3)
    
    plt.xlabel('Experiment Number')
    plt.ylabel('Cumulative Yield (g/L)')
    plt.title('Cumulative Yield - Method Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative yield vs cumulative time (like your example)
    plt.subplot(2, 2, 3)
    for i, method in enumerate(methods):
        cumulative_yields = results[method]['cumulative_yields']
        cumulative_time = np.cumsum(results[method]['time'])
        plt.plot(cumulative_time, cumulative_yields, color=colors[i], linewidth=2, label=method)
    
    plt.xlabel('Cumulative Time [ms]')
    plt.ylabel('Cumulative Titre Conc. [g/L]')
    plt.title('Cumulative Titre Concentration vs. Cumulative Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Best yield vs cumulative time
    plt.subplot(2, 2, 4)
    for i, method in enumerate(methods):
        best_yields = results[method]['best_yields']
        cumulative_time = np.cumsum(results[method]['time'])
        plt.plot(cumulative_time, best_yields, color=colors[i], linewidth=2, label=method)
    
    plt.xlabel('Cumulative Time [ms]')
    plt.ylabel('Best Yield [g/L]')
    plt.title('Best Yield vs. Cumulative Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print final results
    print(f"\n=== FINAL COMPARISON ===")
    for method in methods:
        best = results[method]['best_yields'][-1]
        cumulative = results[method]['cumulative_yields'][-1]
        total_time = np.sum(results[method]['time'])
        print(f"{method:15} -> Best: {best:7.1f} g/L, Cumulative: {cumulative:7.1f} g/L, Time: {total_time:.1f} ms")

# ====================== EXECUTION ======================

if __name__ == "__main__":
    print("Comparing all 5 Bayesian Optimization methods...")
    
    # Generate search space (smaller for faster comparison)
    X_initial = sobol_initial_samples(6)
    X_searchspace = sobol_initial_samples(200)  # Smaller for faster comparison
    
    # Run comparison
    results = compare_methods(X_initial, X_searchspace, iterations=8, batch_size=3)
    
    # Plot results
    plot_comparison(results)