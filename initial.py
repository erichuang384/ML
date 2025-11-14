# Group information
import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
import scipy
import random
import sobol_seq
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

group_names     = ['Name 1','Name 2']
cid_numbers     = ['000000','111111']
oral_assessment = [0, 1]

def objective_func(X):
    return(np.array(virtual_lab.conduct_experiment(X)))

# Helper Class
class RandomSelection:
    def __init__(self, X_searchspace, objective_func, batch): 
        self.X_searchspace = X_searchspace
        self.batch         = batch

        random_searchspace = [self.X_searchspace[random.randrange(len(self.X_searchspace))] for c in range(batch)]
        self.random_Y      = objective_func(random_searchspace)


# BO class
class BO:
    def __init__(self, X_initial, X_searchspace, iterations, batch, objective_func):
        start_time = datetime.timestamp(datetime.now())

        self.X_initial     = X_initial
        self.X_searchspace = X_searchspace
        self.iterations    = iterations
        self.batch         = batch

        self.Y     = objective_func(self.X_initial)
        self.time  = [datetime.timestamp(datetime.now())-start_time]
        self.time += [0]*(len(self.X_initial)-1)
        start_time = datetime.timestamp(datetime.now())
        
        for iteration in range(self.iterations):
            random_selection = RandomSelection(self.X_searchspace, objective_func, self.batch)
            self.Y           = np.concatenate([self.Y, random_selection.random_Y])
            self.time        += [datetime.timestamp(datetime.now())-start_time]
            self.time        += [0]*(len(random_selection.random_Y)-1)
            start_time = datetime.timestamp(datetime.now())


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

    # Combine into list of lists
    X_init = [[temp[i], pH[i], f1[i], f2[i], f3[i], celltype_list[i]] for i in range(n_samples)]
    return X_init
# BO Execution Block

X_initial = sobol_initial_samples(6)


#X_searchspace = [[a,b,c,d,e,f] for a in temp for b in pH for c in f1 for d in f2 for e in f3 for f in celltype]
X_searchspace = sobol_initial_samples(1000)

BO_m = BO(X_initial, X_searchspace, 15, 5, objective_func)

# Assuming 'BO_m' is an instance of the BO class from the provided code
time = np.cumsum(BO_m.time)
cumulative_titre_conc = np.cumsum(BO_m.Y)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(time, cumulative_titre_conc, color='red', label='Cumulative Titre Conc.')
plt.xlabel('Cumulative Time [ms]')
plt.ylabel('Cumulative Titre Conc. [g/L]')
plt.title('Cumulative Titre Concentration vs. Cumulative Time')
plt.legend()
plt.grid(True)
plt.show()