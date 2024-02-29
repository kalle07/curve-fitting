import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from joblib import Parallel, delayed

# Load data from file
data = np.loadtxt('data_bk7.txt')
x_data = data[:, 0]
y_data = data[:, 1]

# Set initial parameters
params_init = [1.5, 1.0, 0.2, 0.05]

# Set Bounds for parameters
bounds = [(0.1, 10), (0.1, 5), (0.001, 1), (0.0001, 0.1)]

# Set bounds for the data
x_bounds = (0.2, 2.5)
y_bounds = (1.0, 5.0)

# Define the function to calculate refractive index
def refractive_index(x, A, B, C, D):
    return np.sqrt(np.maximum(0, A + B * x ** 2 / (x ** 2 - C ** 2) - D * x ** 2))
    # return np.real(np.sqrt(A + B * x ** 2 / ((x ** 2 - C ** 2)+1e-3) - D * x ** 2))

# Define the residual function to be minimized for each set of parameters
def objective(params, x_data, y_data):
    A, B, C, D = params
    y_pred = refractive_index(x_data, A, B, C, D)
    return y_pred - y_data

# Define a function to perform a first optimization for a single set of parameters
def optimize_params(params, x_data, y_data):
    bounds = [(0.1, 10), (0.1, 5), (0.01, 1), (0, 0.1)]
    result_1 = minimize(objective, params, args=(x_data, y_data), bounds=bounds, method='L-BFGS-B', options={'maxiter': 500})
    # print(f"Result for parameters single Step {params}: {result_1}")
    return result_1

# Perform optimization for the initial set of parameters
result_1 = optimize_params(params_init, x_data, y_data)

# Calculate predicted values for the optimized parameters
y_pred = refractive_index(x_data, *result_1.x)

# Calculate R-squared value from first optimization and Print
r_squared_step = r2_score(y_data, y_pred)
print('R-squared value Step 1:', r_squared_step)

# Print the results for the initial set of parameters
# print(f"Result for parameters initial set: {result_1}")

# Generate the new parameter sets within the bounds
a_values = np.geomspace(bounds[0][0], bounds[0][1], num=10)
b_values = np.geomspace(bounds[1][0], bounds[1][1], num=10)
c_values = np.geomspace(bounds[2][0], bounds[2][1], num=10)
d_values = np.geomspace(bounds[3][0], bounds[3][1], num=10)

# Generate 10 new initial randomly parameter sets
new_initial_guesses = []
for i in range(20):
    A = np.random.choice(a_values)
    B = np.random.choice(b_values)
    C = np.random.choice(c_values)
    D = np.random.choice(d_values)
    new_initial_guesses.append([A, B, C, D])

# Define a function to perform a second optimization for 10 sets of parameters
def optimize_params_2(params, x_data, y_data):
    result_2 = minimize(objective, params, args=(x_data, y_data), bounds=bounds, method='trust-constr', options={'maxiter': 100, 'disp':0, 'gtol':1e-8})
    y_pred = refractive_index(x_data, *tuple(result_2.x))
    return result_2, y_pred

# Perform optimization for each set of initial parameters in parallel
results_2 = Parallel(n_jobs=-1)(delayed(optimize_params_2)(params, x_data, y_data) for params in new_initial_guesses)

# Print the results for the initial set of parameters
print(f"Result for parameters initial set: {result_1.x}")

# initialisierung
best_r_squared_2 = None
# best_r_squared_2 = -1

# Calculate R-squared value from second optimization taking y_pred from optimize_params_2
for i, (result_2, y_pred) in enumerate(results_2):
    popt = result_2.x
    r_squared = r2_score(y_data, y_pred)
    
    # Print result
    print(f"Result {i+1}: {popt}")
    print(f"R-squared value: {r_squared}")
    
    # Update best result if necessary
    if r_squared > best_r_squared_2:
        best_result_2 = results_2[0][0].x
        best_r_squared_2 = r_squared

# Print best result and R-squared value for results
print(f"\nBest result for results_2: {best_result_2}")
print(f"Best R-squared value for results_2: {best_r_squared_2}")