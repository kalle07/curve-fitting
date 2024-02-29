import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from joblib import Parallel, delayed

# Define the function to calculate refractive index
def refractive_index(x, A, B, C, D):
    """Calculates refractive index based on given parameters.
    
    Args:
    x (float): input variable
    A (float): first parameter
    B (float): second parameter
    C (float): third parameter
    D (float): fourth parameter
    
    Returns:
    float: refractive index value
    """
    return np.sqrt(A + B * x ** 2 / (x ** 2 - C ** 2) - D * x ** 2)

# Define the function to be minimized for each set of parameters
def objective(params, x_data, y_data):
    """Function to be minimized for each set of parameters.
    
    Args:
    params (list): list of parameters to optimize
    x_data (ndarray): input data
    y_data (ndarray): output data
    
    Returns:
    float: sum of squared errors between predicted and actual y values
    """
    A, B, C, D = params
    y_pred = refractive_index(x_data, A, B, C, D)
    return np.sum((y_pred - y_data) ** 2)

# Define a function to perform the optimization for a single set of parameters
def optimize_params(params, x_data, y_data):
    """Performs optimization for a single set of parameters.
    
    Args:
    params (list): list of parameters to optimize
    x_data (ndarray): input data
    y_data (ndarray): output data
    
    Returns:
    OptimizeResult: object containing optimization result
    """
    bounds = [(0.1, 10), (0.1, 5), (0.001, 1), (0, 0.1)]
    result = minimize(residual, objective, params, args=(x_data, y_data), bounds=bounds, method='L-BFGS-B')
    return result
    print('Best-fit parameters:', result.x)
'''
def optimize_all(new_bounds, x_data, y_data):
    """Optimizes all parameter sets in parallel.
    
    Args:
    new_bounds (list): list of parameter sets to optimize
    x_data (ndarray): input data
    y_data (ndarray): output data
    
    Returns:
    list: list of OptimizeResult objects containing optimization results
    """
    num_cores = 4
    bounds = [(0.1, 10), (0.1, 5), (0.001, 1), (0, 0.1)]
    results = Parallel(n_jobs=num_cores)(
        delayed(minimize)(optimize_params, params, args=(x_bounds, y_bounds), bounds=bounds, method='L-BFGS-B') for params in new_bounds)
    return results
'''
# Set bounds for the data
x_bounds = (0.2, 2.5)
y_bounds = (1.0, 5.0)

# Load data from file
data = np.loadtxt('data.txt')
x_data = data[:, 0]
y_data = data[:, 1]
'''
# Generate the new parameter sets within the bounds
a_values = np.geomspace(0.1, 10, num=10) # create an array of 10 logarithmically spaced values between 0.1 and 10 for A parameter
b_values = np.geomspace(0.1, 5, num=10) # create an array of 10 logarithmically spaced values between 0.1 and 5 for B parameter
c_values = np.geomspace(0.001, 1, num=10) # create an array of 10 logarithmically spaced values between 0.001 and 1 for C parameter
d_values = np.geomspace(0.0001, 0.1, num=10) # create an array of 10 logarithmically spaced values between 0 and 0.1 for D parameter

num_orderings = 1 # set the number of parameter orderings
a_orderings = [np.random.permutation(a_values) for _ in range(num_orderings)] # create 100 permutations of the A parameter array
b_orderings = [np.random.permutation(b_values) for _ in range(num_orderings)] # create 100 permutations of the B parameter array
c_orderings = [np.random.permutation(c_values) for _ in range(num_orderings)] # create 100 permutations of the C parameter array
d_orderings = [np.random.permutation(d_values) for _ in range(num_orderings)] # create 100 permutations of the D parameter array

# Create a list of all parameter sets to optimize
new_bounds = [] # initialize an empty list to store the new parameter sets
for i in range(10): # iterate over the parameter indices and # concatenate the 100 values of each parameter into a list
    params = [a_orderings[j][i] for j in range(num_orderings)] + [b_orderings[j][i] for j in range(num_orderings)] + [c_orderings[j][i] for j in range(num_orderings)] + [d_orderings[j][i] for j in range(num_orderings)]
    new_bounds.append(params) # append the list of parameter sets to the new_bounds list

# Optimize all parameter sets in parallel
results = optimize_all(new_bounds, x_data, y_data)

# Find the best fit
best_fit = min(results, key=lambda x: x[1])
'''
# Calculate R-squared value
# y_pred = refractive_index(x_data, *best_fit[0])
# r2 = r2_score(y_data, y_pred)
# print('R-squared value:', r2)

# Print Parameters
# print('Best-fit parameters:', best_fit[0])
# print('Best-fit objective function value:', best_fit[1])
result = optimize_params
print('Best-fit parameters:', *result)

# Plot the results
x_fit = np.linspace(0.2, 2.5, 100)
# y_fit = refractive_index(x_fit, *best_fit[0])
plt.plot(x_data, y_data, 'o', label='Data')
# plt.plot(x_fit, y_fit, label='Fit')
plt.legend()
plt.show()
