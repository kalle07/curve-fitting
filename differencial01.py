import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score

# Define the function to be optimized
def refractive_index(x, A, B, C, D):
    sqrt_term = B * x ** 2 / ((x ** 2 - C ** 2)+1e-6)
    sqrt_term = np.where(sqrt_term >= 0, sqrt_term, 0)
    y = np.sqrt((A + sqrt_term - D * x ** 2))
    return np.clip(y, y_output_bounds[0], y_output_bounds[1])

# Define the objective function to be minimized
def objective(params, x_data, y_data, y_output_bounds, penalty):
    A, B, C, D = params
    y_pred = refractive_index(x_data, A, B, C, D)
    mask = np.logical_and(x_data >= x_output_bounds[0], x_data <= x_output_bounds[1])
    error = y_data[mask] - y_pred[mask]
    error[np.isnan(error)] = 0  # Set NaN values to 0
    penalty_val = 0
    if y_pred.min() < y_output_bounds[0]:
        penalty_val += 1e12 * (y_output_bounds[0] - y_pred.min()) ** 2
    if y_pred.max() > y_output_bounds[1]:
        penalty_val += 1e12 * (y_pred.max() - y_output_bounds[1]) ** 2
    return np.sum(error ** 2) + penalty + penalty_val

# Load data from file
data = np.loadtxt('data_bk7.txt')
x_data = data[:, 0]
y_data = data[:, 1]

# Set bounds for the output function
x_output_bounds = (0.2, 3.0)
y_output_bounds = (1.0, 5.0)

# Set initial parameters
A_init = 1.5
B_init = 0.5
C_init = 0.2
D_init = 0.05
params_init = [A_init, B_init, C_init, D_init]

# Set bounds for the parameters
bounds = [(0.1, 5), (0.1, 2), (-0.1, 1), (0, 0.5)]

# set panelty for output function bounds
penalty = 1e3  # Set the penalty value here

# Perform first optimization with differential evolution
result = differential_evolution(objective, bounds, tol=1e-12, maxiter=100, popsize=10, strategy='best1bin', args=(x_data, y_data, y_output_bounds, penalty))

# Print the optimized parameters
A_opt, B_opt, C_opt, D_opt = result.x
print("Final parameter values: ", result.x)

# Calculate R-squared value
y_pred = refractive_index(x_data, *result.x)
r2 = r2_score(y_data, y_pred)
print(f"R-squared value: {r2:.8f}")


# Plot the data and the fitted curve
x_plot = np.linspace(0.2, 1.5, 100)
y_plot = refractive_index(x_plot, *result.x)
plt.plot(x_data, y_data, 'bo')
plt.plot(x_plot, y_plot, 'r-', linewidth=2)
plt.xlabel('Wavelength (µm)')
plt.ylabel('Refractive index')
plt.show()