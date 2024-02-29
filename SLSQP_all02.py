import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from scipy.optimize import minimize
from sklearn.metrics import r2_score

# Define the function to be optimized
def refractive_index(x, A, B, C, D, eps=1e-9):
    return np.sqrt(A + B * x ** 2 / ((x ** 2 - C ** 2)+1e-6) - D * x ** 2)

# Define the residual function to be minimized
def residual(params, x_data, y_data):
    A, B, C, D = params
    y_pred = refractive_index(x_data, A, B, C, D)
    return y_pred - y_data

# Load data from file
data = np.loadtxt('data_bk7.txt')
x_data = data[:, 0]
y_data = data[:, 1]

# Set initial parameters
A_init = 1.3
B_init = 1.0
C_init = -0.1
D_init = 0.05
params_init = [A_init, B_init, C_init, D_init]

# Set bounds for the parameters
bounds = [(0.5, 10), (0.001, 5), (-0.02, 0.5), (0.001, 0.05)]

# Set bounds for the data
x_bounds = (0.3, 2.5)
y_bounds = (1.0, 5.0)

# Define the function to be minimized
def objective(params):
    return np.sum(residual(params, x_data, y_data) ** 2)

# Perform the optimization using SLSQP
result = minimize(objective, params_init, method='trust-constr', bounds=bounds, options={'maxiter': 500})



# Print the optimized parameters
print(result.x)

# Calculate R-squared value
y_pred = refractive_index(x_data, *result.x)
r2 = r2_score(y_data, y_pred)
print('R-squared value:', r2)

# Plot the results
x_fit = np.linspace(0.3, 1.8, 100)
y_fit = refractive_index(x_fit, *result.x)
plt.plot(x_data, y_data, 'o', label='Data')
plt.plot(x_fit, y_fit, label='Fit')
plt.legend()
plt.show()
