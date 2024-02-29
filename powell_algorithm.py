import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import r2_score

# Define the function to be optimized
def refractive_index(x, A, B, C, D):
    return np.sqrt(A + B * x ** 2 / ((x ** 2 - C ** 2)+1e-6) - 0.01 * x ** 2)

# Define the residual function to be minimized
def residual(params, x_data, y_data):
    A, B, C, D = params
    y_pred = refractive_index(x_data, A, B, C, D)
    residuals = y_pred - y_data
    ss_res = np.sum(residuals ** 2)
    return ss_res

# Load data from file
data = np.loadtxt('data_bk7_4.txt')
x_data = data[:, 0]
y_data = data[:, 1]

# Set initial parameters
params_init = [1.5, 0.5, 0.1, 0.01]

# Set bounds for the parameters
bounds = [(0.5, 10), (0.001, 5), (-0.2, 0.5), (0.001, 0.1)]

# Optimize using Powell algorithm
result = minimize(residual, params_init, args=(x_data, y_data), method='Powell', options={'xtol':1e-6, 'ftol':1e-6})

# Extract optimized parameters and calculate R2 score
A_opt, B_opt, C_opt, D_opt = result.x
y_pred = refractive_index(x_data, A_opt, B_opt, C_opt, D_opt)
r2 = r2_score(y_data, y_pred)

# print
print('Optimized parameters:', result.x)
print('R-squared value:', r2)

# Plot the data and optimized function
x_fit = np.linspace(0.3, 1.8, 100)
y_fit = refractive_index(x_fit, *result.x)
plt.plot(x_fit, y_fit, label='Fit')
plt.scatter(x_data, y_data, label='Data')
plt.xlabel('Wavelength (µm)')
plt.ylabel('Refractive Index')
plt.legend()
plt.title(f'R2 score: {r2:.4f}')
plt.show()



