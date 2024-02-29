import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from sklearn.metrics import r2_score

# Define the function to be optimized
def refractive_index(x, A, B, C, D):
    return np.sqrt(A + B * x ** 2 / (x ** 2 - C ** 2) - D * x ** 2)

# Define the residual function to be minimized
def residual(params, x_data, y_data):
    A, B, C, D = params
    y_pred = refractive_index(x_data, A, B, C, D)
    residuals = y_pred - y_data
    return residuals

# Load data from file
data = np.loadtxt('data_bk7_4.txt')
x_data = data[:, 0]
y_data = data[:, 1]

# Set initial parameters
params_init = [2.5, 2.0, 0.2, 0.05]

# Set bounds for the parameters
bounds = ([0.1, 0.1, -0.2, 0], [10, 5, 1, 0.1])

# Run LM optimization
params_fit, success = leastsq(residual, params_init, args=(x_data, y_data), ftol=1e-6, xtol=1e-6, gtol=1e-6, maxfev=10000)

# Check if optimization was successful
if success not in [1, 2, 3, 4]:
    print('Optimization failed!')

# Calculate R-squared value
y_pred = refractive_index(x_data, *params_fit)
r2 = r2_score(y_data, y_pred)
print(f"R-squared value: {r2:.20f}")

print('Optimized parameters:', params_fit)
# Plot results
x_plot = np.linspace(np.min(x_data), np.max(x_data), 100)
y_plot = refractive_index(x_plot, *params_fit)
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_plot, y_plot, label='Fit')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Refractive Index')
plt.legend()
plt.show()
