import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import r2_score

# Define the function to be optimized
def refractive_index(x, A, B, C, D):
    return np.sqrt(A + B * x ** 2 / (x ** 2 - C ** 2) - D * x ** 2)

# Define the residual function to be minimized
def residual(params, x_data, y_data):
    A, B, C, D = params
    y_pred = refractive_index(x_data, A, B, C, D)
    residuals = y_pred - y_data
    ss_res = np.sum(residuals ** 2)
    return ss_res

# Load data from file
data = np.loadtxt('data.txt')
x_data = data[:, 0]
y_data = data[:, 1]

# Set initial parameters
params_init = [2.5, 2.0, 0.2, 0.05]

# Set bounds for the parameters
bounds = ([0.1, 0.1, 0.001, 0], [10, 5, 1, 0.1])

# Mini-batch gradient descent
batch_size = 32
n_batches = int(np.ceil(len(x_data) / batch_size))
params = params_init
for i in range(10):
    idx = np.random.permutation(len(x_data))
    x_data = x_data[idx]
    y_data = y_data[idx]
    for j in range(n_batches):
        batch_start = j * batch_size
        batch_end = min((j + 1) * batch_size, len(x_data))
        x_batch = x_data[batch_start:batch_end]
        y_batch = y_data[batch_start:batch_end]
        res = minimize(residual, params, args=(x_batch, y_batch), method='trust-constr', bounds=bounds)
        params = res.x
    if i % 10 == 0:
        print(f"Iteration {i}: Parameters = {params}")

# Calculate R-squared value
y_pred = refractive_index(x_data, *params)
r2 = r2_score(y_data, y_pred)
print(f"R-squared value: {r2:.4f}")

# Plot results
x_plot = np.linspace(np.min(x_data), np.max(x_data), 100)
y_plot = refractive_index(x_plot, *params)
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_plot, y_plot, label='Fit')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Refractive Index')
plt.legend()
plt.show()
