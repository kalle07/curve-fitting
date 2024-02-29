import numpy as np
from sklearn.metrics import r2_score
from skopt import gp_minimize
import matplotlib.pyplot as plt

data = np.loadtxt('data_bk7_4.txt')
x_data = data[:, 0]
y_data = data[:, 1]
#x_data = np.array([0.3, 0.432, 0.63, 0.85, 1.026, 1.312, 1.554, 2.016, 2.434])
#y_data = np.array([1.552770263573871, 1.527078429140649, 1.5151856452759, 1.5098401349174289, 1.507143391738333, 1.503558798360447, 1.500601897832236, 1.49426003597419, 1.487256311781262])

def objective_function(params, debug=False):
    A, B, C, D = params
    arg = A + B * x_data ** 2 / (x_data ** 2 - C ** 2) - D * x_data ** 2
    arg = np.where(arg < 0, 0, arg)
    y_pred = np.sqrt(arg)
    r2 = r2_score(y_data, y_pred)
    if debug:
        print(f'A={A}, B={B}, C={C}, D={D}')
        print(f'r2={r2}')
    return -r2

bounds = [(0.1, 2), (0.1, 5), (0.01, 2), (0.001, 0.5)]

result = gp_minimize(objective_function, acq_func = "PI", xi=0.04 , acq_optimizer = "lbfgs", initial_point_generator = "lhs", dimensions=bounds, n_jobs=-1, n_calls=50, n_random_starts=20, kappa=1.9, noise=1e-6)

# print
print("Result", result.x)


# Plot the results
x_fit = np.linspace(0.2, 2.5, 100)
y_fit = np.sqrt(result.x[0] + result.x[1] * x_fit ** 2 / ((x_fit ** 2 - result.x[2] ** 2)+1e-9) - result.x[3] * x_fit ** 2)


# Calculate R-squared value
y_pred = np.sqrt(result.x[0] + result.x[1] * x_data ** 2 / ((x_data ** 2 - result.x[2] ** 2)+1e-9) - result.x[3] * x_data ** 2)
r2 = r2_score(y_data, y_pred)
print(f"R-squared value: {r2:.8f}")



plt.plot(x_data, y_data, 'o', label='Data')
plt.plot(x_fit, y_fit, label='Fit')
plt.legend()
plt.show()