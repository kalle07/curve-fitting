import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pyswarm import pso
from scipy.special import huber

# Load data from file
try:
    data = np.loadtxt('data_bk7_4.txt')
    x_data = data[:, 0]
    y_data = data[:, 1]
except IOError:
    print('Error: Unable to load data from file')
    # Use data array in the script
    x = np.array([0.3, 0.432, 0.63, 0.85, 1.026, 1.312, 1.554, 2.016, 2.434])
    y = np.array([1.552770263573871, 1.527078429140649, 1.5151856452759, 1.5098401349174289, 1.507143391738333, 1.503558798360447, 1.500601897832236, 1.49426003597419, 1.487256311781262])

def huber_loss(y_true, y_pred, delta):
    """
    Huber loss function.
    """
    residual = y_true - y_pred
    condition = np.abs(residual) < delta
    squared_loss = 0.5 * np.square(residual)
    linear_loss = delta * np.abs(residual) - 0.5 * np.square(delta)
    return np.where(condition, squared_loss, linear_loss)

def objective_function(params):
    A, B, C, D = params
    arg = A + B * x_data ** 2 / (x_data ** 2 - C ** 2) - D * x_data ** 2
    arg = np.where(arg < 0, 0, arg)
    y_pred = np.sqrt(arg)
    return -r2_score(y_data, y_pred)
        # return -mean_absolute_error(y_data, y_pred)
        # return -mean_squared_error(y_data, y_pred)
        # median_absolute_error
        # return -mean_absolute_error(y_data, y_pred)
        # return -mean_squared_error(y_data, y_pred)
        # median_absolute_error

# Define initial guesses and bounds
# guesses = [1.5, 2.0, 0.2, 0.05]
lb = [0.1, 0.1, -0.02, 0]
ub = [5.5, 2, 1, 0.1]
x_bounds = (0.2, 2.5)
y_bounds = (1.0, 5.0)

# Define constraints
ieqcons = [lambda x: x[0]-x_bounds[0] >= 0, 
           lambda x: x[1]-y_bounds[0] >= 0, 
           lambda x: x_bounds[1]-x[0] >= 0, 
           lambda x: y_bounds[1]-x[1] >= 0,
           lambda x: x[2]-x[3] >= 0]


# Run PSO optimization
xopt, fopt = pso(objective_function, swarmsize=300, lb=lb, ub=ub, ieqcons=ieqcons, maxiter=500, debug=0, omega=0.6, phip=0.5, phig=0.4,
                 minstep=1e-6, minfunc=1e-8)

# Print results
print('Best R-squared score:', -fopt)

# Generate x values for plotting
x = np.linspace(0.3, 1.8, 100)

# Get optimized A, B, C, D values
A, B, C, D = xopt
print("Result", xopt)
# Calculate corresponding y values using the objective function
arg = A + B * x ** 2 / (x ** 2 - C ** 2) - D * x ** 2
arg = np.where(arg < 0, 0, arg)
y = np.sqrt(arg)

# Plot the objective function
plt.plot(x, y)
plt.plot(x_data, y_data, 'o', label='Data')

# Set axis labels
plt.xlabel('Wavelength (μm)')
plt.ylabel('Refractive Index')

# Show the plot
plt.show()