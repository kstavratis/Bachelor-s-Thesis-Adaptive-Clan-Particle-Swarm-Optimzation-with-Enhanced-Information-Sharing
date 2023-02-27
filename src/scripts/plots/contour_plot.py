import warnings
warnings.filterwarnings("default")

# Absolute path for easiness' sake.
import scripts.benchmark_functions as bf
#import ..benchmark_functions as bf

import matplotlib.pyplot as plt
#plt.style.use('seaborn-white')
import numpy as np


def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

positions = np.array([[1, 2], [-3, 10], [10, 30]])
Z = np.apply_along_axis(bf.sphere_function_formula, axis=1, arr=positions)

print(positions)
print(Z)


plt.tricontourf([1, -3, 10], [2, 10, 30], Z, 40, cmap='RdGy')
plt.colorbar()
plt.show()