import numpy as np 
import matplotlib.pyplot as plt
from tabulate import tabulate
import warnings

# Ignore numpy - tabulate issue to make output look pretty :)
warnings.simplefilter(action='ignore', category=FutureWarning)

# values for a in our matrices
A_VALUES = [32, 31.9, 31.8, 32.2]

# Create t values to graph
t_data = np.linspace(0, 3, 100)

# Define our function for plotting
def P(A, t): 
    return np.linalg.det(A - t*np.identity(3))

# Find all coefficients for characteristic polynomial
polynomial_coefficients = [] 
for a in A_VALUES: 

    # 'A' matrix with a set to current value
    curr_mat = np.array([[-6, 28, 21], 
                       [4, -15, -12], 
                       [-8, a, 25]])
    
    # Characteristic polynomial of 'A' formed from the Eigenvalues of 'A'
    curr_poly = np.polynomial.Polynomial.fromroots(np.linalg.eigvals(curr_mat))

    # Format the value for a, the Eigenvalues of 'A', and the Characteristic Polynomial of 'A' for printing w/ tabulate
    polynomial_coefficients.append([a, np.linalg.eigvals(curr_mat), [i.real for i in curr_poly.coef]])

    # Plot P(A, t) 
    plt.plot(t_data, [P(curr_mat, t) for t in t_data], label = f'a={a}')

# Print Result
print(tabulate(polynomial_coefficients, headers=['a Value', 'Eigenvalues', 'Coefficients of Characteristic Polynomial']))

# Format our plot
plt.xlabel('t')
plt.ylabel('P(A, t)')
plt.legend()
plt.show()
