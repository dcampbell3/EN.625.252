import numpy as np
from tabulate import tabulate
import warnings

# Ignore numpy - tabulate issue to make output look pretty :)
warnings.simplefilter(action='ignore', category=FutureWarning)

def powerMethod(A, k): 

    # Get our initial x to be a zero column vector except for the first position which is 1
    x = np.zeros((np.shape(A)[1], 1))
    x[0][0] = 1

    # Store the results of each iteration in a list
    results = []

    # main loop
    for i in range(k): 

        # multiply A by x_k
        Ax_k = np.matmul(A, x)

        # find max of Ax_k
        mu = np.max(Ax_k)

        # store result
        results.append([i, x, Ax_k, mu])

        # calculate x_k+1
        x = Ax_k / mu
    
    return results

A = np.array([[6,5,7],
              [1,2,5],
              [1,1,1]])

print(tabulate(powerMethod(A, 10), headers=['k', 'x_k', 'Ax_k', 'mu'], ))
        




