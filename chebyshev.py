import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, newton
import time

import sys

'''
This set of functions was designed to test some methods from Boyd's wonderful
paper, Finding the Zeros of a Univariate Equation: Proxy Rootfinders, Chebyshev
Interpolation, and the Companion Matrix (2013).

It was not designed for general purpose use, but may be enlightening to examine.

Note that in the 2013 paper, there is a typo in Eq. B.2: (-1)a[j - 1]/2/a[N] should
instead be (-1)a[k - 1]/2/a[N]. This is corrected in Boyd's text, Solving
Transcendental Equations (2014).

The subdivision algorithm is not described explicitly in either source, so that
algorithm is of my own design - it could almost certainly use an optimization
pass and could probably be smarter than simply subdividing intervals in half.

Author: Jon Drobny
Email: drobny2@illinois.edu
'''

EV = 1.602E-19
ANGSTROM = 1E-10

def chebyshev_points(a, b, N):
    '''
    Returns the Chebyshev interpolation points (Lobatto grid) for an interval [a,b]

    Eq. A.1 in Boyd (2013)

    Args:
        a, b: lower and upper bound of interval
        N: degree of Chebyshev series, i.e. number of points - 1

    Returns:
        xk: Lobatto grid points on [a, b] with number of points N + 1
    '''
    k = np.arange(0, N + 1)
    return (b - a)/2.*np.cos(np.pi*k/N) + (b + a)/2.

def F(r, p, Er):
    epsilon = 0.343*EV
    sigma = 2*ANGSTROM

    A = 10486*EV/Er
    B = 102E-12*EV/Er

    return (r**6 - r**6*A*np.exp(-r/ANGSTROM/0.273) + B/(1./ANGSTROM)**6 - p**2*r**4)/ANGSTROM**6/(r/ANGSTROM + 1)

def dF(r):
    return (11*r**12 + 12*r*11 + 10*r**6 + 12*r**5 - r**2 - 2*r + 1)/(r + 1)**2

def p(j, N):
    '''
    Constructor for interior elements of the interpolation matrix.
    '''
    if j == 0 or j == N:
        return 2
    else:
        return 1

def interpolation_matrix(N):
    '''
    Chebyshev interpolation matrix such that I.F(xk)=a_j, the Chebyshev coefficients.

    Eq. A.3 in Boyd (2013)

    Args:
        N: degree of Chebyshev series

    Returns:
        I_jk: Interpolation matrix
    '''
    I_jk = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        for k in range(N + 1):
            I_jk[j, k] = 2/p(j, N)/p(k, N)/N*np.cos(j*np.pi*k/N)
    return I_jk

def chebyshev_coefficients(F, a, b, N):
    '''
    Calculates Chebyshev coefficents a_j for F(x) ~ sum( a_j T_j (x))

    Args:
        F: univariate function smooth on a, b
        a, b: lower and upper bound of interval
        N: degree of Chebyshev series

    Returns:
        a_j: Chebyshev coefficients
    '''
    xk = chebyshev_points(a, b, N)
    I_jk = interpolation_matrix(N)
    a_j = I_jk.dot(F(xk))
    return a_j

def chebyshev_approximation_recursive(a_j, a, b, x):
    '''
    Clenshaw-Curtis recurrence method for approximating a function f(x) on the
    interval [a, b] at x given its Chebyshev series coefficents, a_j.

    Eqs. 3.81-3.84 in Boyd (2014)

    Note: there is a typo in the text - the equation b3 = b2 is missing.

    Args:
        a_j: Chebyshev coefficients
        a, b: lower and upper bound of inclusive interval
        x: value at which to approximate F by sum(a_j T_N)

    Returns:
        FN(x): value of Chebyshev approximation at x in [a, b]
    '''
    N = len(a_j) - 1
    j = np.arange(0, N + 1)

    #Clenshaw recurrence relation
    xi = (2.*x - (b + a))/(b - a)
    b1 = 0
    b2 = 0

    for i in range(1, N + 1):
        b0 = 2*xi*b1 - b2 + a_j[N - i]
        b3 = b2
        b2 = b1
        b1 = b0

    fn = (b0 - b3 + a_j[0])/2.
    return fn

def delta(j, k):
    '''
    Kronecker delta
    '''
    return int(j == k)

def companion_matrix(a_j):
    '''
    Chebyshev-Frobenius companion matrix.

    Eq. B.2 in Boyd (2013)

    Args:
        a_j: Chebyshev coefficients

    Returns:
        A_jk: Chebyshev-Frobenius matrix, whose eigenvalues are roots of a_j*T_N
    '''
    N = len(a_j) - 1
    A_jk = np.zeros((N, N))

    for k in range(N):
        A_jk[0, k] = delta(1, k)
        A_jk[N-1, k] = (-1)*(a_j[k]/2/a_j[N]) + (1/2)*delta(k, N - 2)

    for k in range(N):
        for j in range(1, N - 1):
            A_jk[j, k] = (delta(j, k + 1) + delta(j, k - 1))/2.

    return A_jk

def chebyshev_subdivide(F, intervals, N0=2, epsilon=1E-3, N_max=24, interval_limit=1E-3):
    '''
    Adaptive Chebyshev Series interpolation with automatic subdivision.

    This function automatically divides the domain by halves into subintervals
    such that the function F on each subinterval is well approximated (within
    epsilon) by a Chebyshev series of degree N_max or less.

    For each (sub)interval, the adaptive Chebyshev interpolation algorithm,
    which uses degree-doubling, is used to find a Chebyshev series of degree
    N0*2^(N_iterations) < N_max on the interval that is within epsilon of F.

    Args:
        F: univariate function smooth on all intervals
        intervals: list of intervals [a, b] on which to find Chebyshev series for F
        N0: initial degree of Chebyshev series
        epsilon: total error allowed
        N_max: maximum degree of Chebyshev polynomial in approximation
    '''

    coefficients = []
    intervals_out = []

    for interval in intervals:
        a, b = interval

        if (b - a) < interval_limit:
            print("Reached interval limit. Did not converge. Relax epsilon.")
            return None, None

        a_0, error = chebyshev_adaptive_approximation_coefficients(F, a, b, N0, epsilon, N_max)

        if error < epsilon:
            intervals_out.append(interval)
            coefficients.append(a_0)
        else:
            #Splitting interval in 2
            a1 = a
            b1 = a + (b - a)/2

            a2 = a + (b - a)/2
            b2 = b

            #Begin next iteration with current interval divided into two subintervals
            intervals_new, coefficients_new = chebyshev_subdivide(F, [[a1, b1], [a2, b2]], N0, epsilon, N_max)

            #Unpack resulting intervals, coefficients into completed interval lists
            for i, c in zip(intervals_new, coefficients_new):
                intervals_out.append(i)
                coefficients.append(c)

    return intervals_out, coefficients

def chebyshev_adaptive_approximation_coefficients(F, a, b, N0, epsilon, N_max):
    '''
    Adaptive Chebyshev approximation, which starts from degree N0 and doubles
    the degree each iteration.

    Args:
        F: univariate function analytic and smooth on [a, b]
        a, b: defines the interval over which the interpolation will be applied
        N0: initial degree of Chebyshev series
        epsilon: total error of Chebyshev series compared to F over [a, b]
        N_max: maximum allowed degree of Chebyshev series

    Returns:
        a_j: Chebyshev coefficients for F on a, b
        error: Total error - returns 2*N0, the maximal error, when convergence fails
    '''

    a_0 = chebyshev_coefficients(F, a, b, N0)

    while True:
        N1 = 2*N0
        a_1 = chebyshev_coefficients(F, a, b, N1)
        delta = np.append(a_0, np.zeros(N0)) - a_1
        error = np.sum(np.abs(delta))

        #Otherwise, return a_1 when error is small or next N > N_max
        if (error < epsilon) or (2*N1 >= N_max//2):
            return a_1, error

        a_0 = a_1
        N0 = N1
    return a_1

def main():
    a = -11
    b = 100
    N0 = 2
    epsilon = 1E-6
    truncation_threshold = 1E-12
    N_max = 500

    F =lambda x:  (x - 2.)*(x + 3.)*(x - 8.)*(x + 1E-4)*(x - 1E-5)*(x + 1.)*(x + 10)*np.exp(-np.abs(x))
    #G = lambda x: F(x)*np.exp(-np.abs(x))

    print("Subdividing...")
    intervals, coefficients = chebyshev_subdivide(F, [[a, b]], N0=N0, epsilon=epsilon, N_max=N_max, interval_limit=1E-10)
    for interval, coefficient in zip(intervals, coefficients):
        print(interval, np.size(coefficient) - 1)

    N_plot = 100
    x = np.linspace(a, b, N_plot*10)
    plt.plot(x, F(x), linewidth=3)

    roots = []
    for i, c in zip(intervals, coefficients):
        x1 = np.linspace(i[0], i[1], N_plot)
        handle = plt.plot(x1, [chebyshev_approximation_recursive(c, i[0], i[1], x_) for x_ in x1], linestyle='--')
        plt.scatter(i[0], chebyshev_approximation_recursive(c, i[0], i[1], i[0]), color='black', marker='+')
        plt.scatter(i[1], chebyshev_approximation_recursive(c, i[0], i[1], i[1]), color='black', marker='+')

        #If function is numerically identical to zero, it'll break when calculating A
        if np.all(c < truncation_threshold):
            break

        c_list = list(c)
        c_list.reverse()
        for index, a in enumerate(c_list):
            #print(index, a)
            if a > truncation_threshold:
                #print("final: ", index)
                break
        c_truncated = c[:len(c) - index]

        A = companion_matrix(c_truncated)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        for eigenvalue in eigenvalues:
            if np.isreal(eigenvalue) and np.abs(eigenvalue) < 1:
                roots.append((i[1] - i[0])/2*eigenvalue + (i[1] + i[0])/2)
                print((i[1] - i[0])/2*np.real(eigenvalue) + (i[1] + i[0])/2)

    for root in roots:
        handle = plt.scatter(root, 0, marker='*', s=100, color='black')
        transformed_root = np.real(root)

    plt.show()


if __name__ == '__main__':
    main()
