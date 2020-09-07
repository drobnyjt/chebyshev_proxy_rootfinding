import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, newton
import time

import sys
sys.setrecursionlimit(10**6)

'''
This set of functions was designed to test some methods from Boyd's wonderful
paper, Finding the Zeros of a Univariate Equation: Proxy Rootfinders, Chebyshev
Interpolation, and the Companion Matrix (2013).

It was not designed for general purpose use, but may be enlightening to examine.

Note that in the 2013 paper, there is a typo in Eq. B.2. (-1)a[j - 1]/2/a[N] should
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
    '''
    k = np.arange(0, N + 1)
    return (b - a)/2.*np.cos(np.pi*k/N) + (b+a)/2.

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
    Calculates Chebyshev interpolation matrix I_jk

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

    fn = 0.5*(b0 - b3 + a_j[0])
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

def chebyshev_subdivide(F, intervals, N0=2, epsilon=1E-3, N_max=24):
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
        error = np.sum(np.abs(np.append(a_0, np.zeros(N0)) - a_1))

        #Since the last row of the Chebyshev-Frobenius matrix is undefined
        #when a_1[-1] == 0, step back to the previous iteration and return that

        #Since the maximal error is 2*N0 (all a_0 = 1, all a_1 = 0)
        if a_1[-1] == 0:
            return a_0, 2*N0

        #Otherwise, return a_1 when error is small or next N > N_max
        if (error < epsilon) or (2*N1 - 1 >= N_max):
            return a_1, error

        a_0 = a_1
        N0 = N1
    return a_1

def main():
    a = 0.
    b = 200*ANGSTROM

    p = 1*ANGSTROM
    Er = 0.01*EV
    G = lambda x: F(x, p, Er)

    intervals, coefficients = chebyshev_subdivide(G, [[a, b]], 5, 1E-3, N_max=100)
    #print(intervals)

    N_plot = 10000
    x = np.linspace(a, b, N_plot)
    plt.plot(x, G(x))

    roots = []
    for i, c in zip(intervals, coefficients):
        x1 = np.linspace(i[0], i[1], N_plot)
        handle = plt.plot(x1, [chebyshev_approximation_recursive(c, i[0], i[1], x_) for x_ in x1], linestyle='--')
        plt.scatter(i[0], chebyshev_approximation_recursive(c, i[0], i[1], i[0]), color='black', marker='+')
        plt.scatter(i[1], chebyshev_approximation_recursive(c, i[0], i[1], i[1]), color='black', marker='+')

        #If function is numerically identical to zero, it'll break when calculating A
        if np.all(c == 0):
            break

        A = companion_matrix(c)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        for eigenvalue in eigenvalues:
            if np.isreal(eigenvalue) and np.abs(eigenvalue) < 1:
                roots.append((i[1] - i[0])/2*eigenvalue + (i[1] + i[0])/2)


    for root in roots:
        handle = plt.scatter(root, 0, marker='*', s=100, color='black')
        transformed_root = np.real(root)
        print(transformed_root)
    plt.show()

def sweep_p_Er():
    a = 0
    b = 100*ANGSTROM

    impact_parameters = np.linspace(0., 10.*ANGSTROM, 100)
    relative_energies = np.array([1E-5, 1E-4, 1E-3, 1E-2, 1E-1])*EV
    colors = ['red', 'green', 'blue', 'purple', 'orange']

    handles = []
    legends = []
    for energy_index, Er in enumerate(relative_energies):
        fsolve_roots = []
        for p in impact_parameters:

            G = lambda r: F(r, p, Er)

            #N_plot = 1000
            #x = np.linspace(a, b, N_plot)
            #handle = plt.plot(x, G(x))
            #handles = [handle[0]]
            #legends = ['F(x)*S(x)']

            start = time.time()
            intervals, coefficients = chebyshev_subdivide(G, [[a, b]], 5, 0.01, N_max=50)
            #print(intervals)
            stop = time.time()
            #print(f'Chebyshev approximation took {stop - start} s')

            roots = []
            for i, c in zip(intervals, coefficients):
                #x1 = np.linspace(i[0], i[1], N_plot)
                #handle = plt.plot(x1, [chebyshev_approximation_recursive(c, i[0], i[1], x_) for x_ in x1], linestyle='--')
                #plt.scatter(i[0], chebyshev_approximation_recursive(c, i[0], i[1], i[0]), color='black', marker='+')
                #plt.scatter(i[1], chebyshev_approximation_recursive(c, i[0], i[1], i[1]), color='black', marker='+')

                #handles.append(handle[0])
                #legends.append(f'N = {len(c) - 1}')

                #If function is numerically identical to zero, it'll break when calculating A
                if np.all(c == 0):
                    break

                A = companion_matrix(c)
                start = time.time()
                eigenvalues, eigenvectors = np.linalg.eig(A)
                stop = time.time()
                #print(f'Eigenvalue calculation took {stop - start} s')

                for eigenvalue in eigenvalues:
                    if np.isreal(eigenvalue) and np.abs(eigenvalue) < 1:
                        roots.append((i[1] - i[0])/2*eigenvalue + (i[1] + i[0])/2)

            start = time.time()
            fsolve_root = fsolve(G, x0 = p*2, xtol=1E-12)
            stop = time.time()
            fsolve_roots.append(fsolve_root/ANGSTROM)

            for root in roots:
                #handle = plt.scatter(root, 0, marker='*', s=100, color='black')
                handle = plt.scatter(p/ANGSTROM, root/ANGSTROM, marker='.', color=colors[energy_index])
                #handles.append(handle)
                #legends.append(f'CPR root: {np.round(np.real(root/ANGSTROM), 6)}')

        handles.append(handle)
        legends.append(f'E = {np.round(Er/EV,5)} EV')
        plt.plot(impact_parameters/ANGSTROM, fsolve_roots, color=colors[energy_index])
            #print(f'fsolve() took {stop - start} s')

            #print(f'fsolve root: {fsolve_root}')
            #plt.title(f'F(x) with Real Root at x0 = {fsolve_root}')
            #plt.legend(handles, legends)
        #plt.show()
    plt.legend(handles, legends)
    plt.title('DOCA F(x) for Buckingham-type potential CPR: • fsolve(): - ')
    plt.xlabel('p [A]')
    plt.ylabel('R [A]')

    plt.show()


if __name__ == '__main__':
    main()
