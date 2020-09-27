import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, newton
import time

import sys

PI = np.pi
Q = 1.602E-19
EV = Q
AMU = 1.66E-27
ANGSTROM = 1E-10
MICRON = 1E-6
PM = 1E-12
NM = 1E-9
CM = 1E-2
EPS0 = 8.85E-12
A0 = 0.52918E-10
K = 1.11265E-10
ME = 9.11E-31
SQRTPI = 1.772453850906
SQRT2PI = 2.506628274631
C = 299792000.
BETHE_BLOCH_PREFACTOR = 4.*PI*(Q*Q/(4.*PI*EPS0))*(Q*Q/(4.*PI*EPS0))/ME/C/C
LINDHARD_SCHARFF_PREFACTOR = 1.212*ANGSTROM*ANGSTROM*Q
LINDHARD_REDUCED_ENERGY_PREFACTOR = 4.*PI*EPS0/Q/Q

def eam_D(r):
    rn = [
        4.268900000000000,
        3.985680000000000,
        3.702460000000000,
        3.419240000000000,
        3.136020000000000,
        2.852800000000000,
        2.741100000000000,
        2.604045000000000,
        2.466990000000000,
    ]

    vn = [
        -0.1036435865158945,
        -0.2912948318493851,
        -2.096765499656263,
        19.16045452701010,
        -41.01619862085917,
        46.05205617244703,
        26.42203930654883,
        15.35211507804088,
        14.12806259323987,
    ]

    phi = 0.
    for v_i, r_i in zip(vn, rn):
        phi += v_i*(r_i - r)**3*np.heaviside(r_i - r, 1.)
    return phi

def eam_high_e(r):

    r1 = 1.10002200044
    r2 = 2.10004200084

    if r <= r1:
        Za = 74
        Zb = 74
        a = 0.88534*A0/(Za**(0.23) + Zb**(0.23))
        x = r/a*ANGSTROM
        return Za*Zb*Q**2/4./PI/EPS0/(r*ANGSTROM)*(0.02817*np.exp(-0.20162*x) + 0.28022*np.exp(-0.40290*x) + 0.50986*np.exp(-0.94229*x) + 0.18175*np.exp(-3.1998*x))/EV

    elif r <= r2:
        a = [
            1.389653276380862E4,
            -3.596912431628216E4,
            3.739206756369099E4,
            -1.933748081656593E4,
            0.495516793802426E4,
            -0.050264585985867E4
        ]
        x = r
        return a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3 + a[4]*x**4 + a[5]*x**5
    else:
        rn = [
            4.268900000000000,
            3.985680000000000,
            3.702460000000000,
            3.419240000000000,
            3.136020000000000,
            2.852800000000000,
            2.741100000000000,
            2.604045000000000,
            2.466990000000000,
        ]

        vn = [
            -0.1036435865158945,
            -0.2912948318493851,
            -2.096765499656263,
            19.16045452701010,
            -41.01619862085917,
            46.05205617244703,
            26.42203930654883,
            15.35211507804088,
            14.12806259323987,
        ]

        phi = 0.
        for v_i, r_i in zip(vn, rn):
            phi += v_i*(r_i - r)**3*np.heaviside(r_i - r, 1.)
        return phi


def eam(r):
    a=[
        0.960851701343041E2,
        -0.184410923895214E3,
        0.935784079613550E2,
        -0.798358265041677E1,
        0.747034092936229E1,
        -0.152756043708453E1,
        0.125205932634393E1,
        0.163082162159425E1,
        -0.141854775352260E1,
        -0.819936046256149E0,
        0.198013514305908E1,
        -0.696430179520267E0,
        0.304546909722160E-1,
        -0.163131143161660E1,
        0.138409896486177E1
    ]
    delta = [
        2.564897500000000,
        2.629795000000000,
        2.694692500000000,
        2.866317500000000,
        2.973045000000000,
        3.079772500000000,
        3.516472500000000,
        3.846445000000000,
        4.176417500000000,
        4.700845000000000,
        4.895300000000000,
        5.089755000000000,
        5.342952500000000,
        5.401695000000000,
        5.460437500000000
    ]

    phi = 0.
    for a_i, delta_i in zip(a, delta):
        phi += a_i*(delta_i - r)**3*np.heaviside(delta_i - r, 1.)
    return phi

def krc(Za, Zb, r):
    a = 0.885*A0*(np.sqrt(Za) + np.sqrt(Zb))**(-2./3.)
    x = r/a*ANGSTROM
    return Za*Zb*Q**2/4./PI/EPS0/(r*ANGSTROM)*(0.19*np.exp(-0.28*x) + 0.47*np.exp(-0.64*x) + 0.34*np.exp(-1.9*x))/EV

def morse(r, alpha, d, r0):
    return d*(np.exp(-2*alpha*(r*ANGSTROM - r0)) - 2.*np.exp(-alpha*(r*ANGSTROM - r0)))/EV


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
    a_j = I_jk.dot([F(xk_) for xk_ in xk])
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

        if abs(b - a) < interval_limit:
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
            intervals_new, coefficients_new = chebyshev_subdivide(F, [[a1, b1], [a2, b2]], N0, epsilon, N_max, interval_limit)

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

def eam(r):

    r/=ANGSTROM

    a=[
        0.960851701343041E2,
        -0.184410923895214E3,
        0.935784079613550E2,
        -0.798358265041677E1,
        0.747034092936229E1,
        -0.152756043708453E1,
        0.125205932634393E1,
        0.163082162159425E1,
        -0.141854775352260E1,
        -0.819936046256149E0,
        0.198013514305908E1,
        -0.696430179520267E0,
        0.304546909722160E-1,
        -0.163131143161660E1,
        0.138409896486177E1
    ]
    delta = [
        2.564897500000000,
        2.629795000000000,
        2.694692500000000,
        2.866317500000000,
        2.973045000000000,
        3.079772500000000,
        3.516472500000000,
        3.846445000000000,
        4.176417500000000,
        4.700845000000000,
        4.895300000000000,
        5.089755000000000,
        5.342952500000000,
        5.401695000000000,
        5.460437500000000
    ]

    phi = 0.
    for a_i, delta_i in zip(a, delta):
        phi += a_i*(delta_i - r)**3*np.heaviside(delta_i - r, 1.)
    return phi*EV

def main():
    b = 10.*ANGSTROM
    N0 = 2
    epsilon = 0.01
    truncation_threshold = 1E-22
    N_max = 500

    Er = 0.1*EV
    p = 0.1*ANGSTROM

    G = lambda r: ((r/ANGSTROM)**2 - (r/ANGSTROM)**2*eam_D(r/ANGSTROM)*EV/Er - (p/ANGSTROM)**2)

    N_plot = 1000
    x = np.linspace(0.0, b, N_plot*100)*ANGSTROM

    handles = []
    legends = ['F(r) for W-W']

    handle = plt.plot(x/ANGSTROM, [G(x_/ANGSTROM) for x_ in x], linewidth=3)
    handles.append(handle[0])

    print("Subdividing...")
    start = time.time()
    intervals, coefficients = chebyshev_subdivide(G, [[0.0, b/2.], [b/2, b]], N0=N0, epsilon=epsilon, N_max=N_max, interval_limit=0)
    stop = time.time()

    print(f'Chebyshev Adaptive Interpolation with Subdivision took: {stop - start} s')

    for interval, coefficient in zip(intervals, coefficients):
        print(interval, np.size(coefficient) - 1)

    roots = []
    for i, c in zip(intervals, coefficients):
        x1 = np.linspace(i[0], i[1], N_plot)
        handle = plt.plot(x1, [chebyshev_approximation_recursive(c, i[0], i[1], x_) for x_ in x1], linestyle='--')
        plt.scatter(i[0], chebyshev_approximation_recursive(c, i[0], i[1], i[0]), color='black', marker='+', s=100)
        plt.scatter(i[1], chebyshev_approximation_recursive(c, i[0], i[1], i[1]), color='black', marker='+', s=100)
        handles.append(handle[0])
        legends.append(f'N = {len(c) - 1 }')

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
        handles.append(handle)
        legends.append(f'Root = {np.round(np.real(root/ANGSTROM), 2)} A')
        transformed_root = np.real(root)

    plt.xlabel('r [m]')
    plt.ylabel('F(r) A.U.')
    plt.title('DoCA Function for W-W Cubic Spline Potential')
    plt.plot([0., b], [0., 0.], linestyle='--', color='black')
    plt.legend(handles, legends, loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()
