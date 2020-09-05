import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

'''
This set of functions was designed to test some methods from Boyd's wonderful
paper, Finding the Zeros of a Univariate Equation: Proxy Rootfinders, Chebyshev
Interpolation, and the Companion Matrix (2013).

It was not designed for general purpose use, but may be enlightening to examine.

Note that in the paper, there is a typo in Eq. B.2. (-1)a[j - 1]/2/a[N] should
instead be (-1)a[k - 1]/2/a[N]. This is corrected in Boyd's text, Solving
Transcendental Equations (2014).
'''

def chebyshev_points(a, b, N):
    '''
    Returns the Chebyshev interpolation points (Lobatto grid) for an interval [a,b]

    Eq. A.1 in Boyd (2013)
    '''
    k = np.arange(0, N + 1)
    return (b - a)/2.*np.cos(np.pi*k/N) + (b+a)/2.

def F(r):
    return (r**12 - 1 + 2*r**6 - r**2)*np.exp(-r)

def dF(r):
    return np.exp(-r)*(-r**12 + 12*r**11 - 2*r**6 + 12*r**5 + r**2 - 2*r + 1)

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
    '''
    I_jk = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        for k in range(N + 1):
            I_jk[j,k] = 2/p(j, N)/p(k, N)/N*np.cos(j*np.pi*k/N)
    return I_jk

def chebyshev_coefficients(F, a, b, N):
    xk = chebyshev_points(a, b, N)
    I_jk = interpolation_matrix(N)
    a_j = I_jk.dot(F(xk))
    return a_j

def chebyshev_approximation_recursive(a_j, a, b, x):
    '''
    Clenshaw-Curtis recurrence method for approximating a function f(x) on the
    interval [a, b] at x given its Chebyshev series coefficents, a_j.

    Eqs. 3.81-3.84 in Boyd (2014)

    '''
    N = len(a_j) - 1
    j = np.arange(0, N + 1)

    #Clenshaw-Curtis recurrence relations
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

def delta(j,k):
    '''
    Kronecker delta
    '''
    if j == k:
        return 1
    else:
        return 0

def companion_matrix(a_j):
    '''
    Chebyshev-Frobenius companion matrix.

    Eq. B.2 in Boyd (2013)
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

def chebyshev_adaptive_approximation_coefficients(F, a, b, N0, epsilon, maxiter=10):
    '''
    Adaptive Chebyshev approximation, which starts from degree N0 and doubles
    the degree each iteration.
    '''
    N = N0
    a_0 = chebyshev_coefficients(F, a, b, N)

    for i in range(maxiter):
        a_1 = chebyshev_coefficients(F, a, b, 2*N)
        delta = np.append(a_0, np.zeros(N)) - a_1
        error = np.sum(np.abs(delta))
        print(i, error)
        if error < epsilon:
            return a_1
        if a_1[-1] <= 0:
            print('Warning: failure to converge - coefficient == 0')
            return a_0
        a_0 = a_1
        N = 2*N
    return a_1

def is_root_spurious(F, dF, x0, threshold=1E-6):
    '''
    Returns true if the Newton correction for a possible root x0 is greater than
    a threshold, which indicates that x0 is not a root of F.
    '''
    delta = np.abs(F(x0)/dF(x0))
    return delta > threshold

def main():
    a = 0
    b = 2
    a_j = chebyshev_adaptive_approximation_coefficients(F, a, b, 2, 0.1, maxiter=10)

    x = np.linspace(a, b, 1000)
    plt.plot(x, F(x))
    plt.plot(x, [chebyshev_approximation_recursive(a_j, a, b, x_) for x_ in x], linestyle='--')
    plt.show()

    A = companion_matrix(a_j)
    eigenvalues, eigenvectors = np.linalg.eig(A)

    roots = [(b - a)/2*eigenvalue + (b + a)/2 for eigenvalue in eigenvalues[np.isreal(eigenvalues)] if (np.abs(eigenvalue) < 1) and not is_root_spurious(F, dF, (b - a)/2*eigenvalue + (b + a)/2)]
    print(f'CPR roots: {roots}')
    print(f'fsolve root: {fsolve(F, x0=1)}')

if __name__ == '__main__':
    main()
