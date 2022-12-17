import numpy as np

def get_analytic_solution(n,m, L, T):
    """
    Inputs: n is the number of rows, or points is x-space.
            m is the number of columns, or points in t-space.
            L is the length of space.
            T is the length of time.
    """
    x = np.linspace(0, L, n) #spacial points
    t = np.linspace(0, T, m) #time points

    u = np.exp(-np.pi**2 * t[np.newaxis,...]) * np.sin(np.pi*x[...,np.newaxis])

    return x,t,u