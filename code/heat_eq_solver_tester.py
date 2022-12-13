import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from heat_eq_explisit_solver import solve_heateq1D

def f(x):
    """u(x,0) conditions"""
    return np.sin(np.pi * x)

def analytical_solution(x, t):
    """Analytical solution for u(x,t) given that f(x)=sin(pi*x) and u(0,t)=u(1,t)=0"""
    return np.exp(-np.pi**2 * t) * np.sin(np.pi*x)

def setup_analytical_solution_matrix(u, n, m, Δx, Δt):
    """
    Arguments: u is the matrix to hold analytical solution, n is spacial steps, m is time steps,
    Δx is spacial step-size and Δt is time step-size
    """
    for i in range(n):
        for j in range(m):
            x = i*Δx
            t = j*Δt
            u[i,j] = analytical_solution(x,t)

def MSE(target, predicted):
    """Mean squared error"""
    return np.mean(np.square(target - predicted), axis=0)


if __name__ == '__main__':
    #length of rod
    L = 1
    #length of time
    T = 1
    
    sns.set_theme()
    fig, ax = plt.subplots()
    plt.rc('axes', labelsize=14)
    
    '''
    Testing different alpha and Δx combinations and plotting mean squared error
    '''
    for Δx in [0.1, 0.01]:
        for alpha in [1/4, 1/2]:
            #setting up numerical solver
            solver = solve_heateq1D(f, L, T, Δx, alpha)
            #matrix to hold solution
            v = np.zeros((solver.n,solver.m))
            #solving for all time and space steps
            solver.solve(v)

            #setting up analytical solution
            u = np.zeros((solver.n,solver.m))
            setup_analytical_solution_matrix(u, solver.n, solver.m, solver.Δx, solver.Δt)
            sns.lineplot(x=solver.t, y=MSE(u,v), linewidth=1.5, label=f"α={alpha}, Δx={Δx}")
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean squared error')
    ax.set_yscale('log')
    ax.legend()
    plt.show()
    

    '''
    Looking at different timesteps for the numerical solution
    '''
    Δx = 0.1
    alpha = 1/2
    #setting up solver
    solver = solve_heateq1D(f, L, T, Δx, alpha)
    #creating array for solution
    v = np.zeros((solver.n, solver.m))
    #solving
    solver.solve(v)
    #setting up analytical solution
    u = np.zeros((solver.n,solver.m))
    setup_analytical_solution_matrix(u, solver.n, solver.m, solver.Δx, solver.Δt)

    #calculating mean squared error and plotting solution for early times and late time = 4/5*T
    mse = MSE(u, v)
    early = np.argmax(mse)
    late = int(4/5 * solver.m)

    fig, ax = plt.subplots()

    ax.plot(solver.x, v[:, early],label=f't={early*solver.Δt:.2f}')
    ax.plot(solver.x, v[:, late],label=f't={late*solver.Δt:.2f}')

    ax.set_xlabel('x')
    ax.set_ylabel('Numerically calculated solution u(x,t)')
    ax.legend()
    plt.show()


