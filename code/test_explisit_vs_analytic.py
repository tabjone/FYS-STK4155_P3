import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from explisit_solver import solve_heateq1D
from analytic_solution import get_analytic_solution

def MSE(target, predicted):
    """Mean squared error"""
    return np.mean(np.square(target - predicted), axis=0)

if __name__ == '__main__':
    L = 1 #length of rod
    T = 1 #length of time
    
    sns.set_theme()
    fig, ax = plt.subplots()
    plt.rc('axes', labelsize=14)
    
    '''
    Testing different alpha and Δx combinations and plotting mean squared error
    '''
    f = lambda x : np.sin(np.pi*x) #u(x,0) boundary

    for Δx in [0.1, 0.01]:
        for alpha in [1/4, 1/2]:
            #setting up numerical solver
            solver = solve_heateq1D(f, L, T, Δx, alpha)
            #solving for all time and space steps
            x,t,v = solver.solve()
            
            #getting analytical solution
            _, _, u = get_analytic_solution(solver.n, solver.m, L, T)
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
    #getting solution
    x,t,v = solver.solve()

    #getting analytical solution
    _, _, u = get_analytic_solution(solver.n, solver.m, L, T)

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


