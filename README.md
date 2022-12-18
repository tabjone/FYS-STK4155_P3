# FYS-STK4155_P3

The code: 

analytic_solution.py has the function get_analytic_solution(n,m, L, T) that gives the analytic solution for the given boundary conditions in the report.

Inputs: n is the number of rows, or points is x-space.
        m is the number of columns, or points in t-space.
        L is the length of space.
        T is the length of time.


The explicit solver can be found in explisit_solver.py. This has a class solve_heateq1D that initializes with (f, L, T, Δx, alpha=1/4)). 

Inputs: f is intital values u(x,0).
        L is the length of space.
        T is the length of time.
        Δx is spacial step size.
        alpha is the stability criterion alpha<=1/2.

And can be solved by calling the solve() method. This method returns x,t,u, where x is a linspace of spacial steps, t is a linspace of time steps, and u is the solution in space and time, with space in the first dimention and time in the second.


The neural network can be found in NN_solver.py. It initializes with (activation_, optimizer_, n_hidden).

Inputs: activation_ is the activation function 'relu' 'sigmoid' ect
        optimizer_ can be 'Adam', 'rmsprop' ect
        n_hidden is the number of hidden layers minus the first hidden layer

Then you can fit a model using fit_model(X, y, batch_size_, epochs_)
Inputs: X is a 2D array with (x,t) values
        Returns: 1D array y with the corresponding predicted values to X

The X-array and y-array needs to be flattened before sending it to the neural network. There is a function flat_analytic_solution(x,t,u) that does this, where the x,t,u are the outputs of the analytic solution, and this returns X,y which will be used by the NN.

Then predict_model(X, batch_size_) can be used to get a prediction by the neural network.
Inputs: X is a 2D array with (x,t) values
        Returns: 1D array y with the corresponding predicted values to X
        
test_explisit_vs_analytic.py and NN_pde.ipynb are used to produce the plots used in the report.
