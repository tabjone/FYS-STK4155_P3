import numpy as np
import matplotlib.pyplot as plt

class solve_heateq1D:
    def __init__(self, f, L, T, Δx, alpha=1/4):
        """
        Arguments: f is intital values u(x,0). L is length of space, T is lenght of time,
        Δx is spacial step size and alpha is the stability criterion alpha<=1/2
        """
        #Von Neumann stability criterion
        assert alpha <= 1/2, f"α={alpha:.1f}, must be samller than 1/2."

        self.L = L
        self.T = T
        self.f = f
        self.Δx = Δx

        #number of spacial steps
        self.n = int(L/Δx) + 1
        
        #calculating step size in time from stability criterion equation
        self.Δt = Δx**2 * alpha
        #number of time steps
        self.m = int(T/self.Δt) + 1

        #diagonal elements
        self.a = alpha
        self.b = 1 - 2 * alpha

        #array of time points
        self.t = np.linspace(0, T, self.m)
        #array of spacial points
        self.x = np.linspace(0,L,self.n)

    def set_initial_conditions(self, v):
        for i in range(v.shape[0]):
            v[i, 0] = self.f(i * self.Δx)

    def set_boundary_conditions(self, v, j):
        """spacial boundary conditions"""
        v[0,j] = 0
        v[self.n-1,j] = 0

    def forward_step(self, v, j):
        """
        Arguments: v is solution matrix, a,b is diagonal elements, n is the number of spacial steps
        and j is the time step to solve for
        """
        #iteration over spacial steps
        for i in range(1, self.n-1):
            v[i,j] = self.a * v[i-1,j-1] + self.b * v[i,j-1] + self.a * v[i+1,j-1]
        #then set boundary
        self.set_boundary_conditions(v, j)

    def solve(self, v):
        """
        Arguments: v is the solution matrix, a,b is the diagonal elements, n is the number of spacial steps
        and m is the number of time steps
        """
        #setting initial conditions
        self.set_initial_conditions(v)
        self.set_boundary_conditions(v, 0)
        #solving in time
        for j in range(1, self.m):
            self.forward_step(v, j)