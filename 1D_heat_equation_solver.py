import jax

def laplacian():
    return NotImplemented

def pde_solver(x, initial_cond, nx, dt, alpha, T):
    # Spatial step size
    dx = x[1] - x[0]

    # Adjust dt so we end at timestep T
    nt = T // dt
    dt = T / nt

    # Forward time central space 


    return NotImplemented

def main():
    # Number of points in space
    nx = 100

    # Timestep 
    dt = 0.01

    # Alpha from the heat equation
    alpha = 1

    # The final time 
    T = 1

    # Create a grid of points in 1D
    x = jax.numpy.linspace(0, 1, num=nx)

    # Setting initial condition of u at t = 0
    initial_cond = jax.numpy.sin(jax.numpy.pi * x)

    pde_solver(x, initial_cond, nx, dt, alpha, T) 