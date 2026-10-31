import jax.numpy as jnp
from jax.experimental import sparse
import matplotlib.pyplot as plt

### TODO: optimize the for loop w jit

# One way to compute laplacian is using direct differences of neighboring components 
# Another way (used here) is to multiply matrix L by vector u
# Where L contains positive and negative integers which when dot producted with u
# Produce elements that are sums/differences of u
# It would be a matrix with [[1, -2, 1, 0, 0, ...], [0, 1, -2, 1, 0, ...]...[..., 0, 0, 1, -2, 1]]
# The top and bottom rows start from -2 because Dirichlet conditions set u(0, t) and u(1, t) = 0
def laplacian_operator(n, dx):
    mat_sparse = (sparse.eye(n, k=-1) + -2 * sparse.eye(n, k=0) + sparse.eye(n, k=1)) / dx**2
    return mat_sparse

def ftcs_solver(dx, initial_cond, dt, alpha, T):
    # Adjust dt so we end at timestep T in case T is not divisible by dt
    nt = int(round(T / dt))
    dt = T / nt
    r = alpha * dt / dx**2
    if r > 0.5:
        raise ValueError(f"FTCS unstable: r = alpha*dt/dx^2 = {r:.2f} > 0.5")
    u = initial_cond
    # Number of points considered for laplacian
    n = len(initial_cond)
    L = laplacian_operator(n, dx)
    for _ in range(nt):
        # Forward time central space 
        u = u + alpha * dt * L @ u 
        u = u.at[0].set(initial_cond[0])
        u = u.at[-1].set(initial_cond[-1])

    return u

def exact(x, t, alpha):
    return jnp.exp(-alpha * jnp.pi**2 * t) * jnp.sin(jnp.pi * x)

def err(x, dx, dt, initial_cond, alpha, T):
    x, u = ftcs_solver(dx, initial_cond, dt, alpha, T)
    return jnp.max(jnp.abs(u - exact(x, T, alpha)))

def main():
    # Number of intervals in space
    nx = 100

    # Timestep 
    dt = 0.0001

    # NB! Stability requires that aplha * dt / h^2 <= 1/2 
    # Alpha from the heat equation
    alpha = 0.1

    # The final time 
    T = 1

    # Create a grid of points in 1D 
    x = jnp.linspace(0, 1, num=(nx+1))

    # Spatial step size
    dx = x[1] - x[0]
    u0 = x.copy()

    # Dirichlet conditions with u(0, t) and u(1, t) being 0
    u0 = u0.at[0].set(0).at[-1].set(0)

    # Setting initial condition of u, only for internal space 
    # Neighbors for grad computation are not defined at endpoints, instead they are defined by Dirichlet condition (boundary)
    # NB! The function with which intermediate points are set initially 
    # Is respected with the values at endpoints sin(0) = sin(pi*x) = 0
    u0 = u0.at[1:-1].set(jnp.sin(jnp.pi * x[1:-1]))

    u = ftcs_solver(dx, u0, dt, alpha, T)

    # Use analytical solution for the given initial conditions
    u_exact = exact(x, T, alpha)
    error = u - u_exact
    print(f"max error at T={T}: {jnp.max(jnp.abs(error)):.3e}")

    # Plot the estimated u, analytical u and error for t=T
    x_fine = jnp.linspace(0.0, 1.0, 400)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(x, u0, "--", color="gray", label="initial (t=0)")
    ax1.plot(x_fine, exact(x_fine, T, alpha), "k-", label=f"exact (t={T})")
    ax1.plot(x, u, "o", ms=4, label="FTCS")
    ax1.set_ylabel("u")
    ax1.legend()
    ax2.plot(x, error, "r.-")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("error")
    plt.tight_layout()
    plt.show()
 
 
if __name__ == "__main__":
    main()