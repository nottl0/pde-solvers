### Heat equation: 

```math
u_t = \alpha \nabla^2 u
```
The rate of change (time derivative) is proportional to how far u is from its neighbor average. Hills sink, bowls fill, and everything relaxes toward the Laplace (equilibrium) solution.

- Time has a direction. Smoothing destroys information, so it is irreversible.
- Only one time derivative and a second spatial one, so it needs one initial condition plus boundary conditions.
- Information spreads instantly everywhere (infinite speed), but with exponentially small effect far away.

### The goal

With the problems that we are trying to solve using the heat equation we often know the initial conditions of the system at time $t=0$ for all positions in space, and the boundary conditions (values of u) at all times at positions $x=0$ and $x=1$. We need our system to be able to derive the values of $u$ in any location $x$ at any given moment $t$.

### Methods of solving 1D heat equation:

- Analytical
    - Separation of Variables (Fourier)
    - Laplace and Fourier Transform
- Numerical
    - Explicit
        - Forward time centered space (1D_heat_equation_solver.py)
        - Runge Kutta
    - Implicit
        - Backward Euler
        - Crank Nicolson

### Forward Time Centered Space:

    FCTS approcximates the gradient operations in the heat equation with finite differences. Using 

```math
u_t = \frac{u(t+dt)-u(t)}{dt} \\
u_{xx} = \frac{u(t+dt)-2u(t)+u(t-dt)}{dt}
```
   
    We can derive the update rule by combining the equations for $u_t$ and $u_{xx}$ with the heat equation to get the update rule of FCTS: 

```math
u_t = \alpha \nabla^2 u\\
u_t = \frac{u(t+dt)-u(t)}{dt}\\
u_{xx} = \frac{u(x+dx)-2u(x)+u(x-dx)}{dx}\\
\frac{u(t+dt)-u(t)}{dt} = \alpha \times \frac{u(x+dx)-2u(x)+u(x-dx)}{dx}\\
u(t+dt) = u(t) + dt \times \alpha \times \frac{u(x+dx)-2u(x)+u(x-dx)}{dx}
```

    This equation specifies how the value of $u$ changes for each point of space for a very small timestep $dt$. As we can see from the last equation the falue of $u$ for a constant $x$ in the future is given by the current value of u as well as the weighted values of $u$ at its nearest neighbors.

#### PINN Solution:

Instead of using the explicit formulation of $u(t+dt)$ to compute the values at each specific point, we could also train a neural network that would take parameters $x$ and $t$ and output the value of $u$ for the given inputs. However, we need to remember that apart from the heat equation itself, only the initial and boundary conditions are known. So we need to train our model to respect the head equation, such that the predictions ($\hat{u}$) for previously unseen combinations of $x, t$ are still valid.

PINNs incorporate several loss terms to do that. In the formulas below $L_{ic}$ is the loss term that ensures that the initial conditions are satisfied by the predicted $\hat{u}$, $L_{bc}$ is the boundary conditions' loss that ensures that the boundary conditions are met, and $L_{residual}$ is the loss that represents the condition that the heat equation must be satisfied.

```math
L = w_{ic} L_{ic} + w_{bc} L_{bc} + w_{residual} L_{residual} \\
L_{ic} = \sum^{N}_{i=0}[\hat{u}(x_i, t=0) - u(x_i, t=0)]^2\\
L_{bc} = \sum^{N}_{i=0}[\hat{u}(x = 1, t_i) - u(x = 1, t_i)]^2\\
L_{bc} = \sum^{N}_{i=0}[\hat{u}(x = 0, t_i) - u(x = 0, t_i)]^2\\
L_{residual} = \sum^{N}_{i=0}[\hat{u(x_i, x_i)}_t - alpha * \hat{u(x_i, t_i)}_{xx}]^2\\
```
   
#### Additional:

Harmonic oscillator (intro)

- https://beltoforion.de/en/harmonic_oscillator/
- https://colab.research.google.com/github/jcabbasi/SCIML2024workshop/blob/main/tut1_oscillator.ipynb#scrollTo=a4b64298

Jax Training Cookbook

- https://docs.jax.dev/en/latest/the-training-cookbook.html