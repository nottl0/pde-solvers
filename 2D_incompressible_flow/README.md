The equations describing 2D (incompressible) fluid flow are characterised by 2 equations: the continuity equation (conservation of mass) and the Navier-Stokes equation (conservation of momentum).

### The continuity equation: 
$$
\begin{aligned}
\nabla \cdot \textbf{u} = 0
\end{aligned}
$$

The $\textbf{u}$ has 2 components (u(x, y, t), v(x, y, t)), aka the x and y components of the vector. They are defined per position (x, y) at time t.

### The Navier Stokes equation: 
$$
\begin{aligned}
\textbf{u}_t + (\textbf{u} \cdot \nabla) \textbf{u} = -\nabla p + v \Delta \textbf{u} + f
\end{aligned}
$$


### The goal

We want to be able to identify the pressure and flow at each position (x, y) and time t, knowing only the initial conditions: flow (u, v) at each position (x, y) at time $t=0$; and boundary conditions flow (u, v) at all times t for the edges of the spatial domain (inlet, outlet and walls (u=v=0 at the walls because of no-slip condition)). 

### Methods of solving 1D heat equation:

- Analytical

- Numerical
    - Explicit
       
    - Implicit
       
