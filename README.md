Heat equation: `u_t = α∇²u`

The rate of change (time derivative) is proportional to how far u is from its neighbor average. Hills sink, bowls fill, and everything relaxes toward the Laplace (equilibrium) solution.

- Time has a direction. Smoothing destroys information, so it is irreversible.
- Only one time derivative and a second spatial one, so it needs one initial condition plus boundary conditions.
- Information spreads instantly everywhere (infinite speed), but with exponentially small effect far away.

Methods of solving 1D heat equation:
- Analytical
    - Separation of Variables (Fourier)
    - Laplace and Fourier Transform
- Numerical
    - Explicit
        - Forward time central space 
        - Runge Kutta
    - Implicit
        - Backward Euler
        - Crank Nicolson

Additional:

Harmonic oscillator (intro)

- https://beltoforion.de/en/harmonic_oscillator/
- https://colab.research.google.com/github/jcabbasi/SCIML2024workshop/blob/main/tut1_oscillator.ipynb#scrollTo=a4b64298