import jax
import jax.numpy as jnp
from jax.experimental import sparse
import matplotlib.pyplot as plt
import functools as ft
import itertools as it
import hydra
from omegaconf import DictConfig as Config
from typing import Iterator

########## DATASET ##########

# Use exact to sample the data across different x and t
def exact(x, t, alpha):
    return jnp.exp(-alpha * jnp.pi**2 * t) * jnp.sin(jnp.pi * x)

def get_dataset(key, config):
    n = config.batch_size
    while True:
        yield {
            # collocation
            "x_f":  jax.random.uniform(next(key), (n,)),                         
            "t_f":  jax.random.uniform(next(key), (n,), maxval=config.t_max),
            # initial condition for x at t=0
            "x_ic": jax.random.uniform(next(key), (n,)),                     
            # x = 0 and x = 1  at all t  
            "t_b":  jax.random.uniform(next(key), (n,), maxval=config.t_max),
        }

########### MODEL ###########

# Model parsameter initialization
def init_param_state(key, config: Config) -> list:
    zero_init = jax.nn.initializers.zeros
    glorot_init = jax.nn.initializers.glorot_normal()
    dtype = jnp.dtype(config.dtype)

    params = []
    for fan_in, fan_out in zip(config.layer_nodes[:-1], config.layer_nodes[1:]):
        params.append({
            # using xe initialization with tanh activation
            "w": glorot_init(next(key), (fan_in, fan_out), dtype),
            "b": zero_init(next(key), (fan_out,), dtype),
        })
    return params

# u_net function that returns the result of one pass through the model
# to make later grad computation in physics loss (residual) easier
# grad requires the output of a function to be a scalar, vmap will be used for batching loss computation
def u_net(params, x, t):
    h = jnp.stack([x, t])
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["w"] + layer["b"])
    return (h @ params[-1]["w"] + params[-1]["b"])[0]

# Forward step of the model
def model_apply(params, inputs):                       # inputs: (N, 2)
    return jax.vmap(lambda xt: u_net(params, xt[0], xt[1]))(inputs)[:, None]

##### TRAINING #####

# A class that will store the results of the previouysly computed 
# metrics to show them when called instead of the current ones
class RecordWriter:
    prev = None

    def __call__(self, metrics, every=1):
        prev, self.prev = self.prev, metrics
        # print only every few number of steps
        if prev is not None and prev["step"] % every == 0:
            print(*(f"{k}: {float(v):.4g}" for k, v in prev.items()), sep="\t")

    # Flush the last measurement even it the step is not increment of every
    def flush(self):
        if self.prev is not None:
            print(*(f"{k}: {float(v):.4g}" for k, v in self.prev.items()), sep="\t")

def residual(params, x, t, alpha):
    u_t  = jax.grad(u_net, argnums=2)(params, x, t)
    u_xx = jax.grad(jax.grad(u_net, argnums=1), argnums=1)(params, x, t)
    return u_t - alpha * u_xx

def loss_fn(params, batch, alpha, w):
    v = jax.vmap
    # the residual term from pde
    r = v(residual, (None, 0, 0, None))(params, batch["x_f"], batch["t_f"], alpha)
    # initial condition term for t = 0
    u_ic = v(u_net, (None, 0, None))(params, batch["x_ic"], 0.0)
    # boundary conditions terms
    # boundary x = 0 
    u_b0 = v(u_net, (None, None, 0))(params, 0.0, batch["t_b"])
    # boundary x = 1
    u_b1 = v(u_net, (None, None, 0))(params, 1.0, batch["t_b"])
    l_pde = jnp.mean(r**2)
    l_ic = jnp.mean((u_ic - jnp.sin(jnp.pi * batch["x_ic"]))**2)
    l_bc = jnp.mean(u_b0**2) + jnp.mean(u_b1**2)
    total = w["pde"] * l_pde + w["ic"] * l_ic + w["bc"] * l_bc
    return total, {"pde": l_pde, "ic": l_ic, "bc": l_bc}

import optax

# Wrap training step outside jax jit, bcs config of type DictConfig cant be traced by jit
def make_train_step(config):
    opt = optax.adam(config.learning_rate)
    w = {"pde": config.w_pde, "ic": config.w_ic, "bc": config.w_bc}

    @jax.jit
    def train_step(params, opt_state, batch):
        (loss, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, config.alpha, w)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, {"loss": loss, **parts}
    return opt, train_step

def train_loop(config):
    # Initialize a random key from config seed
    root = jax.random.key(config.seed)
    # Separate the data and param_init keys
    init_root, data_root = jax.random.split(root)
    # Create a generator that takes in the root key infinite sequence of integers it.count() 
    # It raps around jax.random_fold_in which derives a random key form the root_key and i
    # key is now a generator that returns a random key on each next(key)
    init_key = map(ft.partial(jax.random.fold_in, init_root), it.count())
    data_key = map(ft.partial(jax.random.fold_in, data_root), it.count())

    params = init_param_state(init_key, config)
    opt, train_step = make_train_step(config)
    opt_state = opt.init(params)
    batches = get_dataset(data_key, config)

    rw = RecordWriter()
    for step in range(config.num_train_steps):
        params, opt_state, metrics = train_step(params, opt_state, next(batches))
        if step % 500 == 0:
            rw({"step": step} | metrics, every=500)
    return params

# Decorator that loads the config
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    params = train_loop(config)
    x = jnp.linspace(0, 1, 200)
    import time
    fig, ax = plt.subplots()
    for t0 in (0.0, 0.25, 0.5, config.t_max):
        # first run takes a bit longer because of compilation in jax jit, faster for following t0
        start = time.time()
        u_pred = jax.vmap(u_net, (None, 0, None))(params, x, t0)
        end = time.time() - start
        u_true = exact(x, t0, config.alpha)
        print(f"t={t0}: max err {jnp.max(jnp.abs(u_pred - u_true)):.2e}, time {end:.2e}")
        ax.plot(x, u_true, "k-", lw=1)
        ax.plot(x, u_pred, "--", label=f"t={t0}")
    ax.legend(); plt.show()
 
if __name__ == "__main__":
    main()