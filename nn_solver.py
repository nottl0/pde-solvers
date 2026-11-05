import jax
import jax.numpy as jnp
from jax.experimental import sparse
import matplotlib.pyplot as plt
import functools as ft
import itertools as it
from collections import defaultdict
from dotted_dict import DottedDict as dot_dict
import hydra
from omegaconf import DictConfig as Config
from typing import Iterator

########## DATASET ##########

# Use exact to sample the data across different x and t
def exact(x, t, alpha):
    return jnp.exp(-alpha * jnp.pi**2 * t) * jnp.sin(jnp.pi * x)


def get_dataset(config: Config) -> Iterator[dict[str, jax.Array]]:
    # Sample points in the domain
    x = jax.random.uniform(
        config.data_key,
        shape=(config.batch_size, 1),
        minval=0.0,
        maxval=1.0,
    )

    t = jax.random.uniform(
        jax.random.fold_in(config.data_key, 1),
        shape=(config.batch_size, 1),
        minval=0.0,
        maxval=config.t_max,
    )

    u = exact(x, t, config.alpha)

    yield {
        "x": x,
        "t": t,
        "u": u,
    }


def get_dataset_on_device(config: Config) -> Iterator[dict[str, jax.Array]]:
    dataset = get_dataset(config)
    sharding = jax.P(config.mesh_axis_names)

    return map(
        ft.partial(jax.make_array_from_process_local_data, sharding),
        dataset,
    )

########### MODEL ###########

# Initialize training parameters: model parameters + optimizer states of those parameters
@jax.jit
def init_train_state(config: Config) -> dot_dict:
  train_state = dot_dict()
  train_state.params = init_param_state(config)
  return train_state

# Model parameter initialization
def init_param_state(config: Config) -> dot_dict:
    # Initialize a random key from config seed
    root_key = jax.random.key(config.seed)

    # Create a generator that takes in the root key infinite sequence of integers it.count() 
    # It raps around jax.random_fold_in which derives a random key form the root_key and i
    # key is now a generator that returns a random key on each next(key)
    key = map(ft.partial(jax.random.fold_in, root_key), it.count())
    zero_init = jax.nn.initializers.constant(0.0)
    he_init = jax.nn.initializers.he_normal(1, 1)

    # Type of the model parameters  
    dtype = config.dtype

    # Inintialize an empty dictionary of weights to store weights of each layer, that will be stored in params.layers
    params = dot_dict(
        layers=dot_dict(),
    )

    for layer in range(config.num_layers):

        params.layers[layer] = dot_dict(
            w=he_init(next(key), (config.layer_nodes[layer], config.layer_nodes[layer+1]), dtype),
            b=zero_init(next(key), (config.layer_nodes[layer], config.layer_nodes[layer+1]), dtype)
        )

    return params

# Forward step of the model
def model_apply(config: Config, params: dot_dict, inputs: jax.Array) -> jax.Array:
  out = inputs

  for layer in range(config.num_layers):
    block = params.layers[layer]
    #sharding?
    out = jnp.einsum("bi,io->bo", out, block.w) + block.b

  logits = out
  return logits

##### TRAINING #####

# A class that will store the results of the previouysly computed metrics to show them when called instead of the current ones
# 
class RecordWriter:
  prev_metrics = None

  def __call__(self, cur_metrics: dict):
    self.prev_metrics, log_metrics = cur_metrics, self.prev_metrics
    if log_metrics is None:
      return
    print(*it.starmap("{}: {}".format, log_metrics.items()), sep="\t")

def sgd_update(config: Config, param: jax.Ref, grad: jax.Array):
  param[...] -= config.learning_rate * grad + config.eps

@jax.jit
def train_step(config: Config, train_state: dot_dict, batch: dict) -> dict:
  def loss_fn(params):
    logits = model_apply(config, params, batch["inputs"])
    labels = batch["targets"]
    return -((labels - logits)**2).mean()

  params = jax.tree.map(jax.ref.get, train_state.params)
  loss, grad = jax.value_and_grad(loss_fn)(params)
  jax.tree.map(ft.partial(sgd_update, config), train_state.params, grad)
  metrics = {"train_loss": loss}
  return metrics

def train_loop(config: Config):
  record_writer = RecordWriter()
  train_state = init_train_state(config)
  train_state = jax.tree.map(jax.ref.new_ref, train_state)
  batch = iter(get_dataset_on_device(config))
  for step in range(config.num_train_steps):
    metrics = train_step(config, train_state, next(batch))
    record_writer({"step": step} | metrics)

# Decorator that loads the config
@hydra.main(version_base=None, config_path="configs", config_name="config")
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