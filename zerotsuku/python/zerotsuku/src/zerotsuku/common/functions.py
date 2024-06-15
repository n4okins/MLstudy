import jax
import jax.numpy as jnp


def linear(x: jnp.ndarray, W: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.dot(x, W) + b

def softmax(a: jnp.ndarray) -> jnp.ndarray:
    c = jnp.max(a)
    exp_a = jnp.exp(a - c)
    sum_exp_a = jnp.sum(exp_a)
    return exp_a / sum_exp_a

def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(0, x)