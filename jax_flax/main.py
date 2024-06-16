# %%
import flax
import jax
import jax.numpy as jnp
import torcivision.datasets as td
from flax import linen as nn

rng = jax.random.PRNGKey(0)

model = nn.Dense(features=5)
key1, key2 = jax.random.split(rng)
x = jax.random.normal(key1, (10, ))
params = model.init(key2, x)

y = model.apply(params, x)
print(y)
# %%