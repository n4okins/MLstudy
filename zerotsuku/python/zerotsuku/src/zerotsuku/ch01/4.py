# 4. ニューラルネットワークの学習
# %%
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import jax
import jax.nn as jnn
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def one_hot(x, num_classes):
    e = jnp.zeros((x.size, num_classes))
    e = e.at[jnp.arange(x.size), x].set(1)
    return e


class DataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.idx = 0
    
    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.dataset):
            self.idx = 0
            raise StopIteration

        x, t = zip(
            *[self.dataset[i] for i in range(self.idx, self.idx + self.batch_size)]
        )
        self.idx += self.batch_size
        return jnp.array(x), jnp.array(t)


class NetworkBase:
    def __init__(self):
        self.parameters = []

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, x):
        raise NotImplementedError


class Linear(NetworkBase):
    def __init__(self, in_features, out_features):
        rng = jax.random.PRNGKey(0)
        w_key, b_key = jax.random.split(rng)
        self.parameters = [
            jax.random.normal(w_key, (in_features, out_features)),
            jax.random.normal(b_key, (out_features,)),
        ]
    
    @property
    def weight(self):
        return self.parameters[0]
    
    @property
    def bias(self):
        return self.parameters[1]

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(x, self.weight) + self.bias


class ReLU(NetworkBase):
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnn.relu(x)


class Softmax(NetworkBase):
    def forward(self, x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
        return jnn.softmax(x, axis=axis)


class Model(NetworkBase):
    def __init__(self, in_features: int = 4, hidden_features: int = 16, out_features: int = 3):
        w1rng, b1rng = jax.random.split(jax.random.PRNGKey(0))
        w2rng, b2rng = jax.random.split(jax.random.PRNGKey(1))
        self.parameters = {
            "weight1": jax.random.normal(w1rng, (in_features, hidden_features)),
            "bias1": jax.random.normal(b1rng, (hidden_features,)),
            "weight2": jax.random.normal(w2rng, (hidden_features, out_features)),
            "bias2": jax.random.normal(b2rng, (out_features,)),
        }
    
    def forward(self, parameters, x):
        x = jnp.dot(x, parameters["weight1"]) + parameters["bias1"]
        x = jnn.relu(x)
        x = jnp.dot(x, parameters["weight2"]) + parameters["bias2"]
        x = jnn.softmax(x)
        return x

class SGD:
    def __init__(self, params, learning_rate):
        self.params = params
        self.learning_rate = learning_rate

    def step(self, grads):
        self.params["weight1"] -= self.learning_rate * grads["weight1"]
        self.params["bias1"] -= self.learning_rate * grads["bias1"]
        self.params["weight2"] -= self.learning_rate * grads["weight2"]
        self.params["bias2"] -= self.learning_rate * grads["bias2"]
        return self.params


class MSELoss:
    def __call__(self, y, t):
        return jnp.mean((y - t) ** 2)


class CrossEntropyLoss:
    def __call__(self, y, t):
        return -jnp.mean(t * jnp.log(y))


dataset = load_iris()
x_data = dataset.data
t_data = one_hot(dataset.target, 3)
x_train, x_test, t_train, t_test = train_test_split(x_data, t_data, test_size=0.3)

criterion = CrossEntropyLoss()
learning_rate = 0.001
model = Model(4, 100, 3)

def train_step(params: dict, x: jnp.ndarray, t: jnp.ndarray):
    y = model(params, x)
    loss = criterion(y, t)
    return loss

plt.scatter(x_data[:, 0], x_data[:, 1], c=dataset.target)
plt.show()

optimizer = SGD(model.parameters, learning_rate)

batch_size = 32
train_loader = DataLoader(list(zip(x_train, t_train)), batch_size)

for epoch in range(100):
    total_loss = 0
    for x, t in train_loader:
        grads = jax.grad(train_step)(model.parameters, x, t)
        model.parameters = optimizer.step(grads)
        total_loss += train_step(model.parameters, x, t)
    
    print(f"epoch: {epoch + 1}, loss: {total_loss / len(train_loader)}")
# %%