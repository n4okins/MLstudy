# 3. ニューラルネットワーク
# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1 / (1 + jnp.exp(-x))


def step_function(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.array(x > 0, dtype=jnp.int32)


def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(0, x)


def identity_function(x: jnp.ndarray) -> jnp.ndarray:
    return x



def init_network() -> dict:
    network = {}
    network["W1"] = jnp.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = jnp.array([0.1, 0.2, 0.3])
    network["W2"] = jnp.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = jnp.array([0.1, 0.2])
    network["W3"] = jnp.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = jnp.array([0.1, 0.2])
    return network


def forward_network(network: dict, x: jnp.ndarray) -> jnp.ndarray:
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = jnp.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = jnp.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = jnp.dot(z2, W3) + b3
    y = identity_function(a3)
    return y


def softmax(a: jnp.ndarray) -> jnp.ndarray:
    c = jnp.max(a)
    exp_a = jnp.exp(a - c)
    sum_exp_a = jnp.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def img_show(img: jnp.ndarray) -> None:
    pil_img = Image.frombuffer("L", (28, 28), img.astype(jnp.uint8).tobytes())
    plt.imshow(pil_img, cmap="gray")
    plt.show()



if __name__ == "__main__":
    mnist_train = datasets.MNIST(
        "data",
        train=True,
        download=True,
    )
    img, label = mnist_train[0]
    img = jnp.frombuffer(img.tobytes(), dtype=jnp.uint8).reshape(28, 28)
    img_show(img)
# %%