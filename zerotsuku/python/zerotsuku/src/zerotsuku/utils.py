import jax.numpy as jnp
import matplotlib.pyplot as plt
from PIL import Image


def img_show(img: jnp.ndarray) -> None:
    shape = img.shape
    if len(shape) == 3:
        img = img.transpose(1, 2, 0)

    pil_img = Image.frombuffer("L", shape, img.astype(jnp.uint8).tobytes())
    plt.imshow(pil_img, cmap="gray" if len(shape) == 2 else None)
    plt.show()


def to_one_hot(x: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    e = jnp.zeros((x.size, num_classes))
    e = e.at[jnp.arange(x.size), x].set(1)
    return e

