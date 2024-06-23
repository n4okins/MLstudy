# %%
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
from tqdm.auto import trange

from mlutils.general.japanize import japanize_matplotlib
japanize_matplotlib()


# dog, cat, lion, bird
# n, m = 300, 300
# a = torch.rand(n)
# b = torch.sin(torch.linspace(0, torch.pi, m))
# c = torch.randint(0, 100, (n, m)) / 10
# c = c * (1 - torch.eye(n, m))

a = torch.tensor([0.2, 0.5, 0.2, 0.1])
b = torch.tensor([0.3, 0.3, 0.4, 0.0])
n, m = len(a), len(b)
c = torch.tensor(
    [
        [0, 2, 2, 2],
        [2, 0, 1, 2],
        [2, 1, 0, 2],
        [2, 2, 2, 0],
    ]
)

ylim = (min(a.min(), b.min()), max(a.max(), b.max()) * 1.05)

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].bar(range(n), a)
ax[0].set_title("Source")
ax[0].set_ylim(ylim)
ax[1].bar(range(m), b)
ax[1].set_title("Target")
ax[1].set_ylim(ylim)
plt.show()

# %%
fig, ax = plt.subplots(1, 4, figsize=(18, 5))
ax[0].set_title("Source")
ax[0].set_ylim(ylim)
ax[1].set_title("Transpoted")
ax[1].set_ylim(ylim)
ax[2].set_title("Target")
ax[2].set_ylim(ylim)

epsilon = 0.1
max_iter = 10
device = "cuda"

a, b, c = a.to(device), b.to(device), c.to(device)
K = torch.exp(-c / epsilon).to(device)
u = torch.ones(n).to(device)

def one_frame(i):
    global K, u
    v = b / (K.T @ u).to(device)
    u = a / (K @ v)
    P = u.reshape(n, 1) * K * v.reshape(1, m)
    fig.suptitle(f"Iteration: {i}")
    ax[0].cla()
    ax[0].bar(range(n), a.cpu().numpy()) 
    ax[1].cla()
    ax[1].bar(range(m), P.cpu().detach().sum(0).numpy())
    ax[2].cla()
    ax[2].bar(range(m), b.cpu().numpy())
    ax[3].cla()
    ax[3].pcolor(P.cpu().detach().numpy(), cmap="Blues")
    ax[0].set_title("Source")
    ax[0].set_ylim(ylim)
    ax[1].set_title("Transpoted")
    ax[1].set_ylim(ylim)
    ax[2].set_title("Target")
    ax[2].set_ylim(ylim)
    ax[3].set_title(f"Cost: {(P * c).sum().cpu().detach().numpy():.2f}")


anim = FuncAnimation(fig, one_frame, frames=trange(max_iter), interval=100)
anim.save("sinkhorn_2.gif", writer="imagemagick")

# %%

