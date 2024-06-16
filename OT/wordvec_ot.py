# %%
from pathlib import Path

import gensim
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
from mlutils.general.environ import load_env
from mlutils.general.japanize import japanize_matplotlib
from tqdm.auto import trange

japanize_matplotlib()

env = load_env(Path(__file__).parents[1])

# https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz
model_path =  Path(env["DATASETS_ROOT_DIR"]) / "gensim_300vec_ja.gz"
# 5分くらいかかる
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)
# %%
source = torch.stack(
    [
        torch.from_numpy(model["犬"]),
        torch.from_numpy(model["猫"]),
        torch.from_numpy(model["ライオン"]),
        torch.from_numpy(model["鳥"]),
    ]
)
costs = torch.rand(300, 300)
print(source.shape, costs.shape)
# %%
import ot

p = ot.sinkhorn(source[0], source[1], costs, 1, epsilon=0.1, numItermax=100, device="cuda")
print(p.shape)
# %%
