# %%
from transformers import BertJapaneseTokenizer, BertModel
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm.auto import trange
from mlutils.general.japanize import japanize_matplotlib
japanize_matplotlib()


class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)
    

def cos_sim(tensor1: torch.Tensor, tensor2: torch.Tensor):
    return torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=0)


MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"  # <- v2です。
model = SentenceBertJapanese(MODEL_NAME)

sentences = [
    "あの日見た花の名前を僕達はまだ知らない。",
    "あの日見た木の名前を僕達はまだ知らない。",
    "あの日見た花の？？を僕達はまだ知らない。",
    "名前",
]

emb = model.encode(sentences)
print(cos_sim(emb[0], emb[1]))

# %%

# sentences = [
#     "あの日見た花の名前を僕達はまだ知らない。",
#     "あの日見た木の名前を僕達はまだ知らない。",
#     "あの日見た花の名前を私達はまだ知らない。",
#     "あの時見た花の名前を私達はまだ知らない。",
# ]
# for sentence in sentences:
#     ss = [sentence[:i] for i in range(1, len(sentence)+1)]
#     sentence_embeddings = model.encode(ss, batch_size=8)
#     sentence_embeddings = sentence_embeddings.clone() / sentence_embeddings[0]

#     fig, axes = plt.subplots(3, 1, figsize=(9, 9))
#     ylim = (sentence_embeddings.min(), sentence_embeddings.max())
#     axes[0].set_title("Start")
#     # axes[0].set_ylim(ylim)
#     axes[0].axis("off")
#     axes[1].set_title("Middle")
#     # axes[1].set_ylim(ylim)
#     axes[1].axis("off")
#     axes[2].set_title("End")
#     # axes[2].set_ylim(ylim)
#     axes[2].axis("off")

#     fig.suptitle(f"{ss[0]}")

#     axes[0].pcolor(sentence_embeddings[0].reshape(16, -1), cmap="viridis")
#     axes[1].pcolor(sentence_embeddings[0].reshape(16, -1), cmap="viridis")
#     axes[2].pcolor(sentence_embeddings[-1].reshape(16, -1), cmap="viridis")

#     # axes[0].bar(range(768), sentence_embeddings[0])
#     # axes[1].bar(range(768), sentence_embeddings[0])
#     # axes[2].bar(range(768), sentence_embeddings[-1])

#     def one_frame(i):
#         emb = sentence_embeddings[i]
#         fig.suptitle(f"{ss[i]}")
#         axes[1].cla()
#         axes[1].pcolor(emb.reshape(16, -1), cmap="viridis")
#         axes[1].set_title("Middle")
#         axes[1].axis("off")


#     ani = FuncAnimation(fig, one_frame, frames=range(len(ss)), interval=400)
#     ani.save(f"sentence_bert_{sentence}.gif", writer="imagemagick")
# %%


    
