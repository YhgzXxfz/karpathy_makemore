import os
import pathlib
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = "cuda:0" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 32

# load corpse
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(os.path.join(pathlib.Path(__file__).parent.resolve(), "input.txt"), "r", encoding="utf-8") as f:
    text = f.read()

vocab = sorted(list(set(text)))
vocab_size = len(vocab)
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}


def encode(s: str) -> List[int]:
    return [stoi[ch] for ch in s]


def decode(lst: List[int]) -> str:
    return "".join(itos[i] for i in lst)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# prepare datasets
def get_batch(split: str):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    xb = torch.stack([data[i : i + block_size] for i in ix])
    yb = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb


@torch.no_grad()
def estimate_loss():
    output = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        output[split] = losses.mean()
    model.train()
    return output


class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


# model
class BiGramLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_head = MultiHeadAttention(num_heads=4, head_size=n_embed // 4)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):  # idx is (B, T)
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # logits is (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.sa_head(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # To fulfill PyTorch cross entropy requirements
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)  # PyTorch requires (B, C, T)
        return logits, loss

    def generate(self, idx, num_examples):
        for _ in range(num_examples):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, C). Only needs the last result of block_size (T)
            probs = F.softmax(logits, dim=-1)  # (B, C). Softmax on dimension C
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


model = BiGramLanguageModel().to(device)

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)

    loss.backward()
    optimizer.step()


# Inference
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, num_examples=500)[0].tolist()))
