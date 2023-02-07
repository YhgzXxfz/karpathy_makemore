from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda:0" if torch.cuda.is_available() else "cpu"
eval_iters = 200

# load corpse
# curl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o input.txt
with open("input.txt", "r", encoding="utf-8") as f:
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


# model
class BiGramLanguageModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):  # idx is (B, T)
        logits = self.token_embedding_table(idx)  # logits is (B, T, C)

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
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # (B, C). Only needs the last result of block_size (T)
            probs = F.softmax(logits, dim=-1)  # (B, C). Softmax on dimension C
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


model = BiGramLanguageModel(vocab_size=vocab_size).to(device)

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