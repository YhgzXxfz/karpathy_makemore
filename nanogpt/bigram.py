import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, n_embed: int) -> None:
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

    def forward(self, idx, targets=None):  # idx is (B, T)
        B, T = idx.shape

        logits = self.token_embedding_table(idx)  # logits is (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # To fulfill PyTorch cross entropy requirements
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)  # PyTorch requires (B, C, T)
        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # (B, C). Only needs the last result of block_size (T)
            probs = F.softmax(logits, dim=-1)  # (B, C). Softmax on dimension C
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx
