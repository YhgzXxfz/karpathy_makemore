import torch
import torch.nn as nn
import torch.nn.functional as F

from nanogpt.models.attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, n_embed: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, n_embed), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, num_heads: int, head_size: int, n_embed: int, block_size: int) -> None:
        super().__init__()
        assert head_size == n_embed // num_heads, "We use n_embed as sum of head_sizes for now."
        self.sa = MultiHeadAttention(num_heads=num_heads, head_size=head_size, n_embed=n_embed, block_size=block_size)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x


class TransformerWithMultiAttentionAndFeedForward(nn.Module):
    def __init__(self, num_heads: int, vocab_size: int, n_embed: int, block_size: int) -> None:
        super().__init__()

        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_head = MultiHeadAttention(
            num_heads=num_heads, head_size=n_embed // num_heads, n_embed=n_embed, block_size=block_size
        )  # We want the total length of concatenated heads (along the C dimension) to be the same as Single Head.
        # - (which is n_embed in our setting)

        self.ffwd = FeedForward(n_embed)

        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):  # idx is (B, T)
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C), C is n_embed
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb  # (B, T, C) + (T, C) => (B, T, C)
        x = self.sa_head(x)  # (B, T, head_size == C)
        x = self.ffwd(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

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
            logits, _ = self(idx[:, -self.block_size :])  # Only forward a sequence of length block_size (at most).
            logits = logits[:, -1, :]  # (B, C). Only needs the last result of block_size (T)
            probs = F.softmax(logits, dim=-1)  # (B, C). Softmax on dimension C
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


class TransformerWithBlocksOfMultiAttentionAndFeedForward(nn.Module):
    def __init__(self, num_heads: int, vocab_size: int, n_embed: int, block_size: int) -> None:
        super().__init__()

        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            Block(num_heads=num_heads, head_size=n_embed // num_heads, n_embed=n_embed, block_size=block_size),
            Block(num_heads=num_heads, head_size=n_embed // num_heads, n_embed=n_embed, block_size=block_size),
            Block(num_heads=num_heads, head_size=n_embed // num_heads, n_embed=n_embed, block_size=block_size),
        )

        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):  # idx is (B, T)
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C), C is n_embed
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb  # (B, T, C) + (T, C) => (B, T, C)
        x = self.blocks(x)  # (B, T, head_size == C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

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
            logits, _ = self(idx[:, -self.block_size :])  # Only forward a sequence of length block_size (at most).
            logits = logits[:, -1, :]  # (B, C). Only needs the last result of block_size (T)
            probs = F.softmax(logits, dim=-1)  # (B, C). Softmax on dimension C
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx
