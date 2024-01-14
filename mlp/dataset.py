import random
import typing as tp

import torch

random.seed(42)


def build_dataset(
    words: tp.List[str],
    block_size: int,
    stoi: tp.Dict[int, str],
):
    X, Y = [], []
    for w in words:
        seq = ["."] * block_size
        for ch in w + ".":
            next_token = ch

            X.append([stoi[c] for c in seq])
            Y.append(stoi[next_token])

            seq = seq[1:] + [ch]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y
