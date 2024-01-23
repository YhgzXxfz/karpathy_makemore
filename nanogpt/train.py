import argparse
import os
import pathlib
import typing as tp

import torch
import torch.nn as nn

from nanogpt.bigram import BiGramLanguageModel, BiGramWithPositionEmbeddingLanguageModel


def encode(s: str, stoi: tp.Dict) -> tp.List[int]:
    return [stoi[ch] for ch in s]


def decode(lst: tp.List[int], itos: tp.Dict) -> str:
    return "".join(itos[i] for i in lst)


def create_data(text: str, stoi: tp.Dict, itos: tp.Dict) -> tp.Dict:
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return {"train": train_data, "val": val_data}


# prepare datasets
def get_batch(raw: tp.Dict, split: str, block_size: int, batch_size: int, device: torch.device):
    data = raw[split]
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    xb = torch.stack([data[i : i + block_size] for i in ix])
    yb = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb


@torch.no_grad()
def estimate_loss(
    model: nn.Module, eval_epochs: int, data: torch.Tensor, block_size: int, batch_size: int, device: torch.device
) -> tp.Dict:
    output = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_epochs)
        for k in range(eval_epochs):
            X, Y = get_batch(data, split, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        output[split] = losses.mean()
    model.train()
    return output


def train(
    model: nn.Module,
    optimizer,
    epochs: int,
    eval_interval: int,
    eval_epochs: int,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> None:
    for iter in range(epochs):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, eval_epochs, data, block_size, batch_size, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch(data, "train", block_size, batch_size, device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)

        loss.backward()
        optimizer.step()


def main():
    parser = argparse.ArgumentParser(description="BiGram Training Scripts.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size for training.")
    parser.add_argument("--block_size", type=int, default=8, help="Context Length of Sequence.")
    parser.add_argument("--epochs", type=int, default=3000, help="Number of iterations of training.")
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=500,
        help="Interval of loss evaluation (both train and val).",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning Rate.")
    parser.add_argument("--eval_epochs", type=int, default=200, help="Number of epochs to evaluate.")
    parser.add_argument("--n_embed", type=int, default=65, help="Number embedding dimension.")
    parser.add_argument("--cuda", action="store_true", help="If True, use GPU for training.")

    args = parser.parse_args()

    # Load Corpse
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open(os.path.join(pathlib.Path(__file__).parent.resolve(), "input.txt"), "r", encoding="utf-8") as fp:
        text = fp.read()

    # Create data
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    data = create_data(text, stoi, itos)

    torch.manual_seed(1337)
    device = "cuda:0" if args.cuda and torch.cuda.is_available() else "cpu"
    model = BiGramWithPositionEmbeddingLanguageModel(
        vocab_size=vocab_size, n_embed=args.n_embed, block_size=args.block_size
    ).to(device)

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train(
        model,
        optimizer,
        args.epochs,
        eval_interval=args.eval_interval,
        eval_epochs=args.eval_epochs,
        data=data,
        block_size=args.block_size,
        batch_size=args.batch_size,
        device=device,
    )

    # Inference
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist(), itos))


if __name__ == "__main__":
    main()
