import argparse
import os
import pathlib
import time
import typing as tp

import torch
import torch.nn as nn

import wandb
from nanogpt.nanogpt import GPT


def encode(s: str, stoi: tp.Dict) -> tp.List[int]:
    return [stoi[ch] for ch in s]


def decode(lst: tp.List[int], itos: tp.Dict) -> str:
    return "".join(itos[i] for i in lst)


def create_data(text: str, stoi: tp.Dict) -> tp.Dict:
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


def init_wandb(args):
    assert (
        args.wandb_base_url is not None
        or args.wandb_api_key is not None
        or args.wandb_entity is not None
        or args.wandb_project is not None
    )

    os.environ["WANDB_BASE_URL"] = args.wandb_base_url
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    os.environ["WANDB_START_METHOD"] = "thread"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=args,
        name="gpt2",
    )


def train(model: nn.Module, optimizer, data: torch.Tensor, device: torch.device, model_args: tp.Dict, args) -> None:
    t0 = time.time()
    best_val_loss = 1e9

    if args.wandb_log:
        init_wandb(args)

    for iter_num in range(args.epochs):
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, args.eval_epochs, data, args.block_size, args.batch_size, device)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if args.wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": args.learning_rate,
                    }
                )
            if losses["val"] < best_val_loss and args.output_dir:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                    }
                    print(f"saving checkpoint to {args.output_dir}")
                    os.makedirs(args.output_dir, exist_ok=True)
                    torch.save(checkpoint, os.path.join(args.output_dir, "ckpt.pt"))

        if iter_num == 0 and args.eval_only:
            break

        xb, yb = get_batch(data, "train", args.block_size, args.batch_size, device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)

        loss.backward()
        optimizer.step()

        if iter_num % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"iter {iter_num}: time {dt*1000:.2f}ms")


def main():
    parser = argparse.ArgumentParser(description="BiGram Training Scripts.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size for training.")
    parser.add_argument("--block_size", type=int, default=8, help="Context Length of Sequence.")
    parser.add_argument("--epochs", type=int, default=3000, help="Number of iterations of training.")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning Rate.")
    parser.add_argument("--n_embed", type=int, default=32, help="Number embedding dimension.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of self-attention heads.")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of self-attention blocks.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout ratio.")
    parser.add_argument("--cuda", action="store_true", help="If True, use GPU for training.")

    # Evaluation
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=500,
        help="Interval of loss evaluation (both train and val).",
    )
    parser.add_argument("--eval_epochs", type=int, default=200, help="Number of epochs to evaluate.")
    parser.add_argument("--eval_only", action="store_true", help="If True, only evaluate the model.")

    # Logging
    parser.add_argument("--log_interval", type=int, default=200, help="Interval to log.")
    parser.add_argument("--wandb_log", action="store_true", help="If True, log to wandb.")
    parser.add_argument("--wandb_base_url", type=str, help="Wandb base url.")
    parser.add_argument("--wandb_api_key", type=str, help="Wandb API key.")
    parser.add_argument("--wandb_entity", type=str, help="Wandb Entity.")
    parser.add_argument("--wandb_project", type=str, help="Wandb Project.")

    # Inference
    parser.add_argument("--output_len", type=int, default=500, help="Number of tokens to generate.")
    parser.add_argument("--output_dir", type=str, help="Path to inference output.")

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
    data = create_data(text, stoi)

    torch.manual_seed(1337)
    device = "cuda:0" if args.cuda and torch.cuda.is_available() else "cpu"
    kwargs = {
        "num_heads": args.num_heads,
        "vocab_size": vocab_size,
        "n_embed": args.n_embed,
        "block_size": args.block_size,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
    }
    model = GPT(**kwargs).to(device)

    print(f"Number of params: {sum(p.numel() for p in model.parameters()) / 1e6} MB.")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train(model, optimizer, data=data, device=device, model_args=kwargs, args=args)

    # Inference
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    output = decode(model.generate(context, max_new_tokens=args.output_len)[0].tolist(), itos)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "inference.txt"), "w") as fp:
            fp.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
