import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        block_size: int,
        hidden_layer_size: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.block_size = block_size
        self.hidden_layer_size = hidden_layer_size
        self.reinitialize_model(seed)

    def reinitialize_model(self, seed: int):
        g = torch.Generator().manual_seed(seed)

        self.C = torch.randn((self.vocab_size, self.embedding_size), generator=g, requires_grad=True)  # (27, 2)
        self.W1 = torch.randn(
            (self.block_size * self.embedding_size, self.hidden_layer_size), generator=g, requires_grad=True
        )  # (6, 100)
        self.b1 = torch.randn((self.hidden_layer_size,), generator=g, requires_grad=True)  # (100,)
        self.W2 = torch.randn((self.hidden_layer_size, self.vocab_size), generator=g, requires_grad=True)  # (100, 27)
        self.b2 = torch.randn((self.vocab_size,), generator=g, requires_grad=True)  # (27,)
        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]

        print(f"Total number of parameters: {sum(p.nelement() for p in self.parameters)}")

    def forward(self, inputs) -> torch.Tensor:
        emb = self.C[inputs]  # (32, 3 ,2)
        h = torch.tanh(emb.view((-1, 6)) @ self.W1 + self.b1)  # (32, 100)
        logits = h @ self.W2 + self.b2  # (32, 27)
        return logits

    @torch.no_grad()
    def generate(self, num_examples: int, seed):
        g = torch.Generator().manual_seed(seed)
        outputs = []
        for _ in range(num_examples):
            output = []
            context = [0] * self.block_size

            while True:
                logits = self.forward(torch.tensor(context))  # (1, block_size, embedding_size)
                probs = F.softmax(logits, dim=1)

                ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
                context = context[1:] + [ix]

                output.append(ix)

                if ix == 0:
                    break
            outputs.append(output)

        return outputs


@torch.no_grad()
def evaluate(model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> float:
    logits = model.forward(inputs)
    loss = F.cross_entropy(logits, labels, reduction="mean")
    return loss.item()


def training_loop(
    model: nn.Module,
    mini_batch_size: int,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    lr: float,
    lr_decay: bool,
):
    for epoch in range(epochs):
        # Forward
        mini_batch_idxs = torch.randint(0, inputs.shape[0], (mini_batch_size,))
        logits = model.forward(inputs[mini_batch_idxs])

        # Loss
        loss = F.cross_entropy(logits, labels[mini_batch_idxs], reduction="mean")

        # Back Propagation
        for p in model.parameters:
            p.grad = None
        loss.backward()

        # Update
        if lr_decay:
            if epoch < 100_000:
                lr = lr
            elif epoch < 150_000:
                lr = lr / 2
            else:
                lr = lr / 10

        for p in model.parameters:
            p.data += -lr * p.grad
