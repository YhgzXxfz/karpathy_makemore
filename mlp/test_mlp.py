import random

import mlp.utils as utils
from mlp.dataset import build_dataset
from mlp.mlp import MLP, evaluate, training_loop

# Model Constants
vocab_size = 27
embedding_size = 2
hidden_layer_size = 100
block_size = 3
mini_batch_size = 32
model_seed = 2147483647

# Dataset Constants
shuffle_seed = 42
tr_ratio = 0.8
val_ratio = 0.9


# Build Dataset
words = utils.load_names("names.txt")
vocab = utils.build_vocab(words)
stoi, itos = utils.build_mapping_between_char_and_index(vocab)

random.seed(shuffle_seed)
random.shuffle(words)
tr_size = int(tr_ratio * len(words))
val_size = int(val_ratio * len(words))
Xtr, Ytr = build_dataset(words[:tr_size], block_size, stoi)
Xval, Yval = build_dataset(words[tr_size:val_size], block_size, stoi)
Xtest, Ytest = build_dataset(words[val_size:], block_size, stoi)


# Init Model
model = MLP(vocab_size, embedding_size, block_size, hidden_layer_size, model_seed)

# Train
training_loop(model, mini_batch_size, Xtr, Ytr, epochs=200_000, lr=0.1, lr_decay=True)

# Evaluate
loss_item = evaluate(model, Xval, Yval)
print(f"Loss on validation set: {loss_item}")

# Generate
outputs = model.generate(20, 1337)
print(["".join(itos[ix] for ix in o) for o in outputs])
