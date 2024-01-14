import typing as tp


def load_names(path: str) -> tp.List[str]:
    """Read names into memory.

    Args:
        path (str): Path the txt file of all names.

    Returns:
        tp.List[str]: A list of names.
    """
    with open(path, "r") as fp:
        words = fp.read().splitlines()
    return words


def build_vocab(words: tp.List[str]) -> tp.List[str]:
    """Fetch unique characters from corpse and sort."""
    vocab = sorted(list(set(ch for w in words for ch in w)))
    return vocab


def build_mapping_between_char_and_index(vocab: tp.List[str]):
    if "." not in vocab:
        vocab = ["."] + vocab

    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    return stoi, itos
