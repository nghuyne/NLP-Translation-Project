# src/vocab.py
from collections import Counter

PAD = "<pad>"
UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"


def build_vocab(tokenized_texts, max_vocab=10000, min_freq=2):
    """
    tokenized_texts: list[list[str]]
    return: (stoi, itos)
    """
    counter = Counter()
    for sent in tokenized_texts:
        counter.update(sent)

    words = [w for w, f in counter.items() if f >= min_freq]
    words = sorted(words, key=counter.get, reverse=True)[: max_vocab - 4]

    itos = [PAD, UNK, SOS, EOS] + words
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos
