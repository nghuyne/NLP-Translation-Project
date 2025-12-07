from collections import Counter

SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


def build_vocab(sentences, tokenizer, max_vocab=20000, min_freq=2):
    counter = Counter()

    # Đếm tần suất token
    for sent in sentences:
        tokens = tokenizer(sent)
        counter.update(tokens)

    # Giữ token đủ tần suất
    vocab = [tok for tok, freq in counter.items() if freq >= min_freq]

    # Giới hạn kích thước vocab
    vocab = vocab[: max_vocab - len(SPECIAL_TOKENS)]

    # stoi: string → index
    itos = SPECIAL_TOKENS + vocab
    stoi = {tok: idx for idx, tok in enumerate(itos)}

    return stoi, itos


import pickle

def save_vocab(vocab, path):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
