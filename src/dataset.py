import os
import torch
from torch.utils.data import Dataset, DataLoader


#############################################
# Load song ngữ song song
#############################################
def load_parallel(src_file, trg_file, limit=None):
    pairs = []
    with open(src_file, encoding="utf-8") as fs, open(trg_file, encoding="utf-8") as ft:
        for i, (s, t) in enumerate(zip(fs, ft)):
            s = s.strip()
            t = t.strip()
            if not s or not t:
                continue
            pairs.append((s, t))
            if limit and len(pairs) >= limit:
                break
    return pairs


#############################################
# Tokenizer đơn giản
#############################################
def get_tokenizers():
    def tokenize_en(text):
        return text.lower().strip().split()

    def tokenize_fr(text):
        return text.lower().strip().split()

    return tokenize_en, tokenize_fr


#############################################
# Dataset class
#############################################
class ParallelDataset(Dataset):
    def __init__(self, pairs, tokenizer_src, tokenizer_trg, src_stoi, trg_stoi, max_len=128):
        self.pairs = pairs
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg
        self.src_stoi = src_stoi
        self.trg_stoi = trg_stoi
        self.max_len = max_len

    def numericalize(self, tokens, stoi):
        ids = [stoi.get(tok, stoi["<unk>"]) for tok in tokens]
        ids = ids[: self.max_len - 2]
        return [stoi["<sos>"]] + ids + [stoi["<eos>"]]

    def __getitem__(self, idx):
        src, trg = self.pairs[idx]

        src_tokens = self.tokenizer_src(src)
        trg_tokens = self.tokenizer_trg(trg)

        src_ids = self.numericalize(src_tokens, self.src_stoi)
        trg_ids = self.numericalize(trg_tokens, self.trg_stoi)

        return torch.tensor(src_ids), torch.tensor(trg_ids)

    def __len__(self):
        return len(self.pairs)


#############################################
# Collate function cho DataLoader
#############################################
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)

    src_lengths = torch.tensor([len(x) for x in src_batch])
    trg_lengths = torch.tensor([len(x) for x in trg_batch])

    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=0)
    trg_padded = torch.nn.utils.rnn.pad_sequence(trg_batch, padding_value=0)

    return src_padded, src_lengths, trg_padded, trg_lengths


#############################################
# Tạo DataLoader
#############################################
def build_dataloader(pairs, tokenizer_src, tokenizer_trg, src_stoi, trg_stoi, batch_size=64, shuffle=True):
    dataset = ParallelDataset(
        pairs, tokenizer_src, tokenizer_trg, src_stoi, trg_stoi
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    return loader
