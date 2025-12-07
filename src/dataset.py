# src/dataset.py
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import spacy

from src.vocab import PAD, UNK, SOS, EOS, build_vocab


def load_parallel(en_path, fr_path):
    with open(en_path, encoding="utf-8") as f1, open(fr_path, encoding="utf-8") as f2:
        ens = [l.strip().lower() for l in f1]
        frs = [l.strip().lower() for l in f2]
    return [(e, f) for e, f in zip(ens, frs) if e and f]


def get_tokenizers():
    print(">>> Loading English tokenizer (en_core_web_sm)...")
    spacy_en = spacy.load("en_core_web_sm")
    print(">>> English tokenizer loaded.")

    print(">>> Loading French tokenizer (fr_core_news_sm)...")
    spacy_fr = spacy.load("fr_core_news_sm")
    print(">>> French tokenizer loaded.")

    def tok_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def tok_fr(text):
        return [tok.text for tok in spacy_fr.tokenizer(text)]

    return tok_en, tok_fr



def numericalize(tokens, stoi):
    return [stoi[SOS]] + [stoi.get(t, stoi[UNK]) for t in tokens] + [stoi[EOS]]


def build_vocabs(train_pairs, tok_en, tok_fr, max_vocab=10000, min_freq=2):
    train_en_tok = [tok_en(e) for e, _ in train_pairs]
    train_fr_tok = [tok_fr(f) for _, f in train_pairs]

    src_stoi, src_itos = build_vocab(train_en_tok, max_vocab, min_freq)
    trg_stoi, trg_itos = build_vocab(train_fr_tok, max_vocab, min_freq)
    return src_stoi, src_itos, trg_stoi, trg_itos


def to_id_pairs(pairs, tok_en, tok_fr, src_stoi, trg_stoi):
    out = []
    for e, f in pairs:
        src_ids = numericalize(tok_en(e), src_stoi)
        trg_ids = numericalize(tok_fr(f), trg_stoi)
        out.append((src_ids, trg_ids))
    return out


def collate_batch(batch, src_pad_idx, trg_pad_idx):
    # batch: list of (src_ids, trg_ids)
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    src_seqs, trg_seqs = zip(*batch)

    src_lens = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    trg_lens = torch.tensor([len(t) for t in trg_seqs], dtype=torch.long)

    src = pad_sequence(
        [torch.tensor(s, dtype=torch.long) for s in src_seqs],
        batch_first=True,
        padding_value=src_pad_idx,
    )
    trg = pad_sequence(
        [torch.tensor(t, dtype=torch.long) for t in trg_seqs],
        batch_first=True,
        padding_value=trg_pad_idx,
    )
    return src, src_lens, trg, trg_lens


def create_dataloaders(
    train_pairs,
    val_pairs,
    tok_en,
    tok_fr,
    src_stoi,
    trg_stoi,
    batch_size=32,
):
    src_pad_idx = src_stoi[PAD]
    trg_pad_idx = trg_stoi[PAD]

    train_ids = to_id_pairs(train_pairs, tok_en, tok_fr, src_stoi, trg_stoi)
    val_ids = to_id_pairs(val_pairs, tok_en, tok_fr, src_stoi, trg_stoi)

    train_loader = DataLoader(
        train_ids,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, src_pad_idx, trg_pad_idx),
    )
    val_loader = DataLoader(
        val_ids,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, src_pad_idx, trg_pad_idx),
    )
    return train_loader, val_loader
