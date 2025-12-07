# main_train.py
import torch

from src.dataset import load_parallel, get_tokenizers, build_vocabs, create_dataloaders
from src.model import Encoder, Decoder, Seq2Seq
from src.utils import (
    set_seed,
    EMB_DIM,
    HID_DIM,
    N_LAYERS,
    DROPOUT,
    BATCH_SIZE,
    MAX_VOCAB,
    MIN_FREQ,
    N_EPOCHS,
    TEACHER_FORCING,
    CLIP,
    LR,
)
from src.vocab import PAD
from src.train import train_model


def main():
    set_seed()

    data_dir = "data"
    train_pairs = load_parallel(f"{data_dir}/train.en", f"{data_dir}/train.fr") 
    val_pairs   = load_parallel(f"{data_dir}/val.en",   f"{data_dir}/val.fr")

    train_pairs = train_pairs[:29000]
    print("Train pairs:", len(train_pairs), "Val pairs:", len(val_pairs))

    tok_en, tok_fr = get_tokenizers()
    src_stoi, src_itos, trg_stoi, trg_itos = build_vocabs(
        train_pairs, tok_en, tok_fr, max_vocab=MAX_VOCAB, min_freq=MIN_FREQ
    )

    src_pad_idx = src_stoi[PAD]
    trg_pad_idx = trg_stoi[PAD]

    train_loader, val_loader = create_dataloaders(
        train_pairs,
        val_pairs,
        tok_en,
        tok_fr,
        src_stoi,
        trg_stoi,
        batch_size=BATCH_SIZE,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    encoder = Encoder(
        vocab_size=len(src_itos),
        emb_dim=EMB_DIM,
        hid_dim=HID_DIM,
        num_layers=N_LAYERS,
        dropout=DROPOUT,
        pad_idx=src_pad_idx,
    )
    decoder = Decoder(
        vocab_size=len(trg_itos),
        emb_dim=EMB_DIM,
        hid_dim=HID_DIM,
        num_layers=N_LAYERS,
        dropout=DROPOUT,
        pad_idx=trg_pad_idx,
    )
    model = Seq2Seq(encoder, decoder, device, trg_pad_idx).to(device)

    train_model(
        model,
        train_loader,
        val_loader,
        n_epochs=N_EPOCHS,
        lr=LR,
        clip=CLIP,
        teacher_forcing=TEACHER_FORCING,
        trg_pad_idx=trg_pad_idx,
        device=device,
        save_path="best_model.pth",
        patience=3,
    )

    # lưu thêm vocab để dùng khi translate
    torch.save(
        {
            "src_stoi": src_stoi,
            "src_itos": src_itos,
            "trg_stoi": trg_stoi,
            "trg_itos": trg_itos,
        },
        "vocab.pt",
    )
    print("Saved vocab.pt")


if __name__ == "__main__":
    main()
