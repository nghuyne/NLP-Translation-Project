# main_translate.py
import torch

from src.dataset import load_parallel, get_tokenizers, build_vocabs
from src.model import Encoder, Decoder, Seq2Seq
from src.translate import translate_sentence
from src.utils import EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, MAX_VOCAB, MIN_FREQ
from src.vocab import PAD


def load_model_and_vocab(device):
    # nếu có vocab.pt thì dùng lại, không thì build từ train
    try:
        vocab_obj = torch.load("vocab.pt", map_location=device)
        src_stoi = vocab_obj["src_stoi"]
        src_itos = vocab_obj["src_itos"]
        trg_stoi = vocab_obj["trg_stoi"]
        trg_itos = vocab_obj["trg_itos"]
    except FileNotFoundError:
        data_dir = "data"
        train_pairs = load_parallel(f"{data_dir}/train.en", f"{data_dir}/train.fr")
        tok_en, tok_fr = get_tokenizers()
        src_stoi, src_itos, trg_stoi, trg_itos = build_vocabs(
            train_pairs, tok_en, tok_fr, max_vocab=MAX_VOCAB, min_freq=MIN_FREQ
        )

    src_pad_idx = src_stoi[PAD]
    trg_pad_idx = trg_stoi[PAD]

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

    state = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, src_stoi, src_itos, trg_stoi, trg_itos


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tok_en, tok_fr = get_tokenizers()
    model, src_stoi, src_itos, trg_stoi, trg_itos = load_model_and_vocab(device)

    while True:
        s = input("Enter English sentence (or 'quit'): ").strip()
        if s.lower() == "quit":
            break
        fr = translate_sentence(
            s, model, tok_en, tok_fr, src_stoi, trg_stoi, trg_itos, device
        )
        print("-> French:", fr)


if __name__ == "__main__":
    main()
