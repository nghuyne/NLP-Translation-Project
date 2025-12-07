# evaluate_bleu.py
import torch
from nltk.tokenize import word_tokenize

from src.dataset import load_parallel, get_tokenizers
from src.model import Encoder, Decoder, Seq2Seq
from src.utils import (
    EMB_DIM,
    HID_DIM,
    N_LAYERS,
    DROPOUT,
    MAX_VOCAB,
    MIN_FREQ,
    compute_bleu,
)
from src.vocab import PAD
from src.translate import translate_sentence


def load_model_and_vocab(device):
    # ưu tiên dùng vocab.pt đã lưu khi train
    vocab_obj = torch.load("vocab.pt", map_location=device)
    src_stoi = vocab_obj["src_stoi"]
    src_itos = vocab_obj["src_itos"]
    trg_stoi = vocab_obj["trg_stoi"]
    trg_itos = vocab_obj["trg_itos"]

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

    # load tập test en-fr
    data_dir = "data"
    test_pairs = load_parallel(f"{data_dir}/test.en", f"{data_dir}/test.fr")
    print("Test pairs:", len(test_pairs))

    tok_en, tok_fr = get_tokenizers()
    model, src_stoi, src_itos, trg_stoi, trg_itos = load_model_and_vocab(device)

    bleu_scores = []
    examples = []  # lưu vài ví dụ để phân tích lỗi

    # có thể giới hạn số câu để chạy nhanh, vd 500
    MAX_SAMPLES = len(test_pairs)  # hoặc 500

    for i, (en, fr_ref) in enumerate(test_pairs[:MAX_SAMPLES]):
        # dịch câu tiếng Anh
        fr_pred = translate_sentence(
            en, model, tok_en, tok_fr, src_stoi, trg_stoi, trg_itos, device
        )

        # tokenizes
        ref_tokens = tok_fr(fr_ref.lower())
        hyp_tokens = fr_pred.split()

        bleu = compute_bleu(ref_tokens, hyp_tokens)
        bleu_scores.append(bleu)

        # lưu vài ví dụ (vd 10 câu đầu)
        if i < 10:
            examples.append((en, fr_ref, fr_pred, bleu))

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"\nAverage BLEU on test ({len(bleu_scores)} sentences): {avg_bleu:.4f}")

    print("\nSome example translations (for error analysis):")
    for en, fr_ref, fr_pred, bleu in examples:
        print("-" * 60)
        print("EN :", en)
        print("REF:", fr_ref)
        print("PRED:", fr_pred)
        print(f"BLEU: {bleu:.4f}")

    # lưu BLEU để đưa vào báo cáo nếu muốn
    with open("bleu_result.txt", "w", encoding="utf-8") as f:
        f.write(f"Average BLEU: {avg_bleu:.4f}\n")
        f.write("\nExamples:\n")
        for en, fr_ref, fr_pred, bleu in examples:
            f.write("-" * 60 + "\n")
            f.write("EN  : " + en + "\n")
            f.write("REF : " + fr_ref + "\n")
            f.write("PRED: " + fr_pred + "\n")
            f.write(f"BLEU: {bleu:.4f}\n")

    print("\nSaved bleu_result.txt (dùng nội dung để đưa vào báo cáo).")


if __name__ == "__main__":
    main()
