import torch
from .evaluate import translate_greedy, translate_beam


def numericalize_sentence(sentence, tokenizer, stoi, sos_idx, eos_idx):
    tokens = tokenizer(sentence)
    ids = [stoi.get(tok, stoi["<unk>"]) for tok in tokens]
    ids = [sos_idx] + ids + [eos_idx]
    return torch.tensor(ids).unsqueeze(1)  # (src_len, 1)


def translate_sentence(
    sentence,
    model,
    tokenizer_src,
    tokenizer_trg,
    src_stoi,
    trg_stoi,
    src_itos,
    trg_itos,
    device,
    beam=False,
    beam_size=5,
    max_len=60
):
    model.eval()

    sos_idx = src_stoi["<sos>"]
    eos_idx = src_stoi["<eos>"]

    src_tensor = numericalize_sentence(sentence, tokenizer_src, src_stoi, sos_idx, eos_idx)
    src_tensor = src_tensor.to(device)

    if beam:
        translation = translate_beam(
            model, src_tensor,
            src_itos, trg_itos,
            trg_stoi["<sos>"], trg_stoi["<eos>"],
            device,
            beam_size=beam_size,
            max_len=max_len
        )
    else:
        translation = translate_greedy(
            model, src_tensor,
            src_itos, trg_itos,
            trg_stoi["<sos>"], trg_stoi["<eos>"],
            device,
            max_len=max_len
        )

    return translation


###############################################
# CHẠY TRỰC TIẾP BẰNG TERMINAL
###############################################
if __name__ == "__main__":
    import argparse
    from .utils import load_checkpoint

    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, required=True, help="English sentence to translate")
    parser.add_argument("--beam", action="store_true", help="Use beam search instead of greedy")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=60)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")

    args = parser.parse_args()

    print("\n[!] This module requires you to load model + vocab manually before CLI can work.")
    print("    Typically used through Jupyter notebooks or main.py.\n")
