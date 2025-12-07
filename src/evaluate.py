import torch
from sacrebleu import corpus_bleu


#############################################
# Chuyển IDs → tokens → câu
#############################################
def ids_to_sentence(ids, itos, eos_idx):
    tokens = []
    for i in ids:
        if i == eos_idx:
            break
        tokens.append(itos[i])
    return " ".join(tokens)


#############################################
# Dịch 1 câu bằng greedy decoding
#############################################
def translate_greedy(model, src_tensor, src_itos, trg_itos, sos_idx, eos_idx, device, max_len=60):
    model.eval()
    src_tensor = src_tensor.to(device)

    output_ids, attention = model.greedy_decode(src_tensor, max_len, sos_idx, eos_idx)
    sentence = ids_to_sentence(output_ids, trg_itos, eos_idx)
    return sentence, attention


#############################################
# Dịch 1 câu bằng beam search
#############################################
def translate_beam(model, src_tensor, src_itos, trg_itos, sos_idx, eos_idx, device, beam_size=5, max_len=60):
    model.eval()
    src_tensor = src_tensor.to(device)

    output_ids = model.beam_search(src_tensor, beam_size, max_len, sos_idx, eos_idx)
    sentence = ids_to_sentence(output_ids, trg_itos, eos_idx)
    return sentence, None


#############################################
# Tính BLEU trên cả test set
#############################################
def evaluate_bleu(
    model,
    pairs,              # [(src_str, trg_str)]
    tokenizer_src,
    tokenizer_trg,
    src_stoi,
    trg_stoi,
    src_itos,
    trg_itos,
    sos_idx,
    eos_idx,
    device,
    beam=False
):
    refs = []
    hyps = []

    for src, trg in pairs:

        # numericalize câu source
        tokens = tokenizer_src(src)
        ids = [src_stoi.get(tok, src_stoi["<unk>"]) for tok in tokens]
        ids = [sos_idx] + ids + [eos_idx]
        src_tensor = torch.tensor(ids).unsqueeze(1)  # (src_len, 1)

        if beam:
            pred, _ = translate_beam(
                model, src_tensor, src_itos, trg_itos,
                sos_idx, eos_idx, device, beam_size=5
            )
        else:
            pred, _ = translate_greedy(
                model, src_tensor, src_itos, trg_itos,
                sos_idx, eos_idx, device
            )

        refs.append([trg])    # sacreBLEU format: list of references per sentence
        hyps.append(pred)

    bleu = corpus_bleu(hyps, refs).score
    return bleu
