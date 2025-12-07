import argparse
import torch
import torch.nn as nn
import os

from src.dataset import load_parallel, get_tokenizers, build_dataloader
from src.vocab import build_vocab, PAD_IDX, SOS_IDX, EOS_IDX, save_vocab, load_vocab
from src.encoder import Encoder
from src.decoder import Decoder
from src.seq2seq import Seq2Seq
from src.train import train_model
from src.evaluate import evaluate_bleu
from src.translate import translate_sentence
from src.utils import load_checkpoint, set_seed
from src.visualization import display_attention


#############################################################
# BUILD MODEL (đã giảm kích thước cho GPU 4GB)
#############################################################
def build_model(
    src_vocab_size,
    trg_vocab_size,
    embed_dim=256,      # tăng lên 256
    enc_hidden=512,     # tăng lên 512 (theo yêu cầu)
    dec_hidden=512,
    attn_dim=128,
    device="cuda"
):

    encoder = Encoder(
        input_dim=src_vocab_size,
        embed_dim=embed_dim,
        enc_hidden_dim=enc_hidden,
        dec_hidden_dim=dec_hidden
    )

    decoder = Decoder(
        output_dim=trg_vocab_size,
        embed_dim=embed_dim,
        enc_hidden_dim=enc_hidden * 2,  # Fix: Encoder is bidirectional
        dec_hidden_dim=dec_hidden,
        attn_dim=attn_dim
    )

    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=PAD_IDX,
        device=device
    ).to(device)

    return model


#############################################################
# MAIN
#############################################################
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--translate", type=str, default=None)
    parser.add_argument("--plot", action="store_true", help="Plot attention map")
    parser.add_argument("--beam", action="store_true")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of training samples")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")

    args = parser.parse_args()
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #############################################################
    # LOAD DATASET
    #############################################################
    print("[*] Loading dataset...")

    train_pairs = load_parallel("data/train.en", "data/train.fr", limit=args.limit)
    val_pairs   = load_parallel("data/val.en",   "data/val.fr", limit=10000) # Limit validation too for speed
    test_pairs  = load_parallel("data/test.en",  "data/test.fr", limit=10000)

    tokenizer_en, tokenizer_fr = get_tokenizers()

    # Check if vocab exists
    if os.path.exists("data/vocab_src.pkl") and os.path.exists("data/vocab_trg.pkl"):
        print("[*] Loading vocabulary...")
        src_stoi = load_vocab("data/vocab_src.pkl")
        src_itos = load_vocab("data/vocab_src_itos.pkl")
        trg_stoi = load_vocab("data/vocab_trg.pkl")
        trg_itos = load_vocab("data/vocab_trg_itos.pkl")
    else:
        print("[*] Building vocabulary...")
        src_stoi, src_itos = build_vocab([p[0] for p in train_pairs], tokenizer_en)
        trg_stoi, trg_itos = build_vocab([p[1] for p in train_pairs], tokenizer_fr)
        
        print("[*] Saving vocabulary...")
        save_vocab(src_stoi, "data/vocab_src.pkl")
        save_vocab(src_itos, "data/vocab_src_itos.pkl")
        save_vocab(trg_stoi, "data/vocab_trg.pkl")
        save_vocab(trg_itos, "data/vocab_trg_itos.pkl")



    # Giảm batch size xuống 8 cho GPU 4GB để tránh OOM
    BATCH_SIZE = 8
    train_loader = build_dataloader(train_pairs, tokenizer_en, tokenizer_fr, src_stoi, trg_stoi, batch_size=BATCH_SIZE)
    val_loader   = build_dataloader(val_pairs,   tokenizer_en, tokenizer_fr, src_stoi, trg_stoi, batch_size=BATCH_SIZE)

    #############################################################
    # BUILD MODEL
    #############################################################
    model = build_model(
        src_vocab_size=len(src_stoi),
        trg_vocab_size=len(trg_stoi),
        device=device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    #############################################################
    # TRAIN
    #############################################################
    if args.train:
        print("\n[✓] Training model...\n")
        train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            device,
            epochs=args.epochs,
            save_path=args.checkpoint
        )
        return

    #############################################################
    # LOAD CHECKPOINT
    #############################################################
    if os.path.exists(args.checkpoint):
        load_checkpoint(args.checkpoint, model, optimizer=None, device=device)
    else:
        print(f"[!] Checkpoint {args.checkpoint} not found.")
        return

    #############################################################
    # EVALUATE BLEU
    #############################################################
    if args.eval:
        print("[*] Evaluating BLEU...")
        bleu = evaluate_bleu(
            model,
            test_pairs,
            tokenizer_en,
            tokenizer_fr,
            src_stoi,
            trg_stoi,
            src_itos,
            trg_itos,
            SOS_IDX,
            EOS_IDX,
            device,
            beam=args.beam
        )
        print(f"\n[✓] BLEU Score: {bleu:.2f}\n")
        return

    #############################################################
    # TRANSLATE
    #############################################################
    if args.translate:
        print("[*] Translating...")

        translation, attention = translate_sentence(
            args.translate,
            model,
            tokenizer_en,
            tokenizer_fr,
            src_stoi,
            trg_stoi,
            src_itos,
            trg_itos,
            device,
            beam=args.beam,
            beam_size=args.beam_size
        )

        print("\nInput :", args.translate)
        print("Output:", translation, "\n")
        
        if args.plot:
            print("[*] Plotting attention...")
            display_attention(args.translate, translation, attention, save_path="attention_map.png")
        
        return

    print("\n[!] No action selected. Use --train / --eval / --translate\n")


if __name__ == "__main__":
    main()
