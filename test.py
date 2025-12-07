import torch
import torch.nn as nn
from src.encoder import Encoder
from src.decoder import Decoder
from src.seq2seq import Seq2Seq
from src.attention import BahdanauAttention

def test_model_shapes():
    print("Running Model Shape Tests...")
    
    # Hyperparameters
    INPUT_DIM = 100
    OUTPUT_DIM = 100
    ENC_EMB_DIM = 32
    DEC_EMB_DIM = 32
    ENC_HID_DIM = 64
    DEC_HID_DIM = 64
    ATTN_DIM = 8
    BATCH_SIZE = 4
    SRC_LEN = 10
    TRG_LEN = 12
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Test Encoder
    print("\n[1] Testing Encoder...")
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM)
    src = torch.randint(0, INPUT_DIM, (SRC_LEN, BATCH_SIZE)) # (src_len, batch)
    
    # Move to device if needed, but testing on CPU is usually fine for shapes
    # encoder.to(DEVICE)
    # src = src.to(DEVICE)
    
    encoder_outputs, hidden, cell = encoder(src)
    
    # Check shapes
    # encoder_outputs: (src_len, batch, enc_hid_dim * 2)
    assert encoder_outputs.shape == (SRC_LEN, BATCH_SIZE, ENC_HID_DIM * 2), f"Encoder output shape mismatch: {encoder_outputs.shape}"
    # hidden: (batch, dec_hid_dim)
    assert hidden.shape == (BATCH_SIZE, DEC_HID_DIM), f"Encoder hidden shape mismatch: {hidden.shape}"
    # cell: (batch, dec_hid_dim)
    assert cell.shape == (BATCH_SIZE, DEC_HID_DIM), f"Encoder cell shape mismatch: {cell.shape}"
    print("    [✓] Encoder shapes correct.")

    # 2. Test Attention
    print("\n[2] Testing Attention...")
    attention = BahdanauAttention(ENC_HID_DIM * 2, DEC_HID_DIM, ATTN_DIM)
    
    # hidden: (batch, dec_hid_dim)
    # encoder_outputs: (src_len, batch, enc_hid_dim * 2)
    # mask: (batch, src_len)
    mask = torch.ones(BATCH_SIZE, SRC_LEN)
    
    attn_out, attn_weights = attention(hidden, encoder_outputs, mask)
    
    # attn_out: (batch, enc_hid_dim * 2) -> Weighted sum
    assert attn_out.shape == (BATCH_SIZE, ENC_HID_DIM * 2), f"Attention output shape mismatch: {attn_out.shape}"
    # attn_weights: (batch, src_len)
    assert attn_weights.shape == (BATCH_SIZE, SRC_LEN), f"Attention weights shape mismatch: {attn_weights.shape}"
    print("    [✓] Attention shapes correct.")

    # 3. Test Decoder
    print("\n[3] Testing Decoder...")
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM * 2, DEC_HID_DIM, ATTN_DIM)
    
    input_token = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,)) # (batch)
    
    prediction, new_hidden, new_cell, attn = decoder(input_token, hidden, cell, encoder_outputs, mask)
    
    # prediction: (batch, output_dim)
    assert prediction.shape == (BATCH_SIZE, OUTPUT_DIM), f"Decoder prediction shape mismatch: {prediction.shape}"
    # new_hidden: (batch, dec_hid_dim)
    assert new_hidden.shape == (BATCH_SIZE, DEC_HID_DIM), f"Decoder hidden shape mismatch: {new_hidden.shape}"
    print("    [✓] Decoder shapes correct.")

    # 4. Test Seq2Seq
    print("\n[4] Testing Seq2Seq Forward...")
    seq2seq = Seq2Seq(encoder, decoder, 0, 'cpu')
    
    trg = torch.randint(0, OUTPUT_DIM, (TRG_LEN, BATCH_SIZE))
    
    outputs = seq2seq(src, SRC_LEN, trg, teacher_forcing_ratio=0.5)
    
    # outputs: (trg_len, batch, output_dim)
    assert outputs.shape == (TRG_LEN, BATCH_SIZE, OUTPUT_DIM), f"Seq2Seq output shape mismatch: {outputs.shape}"
    print("    [✓] Seq2Seq forward pass correct.")

if __name__ == "__main__":
    try:
        test_model_shapes()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\n[!] Test failed: {e}")
        exit(1)
