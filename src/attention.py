import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attn_dim):
        super().__init__()

        self.W_h = nn.Linear(enc_hidden_dim, attn_dim)
        self.W_s = nn.Linear(dec_hidden_dim, attn_dim)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask):

        # encoder_outputs: (src_len, batch, enc_hidden)
        # decoder_hidden:  (batch, dec_hidden)

        src_len = encoder_outputs.shape[0]

        # expand decoder hidden across all source positions
        decoder_hidden = decoder_hidden.unsqueeze(0).repeat(src_len, 1, 1)

        # compute energy (avoid FP16 overflow)
        energy = torch.tanh(
            self.W_h(encoder_outputs.float()) +
            self.W_s(decoder_hidden.float())
        )

        # (src_len, batch, 1) -> (batch, src_len)
        score = self.v_a(energy).squeeze(-1).transpose(0, 1)

        # fix FP16 overflow (no -1e10)
        score = score.masked_fill(mask == 0, -1e4)

        attn_weights = torch.softmax(score, dim=1)

        # context vector
        context = torch.bmm(
            attn_weights.unsqueeze(1),
            encoder_outputs.transpose(0, 1)
        ).squeeze(1)

        return context, attn_weights
