import torch
import torch.nn as nn
from src.attention import BahdanauAttention


class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, enc_hidden_dim, dec_hidden_dim, attn_dim):
        super().__init__()

        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, embed_dim)

        self.attention = BahdanauAttention(
            enc_hidden_dim=enc_hidden_dim,
            dec_hidden_dim=dec_hidden_dim,
            attn_dim=attn_dim
        )

        self.rnn = nn.LSTM(
            embed_dim + enc_hidden_dim,
            dec_hidden_dim
        )

        self.fc_out = nn.Linear(enc_hidden_dim + dec_hidden_dim + embed_dim,
                                output_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, input, hidden, cell, encoder_outputs, mask):

        input = input.unsqueeze(0)  # (1, batch)

        embedded = self.dropout(self.embedding(input))  # (1, batch, emb)

        # context vector
        context, attn_weights = self.attention(hidden, encoder_outputs, mask)

        context = context.unsqueeze(0)  # (1, batch, enc_hidden)

        # LSTM input
        rnn_input = torch.cat((embedded, context), dim=2)

        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))

        hidden = hidden.squeeze(0)
        cell = cell.squeeze(0)

        # final output layer
        output = torch.cat(
            (output.squeeze(0), context.squeeze(0), embedded.squeeze(0)),
            dim=1
        )

        prediction = self.fc_out(output)

        return prediction, hidden, cell, attn_weights
