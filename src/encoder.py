import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    BiGRU Encoder cho NMT (Bahdanau 2014)
    
    Input:
        src: (src_len, batch)
    
    Output:
        encoder_outputs: (src_len, batch, hidden_dim * 2)
        hidden: (batch, hidden_dim) → dùng làm initial hidden cho decoder
    """

    def __init__(self, input_dim, embed_dim, enc_hidden_dim, dec_hidden_dim, dropout=0.5):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=0)

        # BiLSTM → hidden/cell đầu ra = enc_hidden_dim * 2
        self.rnn = nn.LSTM(
            embed_dim,
            enc_hidden_dim,
            bidirectional=True
        )

        # Linear để convert hidden/cell encoder (bi-directional) → hidden/cell decoder (1 chiều)
        self.fc_hidden = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.fc_cell = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        src: (src_len, batch)
        """

        embedded = self.dropout(self.embedding(src))  # (src_len, batch, embed_dim)

        encoder_outputs, (hidden, cell) = self.rnn(embedded)
        # encoder_outputs: (src_len, batch, enc_hidden_dim * 2)
        # hidden: (num_layers*2, batch, enc_hidden_dim)
        # cell:   (num_layers*2, batch, enc_hidden_dim)

        # Ghép 2 chiều forward/backward:
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        cell_cat = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)

        # Đưa về dec_hidden_dim
        hidden_dec = torch.tanh(self.fc_hidden(hidden_cat))  # (batch, dec_hidden_dim)
        cell_dec = torch.tanh(self.fc_cell(cell_cat))        # (batch, dec_hidden_dim)

        return encoder_outputs, hidden_dec, cell_dec
