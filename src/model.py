# src/model.py
import random
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, src, src_lens):
        # src: [batch, src_len]
        emb = self.emb(src)  # [batch, src_len, emb_dim]
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lens.cpu(), batch_first=True, enforce_sorted=True
        )
        _, (h, c) = self.lstm(packed)
        return h, c  # context vector


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hid_dim, vocab_size)
        self.output_dim = vocab_size

    def forward(self, input, hidden, cell):
        # input: [batch]
        input = input.unsqueeze(1)  # [batch, 1]
        emb = self.emb(input)  # [batch, 1, emb_dim]
        output, (h, c) = self.lstm(emb, (hidden, cell))
        pred = self.fc(output.squeeze(1))  # [batch, vocab]
        return pred, h, c


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, trg_pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.trg_pad_idx = trg_pad_idx

    def forward(self, src, src_lens, trg, teacher_forcing_ratio=0.5):
        """
        src: [batch, src_len]
        trg: [batch, trg_len]
        return: outputs [batch, trg_len, trg_vocab]
        """
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=self.device)

        hidden, cell = self.encoder(src, src_lens)
        input_tok = trg[:, 0]  # <sos>

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_tok, hidden, cell)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_tok = trg[:, t] if teacher_force else top1

        return outputs
