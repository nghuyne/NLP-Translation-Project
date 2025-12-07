import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2Seq(nn.Module):
    """
    Wrapper cho Encoder + Decoder
    """

    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def make_src_mask(self, src):
        """
        src: (src_len, batch)
        mask: (batch, src_len)
        """
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask  # 1 là valid, 0 là pad

    ############################################################
    # FORWARD TRAIN (teacher forcing)
    ############################################################
    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        """
        src: (src_len, batch)
        trg: (trg_len, batch)
        trg_len: int
        """

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # Encode
        encoder_outputs, hidden, cell = self.encoder(src)

        # First decoder input = <sos>
        input_token = trg[0, :]

        mask = self.make_src_mask(src)

        for t in range(1, trg_len):
            logits, hidden, cell, _ = self.decoder(
                input_token, hidden, cell, encoder_outputs, mask
            )
            outputs[t] = logits

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            next_input = trg[t] if teacher_force else logits.argmax(1)

            input_token = next_input

        return outputs

    ############################################################
    # GREEDY DECODING
    ############################################################
    def greedy_decode(self, src, max_len, sos_idx, eos_idx):
        """
        src: (src_len, batch=1)
        """

        self.eval()

        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src)
            mask = self.make_src_mask(src)

            input_token = torch.tensor([sos_idx]).to(self.device)

            outputs = []

            attentions = []
            
            for _ in range(max_len):
                logits, hidden, cell, attn = self.decoder(
                    input_token, hidden, cell, encoder_outputs, mask
                )
                
                attentions.append(attn)

                top1 = logits.argmax(1)
                next_token = top1.item()
                outputs.append(next_token)

                input_token = top1

                if next_token == eos_idx:
                    break
        
        # Stack attentions: (trg_len, 1, src_len) -> (trg_len, src_len)
        if attentions:
            attentions = torch.cat(attentions, dim=0) # (trg_len, src_len)
        
        return outputs, attentions

    ############################################################
    # BEAM SEARCH DECODING
    ############################################################
    def beam_search(self, src, beam_size, max_len, sos_idx, eos_idx):
        """
        src: (src_len, 1)
        beam search chuẩn cho NMT
        """

        self.eval()

        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src)
            mask = self.make_src_mask(src)

            # Beam = list of tuples: (sequence, hidden, cell, log_prob)
            beams = [
                ([sos_idx], hidden, cell, 0.0)
            ]

            completed = []

            for _ in range(max_len):

                candidates = []

                for seq, h, c, score in beams:
                    last_token = seq[-1]

                    if last_token == eos_idx:
                        completed.append((seq, score))
                        continue

                    input_token = torch.tensor([last_token]).to(self.device)

                    logits, new_hidden, new_cell, _ = self.decoder(
                        input_token, h, c, encoder_outputs, mask
                    )

                    log_probs = F.log_softmax(logits, dim=1).squeeze(0)

                    topk_log_probs, topk_idx = log_probs.topk(beam_size)

                    for log_p, idx in zip(topk_log_probs, topk_idx):
                        new_seq = seq + [idx.item()]
                        new_score = score + log_p.item()
                        candidates.append((new_seq, new_hidden, new_cell, new_score))

                # Lọc top beam_size
                candidates.sort(key=lambda x: x[3], reverse=True)
                beams = candidates[:beam_size]

            # Nếu chưa có câu hoàn chỉnh → chọn beam tốt nhất
            if len(completed) == 0:
                completed = [(seq, score) for seq, _, _, score in beams]

            completed.sort(key=lambda x: x[1], reverse=True)
            return completed[0][0]
