# src/translate.py
import torch

from src.vocab import SOS, EOS


@torch.no_grad()
def translate_sentence(
    sentence,
    model,
    tok_en,
    tok_fr,          # không dùng trong greedy cơ bản, nhưng để sẵn nếu cần
    src_stoi,
    trg_stoi,
    trg_itos,
    device,
    max_len=50,
):
    model.eval()  # Đặt mô hình vào chế độ đánh giá (không cập nhật gradient)

    from src.dataset import numericalize  # tránh vòng import
    
    # Đặt mô hình vào chế độ đánh giá (không cập nhật gradient)
    src_ids = numericalize(tok_en(sentence.lower()), src_stoi)
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    src_len = torch.tensor([len(src_ids)], dtype=torch.long).to(device)

    # Đặt mô hình vào chế độ đánh giá (không cập nhật gradient)
    hidden, cell = model.encoder(src, src_len)

    # Bắt đầu tạo câu dịch từ token <sos>
    input_tok = torch.tensor([trg_stoi[SOS]], dtype=torch.long, device=device)
    result_ids = []

    for _ in range(max_len):
        output, hidden, cell = model.decoder(input_tok, hidden, cell) #Lấy output từ decoder
        next_tok = output.argmax(1) # Chọn token có xác suất cao nhất
        idx = next_tok.item()
        if idx == trg_stoi[EOS]: # Nếu token là <eos>, dừng dịch
            break
        result_ids.append(idx) # lưu chỉ số từ dự đoán
        input_tok = next_tok #cập nhật tiếp theo từ decoder

    # Chuyển đổi các chỉ số từ về từ ngữ
    tokens = [trg_itos[i] for i in result_ids]
    return " ".join(tokens)
