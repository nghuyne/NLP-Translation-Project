# BÁO CÁO PHẦN MỞ RỘNG (NÂNG CAO)
## Đề tài: Dịch máy Anh-Pháp với Attention Mechanism & Beam Search

**Sinh viên thực hiện:**
- [Trần Ngọc Huy] - [3122411071]
- [Lý Vĩnh Tài] - [3122411180]

---

## 1. Giới thiệu & Điểm mới

So với mô hình cơ bản (NLP), phiên bản nâng cao (NLP1) được cải tiến mạnh mẽ để giải quyết các hạn chế về "nút thắt cổ chai" (bottleneck) và khả năng xử lý câu dài.

| Tiêu chí | Mô hình Cơ bản (NLP) | Mô hình Nâng cao (NLP1) |
|---|---|---|
| **Dataset** | Multi30K (Nhỏ, đơn giản) | **WMT 2014** (Lớn, phức tạp, đa dạng chủ đề) |
| **Encoder** | LSTM 1 chiều (Unidirectional) | **Bi-LSTM** (Bidirectional - 2 chiều xuôi/ngược) |
| **Decoding** | Greedy Decoding | **Beam Search** (Tìm kiếm chùm) |
| **Cơ chế chính** | Context Vector tĩnh (Cố định) | **Bahdanau Attention** (Context Vector động) |

## 2. Kiến trúc chi tiết

### 2.1 Bidirectional LSTM Encoder
Thay vì chỉ đọc câu từ trái sang phải, Encoder của NLP1 đọc theo 2 hướng:
- **Forward:** $x_1 \rightarrow x_2 \rightarrow ... \rightarrow x_T$
- **Backward:** $x_T \rightarrow x_{T-1} \rightarrow ... \rightarrow x_1$
=> Kết quả là mỗi từ được đại diện bởi sự kết hợp của cả ngữ cảnh phía trước và phía sau nó.

### 2.2 Bahdanau Attention (Additive Attention)
Đây là "trái tim" của mô hình.
- **Vấn đề:** Ở mô hình cũ, Encoder phải nén toàn bộ ý nghĩa câu vào 1 vector duy nhất ($z$). Câu quá dài thì vector này bị "quá tải" thông tin.
- **Giải pháp:** Attention cho phép Decoder "nhìn lại" (look back) toàn bộ các từ trong câu gốc tại mỗi bước dịch. Nó tự động gán trọng số (alpha) để biết từ nào quan trọng hợn.
  - Ví dụ: Khi dịch chữ "eat", model sẽ chú ý nhiều vào chữ "manger" bên câu Pháp.

### 2.3 Beam Search Decoding
- **Greedy (Cũ):** Tại mỗi bước chỉ chọn 1 từ có xác suất cao nhất. Dễ bị sai nếu chọn nhầm từ đầu tiên.
- **Beam Search (Mới):** Tại mỗi bước giữ lại $k$ phương án tốt nhất (ví dụ $k=5$). Sau khi dịch xong cả câu, chọn ra phương án có tổng điểm cao nhất. Giúp câu văn mượt mà hơn.

## 3. Thực nghiệm & Kết quả

### Cấu hình huấn luyện
- **Hidden Size:** 512 (Tăng cường khả năng lưu trữ thông tin).
- **Optimizer:** Adam.
- **Dataset Limit:** 5,000 câu (Do giới hạn phần cứng, chỉ train trên tập con nhỏ để kiểm chứng kiến trúc).

### Kết quả (BLEU Score)
- **BLEU Score:** ~2.18% (Thấp hơn mô hình cơ bản).

**Phân tích nguyên nhân:**
1.  **Dữ liệu huấn luyện quá ít:** 5,000 câu là con số rất nhỏ đối với Deep Learning (so với 29,000 câu của bản cơ bản hay 36 triệu câu của WMT14 gốc).
2.  **Độ khó của Dataset:** WMT 2014 là văn bản tin tức/tạp chí, câu văn phức tạp và nhiều từ vựng hiếm (OOV) hơn rất nhiều so với văn bản mô tả tranh của Multi30K.
3.  **Mục tiêu đạt được:** Mặc dù điểm số chưa cao, nhưng mô hình đã **chứng minh được sự hoạt động chính xác** của các thuật toán phức tạp (Attention, Beam Search) và chạy ổn định không lỗi.

## 4. Hướng dẫn chạy (Usage)

### 4.1 Huấn luyện (Train)
```bash
python main.py --train --limit 5000
```
*(Tham số `--limit` giúp chạy nhanh trên máy cấu hình yếu)*

### 4.2 Đánh giá (Evaluate)
```bash
python main.py --eval --limit 5000
```

### 4.3 Dịch thử một câu mới (Inference)
Sử dụng Beam Search để dịch:
```bash
python main.py --translate "A group of people standing in front of a building" --limit 5000
```

## 5. Kết luận & Hướng phát triển

**Kết luận:**
Đồ án mở rộng đã triển khai thành công các kỹ thuật tiên tiến trong Dịch máy (NMT) giai đoạn trước Transformer. Code được tổ chức bài bản, module hóa (Encoder, Decoder, Attention tách biệt).

**Hướng phát triển:**
- Huấn luyện trên toàn bộ dataset WMT 2014 (nếu có GPU mạnh).
- Áp dụng kỹ thuật **Subword Tokenization (BPE)** để xử lý từ hiếm tốt hơn.
- Nâng cấp lên kiến trúc **Transformer** (Self-Attention) để đạt hiệu suất SOTA.
