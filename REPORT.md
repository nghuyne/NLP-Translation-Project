# BÁO CÁO ĐỒ ÁN XỬ LÝ NGÔN NGỮ TỰ NHIÊN
## Đề tài: Dịch máy Anh-Pháp với mô hình Encoder-Decoder LSTM

**Sinh viên thực hiện:**
- [Tên Sinh Viên 1] - [MSSV]
- [Tên Sinh Viên 2] - [MSSV]

---

## 1. Sơ đồ kiến trúc (Architecture)

Mô hình sử dụng kiến trúc **Seq2Seq (Encoder-Decoder)** với **LSTM**.

- **Encoder:**
  - Input: Chuỗi token tiếng Anh (đã qua Embedding).
  - Layer: LSTM (2 layers).
  - Output: Context Vector (Hidden state $h_n$ và Cell state $c_n$ cuối cùng).
- **Decoder:**
  - Input: Context Vector từ Encoder + Token trước đó (bắt đầu bằng `<sos>`).
  - Layer: LSTM (2 layers).
  - Output: Dự đoán token tiếp theo trong chuỗi tiếng Pháp.

*(Bạn có thể chèn hình ảnh sơ đồ mô hình vào đây)*

## 2. Quá trình huấn luyện (Training Process)

### Cấu hình
- **Embedding Dim:** 256
- **Hidden Dim:** 512
- **Layers:** 2
- **Dropout:** 0.3
- **Optimizer:** Adam (lr=1e-3)
- **Loss Function:** CrossEntropyLoss (ignore padding)

### Biểu đồ Loss
*(Chèn hình ảnh `loss_curve.png` vào đây)*

Nhận xét:
- Loss giảm dần qua các epoch, chứng tỏ mô hình đang học tốt.
- Không có dấu hiệu overfitting nặng (val loss giảm cùng train loss).

## 3. Kết quả đánh giá (Evaluation)

### BLEU Score
- **Average BLEU:** [Điền số từ bleu_result.txt, ví dụ: 0.1283]

### Ví dụ dịch và Phân tích lỗi

| STT | Câu gốc (EN) | Câu tham chiếu (REF) | Câu dự đoán (PRED) | Nhận xét / Phân tích lỗi |
|---|---|---|---|---|
| 1 | a group of people standing in front of an igloo. | un groupe de personnes debout devant un igloo. | un groupe de personnes debout devant un bâtiment . | **Tốt:** Cấu trúc đúng. **Lỗi:** Từ vựng (igloo -> tòa nhà). |
| 2 | [Câu EN 2] | [Câu REF 2] | [Câu PRED 2] | [Phân tích] |
| 3 | [Câu EN 3] | [Câu REF 3] | [Câu PRED 3] | [Phân tích] |
| 4 | [Câu EN 4] | [Câu REF 4] | [Câu PRED 4] | [Phân tích] |
| 5 | [Câu EN 5] | [Câu REF 5] | [Câu PRED 5] | [Phân tích] |

**Các loại lỗi phổ biến:**
1.  **OOV (Out of Vocabulary):** Gặp từ lạ chuyển thành `<unk>`.
2.  **Mất thông tin:** Câu dài thường bị mất ý ở đoạn cuối do context vector cố định bị quá tải.
3.  **Ngữ pháp:** Sai trật tự từ trong tiếng Pháp.

## 4. Hướng dẫn chạy mã nguồn

1.  **Cài đặt thư viện:**
    ```bash
    pip install -r requiment.txt
    python -m spacy download en_core_web_sm
    python -m spacy download fr_core_news_sm
    ```
2.  **Huấn luyện:**
    ```bash
    python main_train.py
    ```
3.  **Đánh giá:**
    ```bash
    python evaluate_bleu.py
    ```
4.  **Dịch thử (Inference):**
    ```python
    from inference import translate
    print(translate("a man is walking"))
    ```

---
**Kết luận:** Đồ án đã hoàn thành các yêu cầu cơ bản về xây dựng mô hình Seq2Seq LSTM from scratch.
