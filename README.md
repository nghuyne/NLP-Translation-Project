# Hướng dẫn chạy dự án NLP1 (Advanced)

Đây là phiên bản nâng cao sử dụng **Attention Mechanism (Bahdanau)** và dataset lớn **WMT 2014 (Giga-fren)**.

## 1. Cài đặt môi trường

Đảm bảo đã cài đặt các thư viện cần thiết:

```bash
pip install -r requiment.txt
pip install sacrebleu
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

## 2. Huấn luyện (Training)

Để bắt đầu huấn luyện mô hình:

```bash
python main.py --train --epochs 5 --checkpoint checkpoints/best_model.pt
```

*Lưu ý:* Dataset rất lớn, quá trình train có thể mất nhiều thời gian (vài giờ đến vài ngày tùy GPU).

## 3. Đánh giá (Evaluation)

Tính điểm BLEU trên tập test:

```bash
python main.py --eval --checkpoint checkpoints/best_model.pt
```

## 4. Dịch thử (Translate)

Dịch một câu tiếng Anh sang tiếng Pháp:

```bash
python main.py --translate "A group of people standing in front of a building" --checkpoint checkpoints/best_model.pt
```

## 5. Cấu trúc thư mục

- `src/`: Mã nguồn (encoder, decoder, attention...)
- `data/`: Dữ liệu (train, val, test)
- `checkpoints/`: Nơi lưu model đã train
- `main.py`: File chính để chạy chương trình

## 6. Điểm nổi bật (So với bản cơ bản)

- **Attention Mechanism:** Giúp mô hình tập trung vào các phần quan trọng của câu gốc khi dịch.
- **Bidirectional Encoder:** Encoder 2 chiều giúp nắm bắt ngữ cảnh tốt hơn.
- **Dataset lớn:** Sử dụng WMT 2014 giúp mô hình học được nhiều từ vựng và cấu trúc hơn.
