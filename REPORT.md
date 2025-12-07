# BÁO CÁO ĐỒ ÁN XỬ LÝ NGÔN NGỮ TỰ NHIÊN
## Đề tài: Dịch máy Anh-Pháp với mô hình Encoder-Decoder LSTM

**Sinh viên thực hiện:**
- [Trần Ngọc Huy] - [3122411071]
- [Lý Vĩnh Tài] - [3122411180]

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
![alt text](image.png)

Nhận xét:
- Loss giảm dần qua các epoch, chứng tỏ mô hình đang học tốt.
- Không có dấu hiệu overfitting nặng (val loss giảm cùng train loss).

## 3. Kết quả đánh giá (Evaluation)

### BLEU Score
- **Average BLEU:** [Average BLEU: 0.2564

Examples:
------------------------------------------------------------
EN  : a man in an orange hat starring at something.
REF : un homme avec un chapeau orange regardant quelque chose.
PRED: un homme avec un chapeau orange se quelque chose .
BLEU: 0.6580
------------------------------------------------------------
EN  : a boston terrier is running on lush green grass in front of a white fence.
REF : un terrier de boston court sur l'herbe verdoyante devant une clôture blanche.
PRED: un caniche de court sur sur la herbe verte et blanche .
BLEU: 0.0000
------------------------------------------------------------
EN  : a girl in karate uniform breaking a stick with a front kick.
REF : une fille en tenue de karaté brisant un bâton avec un coup de pied.
PRED: une fille en tenue de karaté attrape un ballon avec un ballon de lui .
BLEU: 0.3943
------------------------------------------------------------
EN  : five people wearing winter jackets and helmets stand in the snow, with snowmobiles in the background.
REF : cinq personnes avec des vestes d'hiver et des casques sont debout dans la neige, avec des motoneiges en arrière-plan.
PRED: cinq personnes portant des casques et des casques de soleil sont debout dans la rue avec des arbres en arrière-plan .
BLEU: 0.2361
------------------------------------------------------------
EN  : people are fixing the roof of a house.
REF : des gens réparent le toit d'une maison.
PRED: des gens se l' arrière d' une maison .
BLEU: 0.3549
------------------------------------------------------------
EN  : a man in light colored clothing photographs a group of men wearing dark suits and hats standing around a woman dressed in a strapless gown.
REF : un homme en tenue claire photographie un groupe d'hommes portant des costumes sombres et des chapeaux, debout autour d'une femme vêtue d'une robe bustier.
PRED: un homme avec des vêtements de cérémonie en costume de une femme en costume et une femme debout debout d' une femme en costumes de de et d' un restaurant .
BLEU: 0.0000
------------------------------------------------------------
EN  : a group of people standing in front of an igloo.
REF : un groupe de personnes debout devant un igloo.
PRED: un groupe de personnes debout devant un public .
BLEU: 0.7506
------------------------------------------------------------
EN  : a boy in a red uniform is attempting to avoid getting out at home plate, while the catcher in the blue uniform is attempting to catch him.
REF : un garçon en uniforme rouge essaie d'éviter de sortir du marbre, tandis que le receveur en tenue bleue essaie de l'attraper.
PRED: un garçon en maillot rouge tente de frapper le ballon tandis que il est est <unk> tandis que le receveur essaie de lui qui est en l' air .
BLEU: 0.1450
------------------------------------------------------------
EN  : a guy works on a building.
REF : un gars travaille sur un bâtiment.
PRED: un gars travaillant sur un bâtiment .
BLEU: 0.4889
------------------------------------------------------------
EN  : a man in a vest is sitting in a chair and holding magazines.
REF : un homme en gilet est assis dans une chaise et tient des magazines.
PRED: un homme en veste est assis dans une chaise et et de la .
BLEU: 0.4572]

### Ví dụ dịch và Phân tích lỗi

| STT | Câu gốc (EN) | Câu tham chiếu (REF) | Câu dự đoán (PRED) | Nhận xét / Phân tích lỗi |
|---|---|---|---|---|
| 1 | a group of people standing in front of an igloo. | un groupe de personnes debout devant un igloo. | un groupe de personnes debout devant un bâtiment . | **Tốt:** Cấu trúc đúng. **Lỗi:** Từ vựng (igloo -> tòa nhà). |
| 2 | a guy works on a building. | un gars travaille sur un bâtiment. | un gars travaillant sur un bâtiment . | **Tốt:** Cấu trúc đúng. **Lỗi:** sai ngữ pháp câu gốc là động từ (travaillant) còn câu dịch là danh từ (travaille -> công việc ) |
| 3 | a girl in karate uniform breaking a stick with a front kick. | une fille en tenue de karaté brisant un bâton avec un coup de pied.	 | une fille en tenue de karaté attrape un ballon avec un ballon de lui .	 | **Sai nghĩa hoàn toàn (Hallucination):** Model đoán mờ ám dựa trên ngữ cảnh ("chơi thể thao") nên dịch sai hành động (bẻ gậy -> bắt bóng) và vật thể (gậy -> bóng).
 |
| 4 | five people wearing winter jackets and helmets stand in the snow...	 | cinq personnes avec des vestes d'hiver et des casques...	 | cinq personnes portant des casques et des casques de soleil...	 | **Lỗi lặp từ (Repetition):** Model bị "kẹt", lặp lại cụm từ "des casques" hai lần liên tiếp mà không thoát ra được để dịch vế sau.
 |
| 5 | a boston terrier is running on lush green grass in front of a white fence.	 | un terrier de boston court sur l'herbe verdoyante devant une clôture blanche. | un caniche de court sur sur la herbe verte et blanche  | **Mất thông tin & Sai từ:** Model không biết từ "Boston Terrier" nên đoán thành "caniche" (chó xù), và bị mất hoàn toàn đoạn cuối "fence" (hàng rào) do câu quá dài (Short-term memory problem). |

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
