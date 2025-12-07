# src/utils.py
import random
import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu

# cấu hình NHẸ cho CPU nhưng đã đổi lại gần như là theo yêu cầu vì đã chuyển sang GPU
EMB_DIM = 256        # trước là 256
HID_DIM = 512        # trước là 512
N_LAYERS = 2         # trước là 2
DROPOUT = 0.3        # nhẹ hơn
BATCH_SIZE = 64      # to hơn để ít batch hơn (nếu RAM đủ)
MAX_VOCAB = 10000     # giảm vocab để embedding nhỏ hơn
MIN_FREQ = 2
N_EPOCHS = 10         # 3–5 epoch, tuỳ
TEACHER_FORCING = 0.5
CLIP = 1.0
LR = 1e-3
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_bleu(ref_tokens, hyp_tokens):
    return sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    #   Tính BLEU score giữa câu tham chiếu và câu dự đoán.
    # - ref_tokens: list[str] - danh sách các từ trong câu tham chiếu
    # - hyp_tokens: list[str] - danh sách các từ trong câu dịch dự đoán
