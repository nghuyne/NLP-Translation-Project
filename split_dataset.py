# FILE: split_dataset.py
import os

SRC_IN = "data/giga-fren.release2.fixed.en"
TRG_IN = "data/giga-fren.release2.fixed.fr"

TRAIN_EN = "data/train.en"
TRAIN_FR = "data/train.fr"

VAL_EN   = "data/val.en"
VAL_FR   = "data/val.fr"

TEST_EN  = "data/test.en"
TEST_FR  = "data/test.fr"

# Số lượng cần chia
NUM_TRAIN = 2_000_000
NUM_VAL   = 10_000
NUM_TEST  = 10_000

MAX_NEED = NUM_TRAIN + NUM_VAL + NUM_TEST

print("Đang đếm số dòng... (cần dòng để chia đúng)")

# --- Đếm số dòng nhanh, KHÔNG đọc toàn bộ file vào RAM ---
def count_lines(path):
    count = 0
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        for _ in f:
            count += 1
            if count % 500000 == 0:
                print("  đã đếm:", count)
    return count

total_lines = count_lines(SRC_IN)
print(f"Tổng số câu EN: {total_lines:,}")

# Kiểm tra đủ dữ liệu để chia
if total_lines < MAX_NEED:
    raise RuntimeError(f"Dataset quá nhỏ, cần {MAX_NEED} câu.")

print("Bắt đầu chia dataset (streaming, không load toàn file)...")

# --- Mở tất cả file output ---
out_train_en = open(TRAIN_EN, "w", encoding="utf8")
out_train_fr = open(TRAIN_FR, "w", encoding="utf8")
out_val_en   = open(VAL_EN, "w", encoding="utf8")
out_val_fr   = open(VAL_FR, "w", encoding="utf8")
out_test_en  = open(TEST_EN, "w", encoding="utf8")
out_test_fr  = open(TEST_FR, "w", encoding="utf8")

# --- Đọc từng dòng song song (tiết kiệm RAM) ---
with open(SRC_IN, "r", encoding="utf8", errors="ignore") as src_f, \
     open(TRG_IN, "r", encoding="utf8", errors="ignore") as trg_f:

    idx = 0
    for src_line, trg_line in zip(src_f, trg_f):

        if idx < NUM_TRAIN:
            out_train_en.write(src_line)
            out_train_fr.write(trg_line)

        elif idx < NUM_TRAIN + NUM_VAL:
            out_val_en.write(src_line)
            out_val_fr.write(trg_line)

        elif idx < NUM_TRAIN + NUM_VAL + NUM_TEST:
            out_test_en.write(src_line)
            out_test_fr.write(trg_line)

        else:
            break

        idx += 1
        if idx % 500000 == 0:
            print(f"  Đã xử lý {idx:,} dòng...")

# Đóng file output
out_train_en.close()
out_train_fr.close()
out_val_en.close()
out_val_fr.close()
out_test_en.close()
out_test_fr.close()

print("\nHoàn tất chia dataset!")
print(f"train: {NUM_TRAIN:,} câu")
print(f"val:   {NUM_VAL:,} câu")
print(f"test:  {NUM_TEST:,} câu")
