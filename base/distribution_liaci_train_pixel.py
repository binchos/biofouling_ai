import os
import numpy as np
import pandas as pd
from PIL import Image

# --- [1] 경로 설정 ---
mask_dir = r"C:\Users\chsobn0710\Desktop\biofouling\data\liaci\masks"
split_csv = r"C:\Users\chsobn0710\Desktop\biofouling\data\liaci\splits.csv"

# --- [2] split.csv 읽기 ---
df = pd.read_csv(split_csv)
train_ids = set(df[df["split"] == "train"]["id"])

# --- [3] 통계 저장용 리스트 ---
count_M, count_S = [], []

for fname in os.listdir(mask_dir):
    if not fname.endswith(".png"):
        continue

    # ID 추출 방식 수정!
    image_id = "_".join(fname.split("_")[:2])
    if image_id not in train_ids:
        continue

    path = os.path.join(mask_dir, fname)
    mask = np.array(Image.open(path))

    if fname.endswith("_M.png"):
        count_M.append((mask == 255).sum())
    elif fname.endswith("_S.png"):
        count_S.append((mask == 255).sum())


# --- [4] 통계 출력 함수 ---
def summarize(name, counts):
    total = len(counts)
    avg = np.mean(counts)
    zero = sum(c == 0 for c in counts)
    le10 = sum(c <= 10 for c in counts)
    le100 = sum(c <= 100 for c in counts)
    gt100 = sum(c > 100 for c in counts)

    print(f"\n=== [LIACi TRAIN] {name} 마스크 픽셀 통계 ===")
    print(f"총 이미지 수: {total}")
    print(f" 0픽셀: {zero}장")
    print(f" ≤10픽셀: {le10}장")
    print(f" ≤100픽셀: {le100}장")
    print(f" >100픽셀: {gt100}장")
    print(f" 평균 픽셀 수: {avg:.2f}")

# --- [5] 결과 출력 ---
summarize("Marine(M)", count_M)
summarize("Structure(S)", count_S)
