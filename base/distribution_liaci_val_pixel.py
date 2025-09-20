import os
import numpy as np
from PIL import Image
import pandas as pd

# 경로 설정
mask_dir = r"C:\Users\chsobn0710\Desktop\biofouling\data\liaci\masks"
split_csv = r"C:\Users\chsobn0710\Desktop\biofouling\data\liaci\splits.csv"

# split.csv 불러오기
df = pd.read_csv(split_csv)
val_ids = set(df[df['split'] == 'val']['id'])

count_M, count_S = [], []

for fname in os.listdir(mask_dir):
    if not any(fname.endswith(suffix) for suffix in ['_M.png', '_S.png']):
        continue

    stem = fname.split('_')[1].split('.')[0]
    if f"image_{stem}" not in val_ids:
        continue  # VAL이 아닌 경우 skip

    path = os.path.join(mask_dir, fname)
    mask = np.array(Image.open(path))

    if fname.endswith("_M.png"):
        count_M.append((mask == 255).sum())
    elif fname.endswith("_S.png"):
        count_S.append((mask == 255).sum())

def summarize(name, counts):
    total = len(counts)
    avg = np.mean(counts)
    zero = sum(c == 0 for c in counts)
    le10 = sum(c <= 10 for c in counts)
    le100 = sum(c <= 100 for c in counts)
    gt100 = sum(c > 100 for c in counts)

    print(f"\n=== [LIACi VAL] {name} 마스크 픽셀 통계 ===")
    print(f"총 이미지 수: {total}")
    print(f" 0픽셀: {zero}장")
    print(f" ≤10픽셀: {le10}장")
    print(f" ≤100픽셀: {le100}장")
    print(f" >100픽셀: {gt100}장")
    print(f" 평균 픽셀 수: {avg:.2f}")

summarize("Marine(M)", count_M)
summarize("Structure(S)", count_S)
