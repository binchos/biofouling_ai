import os
import numpy as np
from PIL import Image

mask_dir = r"C:\Users\chsobn0710\PycharmProjects\biofouling\data\liaci\masks"
count_M, count_S = [], []

for fname in os.listdir(mask_dir):
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

    print(f"\n=== [LIACi] {name} 마스크 픽셀 통계 ===")
    print(f"총 이미지 수: {total}")
    print(f" 0픽셀: {zero}장")
    print(f" ≤10픽셀: {le10}장")
    print(f" ≤100픽셀: {le100}장")
    print(f" >100픽셀: {gt100}장")
    print(f" 평균 픽셀 수: {avg:.2f}")

summarize("Marine(M)", count_M)
summarize("Structure(S)", count_S)
