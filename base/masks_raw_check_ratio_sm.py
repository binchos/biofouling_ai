from pathlib import Path
import numpy as np
from PIL import Image

def bin_load(p):
    arr = np.array(Image.open(p).convert("L"))
    return arr > 127

def main():
    root = Path("/data/liaci/masks")
    s_files = sorted(root.glob("*_S.png"))

    total_S, total_M = 0, 0
    counts = []  # 샘플별 비율 기록

    for s_path in s_files:
        id_ = s_path.stem[:-2]  # "_S" 잘라내기
        m_path = root / f"{id_}_M.png"
        if not m_path.exists():
            continue

        S = bin_load(s_path)
        M = bin_load(m_path)

        nS, nM = int(S.sum()), int(M.sum())
        total_S += nS
        total_M += nM
        ratio = (nM / nS * 100) if nS > 0 else 0
        counts.append((id_, nS, nM, ratio))

    print(f"[전체] 구조물 픽셀 = {total_S:,}")
    print(f"[전체] 마린 픽셀   = {total_M:,}")
    print(f"[전체] 평균 비율   = {total_M/total_S*100:.2f}%")

    # 예시 5개 출력
    print("\n샘플 예시:")
    for id_, nS, nM, ratio in counts[:5]:
        print(f"{id_}: S={nS:,}, M={nM:,}, 비율={ratio:.2f}%")

if __name__ == "__main__":
    main()
