from pathlib import Path
from PIL import Image
import csv, random

SRC = Path(r"C:\Users\chsobn0710\Desktop\biofouling\kowp")   # 원본
DST = SRC.parent / "kowp_prepared"                           # 결과

IMAGES_SRC = SRC / "images"
MG_SRC     = SRC / "masks" / "marine_growth"
S_SRC      = SRC / "masks" / "structure"

IMAGES_DST = DST / "images"
MASKS_DST  = DST / "masks"
SPLIT_CSV  = DST / "splits.csv"

VAL_RATIO = 0.2  # 대략 20%를 val로

def main():
    assert IMAGES_SRC.exists(), f"Missing: {IMAGES_SRC}"
    assert MG_SRC.exists(),     f"Missing: {MG_SRC}"
    assert S_SRC.exists(),      f"Missing: {S_SRC}"
    IMAGES_DST.mkdir(parents=True, exist_ok=True)
    MASKS_DST.mkdir(parents=True, exist_ok=True)

    images = sorted(IMAGES_SRC.glob("*.png"))
    assert images, "images 폴더에 png가 없습니다."

    ids = []
    for img in images:
        stem = img.stem                         # 예: frame_000001
        mg = (MG_SRC / f"{stem}.png")
        s  = (S_SRC  / f"{stem}.png")
        if not mg.exists() or not s.exists():
            print(f"[WARN] 마스크 없음: {stem} (MG={mg.exists()}, S={s.exists()})")
            continue

        Image.open(img).save(IMAGES_DST / f"{stem}.png")
        Image.open(s).convert("L").save(MASKS_DST / f"{stem}_S.png")
        Image.open(mg).convert("L").save(MASKS_DST / f"{stem}_M.png")
        ids.append(stem)

    assert ids, "유효한 (이미지+S+M) 페어가 없습니다."
    random.seed(42); random.shuffle(ids)
    n = len(ids); n_val = max(1, int(round(n * VAL_RATIO)))
    val_ids = set(ids[:n_val]); train_ids = ids[n_val:]

    with SPLIT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "split"])
        for sid in train_ids: w.writerow([sid, "train"])
        for sid in sorted(val_ids): w.writerow([sid, "val"])

    print("=== Done ===")
    print(f"Prepared at: {DST}")
    print(f"counts -> train: {len(train_ids)}, val: {len(val_ids)}")

if __name__ == "__main__":
    main()
