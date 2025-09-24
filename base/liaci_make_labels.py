from pathlib import Path
import argparse, csv
import numpy as np
from PIL import Image
import shutil

try:
    import cv2
except Exception:
    cv2 = None

STRUCTURE_CLASSES = [
    "anode", "bilge_keel", "over_board_valves", "paint_peel",
    "propeller", "sea_chest_grating", "corrosion", "defect","ship_hull"
]
MARINE_CLASS = "marine_growth"
MASK_EXTS = [".png", ".bmp", ".tif", ".tiff"]

def find_mask(class_dir: Path, stem: str):
    for ext in MASK_EXTS:
        p = class_dir / f"{stem}{ext}"
        if p.exists():
            return p
    cands = list(class_dir.glob(f"{stem}.*"))
    return cands[0] if cands else None

def load_bin(path_img: Path, ref_shape=None):
    if path_img is None or not path_img.exists():
        return None
    arr = np.array(Image.open(path_img).convert("L"))
    m = arr > 127
    if ref_shape is not None and m.shape != ref_shape:
        raise ValueError(f"[해상도 불일치] {path_img} {m.shape} != {ref_shape}")
    return m

def save_bin(mask_bool: np.ndarray, path_png: Path):
    path_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_bool.astype(np.uint8) * 255).save(path_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_csv", type=str,
                    default="C:/Users/chsobn0710/Desktop/biofouling/data/liaci/splits.csv")
    ap.add_argument("--raw_root", type=str,
                    default="C:/Users/chsobn0710/Desktop/biofouling/raw_data1/liaci")
    ap.add_argument("--out_root", type=str,
                    default="C:/Users/chsobn0710/Desktop/biofouling/data_1/liaci")
    args = ap.parse_args()

    splits_csv = Path(args.splits_csv)
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    raw_masks = raw_root / "masks"
    raw_images = raw_root / "images"

    out_masks = out_root / "masks"
    out_proc = out_root / "masks_proc"
    out_images = out_root / "images"

    out_masks.mkdir(parents=True, exist_ok=True)
    out_proc.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    # === 1. splits.csv 읽기 ===
    ids = []
    with open(splits_csv, "r") as f:
        rdr = csv.reader(f)
        next(rdr)  # header skip
        for row in rdr:
            ids.append(row[0])
    print(f"[Info] splits.csv 기반 id 수: {len(ids)}")

    kept, skipped = [], []
    for id_ in ids:
        ref_shape = None
        S = None

        # === 구조물 마스크 합치기 ===
        for cls in STRUCTURE_CLASSES:
            cdir = raw_masks / cls
            mp = find_mask(cdir, id_)
            if mp is None:
                continue
            m = load_bin(mp, ref_shape=None)
            if m is None:
                continue
            if ref_shape is None:
                ref_shape = m.shape
            elif m.shape != ref_shape:
                raise ValueError(f"[해상도 불일치] {mp} {m.shape} vs {ref_shape}")
            S = m if S is None else (S | m)

        if S is None or S.sum() == 0:
            skipped.append((id_, "no_structure_or_S0"))
            continue

        # === 해양생물 마스크 ===
        mdir = raw_masks / MARINE_CLASS
        M = None
        mp = find_mask(mdir, id_)
        if mp:
            M = load_bin(mp, ref_shape=ref_shape)
        if M is None:
            M = np.zeros_like(S, dtype=bool)

        M = (M & S)

        # === 저장 ===
        save_bin(S, out_masks / f"{id_}_S.png")
        save_bin(M, out_masks / f"{id_}_M.png")

        if cv2 is not None:
            r = 20
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
            M_d = cv2.dilate(M.astype(np.uint8), k)
            IGN_band = (M_d.astype(bool) & S)
            IGN = IGN_band
            valid_S = (S & (~IGN))
            save_bin(IGN, out_proc / f"{id_}_IGN.png")
            save_bin(valid_S, out_proc / f"{id_}_validS.png")

        # === 이미지 복사 ===
        src_img = raw_images / f"{id_}.jpg"
        if not src_img.exists():
            src_img = raw_images / f"{id_}.png"
        if src_img.exists():
            shutil.copy(src_img, out_images / src_img.name)

        kept.append(id_)

    print(f"[Info] kept={len(kept)} | skipped={len(skipped)}")
    if skipped:
        (out_root / "prep_skipped.csv").write_text(
            "id,reason\n" + "\n".join([f"{i},{r}" for i,r in skipped]),
            encoding="utf-8"
        )

    # === 기존 splits.csv도 같이 복사해 두기 ===
    shutil.copy(splits_csv, out_root / "splits.csv")
    print(f"[OK] splits.csv 복사 완료: {out_root/'splits.csv'}")
    print("[완료] data_1/liaci 정리 끝")

if __name__ == "__main__":
    main()
