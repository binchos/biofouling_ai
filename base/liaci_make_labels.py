from pathlib import Path
import argparse, csv, random
import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

# S(Structure)에 포함할 클래스들 (segmentation 제외)
STRUCTURE_CLASSES = [
    "anode", "bilge_keel", "over_board_valves", "paint_peel",
    "propeller", "sea_chest_grating", "corrosion", "defect",
]
MARINE_CLASS = "marine_growth"

# 허용 확장자
MASK_EXTS = [".png", ".bmp", ".tif", ".tiff"]

def find_mask(class_dir: Path, stem: str):
    """클래스 폴더에서 stem(확장자 없는 파일명)에 맞는 마스크 파일을 찾음."""
    for ext in MASK_EXTS:
        p = class_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # 백업: stem.* 첫 번째
    cands = list(class_dir.glob(f"{stem}.*"))
    return cands[0] if cands else None

def load_bin(path_img: Path, ref_shape=None):
    """0/255 보장: 회색값은 >127 임계로 이진화. 해상도 체크."""
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

def collect_ids(raw_masks_dir: Path):
    """모든 클래스 폴더에서 stem을 모아 id 리스트 생성 (확장자 혼용 대응)."""
    ids = set()
    for cls_dir in raw_masks_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        for ext in MASK_EXTS:
            for p in cls_dir.glob(f"*{ext}"):
                ids.add(p.stem)  # image_0001
    return sorted(ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str,
                    default="C:/Users/chsobn0710/PycharmProjects/biofouling/raw_data1/liaci",
                    help="LIACI 데이터 루트 경로")
    ap.add_argument("--raw_masks_dir", type=str, default="masks", help="클래스별 마스크 최상위 폴더(예: masks)")
    ap.add_argument("--out_masks", type=str, default="masks")
    ap.add_argument("--out_masks_proc", type=str, default="masks_proc")
    ap.add_argument("--dilate_r", type=int, default=20, help="IGN 띠 반경(px, OpenCV 필요)")
    ap.add_argument("--split_train", type=float, default=0.8)
    ap.add_argument("--split_val", type=float, default=0.2)  # ← test 안 씀, train/val만
    args = ap.parse_args()

    root = Path(args.root)
    raw_masks = root / args.raw_masks_dir
    out_raw = root / args.out_masks
    out_proc = root / args.out_masks_proc
    assert raw_masks.exists(), f"[Error] raw_masks 폴더 없음: {raw_masks}"
    out_raw.mkdir(parents=True, exist_ok=True)
    out_proc.mkdir(parents=True, exist_ok=True)

    ids = collect_ids(raw_masks)
    print(f"[Info] 후보 id 수: {len(ids)} (예시: {ids[:5]})")

    kept, skipped = [], []
    for id_ in ids:
        ref_shape = None
        S = None

        # S: 구조 클래스 OR 합
        for cls in STRUCTURE_CLASSES:
            cdir = raw_masks / cls
            if not cdir.exists():
                continue
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

        # M: marine_growth
        mdir = raw_masks / MARINE_CLASS
        M = None
        if mdir.exists():
            mp = find_mask(mdir, id_)
            if mp:
                M = load_bin(mp, ref_shape=ref_shape)
        if M is None:
            M = np.zeros_like(S, dtype=bool)

        # 강제 클리핑
        M = (M & S)

        # 저장 (PNG, 0/255)
        save_bin(S, out_raw / f"{id_}_S.png")
        save_bin(M, out_raw / f"{id_}_M.png")

        # IGN/valid_S (선택)
        if cv2 is not None and args.dilate_r > 0:
            r = int(args.dilate_r)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
            M_d = cv2.dilate(M.astype(np.uint8), k)
            IGN_band = (M_d.astype(bool) & S)

            ignm_path = out_raw / f"{id_}_IGN_manual.png"
            if ignm_path.exists():
                IGN_m = np.array(Image.open(ignm_path).convert("L")) > 127
                if IGN_m.shape != S.shape:
                    raise ValueError(f"[IGN_manual 해상도 불일치] {ignm_path}")
            else:
                IGN_m = np.zeros_like(S, bool)

            IGN = (IGN_band | IGN_m)
            valid_S = (S & (~IGN))
            save_bin(IGN, out_proc / f"{id_}_IGN.png")
            save_bin(valid_S, out_proc / f"{id_}_validS.png")

        kept.append(id_)

    print(f"[Info] kept={len(kept)} | skipped={len(skipped)}")
    if skipped:
        (root / "prep_skipped.csv").write_text(
            "id,reason\n" + "\n".join([f"{i},{r}" for i, r in skipped]),
            encoding="utf-8"
        )
        print(f"[Info] 제외 목록 저장: {root/'prep_skipped.csv'}")

    # splits.csv (train/val만)
    random.shuffle(kept)
    n = len(kept)
    a = int(args.split_train * n)
    rows = [("id", "split")] \
           + [(i, "train") for i in kept[:a]] \
           + [(i, "val") for i in kept[a:]]
    with open(root / "splits.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"[OK] splits.csv 저장: {root/'splits.csv'}")
    print("[완료] LIACI 정리 끝 (혼합 확장자 대응, segmentation 미포함)")

if __name__ == "__main__":
    main()
