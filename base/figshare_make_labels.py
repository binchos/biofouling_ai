# figshare_make_labels.py
# 사용 예:
#   python figshare_make_labels.py --root raw_data/figshare --meta metadata.csv --images images --make_stratified_if_missing
#   (meta가 tsv면 --meta metadata.tsv 로만 바꾸면 됨. 구분자는 자동 감지)

from pathlib import Path
import argparse
import pandas as pd

SPLIT_MAP = {
    "training": "train",
    "validation": "val",
    "testing": "test",
    "train": "train",
    "val": "val",
    "test": "test",
}

def resolve_paths(args):
    # 스크립트 파일 기준으로 프로젝트 루트 추정 (…/base/ 상위가 루트라고 가정)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    root = (project_root / args.root).resolve()
    meta_path = (root / args.meta).resolve()
    img_dir = (root / args.images).resolve()
    out_path = (root / args.out).resolve()
    return project_root, root, meta_path, img_dir, out_path

def find_meta_fallback(root, meta_path):
    if meta_path.exists():
        return meta_path
    candidates = [
        root / "metadata.csv",
        root / "metadata.tsv",
        root / "meta.csv",
        root / "meta.tsv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return meta_path

def load_meta(meta_path):
    # 쉼표/탭 자동 감지
    try:
        return pd.read_csv(meta_path, sep=None, engine="python")
    except Exception as e:
        raise SystemExit(f"[Error] 메타 파일 읽기 실패: {meta_path}\n{e}")

def normalize_split_column(df, make_stratified_if_missing):
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.lower().map(SPLIT_MAP).fillna("train")
        return df

    if not make_stratified_if_missing:
        df["split"] = "train"
        return df

    # 층화 8:1:1
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise SystemExit("[Error] scikit-learn 미설치: pip install scikit-learn "
                         "또는 메타에 split 컬럼을 넣어주세요.")

    idx = df.index
    tr, tmp = train_test_split(idx, test_size=0.2, random_state=42, stratify=df["label_cls"])
    va, te = train_test_split(tmp, test_size=0.5, random_state=42, stratify=df.loc[tmp, "label_cls"])
    df["split"] = "train"
    df.loc[va, "split"] = "val"
    df.loc[te, "split"] = "test"
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="raw_data1/figshare", help="Figshare 데이터 루트(스크립트 상위가 기준)")
    ap.add_argument("--meta", type=str, default="metadata.csv", help="메타 파일명(csv/tsv 자동 감지)")
    ap.add_argument("--images", type=str, default="images", help="이미지 폴더명")
    ap.add_argument("--out", type=str, default="labels.csv", help="출력 labels.csv 경로(루트 기준)")
    ap.add_argument("--make_stratified_if_missing", action="store_true",
                    help="원본에 split이 없으면 층화(8:1:1)로 생성")
    args = ap.parse_args()

    project_root, root, meta_path, img_dir, out_path = resolve_paths(args)

    if not root.exists():
        raise SystemExit(f"[Error] --root 경로가 없습니다: {root}")

    meta_path = find_meta_fallback(root, meta_path)
    if not meta_path.exists():
        listing = "\n".join(f" - {p.name}" for p in root.glob("*"))
        raise SystemExit(
            f"[Error] 메타 파일을 찾지 못했습니다.\n"
            f"루트: {root}\n"
            f"시도한 경로: {meta_path}\n"
            f"루트 목록:\n{listing}\n"
            f"→ --root/--meta 인자를 확인하거나 파일명을 metadata.csv/tsv로 맞춰주세요."
        )
    if not img_dir.exists():
        listing = "\n".join(f" - {p.name}" for p in root.glob("*"))
        raise SystemExit(
            f"[Error] 이미지 폴더가 없습니다: {img_dir}\n"
            f"루트 목록:\n{listing}"
        )

    print(f"[Info] PROJECT_ROOT: {project_root}")
    print(f"[Info] ROOT:         {root}")
    print(f"[Info] META:         {meta_path}")
    print(f"[Info] IMAGES:       {img_dir}")

    df = load_meta(meta_path)

    # 필수 컬럼 확인
    need = {"image.name", "SLoF"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"[Error] 필요한 컬럼 누락: {missing}\n존재 컬럼: {list(df.columns)}")

    # 라벨 생성
    df["label_cls"] = df["SLoF"].astype(int)                     # 0/1/2
    df["label_bin"] = (df["SLoF"].astype(int) > 0).astype(int)   # 0/1

    # split 처리
    df = normalize_split_column(df, args.make_stratified_if_missing)

    # 실제 이미지 존재 여부 필터
    df["exists"] = df["image.name"].apply(lambda x: (img_dir / str(x)).exists())
    missing_imgs = df[~df["exists"]]
    if len(missing_imgs):
        print(f"[경고] 이미지 누락 {len(missing_imgs)}개 → labels에서 제외 (예시 5개):")
        print(missing_imgs[["image.name", "split"]].head())

    out = df[df["exists"]][["image.name", "label_cls", "label_bin", "split"]].rename(columns={"images.name": "id"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] 저장 완료 → {out_path} | rows={len(out)}")

    # 3-class 분포/권장 가중치
    counts = out["label_cls"].value_counts().sort_index()
    total = counts.sum()
    weights = (total / (len(counts) * counts)).round(2)  # inverse frequency
    print("[Info] 3-class 분포:", counts.to_dict())
    print("[Info] CE class weights (suggested):", weights.to_list())

    # split별 분포
    for sp in ["train", "val", "test"]:
        sub = out[out["split"] == sp]
        if len(sub):
            c = sub["label_cls"].value_counts().sort_index()
            print(f"[Info] {sp}: n={len(sub)}, dist={c.to_dict()}")

if __name__ == "__main__":
    main()
