import argparse, os, csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from model import MultiHeadNet
from dataio import letterbox  # 학습과 동일 전처리

IMNET_MEAN=(0.485, 0.456, 0.406)
IMNET_STD =(0.229, 0.224, 0.225)

EXTS = [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]

def to_tensor(img_pil):
    tfm = T.Compose([T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])
    return tfm(img_pil)

def load_mask_if(path):
    if path is None or not Path(path).exists(): return None
    m = np.array(Image.open(path).convert("L"))
    return (m>127).astype(np.uint8)

def save_png01(arr01, path):
    Image.fromarray((arr01*255).astype(np.uint8)).save(path)

def color_overlay(rgb, S_bin, M_bin, alpha=0.4):
    """S=blue, M=red 오버레이"""
    base = rgb.copy().astype(np.float32)
    color = np.zeros_like(base)
    color[...,2] += (S_bin>0)*255   # S=blue
    color[...,0] += (M_bin>0)*255   # M=red
    out = (base*(1-alpha) + color*alpha).clip(0,255).astype(np.uint8)
    return out

def make_panel(rgb, S_bin, M_bin, overlay):
    """원본/ S / M / 오버레이 2x2 패널"""
    def to_pil_gray(a01):
        return Image.fromarray((a01*255).astype(np.uint8)).convert("L")
    H,W,_ = rgb.shape
    pad = 8
    orig = Image.fromarray(rgb)
    s_img = ImageOps.colorize(to_pil_gray(S_bin), black="black", white="blue")
    m_img = ImageOps.colorize(to_pil_gray(M_bin), black="black", white="red")
    ov_img = Image.fromarray(overlay)
    for im in (orig, s_img, m_img, ov_img):
        if im.size != (W,H): im = im.resize((W,H), Image.NEAREST)
    row1 = Image.new("RGB", (W*2+pad, H))
    row1.paste(orig, (0,0)); row1.paste(s_img, (W+pad,0))
    row2 = Image.new("RGB", (W*2+pad, H))
    row2.paste(m_img, (0,0)); row2.paste(ov_img, (W+pad,0))
    panel = Image.new("RGB", (W*2+pad, H*2+pad), (30,30,30))
    panel.paste(row1, (0,0)); panel.paste(row2, (0,H+pad))
    return panel

def dice_all(pred_bin, tgt_bin, eps=1e-6):
    if tgt_bin is None: return None
    if tgt_bin.sum()==0:
        return 1.0 if pred_bin.sum()==0 else 0.0
    inter = 2.0 * ((pred_bin & tgt_bin).sum())
    den   = pred_bin.sum() + tgt_bin.sum() + eps
    return float(inter/den)

def compute_coverage(S_bin, M_bin):
    area_S = int((S_bin>0).sum())
    area_M = int(((M_bin>0) & (S_bin>0)).sum())  # S 내부 M만
    cov = area_M / max(area_S, 1)
    return area_S, area_M, float(cov)

def find_image_paths(img_dir):
    paths=[]
    for p in sorted(Path(img_dir).iterdir()):
        if p.suffix.lower() in EXTS: paths.append(p)
    return paths

def find_masks(mask_dir, stem):
    s = Path(mask_dir)/f"{stem}_S.png"
    m = Path(mask_dir)/f"{stem}_M.png"
    if s.exists() or m.exists():
        return (str(s) if s.exists() else None), (str(m) if m.exists() else None)
    for ext in EXTS:
        cand = Path(mask_dir)/f"{stem}{ext}"
        if cand.exists(): return None, str(cand)
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True, help=".../data/test (images/, masks/)")
    ap.add_argument("--save_dir", default="out_test_vis")
    ap.add_argument("--thr_s", type=float, default=0.5)
    ap.add_argument("--thr_m", type=float, default=0.65)
    ap.add_argument("--size_h", type=int, default=736)
    ap.add_argument("--size_w", type=int, default=1280)
    ap.add_argument("--save_pred_png", action="store_true")
    ap.add_argument("--save_overlay", action="store_true")
    ap.add_argument("--save_panel", action="store_true")
    args = ap.parse_args()

    img_dir = Path(args.data_root)/"images"
    msk_dir = Path(args.data_root)/"masks"
    os.makedirs(args.save_dir, exist_ok=True)

    # ---------------- 모델 로드 (cls_head 제외) ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model = MultiHeadNet(backbone_name="convnext_tiny", n_cls=2).to(device).eval()

    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    filtered = {k: v for k, v in state.items() if not k.startswith("cls_head")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print("[viz] loaded with filtering. missing:", missing, "| unexpected:", unexpected)

    # ---------------- CSV 헤더 ----------------
    csv_path = Path(args.save_dir)/"coverage_report.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image","S_dice","M_dice_all","M_dice_pos","is_empty_gt",
                    "S_area","M_area","coverage","thr_s","thr_m"])

    # ---------------- 평가 루프 ----------------
    s_sum, s_cnt = 0.0, 0
    mall_sum, mall_cnt = 0.0, 0
    mpos_sum, mpos_cnt = 0.0, 0
    empty_cnt, empty_fp = 0, 0

    img_paths = find_image_paths(img_dir)
    for ip in img_paths:
        pil = Image.open(ip).convert("RGB")
        pil_in = letterbox(pil, (args.size_h, args.size_w))
        x = to_tensor(pil_in).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)
            S_prob = torch.sigmoid(out["S"]).cpu().numpy()[0,0]
            M_prob = torch.sigmoid(out["M"]).cpu().numpy()[0,0]

        S_bin = (S_prob > args.thr_s).astype(np.uint8)
        M_prob_masked = M_prob * (S_bin>0)
        M_bin = (M_prob_masked > args.thr_m).astype(np.uint8)

        rgb_in = np.array(pil_in)
        overlay = color_overlay(rgb_in, S_bin, M_bin)

        stem = Path(ip).stem
        s_gt_path, m_gt_path = find_masks(msk_dir, stem)
        S_gt = load_mask_if(s_gt_path)
        M_gt = load_mask_if(m_gt_path)

        S_dice = dice_all(S_bin, S_gt)
        M_dice_all = dice_all(M_bin, M_gt)
        M_dice_pos = None
        is_empty_gt = False
        if M_gt is not None:
            if M_gt.sum()==0:
                is_empty_gt = True
                empty_cnt += 1
                if M_bin.sum()>0: empty_fp += 1
            else:
                inter = 2.0 * ((M_bin & M_gt).sum())
                den   = M_bin.sum() + M_gt.sum() + 1e-6
                M_dice_pos = float(inter/den)

        if S_dice is not None: s_sum += S_dice; s_cnt += 1
        if M_dice_all is not None: mall_sum += M_dice_all; mall_cnt += 1
        if M_dice_pos is not None: mpos_sum += M_dice_pos; mpos_cnt += 1

        S_area, M_area, cov = compute_coverage(S_bin, M_bin)

        if args.save_pred_png:
            d = Path(args.save_dir)/"pred_png"; d.mkdir(parents=True, exist_ok=True)
            save_png01(S_bin, d/f"{stem}_S.png")
            save_png01(M_bin, d/f"{stem}_M.png")
        if args.save_overlay:
            d = Path(args.save_dir)/"overlay"; d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(overlay).save(d/f"{stem}_overlay.png")
        if args.save_panel:
            d = Path(args.save_dir)/"panel"; d.mkdir(parents=True, exist_ok=True)
            panel = make_panel(rgb_in, S_bin, M_bin, overlay)
            panel.save(d/f"{stem}_panel.png")

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                stem,
                f"{S_dice:.4f}" if S_dice is not None else "NA",
                f"{M_dice_all:.4f}" if M_dice_all is not None else "NA",
                f"{M_dice_pos:.4f}" if M_dice_pos is not None else "NA",
                int(is_empty_gt),
                S_area, M_area, f"{cov:.6f}",
                args.thr_s, args.thr_m
            ])

    # ---------------- 요약 ----------------
    print("=== TEST SUMMARY ===")
    if s_cnt:     print(f"S_dice     : {s_sum/s_cnt:.4f}  (n={s_cnt})")
    if mpos_cnt:  print(f"M_dice_pos : {mpos_sum/mpos_cnt:.4f}  (n_pos={mpos_cnt})")
    if mall_cnt:  print(f"M_dice_all : {mall_sum/mall_cnt:.4f}  (n={mall_cnt})")
    if empty_cnt:
        print(f"empty_FPR  : {empty_fp/empty_cnt:.4f} ({empty_fp}/{empty_cnt})")
    else:
        print("empty_FPR  : NA (no empty frames)")
    if s_cnt and mpos_cnt:
        dice_mean = ((s_sum/s_cnt) + (mpos_sum/mpos_cnt))/2
        print(f"dice_mean  : {dice_mean:.4f}")
    print(f"Coverage CSV: {csv_path}")
    print("Done.")

if __name__ == "__main__":
    main()
