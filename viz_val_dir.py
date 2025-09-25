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

def to_tensor(img_pil):
    tfm = T.Compose([T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])
    return tfm(img_pil)

def save_png01(arr01, path):
    Image.fromarray((arr01*255).astype(np.uint8)).save(path)

def color_overlay(rgb, S_bin, M_bin, alpha=0.4):
    base = rgb.copy().astype(np.float32)
    color = np.zeros_like(base)
    color[...,2] += (S_bin>0)*255   # S=blue
    color[...,0] += (M_bin>0)*255   # M=red
    return (base*(1-alpha) + color*alpha).clip(0,255).astype(np.uint8)

def make_panel(rgb, S_bin, M_bin, overlay):
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True, help=".../kowp_prepared (images/, masks/, splits.csv)")
    ap.add_argument("--save_dir", default="out_val_vis")
    ap.add_argument("--thr_s", type=float, default=0.5)
    ap.add_argument("--thr_m", type=float, default=0.65)
    ap.add_argument("--size_h", type=int, default=736)
    ap.add_argument("--size_w", type=int, default=1280)
    ap.add_argument("--save_overlay", action="store_true")
    ap.add_argument("--save_panel", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_root)
    img_dir = root/"images"
    msk_dir = root/"masks"
    split_csv = root/"splits.csv"

    os.makedirs(args.save_dir, exist_ok=True)

    # 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model = MultiHeadNet(backbone_name="convnext_tiny", n_cls=2).to(device).eval()
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    filtered = {k: v for k, v in state.items() if not k.startswith("cls_head")}
    model.load_state_dict(filtered, strict=False)

    # splits.csv 읽기 → val 만 선택
    val_list = []
    with open(split_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "val":
                val_list.append(row["filename"])  # filename 컬럼 가정

    print(f"[viz] found {len(val_list)} val samples")

    for stem in val_list:
        ip = img_dir / stem
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

        if args.save_overlay:
            d = Path(args.save_dir)/"overlay"; d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(overlay).save(d/f"{stem}_overlay.png")
        if args.save_panel:
            d = Path(args.save_dir)/"panel"; d.mkdir(parents=True, exist_ok=True)
            panel = make_panel(rgb_in, S_bin, M_bin, overlay)
            panel.save(d/f"{stem}_panel.png")

    print("Done.")

if __name__ == "__main__":
    main()
