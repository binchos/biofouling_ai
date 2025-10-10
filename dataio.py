# dataio.py
from pathlib import Path
from typing import Optional, Tuple
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


def _find_first_with_stem(folder: Path, stem: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")) -> Optional[Path]:
    for e in exts:
        p = folder / f"{stem}{e}"
        if p.exists():
            return p
    return None
def letterbox(im: Image.Image, size_hw, fill=0, interp=InterpolationMode.BILINEAR):
    H,W = size_hw
    w0, h0 = im.size
    scale = min(W / w0, H / h0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    im_r = TF.resize(im, [nh, nw], interpolation=interp)
    top = (H - nh) // 2
    left = (W - nw) // 2
    canvas = Image.new(im_r.mode, (W, H), (fill, fill, fill) if im_r.mode == "RGB" else fill)
    canvas.paste(im_r, (left, top))
    return canvas






class FigshareDataset(Dataset):
    """
    data/figshare/
      images/<id>.jpg
      labels.csv: id,label_cls,label_bin,split
    """
    def __init__(self, root="data/figshare", split="train", use_bin=False, transform=None):
        self.root = Path(root)
        self.meta = pd.read_csv(self.root / "labels.csv")
        self.meta = self.meta[self.meta["split"] == split].reset_index(drop=True)
        self.use_bin = use_bin
        self.transform = transform

        self.img_dir = self.root / "images"
        assert self.img_dir.exists(), f"Not found: {self.img_dir}"

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_path = self.img_dir / row["id"]
        img = Image.open(img_path).convert("RGB")
        img = TF.resize(img, [512, 512], interpolation=InterpolationMode.BILINEAR)
        if self.transform:
            img = self.transform(img)
        else:
            img=TF.to_tensor(img)

        label = int(row["label_bin"] if self.use_bin else row["label_cls"])
        return {"image": img, "cls": torch.tensor(label, dtype=torch.long)}

class LiaciDataset(Dataset):
    """
    data/liaci/
      images/<id>.(jpg|png|bmp)
      masks_raw/<id>_S.png
      masks_raw/<id>_M.png
      splits.csv: id,split
    """
    def __init__(self, root="data/liaci", synth_root="data/synth_data", split="train", transform=None, strict=True, size: Optional[Tuple[int,int]]=None):
        self.root = Path(root)
        self.synth_root = Path(synth_root)
        self.split = split
        self.meta = pd.read_csv(self.root / "splits.csv")
        self.meta = self.meta[self.meta["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.strict = strict
        self.size = size

        self.img_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.synth_img_dir = self.synth_root / "images"
        self.synth_mask_dir = self.synth_root / "masks"
        #원본 이미지 목록
        self.orig_items = []
        for id_ in self.meta["id"]:
            self.orig_items.append({
                "img": _find_first_with_stem(self.img_dir, id_),
                "S": self.masks_dir / f"{id_}_S.png",
                "M": self.masks_dir / f"{id_}_M.png"
            })
        #합성 이미지 목록
        self.synth_items = []
        if self.synth_img_dir.exists():
            synth_imgs = sorted(glob.glob(str(self.synth_img_dir / "*.png")))
            for img_path in synth_imgs:
                id_ = Path(img_path).stem  # '00001' from '00001.png'
                self.synth_items.append({
                    "img": Path(img_path),
                    "S": self.synth_mask_dir / f"{id_}_S.png",
                    "M": self.synth_mask_dir / f"{id_}_M.png"
                })

        #병합
        self.items = self.orig_items + self.synth_items
        print(f"[LiaciDataset] Loaded {len(self.orig_items)} original + {len(self.synth_items)} synth = {len(self.items)} total samples")

        assert self.img_dir.exists(), f"Not found: {self.img_dir}"
        assert self.masks_dir.exists(), f"Not found: {self.masks_dir}"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        paths = self.items[idx]
        img_path, s_path, m_path = paths["img"], paths["S"], paths["M"]

        if img_path is None:
            if self.strict:
                raise FileNotFoundError(f"Image not found for id={id_} in {self.img_dir}")

            img_path = list(self.img_dir.glob(f"{id_}.*"))[0]

        if not s_path.exists() or not m_path.exists():
            raise FileNotFoundError(f"Mask not found: {s_path} or {m_path}")

        img = Image.open(img_path).convert("RGB")
        S_pil = Image.open(s_path).convert("L")
        M_pil = Image.open(m_path).convert("L")

        if self.size is not None:
            H, W = self.size
            img = letterbox(img, (H, W), fill=0, interp=InterpolationMode.BILINEAR)
            S_pil = letterbox(S_pil, (H, W), fill=0, interp=InterpolationMode.NEAREST)
            M_pil = letterbox(M_pil, (H, W), fill=0, interp=InterpolationMode.NEAREST)

            # Tensor 변환
        if self.transform:
            img = self.transform(img)
        else:
            img = TF.to_tensor(img)

        S = torch.tensor((np.array(S_pil) > 127).astype(np.float32)).unsqueeze(0)
        M = torch.tensor((np.array(M_pil) > 127).astype(np.float32)).unsqueeze(0)

        # M ⊆ S 보장 (전처리에서 이미  했지만 안전망)
        M = (M * (S > 0.5)).float()



        return {"image": img, "S": S, "M": M}
