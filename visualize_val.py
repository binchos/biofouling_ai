import torch
from pathlib import Path
from dataio import LiaciDataset
from model import MultiHeadNet
from torchvision.utils import save_image
from torchvision import transforms as T

VAL_PATH = Path("/home/shared_project/data/kowp/kowp_prepared")

CKPT_PATH = Path(r"C:\Users\kksy0316\Desktop\finetuning\kowp_finetune_data_50_edit.pt")

OUTDIR = Path("exp/vis_kowp_val_ep100")
OUTDIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
# 🔧 tfm 인자가 아니라 transforms 내부에 이미 적용된 것으로 간주
ds = LiaciDataset(VAL_PATH, split="val", size=(736, 1280))
dl = torch.utils.data.DataLoader(ds, batch_size=1)

# Model
model = MultiHeadNet()
ckpt = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval().to(device)

# Run

# Run
for i, sample in enumerate(dl):
    img = sample["image"].to(device)
    mask_s = sample["S"]
    mask_m = sample["M"]

    with torch.no_grad():
        out = model(img)
        logits_s = out["S"]
        logits_m = out["M"]

    prob_s = torch.sigmoid(logits_s)
    prob_m = torch.sigmoid(logits_m)
    print(torch.equal(mask_s, mask_m))  # True면 문제 있음
    print(mask_s.unique(), mask_m.unique())  # 값 분포도 확인

    # Save
    save_image(img,      OUTDIR / f"{i:03d}_img.png")
    save_image(mask_s,   OUTDIR / f"{i:03d}_gtS.png")
    save_image(mask_m,   OUTDIR / f"{i:03d}_gtM.png")
    save_image(prob_s,   OUTDIR / f"{i:03d}_predS.png")
    save_image(prob_m,   OUTDIR / f"{i:03d}_predM.png")
