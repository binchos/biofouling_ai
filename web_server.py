import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io, base64
import numpy as np
from model import MultiHeadNet
from torchvision import transforms as T
from pathlib import Path
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiHeadNet(backbone_name="convnext_tiny", n_cls=3)
WEIGHT_PATH = Path(r"C:\Users\chsobn0710\Desktop\bio\kowp_finetune_data_50_edit.pt")
ckpt = torch.load(WEIGHT_PATH, map_location=device)
state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
model.load_state_dict(state, strict=False)
model.eval().to(device)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
])

def mask_to_base64(mask: np.ndarray, cmap="inferno") -> str:
    plt.figure(figsize=(3, 2))
    plt.axis("off")
    plt.imshow(mask, cmap=cmap)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 이미지 읽기
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)


    S_mask = torch.sigmoid(out["S"]).cpu().numpy()[0, 0]
    M_mask = torch.sigmoid(out["M"]).cpu().numpy()[0, 0]


    S_bin = (S_mask > 0.5).astype(np.uint8)
    M_bin = (M_mask > 0.5).astype(np.uint8)
    both = (S_bin & M_bin).astype(np.uint8)


    S_area = float(S_bin.mean() * 100)
    M_area = float(M_bin.mean() * 100)
    overlap_area = float(both.mean() * 100)


    rgb = np.array(img).astype(np.float32)
    overlay = rgb.copy()


    purple = np.array([200, 0, 200], dtype=np.float32)
    alpha = 0.6
    overlay[both == 1] = overlay[both == 1] * (1 - alpha) + purple * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)


    S_img_b64 = mask_to_base64(S_mask, cmap="Reds")
    M_img_b64 = mask_to_base64(M_mask, cmap="Blues")


    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="JPEG", quality=95)
    overlay_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    return JSONResponse({
        "S_mask": S_img_b64,
        "M_mask": M_img_b64,
        "overlay": overlay_b64,
        "S_area": round(S_area, 2),
        "M_area": round(M_area, 2),
        "Overlap_area": round(overlap_area, 2)
    })
