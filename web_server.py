import os
from fastapi import FastAPI, File, UploadFile, Form   # ✅ Form 추가
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


model_img = MultiHeadNet(backbone_name="convnext_tiny", n_cls=3)
ckpt_img = torch.load(r"C:\Users\kksy0316\Desktop\finetuning\kowp_finetune_data_edit.pt", map_location=device)
state_img = ckpt_img["state_dict"] if "state_dict" in ckpt_img else ckpt_img
model_img.load_state_dict(state_img, strict=False)
model_img.eval().to(device)


model_vid = MultiHeadNet(backbone_name="convnext_tiny", n_cls=3)
ckpt_vid = torch.load(r"C:\Users\kksy0316\Desktop\finetuning\kowp_finetune_data_50_edit.pt", map_location=device)
state_vid = ckpt_vid["state_dict"] if "state_dict" in ckpt_vid else ckpt_vid
model_vid.load_state_dict(state_vid, strict=False)
model_vid.eval().to(device)


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


def run_inference(model, img):
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

    return {
        "S_mask": S_img_b64,
        "M_mask": M_img_b64,
        "overlay": overlay_b64,
        "S_area": round(S_area, 2),
        "M_area": round(M_area, 2),
        "Overlap_area": round(overlap_area, 2)
    }



@app.post("/predict")
async def predict(file: UploadFile = File(...), mode: str = Form("image")):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")


    if mode == "video":
        result = run_inference(model_vid, img)
    else:
        result = run_inference(model_img, img)

    return JSONResponse(result)
