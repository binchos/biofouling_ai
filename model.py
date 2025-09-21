# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SegHead(nn.Module):
    """
    간단한 1x1 Conv 헤드 (+ bilinear upsample to input size)
    out_ch=2 → [S, M] 로짓
    """
    def __init__(self, in_ch: int, out_ch: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        with torch.no_grad():
            if self.conv.bias is not None and self.conv.bias.numel() >= 2:
                # [S, M] 순서 그대로이므로 M 채널 = index 1
                self.conv.bias.data[1] = -2.0

    def forward(self, feat, out_hw: tuple):
        x = self.conv(feat)  # Bx2xhxw
        if (x.shape[-2], x.shape[-1]) != out_hw:
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return x

class ClsHead(nn.Module):
    def __init__(self, in_ch: int, n_cls: int = 3, hidden: int = 256, p: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, feat):
        pooled = feat.mean(dim=[2,3])  # GAP
        return self.fc(pooled)

class MultiHeadNet(nn.Module):
    """
    공유 백본(features_only=True) → 마지막 스테이지 특징을
    - SegHead로 보내 S/M 로짓
    - ClsHead로 보내 클래스 로짓
    항상 두 헤드 forward 하고, 손실은 있는 라벨만 계산
    """
    def __init__(self, backbone_name="convnext_tiny", n_cls=3):
        super().__init__()
        # LN/GN 계열 백본 권장(도메인 차이 안정) → convnext_tiny default
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        feat_ch = self.backbone.feature_info[-1]["num_chs"]

        self.seg_head = SegHead(feat_ch, out_ch=2)
        self.cls_head = ClsHead(feat_ch, n_cls=n_cls)

    def forward(self, x):
        feats = self.backbone(x)[-1]      # BxCxHxW (마지막 스테이지)
        H, W = x.shape[-2], x.shape[-1]

        seg_logits = self.seg_head(feats, (H, W))  # Bx2xHxW
        out = {
            "S": seg_logits[:, 0:1, ...],
            "M": seg_logits[:, 1:1+1, ...],
            "cls": self.cls_head(feats)            # BxC
        }
        return out
