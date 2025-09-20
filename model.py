# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class TinyFPN(nn.Module):
    def __init__(self, ch3: int, ch4: int, out_ch: int = 2):
        super().__init__()
        self.l3 = nn.Conv2d(ch3, 256, 1)
        self.l4 = nn.Conv2d(ch4, 256, 1)
        self.fuse3 = nn.Conv2d(256, 256, 3, padding=1)
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128,  64, 3, padding=1),
            nn.Conv2d(64, out_ch, 1)
        )

    def forward(self, c3, c4, out_hw):
        p4 = self.l4(c4)                      # stride 32 → 256ch
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)  # fuse at stride 16
        p3 = self.fuse3(p3)
        x  = self.head(p3)                    # stride 16 → stride 8
        if x.shape[-2:] != out_hw:
            x = F.interpolate(x, size=out_hw, mode='bilinear', align_corners=False)
        return x  # Bx2xHxW

class ClsHead(nn.Module):
    def __init__(self, in_ch: int, n_cls: int = 3, hidden: int = 256, p: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(hidden, n_cls),
        )
    def forward(self, feat):
        pooled = feat.mean(dim=[2,3])
        return self.fc(pooled)

class MultiHeadNet(nn.Module):
    def __init__(self, backbone_name="convnext_tiny", n_cls=3):
        super().__init__()
        # 여러 스테이지를 뽑도록 features_only 사용
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        chs = [fi["num_chs"] for fi in self.backbone.feature_info]  # [..., ch3, ch4]
        ch4, ch3 = chs[-1], chs[-2]

        self.fpn = TinyFPN(ch3=ch3, ch4=ch4, out_ch=2)
        self.cls_head = ClsHead(ch4, n_cls=n_cls)

    def forward(self, x):
        feats = self.backbone(x)  # [..., c3, c4]
        c3, c4 = feats[-2], feats[-1]
        H, W = x.shape[-2], x.shape[-1]

        seg_logits = self.fpn(c3, c4, (H, W))      # Bx2xHxW
        out = {
            "S":  seg_logits[:, 0:1, ...],
            "M":  seg_logits[:, 1:2, ...],
            "cls": self.cls_head(c4)
        }
        return out
