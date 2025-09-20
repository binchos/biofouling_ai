# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps=1e-6):
    """
    logits: Bx1xHxW, target: Bx1xHxW in {0,1}
    """
    prob = torch.sigmoid(logits)
    inter = 2.0 * (prob * target).sum()
    den = prob.sum() + target.sum() + eps
    return 1.0 - inter / den

def _bce_logits_pos_weight(pred, target, eps=1e-6):
    # pred, target: Bx1xHxW
    pos = target.sum()
    neg = target.numel() - pos
    # 양성 픽셀이 적을수록 더 큰 가중치 (너무 커지지 않게 clip)
    pw = (neg / (pos + eps)).clamp(min=1.0, max=100.0)
    return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pw)


class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=0.5, class_weight=None):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.bce = nn.BCEWithLogitsLoss()
        self.ce  = nn.CrossEntropyLoss(weight=class_weight.to("cuda")) if class_weight is not None else nn.CrossEntropyLoss()

    def forward(self, outputs: dict, batch: dict):
        loss = 0.0

        # S: 구조물
        if "S" in outputs and ("S" in batch) and batch["S"] is not None:
            pred_S = F.interpolate(outputs["S"], size=batch["S"].shape[2:], mode="bilinear", align_corners=False)
            loss_S = self.bce(pred_S, batch["S"]) + (1.0 - (2*(torch.sigmoid(pred_S)*batch["S"]).sum() / (torch.sigmoid(pred_S).sum() + batch["S"].sum() + 1e-6)))
            loss = loss + loss_S

        # M: 마린(희소) → pos_weight 적용
        # Seg - Marine
        if "M" in outputs and ("M" in batch) and batch["M"] is not None:
            pred_M = F.interpolate(outputs["M"], size=batch["M"].shape[2:], mode="bilinear", align_corners=False)
            bce_M = self.bce(pred_M, batch["M"])
            if batch["M"].sum() > 0:
                loss_M = bce_M + dice_loss_from_logits(pred_M, batch["M"])
            else:
                loss_M = bce_M  # 빈 GT에서는 Dice 생략 (정의상 불안정/왜곡 방지)
            loss = loss + self.alpha * loss_M

        # 분류
        if ("cls" in outputs) and ("cls" in batch) and (batch["cls"] is not None):
            loss_C = self.ce(outputs["cls"], batch["cls"])
            loss = loss + self.beta * loss_C

        return loss
