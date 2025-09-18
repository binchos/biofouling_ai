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


class MultiTaskLoss(nn.Module):
    """
    loss = (BCE+Dice for S) + alpha*(BCE+Dice for M) + beta*CE(cls)
    배치에 해당 라벨이 없으면 그 항은 0으로 스킵.
    """
    def __init__(self, alpha=2.0, beta=0.5, class_weight=None):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.bce = nn.BCEWithLogitsLoss()
        if class_weight is not None:
            self.ce = nn.CrossEntropyLoss(weight=class_weight.to("cuda"))
        else:
            self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs: dict, batch: dict):
        loss = 0.0

        # Seg - Structure
        if "S" in outputs and ("S" in batch) and batch["S"] is not None:
            # 출력 크기를 GT와 일치시킴
            pred_S = F.interpolate(outputs["S"], size=batch["S"].shape[2:], mode="bilinear", align_corners=False)
            loss_S = self.bce(pred_S, batch["S"]) + dice_loss_from_logits(pred_S, batch["S"])
            loss = loss + loss_S

        # Seg - Marine
        if "M" in outputs and ("M" in batch) and batch["M"] is not None:
            if batch["M"].sum() > 10:
                pred_M = F.interpolate(outputs["M"], size=batch["M"].shape[2:], mode="bilinear", align_corners=False)
                loss_M = self.bce(pred_M, batch["M"]) + dice_loss_from_logits(pred_M, batch["M"])
                loss = loss + self.alpha * loss_M
        # Classification
        if ("cls" in outputs) and ("cls" in batch) and (batch["cls"] is not None):
            loss_C = self.ce(outputs["cls"], batch["cls"])
            loss = loss + self.beta * loss_C

        return loss
