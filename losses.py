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
    loss = (BCE+Dice for S) + alpha*(BCE(+Dice_if_pos) for M) + beta*CE(cls)
    - M: 빈 GT에서는 BCE만, 양성일 때만 Dice 더함
    - cls: 클래스 가중치(불균형 대응) 지원
    - pos_weight_M: M 희소성 대응 (양성 가중)
    """
    def __init__(self, alpha=2.0, beta=0.5, class_weight=None, pos_weight_M: float = None):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

        self.bce_S = nn.BCEWithLogitsLoss()
        if pos_weight_M is None:
            self.bce_M = nn.BCEWithLogitsLoss()
        else:
            # pos_weight는 출력 채널 크기(1)에 맞춰 텐서로 등록
            self.register_buffer("poswM", torch.tensor([pos_weight_M], dtype=torch.float32))
            self.bce_M = nn.BCEWithLogitsLoss(pos_weight=self.poswM)

        self.ce = nn.CrossEntropyLoss(weight=class_weight) if class_weight is not None else nn.CrossEntropyLoss()

    def forward(self, outputs: dict, batch: dict):
        loss = 0.0

        # Seg - Structure (항상 BCE+Dice)
        if "S" in outputs and ("S" in batch) and batch["S"] is not None:
            pred_S = F.interpolate(outputs["S"], size=batch["S"].shape[2:], mode="bilinear", align_corners=False)
            loss_S = self.bce_S(pred_S, batch["S"]) + dice_loss_from_logits(pred_S, batch["S"])
            loss = loss + loss_S

        # Seg - Marine (빈 GT에서는 BCE만, 양성일 때만 Dice 추가)
        if "M" in outputs and ("M" in batch) and batch["M"] is not None:
            pred_M = F.interpolate(outputs["M"], size=batch["M"].shape[2:], mode="bilinear", align_corners=False)
            bce_M  = self.bce_M(pred_M, batch["M"])
            if batch["M"].sum() > 0:
                loss_M = bce_M + dice_loss_from_logits(pred_M, batch["M"])
            else:
                loss_M = bce_M
            loss = loss + self.alpha * loss_M

        # Classification
        if ("cls" in outputs) and ("cls" in batch) and (batch["cls"] is not None):
            loss = loss + self.beta * self.ce(outputs["cls"], batch["cls"])

        return loss
