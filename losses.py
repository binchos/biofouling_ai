# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


@torch.no_grad()
def _ensure_same_hw(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """pred을 tgt와 같은 (H, W)로 보간."""
    if pred.shape[-2:] != tgt.shape[-2:]:
        pred = F.interpolate(pred, size=tgt.shape[-2:], mode="bilinear", align_corners=False)
    return pred


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    배치 평균 Dice loss (threshold 없이 확률로 계산).
    logits: Bx1xHxW, target: Bx1xHxW (0/1 float)
    """
    prob = torch.sigmoid(logits)
    inter = 2.0 * (prob * target).sum()
    den = prob.sum() + target.sum() + eps
    return 1.0 - inter / den


def dice_loss_per_sample_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    샘플별 Dice loss 벡터 반환 (양성 샘플 마스크로 평균할 때 사용).
    logits: Bx1xHxW, target: Bx1xHxW (0/1 float)
    return: B (각 샘플의 dice loss)
    """
    prob = torch.sigmoid(logits)
    prob_f = prob.flatten(1)
    tgt_f = target.flatten(1)
    inter = 2.0 * (prob_f * tgt_f).sum(dim=1)              # [B]
    den = prob_f.sum(dim=1) + tgt_f.sum(dim=1) + eps       # [B]
    return 1.0 - inter / den                               # [B]


class MultiTaskLoss(nn.Module):
    """
    loss = (BCE+Dice for S) + alpha * (BCE (+Dice_if_pos) for M) + beta * CE(cls)

    - S: 항상 BCE + Dice
    - M: BCE는 항상, GT>0(양성) 샘플에 대해서만 Dice를 추가 (빈 GT 안정성)
    - cls: Figshare 불균형 대응을 위한 class_weight 지원
    - pos_weight_M: LIACi Marine의 희소 양성 픽셀 가중 (BCEWithLogitsLoss의 pos_weight)
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 0.5,
        class_weight: torch.Tensor | None = None,
        pos_weight_M: float | None = None,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

        # BCE for S
        self.bce_S = nn.BCEWithLogitsLoss()

        # BCE for M (pos_weight 지원)
        if pos_weight_M is None:
            self.bce_M = nn.BCEWithLogitsLoss()
            self.register_buffer("poswM", None, persistent=False)
        else:
            self.register_buffer("poswM", torch.tensor([pos_weight_M], dtype=torch.float32))
            self.bce_M = nn.BCEWithLogitsLoss(pos_weight=self.poswM)

        # CE for classification (class_weight 지원)
        if class_weight is not None:
            # CrossEntropyLoss는 내부적으로 weight를 buffer로 등록하므로 .to(device)로 이동 가능
            self.ce = nn.CrossEntropyLoss(weight=class_weight)
        else:
            self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = torch.zeros((), dtype=torch.float32, device=next(self.parameters()).device)

        # ----- Structure Segmentation (S): BCE + Dice -----
        if "S" in outputs and "S" in batch and batch["S"] is not None:
            pred_S = _ensure_same_hw(outputs["S"], batch["S"])
            loss_S = self.bce_S(pred_S, batch["S"]) + dice_loss_from_logits(pred_S, batch["S"])
            loss = loss + loss_S

        # ----- Marine Segmentation (M): BCE (+ Dice for positive-only) -----
        if "M" in outputs and "M" in batch and batch["M"] is not None:
            pred_M = _ensure_same_hw(outputs["M"], batch["M"])
            bce_M = self.bce_M(pred_M, batch["M"])

            # 샘플별 양성 여부 (GT 합이 0보다 큰 경우)
            with torch.no_grad():
                pos_mask = (batch["M"].flatten(1).sum(dim=1) > 0)  # [B] bool

            if pos_mask.any():
                dice_vec = dice_loss_per_sample_from_logits(pred_M[pos_mask], batch["M"][pos_mask])  # [B_pos]
                loss_M = bce_M + dice_vec.mean()
            else:
                loss_M = bce_M

            loss = loss + self.alpha * loss_M

        # ----- Classification (Figshare) -----
        if "cls" in outputs and "cls" in batch and batch["cls"] is not None:
            loss_C = self.ce(outputs["cls"], batch["cls"])
            loss = loss + self.beta * loss_C

        return loss
