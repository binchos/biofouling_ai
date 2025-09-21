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


# losses.py

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=0.5, class_weight=None, pos_weight_M=None, lambda_empty=0.05):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lambda_empty = float(lambda_empty)
        # device reference (파라미터가 없는 모듈 대비)
        self.register_buffer("_dev_ref", torch.tensor(0.), persistent=False)

        self.bce_S = nn.BCEWithLogitsLoss()
        if pos_weight_M is None:
            self.bce_M = nn.BCEWithLogitsLoss()
            self.register_buffer("poswM", None, persistent=False)
        else:
            self.register_buffer("poswM", torch.tensor([pos_weight_M], dtype=torch.float32))
            self.bce_M = nn.BCEWithLogitsLoss(pos_weight=self.poswM)

        if class_weight is not None:
            self.ce = nn.CrossEntropyLoss(weight=class_weight)
        else:
            self.ce = nn.CrossEntropyLoss()

    def _pick_device(self, outputs: Dict[str, torch.Tensor]) -> torch.device:
        for k in ("S", "M", "cls"):
            t = outputs.get(k, None)
            if t is not None:
                return t.device
        return self._dev_ref.device  # fallback

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = self._pick_device(outputs)
        loss = torch.zeros((), dtype=torch.float32, device=device)

        # ----- Structure Segmentation (S): BCE + Dice -----
        if "S" in outputs and "S" in batch and batch["S"] is not None:
            pred_S = _ensure_same_hw(outputs["S"], batch["S"])
            loss_S = self.bce_S(pred_S, batch["S"]) + dice_loss_from_logits(pred_S, batch["S"])
            loss = loss + loss_S

        # ----- Marine Segmentation (M): BCE (+ Dice for positive-only) -----
        if "M" in outputs and "M" in batch and batch["M"] is not None:
            pred_M = _ensure_same_hw(outputs["M"], batch["M"])
            bce_M = self.bce_M(pred_M, batch["M"])

            with torch.no_grad():
                flat_sum = batch["M"].flatten(1).sum(dim=1)
                pos_mask = (flat_sum > 0)
                neg_mask = (flat_sum == 0)  # ← empty 이미지

            # M Dice(워밍업) – 기존 그대로
            use_m_dice = (getattr(self, "_epoch", 1) >= 4)
            if pos_mask.any():
                dice_vec = dice_loss_per_sample_from_logits(pred_M[pos_mask], batch["M"][pos_mask])
                dice_term = (dice_vec.mean() if use_m_dice else 0.0)
            else:
                dice_term = 0.0

            # --- ADD: empty 억제항 ---
            empty_term = 0.0
            if neg_mask.any() and self.lambda_empty > 0.0:
                # empty 이미지에서 "양성 확률"을 평균으로 눌러줌
                p_empty = torch.sigmoid(pred_M[neg_mask])
                empty_term = p_empty.mean() * self.lambda_empty

            loss_M = bce_M + dice_term + empty_term
            loss = loss + self.alpha * loss_M

        # ----- Classification (Figshare) -----
        if "cls" in outputs and "cls" in batch and batch["cls"] is not None:
            loss_C = self.ce(outputs["cls"], batch["cls"])
            loss = loss + self.beta * loss_C

        return loss

