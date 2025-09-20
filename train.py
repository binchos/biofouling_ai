import argparse
from itertools import cycle
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataio import FigshareDataset, LiaciDataset
from model import MultiHeadNet
from losses import MultiTaskLoss


# -------------------- Utils --------------------
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.default_collate(batch) if len(batch) > 0 else None


# -------------------- Metrics --------------------
@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, target: torch.Tensor):
    if logits is None or target is None:
        return None
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()


@torch.no_grad()
def dice_all_batch(logits: torch.Tensor, target: torch.Tensor, thr=0.5, eps=1e-6):
    """
    배치 단위 Dice 평균 (이미지별로 계산 후 평균)
    - GT가 빈 이미지: pred도 빈 경우 1.0, 아니면 0.0
    """
    prob = torch.sigmoid(logits)
    pred = (prob > thr).float()
    B = target.size(0)

    tgt_sum = target.flatten(1).sum(1)   # [B]
    pred_sum = pred.flatten(1).sum(1)    # [B]
    empty = (tgt_sum == 0)

    dice = torch.zeros(B, dtype=torch.float32, device=target.device)

    # 빈 GT 처리
    if empty.any():
        dice[empty] = (pred_sum[empty] == 0).float()

    # 양성 GT 처리
    if (~empty).any():
        inter = 2.0 * (pred[~empty].flatten(1) * target[~empty].flatten(1)).sum(1)
        den = pred_sum[~empty] + tgt_sum[~empty] + eps
        dice[~empty] = inter / den

    return dice.mean().item()


@torch.no_grad()
def dice_pos_batch(logits: torch.Tensor, target: torch.Tensor, thr=0.5, eps=1e-6):
    """
    양성(GT>0) 이미지들만 Dice를 이미지별로 계산해 벡터로 반환.
    또한 빈(GT=0) 이미지 개수와 그 중 FP가 난 이미지 개수도 함께 반환.
    """
    prob = torch.sigmoid(logits)
    pred = (prob > thr).float()

    tgt_sum = target.flatten(1).sum(1)   # [B]
    pred_sum = pred.flatten(1).sum(1)    # [B]

    pos = (tgt_sum > 0)
    neg = ~pos

    dice_vec = None
    if pos.any():
        inter = 2.0 * (pred[pos].flatten(1) * target[pos].flatten(1)).sum(1)
        den = pred[pos].flatten(1).sum(1) + tgt_sum[pos] + eps
        dice_vec = (inter / den)  # [N_pos]

    n_empty = int(neg.sum().item())
    n_empty_fp = int((pred_sum[neg] > 0).sum().item())

    return dice_vec, n_empty, n_empty_fp


# -------------------- Train loops --------------------
def train_interleaved_epoch(dl_fig, dl_liaci, model, criterion, optim, device):
    model.train()
    total_loss, steps = 0.0, 0
    for b_fig, b_seg in zip(dl_fig, cycle(dl_liaci)):
        for batch in (b_fig, b_seg):  # Figshare -> LIACI 교대 학습
            if batch is None:
                continue
            imgs = batch["image"].to(device)
            out = model(imgs)
            # 필요한 라벨만 디바이스로
            for k in ("S", "M", "cls"):
                if k in batch and batch[k] is not None and torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device)
            loss = criterion(out, batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
            steps += 1
    return total_loss / max(1, steps)


@torch.no_grad()
def eval_figshare(dl, model, device):
    if dl is None:
        return None
    model.eval()
    accs, cnt = 0.0, 0
    for batch in dl:
        imgs = batch["image"].to(device)
        out = model(imgs)
        if "cls" in batch:
            acc = accuracy_top1(out["cls"].cpu(), batch["cls"])
            if acc is not None:
                accs += acc
                cnt += 1
    return (accs / cnt) if cnt > 0 else None


@torch.no_grad()
def eval_liaci(dl, model, device):
    if dl is None:
        return None
    model.eval()
    sumS=cntS=sumMall=cntMall=sumMpos=cntMposB=empty_fp=cnt_empty=0

    for batch in dl:
        if batch is None:
            continue
        imgs = batch["image"].to(device)
        out = model(imgs)

        # S: 배치 평균 (빈 마스크 규칙 포함)
        dS = dice_all_batch(out["S"].cpu(), batch["S"])
        sumS += dS; cntS += 1

        # M: 배치 평균 (빈 마스크 규칙 포함)
        dAll = dice_all_batch(out["M"].cpu(), batch["M"])
        sumMall += dAll; cntMall += 1

        # M: 양성 샘플만
        dice_vec, n_empty, n_empty_fp = dice_pos_batch(out["M"].cpu(), batch["M"])
        if dice_vec is not None and dice_vec.numel() > 0:
            sumMpos += float(dice_vec.mean().item())
            cntMposB += 1
        empty_fp += n_empty_fp
        cnt_empty += n_empty

    dice_S     = (sumS/cntS) if cntS else 0.0
    dice_M_all = (sumMall/cntMall) if cntMall else 0.0
    dice_M_pos = (sumMpos/cntMposB) if cntMposB else 0.0
    empty_FPR  = (empty_fp/cnt_empty) if cnt_empty else 0.0

    return {
        "dice_S": dice_S,
        "dice_M_all": dice_M_all,
        "dice_M_pos": dice_M_pos,
        "empty_FPR": empty_FPR,
        "n_pos": cntMposB,
        "n_empty": cnt_empty,
        # 핵심 평균은 보통 S와 M_pos를 사용
        "dice_mean": (dice_S + dice_M_pos) / 2 if cntMposB else 0.0,
    }


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16, help="Figshare batch size")
    ap.add_argument("--seg_batch", type=int, default=4, help="LIACi batch size")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=3.0, help="weight for Marine seg loss")
    ap.add_argument("--beta", type=float, default=0.5, help="weight for cls loss")
    ap.add_argument("--use_bin", action="store_true", help="Figshare: use binary labels (0/1)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", type=str, default="exp/checkpoints/best_liaci_first_edit_val.pt")
    ap.add_argument("--mode", type=str, choices=["multitask", "sequential-A", "sequential-B"], default="multitask",
                    help="multitask: Figshare+LIACI 교대 / sequential-A: Figshare만 / sequential-B: LIACI만")
    ap.add_argument("--posw_M", type=float, default=14.2, help="BCE pos_weight for Marine (handles sparsity)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transforms (ImageNet 정규화 필수)
    tfm_train = T.Compose([
        T.ToTensor(),
        T.Normalize(IMNET_MEAN, IMNET_STD),
    ])
    tfm_val = T.Compose([
        T.ToTensor(),
        T.Normalize(IMNET_MEAN, IMNET_STD),
    ])

    # Datasets & Loaders
    dl_fig_train = dl_fig_val = dl_liaci_train = dl_liaci_val = None

    if args.mode in ["multitask", "sequential-A"]:
        fig_train = FigshareDataset(split="train", use_bin=args.use_bin, transform=tfm_train)
        fig_val   = FigshareDataset(split="val",   use_bin=args.use_bin, transform=tfm_val)
        dl_fig_train = DataLoader(fig_train, batch_size=args.batch, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        dl_fig_val   = DataLoader(fig_val,   batch_size=args.batch, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

    if args.mode in ["multitask", "sequential-B"]:
        # LIACi 입력 크기 = (H=736, W=1280)
        liaci_train = LiaciDataset(split="train", transform=tfm_train, size=(736, 1280))
        liaci_val   = LiaciDataset(split="val",   transform=tfm_val,   size=(736, 1280))
        dl_liaci_train = DataLoader(liaci_train, batch_size=args.seg_batch, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True, collate_fn=collate_skip_none)
        dl_liaci_val   = DataLoader(liaci_val,   batch_size=args.seg_batch, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True, collate_fn=collate_skip_none)

    # Model
    model = MultiHeadNet(backbone_name="convnext_tiny", n_cls=(2 if args.use_bin else 3)).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    # Classification class weights (handle imbalance)
    weights_cls = None
    if args.mode in ["multitask", "sequential-A"]:
        # 역비 가중치(클래스 빈도에 반비례)
        import pandas as pd
        if args.use_bin:
            vc = fig_train.meta["label_bin"].value_counts().sort_index()
        else:
            vc = fig_train.meta["label_cls"].value_counts().sort_index()
        total = vc.sum()
        inv = total / vc.clip(lower=1)
        weights_cls = torch.tensor(inv.values, dtype=torch.float32, device=device)
        # 🔴 평균이 1이 되도록 정규화 (sum이 아니라 mean!)
        weights_cls = weights_cls / weights_cls.mean()

    # Criterion / Optimizer
    criterion = MultiTaskLoss(
        alpha=args.alpha,
        beta=(0.0 if args.mode == "sequential-B" else args.beta),
        class_weight=weights_cls,
        pos_weight_M=args.posw_M,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # -------------------- Train --------------------
    best_liaci = -1.0
    for epoch in range(1, args.epochs + 1):
        if args.mode == "multitask":
            train_loss = train_interleaved_epoch(dl_fig_train, dl_liaci_train, model, criterion, optimizer, device)
        elif args.mode == "sequential-A":  # Figshare only
            model.train(); total, steps = 0.0, 0
            for batch in dl_fig_train:
                imgs = batch["image"].to(device)
                out = model(imgs)
                batch["cls"] = batch["cls"].to(device)
                loss = criterion(out, batch)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += float(loss.item()); steps += 1
            train_loss = total / max(1, steps)
        else:  # sequential-B: LIACi only
            model.train(); total, steps = 0.0, 0
            for batch in dl_liaci_train:
                if batch is None:
                    continue
                imgs = batch["image"].to(device)
                out = model(imgs)
                batch["S"] = batch["S"].to(device)
                batch["M"] = batch["M"].to(device)
                loss = criterion(out, batch)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += float(loss.item()); steps += 1
            train_loss = total / max(1, steps)

        # -------------------- Validation --------------------
        acc_fig = eval_figshare(dl_fig_val, model, device) if dl_fig_val is not None else None
        liaci_metrics = eval_liaci(dl_liaci_val, model, device) if dl_liaci_val is not None else None

        msg = f"[Epoch {epoch:03d}] loss={train_loss:.4f}"
        if acc_fig is not None:
            msg += f" | fig_acc={acc_fig:.3f}"
        if liaci_metrics is not None:
            msg += (
                f" | liaci: S={liaci_metrics['dice_S']:.3f}"
                f" M_pos={liaci_metrics['dice_M_pos']:.3f}"
                f" M_all={liaci_metrics['dice_M_all']:.3f}"
                f" empty_FPR={liaci_metrics['empty_FPR']:.3f}"
                f" (n_pos={liaci_metrics['n_pos']}, n_empty={liaci_metrics['n_empty']})"
            )
        print(msg)

        # Save best (seg 목적 핵심이므로 S & M_pos의 평균인 dice_mean 기준)
        if liaci_metrics is not None:
            if liaci_metrics["dice_mean"] > best_liaci:
                best_liaci = liaci_metrics["dice_mean"]
                Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "state_dict": state_dict,
                    "best_liaci": best_liaci,
                    "args": vars(args)
                }, args.save)
                print(f" -> saved best to {args.save} (dice_mean={best_liaci:.4f})")


if __name__ == "__main__":
    main()
