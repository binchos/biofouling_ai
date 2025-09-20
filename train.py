# train.py


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



tfm_train = T.Compose([
    T.ToTensor()
])
tfm_val = T.Compose([
    T.ToTensor()
])


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ------- Metrics -------
@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, target: torch.Tensor):
    if logits is None or target is None:
        return None
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()


@torch.no_grad()
def dice_from_logits(logits: torch.Tensor, target: torch.Tensor, thr=0.5, eps=1e-6):
    if logits is None or target is None:
        return None
    prob = torch.sigmoid(logits)
    pred = (prob > thr).float()
    inter = 2.0 * (pred * target).sum()
    den = pred.sum() + target.sum() + eps
    return (inter / den).item()

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.default_collate(batch) if len(batch) > 0 else None


# ------- Train loops -------
def train_interleaved_epoch(dl_fig, dl_liaci, model, criterion, optim, device):
    model.train()
    total_loss, steps = 0.0, 0
    for b_fig, b_seg in zip(dl_fig, cycle(dl_liaci)):
        for batch in (b_fig, b_seg):  # Figshare → LIACI 교대
            if batch is None:
                continue
            imgs = batch["image"].to(device)
            # 항상 두 헤드 forward (공유 백본)
            out = model(imgs)
            # 사용할 라벨만 디바이스로
            for k in ("S", "M", "cls"):
                if k in batch and batch[k] is not None:
                    batch[k] = batch[k].to(device)
            loss = criterion(out, batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
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
        out = model(imgs)  # 항상 둘 다 forward
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
    dices_S, dices_M_all, dices_M_filtered = 0.0, 0.0, 0.0
    cnt_S, cnt_M_all, cnt_M_filtered = 0, 0, 0
    for batch in dl:
        if batch is None:
            continue
        imgs = batch["image"].to(device)
        out = model(imgs)

        # Structure는 항상 존재
        dS = dice_from_logits(out["S"].cpu(), batch["S"])
        if dS is not None:
            dices_S += dS
            cnt_S += 1

        # Marine - 모든 이미지 기준
        dM_all = dice_from_logits(out["M"].cpu(), batch["M"])
        if dM_all is not None:
            dices_M_all += dM_all
            cnt_M_all += 1

        # Marine - 충분한 픽셀 있는 경우만 (Filtered)
        if batch["M"].sum() > 100:
            dM_filtered = dice_from_logits(out["M"].cpu(), batch["M"])
            if dM_filtered is not None:
                dices_M_filtered += dM_filtered
                cnt_M_filtered += 1

    if cnt_S == 0:
        return None
    avg_dice_S = dices_S / cnt_S
    avg_dice_M_filtered = (dices_M_filtered / cnt_M_filtered) if cnt_M_filtered > 0 else 0.0

    return {
        "dice_S": avg_dice_S,
        "dice_M": avg_dice_M_filtered,  # ← 핵심 지표로 사용됨
        "dice_M_all": dices_M_all / cnt_M_all if cnt_M_all > 0 else 0.0,
        "dice_mean": (avg_dice_S + avg_dice_M_filtered) / 2
    }

#train에서는 M 픽셀 < 100인 샘플을 return None으로 걸렀는데 평가시에는 포함되는 문제 해결
#이것때문에 liaci_diceM = 0.000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=1.0, help="weight for Marine seg loss")
    ap.add_argument("--beta", type=float, default=0.5, help="weight for cls loss")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--use_bin", action="store_true", help="Figshare: use binary labels (0/1)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", type=str, default=" exp/checkpoints/best_liaci_first_edit_val.pt")
    ap.add_argument("--mode", type=str,
                    choices=["multitask", "sequential-A", "sequential-B"],
                    default="multitask",
                    help="multitask: Figshare+LIACI 교대 / sequential-A: Figshare만 / sequential-B: LIACI만")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transform (두 도메인 동일 해상도로 맞춤)
    tfm_train = T.Compose([
        T.ToTensor()
    ])
    tfm_val = T.Compose([
        T.ToTensor()
    ])

    # Datasets & Loaders
    dl_fig_train = dl_fig_val = dl_liaci_train = dl_liaci_val = None

    if args.mode in ["multitask", "sequential-A"]:
        fig_train = FigshareDataset(split="train", use_bin=args.use_bin, transform=tfm_train)
        fig_val = FigshareDataset(split="val", use_bin=args.use_bin, transform=tfm_val)
        dl_fig_train = DataLoader(fig_train, batch_size=16, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        dl_fig_val = DataLoader(fig_val, batch_size=16, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

    if args.mode in ["multitask", "sequential-B"]:
        # LIACi 입력 크기 = (H=736, W=1280)
        liaci_train = LiaciDataset(split="train", transform=tfm_train, size=(736, 1280))
        liaci_val = LiaciDataset(split="val", transform=tfm_val, size=(736, 1280))
        dl_liaci_train = DataLoader(liaci_train, batch_size=4 if args.mode == "multitask" else 4,
                                    shuffle=True, num_workers=args.num_workers, pin_memory=True,collate_fn=collate_skip_none)
        dl_liaci_val = DataLoader(liaci_val, batch_size=4, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True,collate_fn=collate_skip_none)

    # Model / Loss / Optim
    model = MultiHeadNet(backbone_name="convnext_tiny",
                         n_cls=(2 if args.use_bin else 3)).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    if args.use_bin:
        total = 7822 + 2441
        w0 = total / 7822
        w1 = total / 2441
        weights_bin = torch.tensor([w0, w1], dtype=torch.float32)
        weights_bin = weights_bin / weights_bin.sum()
        weights_bin = weights_bin.to(device)
    else:
        weights_bin = None

    criterion = MultiTaskLoss(alpha=args.alpha,
                              beta=(0.0 if args.mode == "sequential-B" else args.beta),
                              class_weight=weights_bin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_liaci = -1.0
    for epoch in range(1, args.epochs + 1):
        if args.mode == "multitask":
            train_loss = train_interleaved_epoch(dl_fig_train, dl_liaci_train,
                                                 model, criterion, optimizer, device)
        elif args.mode == "sequential-A":  # Figshare만
            model.train()
            total, steps = 0.0, 0
            for batch in dl_fig_train:
                imgs = batch["image"].to(device)
                out = model(imgs)
                batch["cls"] = batch["cls"].to(device)
                loss = criterion(out, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
                steps += 1
            train_loss = total / max(1, steps)
        else:  # LIACI만
            model.train()
            total, steps = 0.0, 0
            for batch in dl_liaci_train:
                if batch is None:
                    continue
                imgs = batch["image"].to(device)
                out = model(imgs)
                batch["S"] = batch["S"].to(device)
                batch["M"] = batch["M"].to(device)
                loss = criterion(out, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
                steps += 1
            train_loss = total / max(1, steps)

        # Validation
        acc_fig = eval_figshare(dl_fig_val, model, device) if dl_fig_val is not None else None
        liaci_metrics = eval_liaci(dl_liaci_val, model, device) if dl_liaci_val is not None else None

        msg = f"[Epoch {epoch:03d}] loss={train_loss:.4f}"
        if acc_fig is not None:
            msg += f" | fig_acc={acc_fig:.3f}"
        if liaci_metrics is not None:
            msg += f" | liaci_diceS={liaci_metrics['dice_S']:.3f} liaci_diceM={liaci_metrics['dice_M']:.3f} (all={liaci_metrics['dice_M_all']:.3f})"

        print(msg)

        # best 저장 (세그 목적이 핵심이라 LIACI dice_mean 기준)
        if liaci_metrics is not None:
            if liaci_metrics["dice_mean"] > best_liaci:
                best_liaci = liaci_metrics["dice_mean"]
                Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_liaci": best_liaci,
                    "args": vars(args)
                }, args.save)
                print(f" -> saved best to {args.save} (dice_mean={best_liaci:.4f})")


if __name__ == "__main__":
    main()
