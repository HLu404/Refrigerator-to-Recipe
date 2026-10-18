"""Train the multi-label FoodCNN on Clean_Data.csv + the flat image folder.

Usage:
    python train.py --images /Users/harry/Downloads/All --csv Clean_Data.csv

Outputs (in --out-dir): best_model.pth, class_mapping.json
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from model import FoodCNN

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def clean_name(col):
    return col.strip().lower().replace(" ", "_")


def load_labels(csv_path, images_dir, drop_empty=True):
    """Return (class_names, [(image_path, multi-hot np.float32 vector)])."""
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        fn_idx = header.index("filename")
        label_idx = [i for i, h in enumerate(header) if i != fn_idx and h != ""]
        classes = [clean_name(header[i]) for i in label_idx]
        assert len(set(classes)) == len(classes), "duplicate class names after cleaning"
        samples, missing, empty = [], 0, 0
        for row in reader:
            path = Path(images_dir) / row[fn_idx]
            if not path.exists():
                missing += 1
                continue
            y = np.array([float(row[i]) > 0 for i in label_idx], dtype=np.float32)
            if drop_empty and y.sum() == 0:
                empty += 1
                continue
            samples.append((path, y))
    print(f"{len(samples)} samples, {len(classes)} classes "
          f"(skipped {missing} missing files, {empty} with no labels)")
    return classes, samples


class FridgeDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.from_numpy(y)


def make_transforms(size):
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.05),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    val = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    return train, val


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def f1_scores(probs, targets, thr):
    """Micro and macro F1 given (N, C) probs/targets and thr (scalar or (C,))."""
    pred = probs >= thr
    t = targets.astype(bool)
    tp = (pred & t).sum(0)
    fp = (pred & ~t).sum(0)
    fn = (~pred & t).sum(0)
    micro = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
    per_class = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(micro), float(per_class.mean())


def tune_thresholds(probs, targets, grid=np.arange(0.1, 0.91, 0.05), default=0.5):
    """Per-class threshold maximizing F1 on validation; falls back to default for rare classes."""
    thr = np.full(probs.shape[1], default, dtype=np.float32)
    for c in range(probs.shape[1]):
        if targets[:, c].sum() < 3:
            continue
        scores = [f1_scores(probs[:, [c]], targets[:, [c]], g)[0] for g in grid]
        thr[c] = grid[int(np.argmax(scores))]
    return thr


def run_epoch(net, loader, device, criterion, optimizer=None, scheduler=None):
    train = optimizer is not None
    net.train(train)
    total, n = 0.0, 0
    all_p, all_t = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()  # OneCycle steps per batch
            total += loss.item() * x.size(0)
            n += x.size(0)
            if not train:
                all_p.append(torch.sigmoid(logits).cpu().numpy())
                all_t.append(y.cpu().numpy())
    out = {"loss": total / n}
    if not train:
        out["probs"], out["targets"] = np.concatenate(all_p), np.concatenate(all_t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder holding all images (flat)")
    ap.add_argument("--csv", default="Clean_Data.csv")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=10, help="early-stopping patience (epochs)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-empty", action="store_true", help="keep images with no positive label")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = pick_device()
    print("device:", device)

    classes, samples = load_labels(args.csv, args.images, drop_empty=not args.keep_empty)
    random.Random(args.seed).shuffle(samples)
    n_val = int(len(samples) * args.val_frac)
    val_s, train_s = samples[:n_val], samples[n_val:]
    print(f"train {len(train_s)} / val {len(val_s)}")

    tf_train, tf_val = make_transforms(args.img_size)
    pin = device.type == "cuda"
    train_dl = DataLoader(FridgeDataset(train_s, tf_train), args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=pin, drop_last=True,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(FridgeDataset(val_s, tf_val), args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=pin,
                        persistent_workers=args.workers > 0)

    # Class imbalance (31 .. 2847 positives): weight positives by sqrt(neg/pos), capped.
    Y = np.stack([y for _, y in train_s])
    pos = Y.sum(0).clip(min=1)
    pos_weight = torch.tensor(np.sqrt((len(Y) - pos) / pos).clip(1, 10), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    net = FoodCNN(len(classes)).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_dl), pct_start=0.15)

    out_dir = Path(args.out_dir)
    best_f1, bad = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(net, train_dl, device, criterion, optimizer, scheduler)
        va = run_epoch(net, val_dl, device, criterion)
        micro, macro = f1_scores(va["probs"], va["targets"], 0.5)
        print(f"epoch {epoch:3d}  train_loss {tr['loss']:.4f}  val_loss {va['loss']:.4f}  "
              f"F1 micro {micro:.3f} macro {macro:.3f}")

        if macro > best_f1:
            best_f1, bad = macro, 0
            thr = tune_thresholds(va["probs"], va["targets"])
            torch.save({
                "model_state": net.state_dict(),
                "num_classes": len(classes),
                "classes": classes,
                "img_size": args.img_size,
                "mean": MEAN, "std": STD,
                "thresholds": thr.tolist(),
                "val_macro_f1": macro, "val_micro_f1": micro, "epoch": epoch,
            }, out_dir / "best_model.pth")
            with open(out_dir / "class_mapping.json", "w") as f:
                json.dump({"classes": classes,
                           "idx_to_class": {i: c for i, c in enumerate(classes)},
                           "thresholds": dict(zip(classes, map(float, thr)))}, f, indent=2)
            print(f"  saved best_model.pth (macro F1 {macro:.3f}, "
                  f"tuned-threshold micro/macro {f1_scores(va['probs'], va['targets'], thr)})")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stopping: no macro-F1 improvement for {args.patience} epochs")
                break
    print(f"done. best val macro F1 = {best_f1:.3f}")


if __name__ == "__main__":
    main()
