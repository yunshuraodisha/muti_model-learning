"""Linear probe:冻结训好的 context encoder,训一个线性头评估表示质量。

用法:
  python linear_eval.py --config mini --ckpt ./ckpt/context_encoder_mini.pth

🔑 自监督评估的铁律:loss 低 ≠ 表示好。必须看 linear probe 的分类准确率。
   若 loss 一直降但 linear acc 不涨 → 很可能是表示坍塌(去 stop-grad 会触发)。
"""
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import MINI, SERIOUS
from models import ViT


def get_loaders(cfg):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tf = T.Compose([T.Resize((cfg.img_size, cfg.img_size)),
                    T.ToTensor(), T.Normalize(mean, std)])
    tr = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=tf)
    return (DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers),
            DataLoader(te, batch_size=256, shuffle=False, num_workers=cfg.num_workers))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mini", choices=["mini", "serious"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    cfg = MINI if args.config == "mini" else SERIOUS
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = ViT(img_size=cfg.img_size, patch_size=cfg.patch_size,
                  embed_dim=cfg.embed_dim, depth=cfg.depth, num_heads=cfg.num_heads)
    encoder.load_state_dict(torch.load(args.ckpt, map_location=device))
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # 图表示 = 所有 patch token 的平均(本实现 encoder 无 CLS token)
    clf = nn.Linear(cfg.embed_dim, 10).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)

    trl, tel = get_loaders(cfg)
    for epoch in range(args.epochs):
        clf.train()
        for x, y in trl:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feats = encoder(x).mean(dim=1)        # (B, D)
            logits = clf(feats)
            loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()

        clf.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in tel:
                x, y = x.to(device), y.to(device)
                feats = encoder(x).mean(dim=1)
                correct += (clf(feats).argmax(1) == y).sum().item()
                total += y.size(0)
        print(f"[epoch {epoch + 1}/{args.epochs}] test acc {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
