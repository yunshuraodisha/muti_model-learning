"""mini-JEPA 预训练。

用法:
  python train.py --config mini         # ViT-Tiny + CIFAR-10 + 50ep(单卡够)
  python train.py --config serious      # ViT-Small + 200ep + batch512(发挥 4×48G)

一次 step 的数据流(对照 models.py):
  图 --context_enc--> context 表示 --predictor(+target 位置)--> 预测 ŝ_y
  图 --target_enc(EMA / stop-grad)--> target 表示 s_y
  loss = smooth_l1( ŝ_y , s_y.detach() )        # 🔑 在潜空间,不是像素重建
  反传只更新 context_encoder + predictor;target_encoder 由 EMA 更新
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import MINI, SERIOUS
from masks import sample_masks
from models import IJEPA, init_weights


def get_loader(cfg):
    # 🔑 I-JEPA 不用强数据增强(这是它的特点之一!)—— 只 resize + 水平翻转 + normalize
    tf = T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=tf)
    return DataLoader(train, batch_size=cfg.batch_size, shuffle=True,
                      num_workers=cfg.num_workers, drop_last=True)


def lr_scale(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mini", choices=["mini", "serious"])
    ap.add_argument("--out_dir", default="./ckpt")
    args = ap.parse_args()

    cfg = MINI if args.config == "mini" else SERIOUS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[config] {args.config} | device={device} | {cfg}")

    model = IJEPA(cfg).to(device)
    model.apply(init_weights)
    model.target_encoder.load_state_dict(model.context_encoder.state_dict())  # EMA 起点对齐

    loader = get_loader(cfg)
    params = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs * len(loader)
    warmup = max(1, total_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_scale(s, warmup, total_steps))

    model.train()
    step = 0
    for epoch in range(cfg.epochs):
        t0, loss_sum, n = time.time(), 0.0, 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            B = x.size(0)

            target_idx, context_idx = sample_masks(
                B, cfg.grid, device,
                target_blocks=cfg.target_blocks,
                block_size=cfg.block_size,
                num_context=cfg.num_context,
            )

            # 🔑 target 表示(stop-gradient):target_encoder 只看 target 区域
            with torch.no_grad():
                target_repr = model.target_encoder(x, patch_idx=target_idx)   # (B, Nt, D)

            # context 表示:context_encoder 只看 context 区域
            context_repr = model.context_encoder(x, patch_idx=context_idx)    # (B, Nc, D)

            # 🔑 predictor 拿 context 表示 + target 位置 → 预测 target 表示
            pred = model.predictor(context_repr, target_idx)                  # (B, Nt, D)

            # 🔑 损失在「潜空间」(smooth L1),不是像素重建;target 端 stop-grad
            loss = nn.functional.smooth_l1_loss(pred, target_repr.detach())

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            # 🔑 EMA 更新 target_encoder(不反传)
            model.update_target_ema(cfg.ema_tau)

            loss_sum += loss.item() * B
            n += B
            step += 1
            if step % 50 == 0:
                print(f"  step {step}/{total_steps} | loss {loss.item():.4f} "
                      f"| lr {opt.param_groups[0]['lr']:.2e}")

        print(f"[epoch {epoch + 1}/{cfg.epochs}] loss {loss_sum / n:.4f} "
              f"| {time.time() - t0:.0f}s")

    ckpt = os.path.join(args.out_dir, f"context_encoder_{args.config}.pth")
    torch.save(model.context_encoder.state_dict(), ckpt)
    print(f"[saved] {ckpt}")


if __name__ == "__main__":
    main()
