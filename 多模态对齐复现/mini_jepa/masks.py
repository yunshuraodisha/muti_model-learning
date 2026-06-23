"""随机 block mask 采样:target blocks(要预测的)+ context(可见的)。

对应论文 Sec. 3.2 / 4.2:masking 策略是 I-JEPA 成败的关键。
  · target 要「够大」(semantic scale)—— 这里用 2×2 block。
  · context 要 spatially distributed —— 这里从剩余 patch 随机散布。

🔑 Ablation:把 block_size 从 2 改成 1(即随机散点 target),linear probe 准确率会
下降 —— 印证论文「target 要够大才学到语义」的结论。
"""
import random
import torch


def sample_masks(B, grid, device, target_blocks=4, block_size=2, num_context=12):
    """每个样本:
       1) 随机放 target_blocks 个 block_size×block_size 的「不重叠」target block;
       2) 从剩余 patch 随机选 num_context 个作 context。

    返回:
       target_idx  (B, target_blocks * block_size**2) —— 要预测的 patch 位置
       context_idx (B, num_context)                   —— 可见的 patch 位置
    """
    num_patches = grid * grid
    Nt = target_blocks * (block_size ** 2)
    target_idx = torch.zeros(B, Nt, dtype=torch.long, device=device)
    context_idx = torch.zeros(B, num_context, dtype=torch.long, device=device)

    for b in range(B):
        used, t, attempts = set(), [], 0
        while len(t) < Nt and attempts < 200:
            r = random.randint(0, grid - block_size)
            c = random.randint(0, grid - block_size)
            new = [(r + dr) * grid + (c + dc)
                   for dr in range(block_size) for dc in range(block_size)]
            if not any(x in used for x in new):
                used.update(new)
                t.extend(new)
            attempts += 1
        # 兜底(几乎不会触发):用未占用 patch 补满
        if len(t) < Nt:
            rem = [i for i in range(num_patches) if i not in used]
            random.shuffle(rem)
            t = (t + rem)[:Nt]

        target_used = set(t)
        remaining = [i for i in range(num_patches) if i not in target_used]
        ctx = random.sample(remaining, min(num_context, len(remaining)))
        while len(ctx) < num_context:           # grid=6, Nt=16 时必足,兜底
            ctx.append(0)

        target_idx[b] = torch.tensor(t[:Nt], dtype=torch.long, device=device)
        context_idx[b] = torch.tensor(ctx, dtype=torch.long, device=device)

    return target_idx, context_idx
