from dataclasses import dataclass


@dataclass
class Config:
    """mini-JEPA 超参。grid = img_size // patch_size 必须整除。"""
    dataset: str = "cifar10"
    img_size: int = 96
    patch_size: int = 16
    grid: int = 6                 # = img_size // patch_size,patch grid = 6×6 = 36

    # ViT encoder(ViT-Tiny 默认)
    embed_dim: int = 192
    depth: int = 4
    num_heads: int = 3

    # Predictor(窄 ViT)
    pred_depth: int = 2
    pred_num_heads: int = 3

    # Mask
    target_blocks: int = 4        # 放 4 个 target block
    block_size: int = 2           # 每个 target block = 2×2 patch(够大、够 semantic)
    num_context: int = 12         # 从剩余 patch 随机选 12 个作 context

    # 训练
    batch_size: int = 256
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.05
    ema_tau: float = 0.996        # 🔑 EMA 系数,接近 1

    num_workers: int = 4


# mini 档:理解机制,先跑通(单卡甚至 CPU 都能跑)
MINI = Config()

# 认真档:发挥 4×48G(ViT-Small + 长 epoch + 大 batch)
SERIOUS = Config(
    embed_dim=384, depth=6, num_heads=6,        # ViT-Small
    pred_depth=3, pred_num_heads=6,
    epochs=200,
    batch_size=512,
)
