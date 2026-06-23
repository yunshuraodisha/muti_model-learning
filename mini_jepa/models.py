"""mini-JEPA 模型:Context/Target Encoder(ViT)+ Predictor + EMA。

对应论文:Assran et al., "Masked Image Modeling via Joint-Embedding Predictive
Architecture" (I-JEPA), CVPR 2023, Sec. 3。

🔑 学生必须能脱稿讲清:
  1. 为什么有 *两个* encoder 且参数不同 —— target 是 context 的 EMA 慢更新副本。
  2. stop-gradient 切在哪里(见 train.py 的 .detach() + no_grad)。
  3. Predictor 为什么 *必须* 接收 target 位置编码作为条件(不给位置 → 只能输出常数 → 坍塌)。
"""
import torch
import torch.nn as nn


def trunc_normal_(tensor, mean=0.0, std=0.02):
    """简化版 truncated normal 初始化(截断在 ±2σ)。"""
    with torch.no_grad():
        tensor.normal_(mean, std)
        tensor.clamp_(mean - 2 * std, mean + 2 * std)
    return tensor


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)        # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    """标准 pre-norm Transformer block:self-attn + MLP + 残差。"""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """ViT encoder,输出 patch-level tokens(不做 pooling)。

    支持传入 patch_idx:只编码选中的 patch 子集,并加上这些 patch 对应 *绝对位置*
    的 pos_embed。这是 I-JEPA 的关键 ——
      · context encoder 只看 context 区域
      · target  encoder 只看 target 区域
    但两者共享同一套「按 grid 位置定义」的 pos_embed。
    """

    def __init__(self, img_size=96, patch_size=16, in_chans=3,
                 embed_dim=192, depth=4, num_heads=3, mlp_ratio=4.0):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid = img_size // patch_size
        self.num_patches = self.grid ** 2
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, patch_idx=None):
        x = self.patch_embed(x)                  # (B, D, grid, grid)
        x = x.flatten(2).transpose(1, 2)         # (B, num_patches, D)
        pos = self.pos_embed.expand(x.size(0), -1, -1)

        if patch_idx is not None:
            # 每个样本选不同 patch:用 gather(每个样本可有自己的 mask)
            idx = patch_idx.unsqueeze(-1).expand(-1, -1, x.size(-1))  # (B, K, D)
            x = torch.gather(x, dim=1, index=idx)                      # (B, K, D)
            pos = torch.gather(pos, dim=1, index=idx)                  # (B, K, D)
        x = x + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x                                  # (B, K, D) patch-level


class Predictor(nn.Module):
    """窄 ViT:拿 context 表示 + target 位置的 query token,预测 target 的潜表示。

    🔑 关键:target query = learnable token + target 位置编码。
    没有位置编码,predictor 不知道「该预测哪个位置」,只能输出常数 → 表示坍塌。
    """

    def __init__(self, num_patches=36, embed_dim=192, pred_dim=192,
                 depth=2, num_heads=3, mlp_ratio=4.0):
        super().__init__()
        self.pred_dim = pred_dim

        self.ctx_proj = nn.Linear(embed_dim, pred_dim)
        self.target_token = nn.Parameter(torch.zeros(1, 1, pred_dim))        # learnable query
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, pred_dim)) # target 位置编码
        self.blocks = nn.ModuleList([
            Block(pred_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(pred_dim)
        trunc_normal_(self.target_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, ctx_tokens, target_idx):
        # ctx_tokens: (B, Nc, embed_dim) 来自 context encoder
        # target_idx: (B, Nt) 要预测的 patch 位置
        B, Nt = target_idx.shape
        ctx = self.ctx_proj(ctx_tokens)                                       # (B, Nc, pred_dim)

        tgt_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1), dim=1,
            index=target_idx.unsqueeze(-1).expand(-1, -1, self.pred_dim)
        )                                                                     # (B, Nt, pred_dim)
        tgt = self.target_token.expand(B, Nt, -1) + tgt_pos                   # (B, Nt, pred_dim)

        x = torch.cat([ctx, tgt], dim=1)                                      # (B, Nc+Nt, pred_dim)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, -Nt:, :]                                                   # 取 target 位置输出 (B, Nt, pred_dim)


class IJEPA(nn.Module):
    """组合:context encoder + target encoder(EMA)+ predictor。"""

    def __init__(self, cfg):
        super().__init__()
        vit_kwargs = dict(
            img_size=cfg.img_size, patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim, depth=cfg.depth, num_heads=cfg.num_heads,
        )
        self.context_encoder = ViT(**vit_kwargs)
        self.target_encoder = ViT(**vit_kwargs)
        self.predictor = Predictor(
            num_patches=cfg.grid ** 2,
            embed_dim=cfg.embed_dim, pred_dim=cfg.embed_dim,
            depth=cfg.pred_depth, num_heads=cfg.pred_num_heads,
        )
        # target encoder 初始 = context,且不参与反传
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_ema(self, tau=0.996):
        """🔑 EMA 更新:θ̃ ← τ·θ̃ + (1−τ)·θ。target_encoder 永远不反传,只靠这里慢更新。"""
        for p_t, p_c in zip(self.target_encoder.parameters(),
                            self.context_encoder.parameters()):
            p_t.data.mul_(tau).add_(p_c.data, alpha=1 - tau)


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
