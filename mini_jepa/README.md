# mini-JEPA · 从零实现的 I-JEPA 起步骨架(完整可运行版)

> 配合 `../02-IJEPA-进阶篇-任务书.md` 的**路线 B** 使用。
> 这是一个**只保留 JEPA 灵魂、纯 PyTorch、单卡就能跑**的最小 I-JEPA 实现。
> 目的是让你**亲手跑通 → 读懂 → 改做 ablation**,而不是去和 Meta 原版的 ViT-Huge + ImageNet 较劲。

---

## 1. 这是什么

I-JEPA 的灵魂 = **4 件东西**:

1. **Context encoder**(可训练 ViT)— 只看图的 context 区域
2. **Target encoder**(EMA 慢更新 + stop-grad)— 只看 target 区域,产生预测目标
3. **Predictor**(窄 ViT)— 拿 context 表示 + target 位置编码,猜 target 的潜表示
4. **潜空间 L1 损失** — 在表示空间回归,**不重建像素**

本目录把这 4 件用 ~300 行纯 PyTorch 搭起来,CIFAR-10 上能跑、能学、能评估。**VISSL、分布式样板、复杂 mask 采样优化一概不要。**

---

## 2. 怎么跑

```bash
pip install -r requirements.txt                  # torch + torchvision

# ① 预训练(mini 档:ViT-Tiny + CIFAR-10 + 50ep,单卡甚至 CPU 都能跑)
python train.py --config mini --out_dir ./ckpt

# ② Linear probe 评估表示质量
python linear_eval.py --config mini --ckpt ./ckpt/context_encoder_mini.pth
```

跑完你会看到 linear probe 的 CIFAR-10 测试准确率从随机水平(~10%)逐步上升 —— **这就是「学到了语义表示」的证据**。

## 3. 认真档(发挥 4×48G)

```bash
python train.py --config serious     # ViT-Small + 200ep + batch512
```

认真档在**单张 48G 卡**上就能跑;想用满 4 卡,用 `torchrun --nproc_per_node=4 train.py`(需自行把模型/优化器套 DDP,训练循环不用改)。**但不建议追求更大模型** —— 理解机制比堆规模重要,堆规模是 Meta 工程师的事。

---

## 4. 代码结构(对照论文)

| 文件 | 内容 | 对应论文 |
|---|---|---|
| `models.py` | ViT encoder / Predictor / EMA / IJEPA 组合 | Sec. 3 方法 |
| `masks.py` | target block + context mask 采样 | Sec. 3.2 / 4.2 masking |
| `train.py` | 预训练循环(前向 + loss + EMA) | Sec. 3 + Alg. 1 |
| `linear_eval.py` | linear probe 评估 | Sec. 4.1 评估协议 |
| `config.py` | 超参(mini / serious 两档) | Sec. 4 实验设置 |

---

## 5. 🔑 你必须能脱稿讲清的清单(导师会逐条抽查)

代码里这些位置打了 `🔑` 注释,你必须能讲清「为什么」:

- [ ] **`models.py · IJEPA.__init__`** — 为什么有 *两个* encoder?为什么 `target_encoder.requires_grad = False`?
- [ ] **`train.py · target_repr`** — 为什么在 `torch.no_grad()` 里算?stop-gradient 切在哪?
- [ ] **`train.py · update_target_ema`** — 写出 EMA 公式 `θ̃ ← τ·θ̃ + (1−τ)·θ`;τ≈0.996 接近 1 的直觉是什么?(τ 太小会怎样?)
- [ ] **`models.py · Predictor.forward`** — 为什么 target query 要加位置编码?不给位置会怎样?(答:predictor 只能输出常数 → 表示坍塌)
- [ ] **`train.py · smooth_l1_loss(pred, target_repr.detach())`** — 为什么是潜空间 L1 而不是像素重建?
- [ ] **灵魂题(口头)** — 没有负样本,为什么表示不坍塌?(三点:EMA 慢靶 + stop-grad + predictor 位置条件)

> **讲不清这些,即使代码跑通了,也不算过关。** 导师要的是你懂「为什么」,不是你跑通了。

---

## 6. Ablation 建议(用认真档跑,这些最能出结论)

改 `config.py` 或代码,观察 linear probe 准确率:

| 实验 | 改哪里 | 预期结论 |
|---|---|---|
| **去 stop-gradient** | `train.py` 去掉 `target_repr.detach()` 和 `no_grad` | **表示坍塌**(acc 掉到随机水平)← 最有教学价值 |
| **去 predictor 位置编码** | `models.py · Predictor` 去掉 `tgt_pos` | **坍塌** |
| **像素重建 vs 潜空间** | `train.py` loss 换成对像素的 MSE(需加 decode 头) | 像素重建表示更差 → 印证「预测潜空间更 semantic」 |
| **target block 尺寸** | `config.py` `block_size: 2 → 1` | 散点 target 学不到语义 → 印证「target 要够大」 |
| **EMA τ** | `config.py` `ema_tau: 0.996 → 0.5` | τ 太小训练不稳 |

> **「去 stop-grad」和「去位置编码」这两个实验,会让你亲眼看到坍塌** —— 这比读十遍论文都管用。

---

## 7. 常见坑

- **CIFAR-10 下载**:首次运行自动下到 `./data`;网络不通就手动下放到 `./data/cifar-10-batches-py`。
- **loss 一直降但 linear acc 不涨** → 多半是表示坍塌的信号(去 stop-grad 会触发)。**一定要同时盯 linear probe,不能只盯 loss。**
- **OOM** → 调小 `config.py` 的 `batch_size`。
- **CPU 跑** → mini 档能跑,50ep 几小时;有卡就用卡。

---

## 8. 和课题的衔接

跑通 + 读懂 + 做完 ablation 后,你应该能回答:

> I-JEPA 是**单模态(图像)**的潜空间预测;要把它变成**时序-文本-图像三模态对齐**,你会怎么改造「编码器数量」「预测目标(谁预测谁)」「位置条件」?

把答案写进 `../02-IJEPA-进阶篇-任务书.md` 的 **L4 第 13 题**。这才是这篇训练真正服务的那个问题。
