# 进阶篇 · I-JEPA 复现与分析任务书

> 这是三篇里的**第二篇(进阶篇)**。基础篇 TimeCMA "用交叉注意力做对齐";这一篇带你进入**课题真正要模仿的路线——JEPA(预测潜空间而非重建像素)**。这篇的数学比 TimeCMA 重,是检验你"深度学习数学原理到底懂没懂"的关键一篇。
>
> **前置**:先把基础篇 TimeCMA 交付了再来。两篇的对齐范式要能对比(见本篇 L4)。

---

## 0. 这篇论文是什么 · 为什么让你读

| 项目 | 内容 |
|---|---|
| 标题 | Masked Image Modeling via Joint-Embedding Predictive Architecture(I-JEPA) |
| 会议/年份 | **CVPR 2023** |
| arXiv | https://arxiv.org/abs/2301.08243 |
| 代码仓库 | https://github.com/facebookresearch/ijepa(Meta 官方) |
| 本地 PDF | `papers/I-JEPA_2023_Assran_Masked_Image_Modeling_JEPA.pdf` |

**一句话**:不让模型重建像素、也不用负样本做对比,而是让一个编码器**看图像的一小块 context,去预测另一块被遮住区域在潜空间里的表示**——这个表示由一个慢更新的"目标编码器"给出。

**为什么读它(和课题的关系)**:课题决定"模仿 **JEPA / V-JEPA2 路线**做时序-文本-图像对齐",而 **I-JEPA 就是 JEPA 思想的根**(V-JEPA、V-JEPA2 都是它的视频/时序延伸)。把 I-JEPA 的机制吃透,你才能理解为什么"对比学习不够、要结构化潜空间对齐"。**这一篇是该课题方法论的直接原型。**

---

## 1. 资源

- **PDF**:`papers/I-JEPA_2023_Assran_Masked_Image_Modeling_JEPA.pdf`
- **仓库**:clone `https://github.com/facebookresearch/ijepa`
- **环境**:纯 PyTorch(已在 `mini_jepa/` 目录提供**完整可运行的起步骨架**,不依赖 Meta 的 VISSL,免去环境配置)
- **你的机器**:算力够跑 ViT-Small + ImageNet 子集;但 I-JEPA 原版是 ViT-Huge/14 + ImageNet 全量用 16×A100-80G(仍跑不动,也没必要)。复现策略 = **基于 mini-JEPA 骨架验证 JEPA 思想 + 用算力做严肃 ablation**,不是刷 SOTA。

---

## 2. 导读:带着这些问题去读(自己写答案)

### L1 · 概念层(JEPA 在自监督里处在什么位置)

1. 自监督表示学习大致有**三大范式**,各举一个代表工作,说清它们各自**优化什么**:
   - ① **对比学习**(实例判别 / 负样本)——例:SimCLR / CLIP
   - ② **掩码重建**(重建像素或 token)——例:MAE / BEiT
   - ③ **预测潜空间**(JEPA)——例:I-JEPA
2. I-JEPA 明确**拒绝三样东西**:① 手工数据增强 ② 负样本 ③ 像素重建。它**为什么拒绝每一个**?拒绝了这些,它靠什么还能学到好表示?
3. LeCun 的 JEPA 哲学一句话:"**预测潜空间(抽象表示),而不是重建像素或对比负样本**"。用你自己的话解释:为什么"预测抽象表示"比"重建像素"更适合学语义?

### L2 · 机制层(I-JEPA 的三个组件 + masking)

4. I-JEPA 有三个核心组件:**Context-Encoder、Target-Encoder、Predictor**。分别说清:
   - 各自的输入是什么、输出是什么?
   - 哪些有梯度回传,哪些没有(stop-gradient)?

**5，6问题不用回答，太难**
 
5. **Target-Encoder 为什么用 EMA(指数移动平均)慢更新**,而不是直接和 Context-Encoder 共享参数?如果直接共享参数会出什么问题?
6. **Masking 策略是 I-JEPA 成败的关键**(论文强调)。解释为什么:
   - ① **Target blocks 要"足够大"**(semantic scale)——太小会学到什么坏东西?
   - ② **Context block 要"空间上分散且信息足够"**(spatially distributed)——如果 context 很集中、或信息太少(几乎全 mask 掉)会怎样?


7. **Predictor** 的输入除了 Context-Encoder 的输出,**还接收什么作为条件**?(提示:被预测的 target block 的**位置编码**。)为什么没有这个条件,整个方法就崩了?

### L3 · 数学层 （尽可能回答吧）

8. **写出 I-JEPA 的损失函数**。它是什么形式(L1 / L2 / smooth-L1 回归)?**在什么空间上回归**?写出公式,标清:① 哪一项是 Predictor 的输出 ② 哪一项是 Target-Encoder 的输出(且 stop-gradient)。

9.  **(全篇最关键 · 防坍塌)** I-JEPA **没有负样本**。对比学习(CLIP/SimCLR)必须靠**负样本**把不同样本推开,否则表示会**坍塌**(所有样本映射到同一点,损失为 0 但啥也没学)。**I-JEPA 靠什么机制防止坍塌?**
    - 请解释 **EMA + stop-gradient + Predictor 的非对称结构**如何共同起作用。
    - (可类比:BYOL、SimSiam 也是无负样本方法,它们靠什么防坍塌?和 I-JEPA 异同?)
    - **这很难**。
  
10. **(和 MAE 对照)** MAE 的损失是**像素空间 L2 重建**,I-JEPA 的损失是**潜空间回归**。请解释:为什么"不重建像素"反而能学到**更 semantic** 的表示?(提示:像素里有大量高频噪声/纹理细节,强迫模型重建它们会浪费容量在"非语义"信息上。)
11. **(基础自查)** 看到这篇之前,独立写清:① ViT 把图像切成 patch 后如何变 token;② LayerNorm vs BatchNorm;③ 残差连接。

### L4 · 批判与迁移层

13. **(课题核心题)** I-JEPA 是纯图像**单模态**。如果要把它迁移到**时序-文本-图像三模态对齐**,你会怎么设计?
    - 几个编码器(每模态一个?共享?)?
    - "预测谁"——让时序编码器预测图像编码器的潜表示?还是反过来?还是有共享枢纽?
    - **这道题写下你的初步设计,不用完美——导师要看的是你的思路。**

---

## 3. 复现 Roadmap(4×48G · 基于 mini-JEPA 骨架)

> 在 `mini_jepa/` 目录放了一个**完整可运行的 mini-JEPA 起步骨架**(纯 PyTorch,~300 行,机制完整,关键处用 `🔑` 标注)。你的任务是:**跑通 → 读懂(对照 `🔑` 清单)→ 改做 ablation**。**先读 `mini_jepa/README.md`。**

### Step 0 · 跑通
- `pip install -r mini_jepa/requirements.txt`
- `python mini_jepa/train.py --config mini`(ViT-Tiny + CIFAR-10 + 50ep,**单卡甚至 CPU 都能跑通**)
- `python mini_jepa/linear_eval.py --config mini --ckpt ckpt/context_encoder_mini.pth`
- 看到 linear probe 准确率从 ~10% 往上爬 = 表示学到了语义 = 跑通。

### Step 1 · 读懂
- 对照 `mini_jepa/README.md` 第 5 节的 **🔑 清单**,逐一吃透 6 个"必须能讲清"的点:
  ① 两个 encoder 为何参数不同 ② stop-grad 切在哪 ③ EMA 公式与 τ 直觉 ④ predictor 为何要位置编码 ⑤ 损失为何在潜空间 ⑥ 无负样本为何不坍塌。
- **这 6 点就是本篇 L2/L3 问题的答案落点,讲不清就回论文对应章节重读。**

### Step 2 · 用 服务器 跑认真档 + Ablation
- `python mini_jepa/train.py --config serious`(ViT-Small + 200ep + batch512,单张 48G 够)。
- **必做 ablation**(见 `mini_jepa/README.md` 第 6 节,每组只改一两行):
  - ① **去 stop-gradient** → 亲眼看到**表示坍塌**(linear acc 掉到随机水平)← 全篇最有教学价值的实验
  - ② **像素重建 vs 潜空间回归** → 印证"预测潜空间更 semantic"
- 把每组的 linear probe 曲线画出来对比,写进报告。

---

## 4. 你要交付什么

| 交付物 | 说明 | 暴露的能力 |
|---|---|---|
| **复现报告** | 走了哪条路线、small config 细节、linear eval 数值 | 工程复现 |
| **代码** | 基于 `mini_jepa/` 骨架的运行记录 + ablation 改动 + 🔑 清单的讲解 | 实现能力 |
| **Ablation 结果表** | 至少 2 组对比(尤其"像素重建 vs 潜空间预测") | 实验设计 |
| **L1/L2/L4 问题书面回答** | 用自己的话,第 13 题要写出你的设计思路 | 理解 + 课题迁移 |

---

## 5. 延伸(和课题的衔接)

读完这篇,应该能用一段话回答:**"JEPA 的'预测潜空间'思想,比 TimeCMA 的'交叉注意力检索'更接近我们课题要的结构化潜空间对齐——原因是……;但 I-JEPA 本身是单模态预测、不是多模态对齐,要把它变成三模态对齐,我打算这样改造……"**

这篇是你下一篇视野篇(LatentMAS)的桥梁:LatentMAS 会告诉你"对齐/预测出来的潜空间,在多智能体系统里到底怎么被用来通信"。

---

## 附:建议节奏(1 周)
