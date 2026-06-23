# 基础篇 · TimeCMA 复现与分析任务书

> 这是这次任务的三篇里的**第一篇(基础篇)**。目标不是"跑通别人的代码",而是让你把**"跨模态对齐"这件事从数学到工程彻底吃透**。读完 + 复现完,你应该能脱稿回答:为什么要对齐?对齐在优化什么?交叉注意力里的 Q/K/V 分别从哪来、为什么这么分?
>
> 带着本任务书第 2 节的问题去读论文,自己写答案,不许只抄论文原话。

---

## 0. 这篇论文是什么 · 为什么让你读

| 项目 | 内容 |
|---|---|
| 标题 | TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment |
| 会议/年份 | **AAAI 2025** |
| arXiv | https://arxiv.org/abs/2406.01638 |
| 代码仓库 | https://github.com/ChenxiLiu-HNU/TimeCMA(官方) |
| 本地 PDF | `papers/TimeCMA_2025_Liu_Cross_Modality_Alignment_MTSF.pdf` |

**一句话**:把"数值时间序列"和"文本(借由 LLM)"两种模态,对齐到一个共享潜空间,用 LLM 学到的高质量表示去**增强**时序预测——而不是让 LLM 直接预测数值。

**为什么读它(和课题的关系)**:整体课题要做**"时序-文本-图像三模态语义对齐"**(这是底层基石,上面才接多智能体通信)。TimeCMA 把其中最核心的**"时序 ↔ 文本"子对齐**做透了:数学干净、代码可跑、动机讲得清楚。**先把两模态怎么对齐吃透,再谈三模态、再谈 JEPA。** 它是你入门"模态对齐"领域的最佳起点。

---

## 1. 资源

- **PDF**:`papers/TimeCMA_2025_Liu_Cross_Modality_Alignment_MTSF.pdf`
- **仓库**:clone `https://github.com/ChenxiLiu-HNU/TimeCMA`
- **环境**:Python 3.11 + PyTorch 2.1.2 + CUDA 12.1(README 提供 `env_ubuntu.yaml` / `env_windows.yaml`)
- **数据**:TimesNet / TFB 的 8 个多变量时序数据集(ETT、Weather、Electricity 等,README 有 Google Drive 链接)
- **你的机器**:TimeCMA 用冻结的小 LLM(如 GPT-2 / Qwen 系小模型),**小数据集上 6G 可以跑**;跑不动就降 LLM backbone 或用 CPU/Colab。

---

## 2. 导读:带着这些问题去读

> 这一节的问题**分层**,从浅到深。**答案不许只贴论文原话**——要用自己的话 + 公式 + 例子。

### L1 · 概念层

1. 现有"LLM-based 时序预测"方法分哪两大类(论文 Section 1)?它们各自的输入是什么、各自的痛点是什么?
2. 论文反复说的 **"data entanglement(数据纠缠)"** 到底是什么意思?用**一个具体例子**解释:为什么把时序数值包进 text prompt 喂给 LLM,得到的 embedding 会"既强又纠缠"?
3. "disentangled yet weak" vs "entangled yet robust" —— 这是两个怎样的权衡(trade-off)?TimeCMA 的核心目标用一句话说,是想要什么样的 embedding?

### L2 · 机制层

4. **Dual-modality encoding** 有两个分支:
   - 时序分支为什么用 **inverted Transformer**(而非普通 Transformer)?(提示:去查 iTransformer,"变量当 token"是什么意思。)
   - LLM 分支为什么是 **frozen(冻结)** 的?它在这个框架里扮演什么角色?(为什么不是微调 LLM?)
5. **Cross-modality alignment module** 具体是怎么做"检索/对齐"的?
   - 它基于什么**相似度**计算?
   - 谁是 query,谁是 key/value?(这直接对应第 3 节你要推的交叉注意力公式。)
   - 为什么这一步能把"纠缠"变"不纠缠"?
6. 为什么要把时序信息**压到 last token**?**last token embedding storage** 这个设计在推理时省了什么计算?(提示:论文强调的 efficiency。)

### L3 · 数学层

7. **写出交叉注意力(cross-attention)的完整公式**,并标清在 TimeCMA 里 **Q / K / V 分别来自哪个分支的什么输出**。这一题答不清楚,等于没懂这篇论文。
8. 对齐 / 预测的**损失函数**是什么形式?是回归(MSE)?对比(contrastive)?还是两者结合?**写出它的数学目标和每一项的含义**。
9. 论文里"模态相似度"是用 **dot product、cosine、还是带温度的 softmax**?把它的计算式写出来,并解释温度系数 τ 在这里会起什么作用(如果你不确定论文用了哪个,就去**代码里找到那一行**确认——这是基本工)。
10. **(基础自查)** 看到这篇之前,你能否独立写出:① 标准 self-attention 公式;② LayerNorm 的作用;③ 残差连接为什么能训深网络?**如果这三个你有任何一个写不顺,停下来先补 Transformer 基础**——TimeCMA 的每一层都建立在这三个之上。

### L4 · 批判与迁移层

1.  TimeCMA 对齐的是**两个模态**。如果要再加**图像**做三模态,直接把这个框架套上去会撞到什么问题？
2.  TimeCMA 的对齐是**"检索/注意力式"**的;CLIP 是**"对比学习式"**的。这两种对齐范式在**目标函数**和**负样本依赖**上有什么本质区别?各自什么时候更好?
3.  TimeCMA 用一个 **frozen LLM 当增强器**。如果换成**"JEPA 式的预测潜空间"**来做对齐(即不冻结、也不重建,而是让一个编码器预测另一个编码器的潜表示),整体思路会怎么变?——**这道题是进阶篇 I-JEPA 的伏笔,先记下你的直觉,读完 I-JEPA 再回来对比。**

---

## 3. 复现 Roadmap

> 复现**允许用大模型辅助**(查 API、解释报错、调试、辅助写代码)，下面每一步都要有产出物。

- **Step 0 · 环境**(半天)
  - clone 仓库,`conda env create -f env_ubuntu.yaml`(或按你系统),下一个小数据集(**先只用 ETTh1**,别一上来全量)。
  - 跑通 `Store_ETTh1.sh`(预存 last token embedding)。
- **Step 1 · 跑通 baseline**(1-2 天)
  - `bash ETTh1.sh` 训练,记录你复现的 **MSE / MAE**,和论文 Table 里 ETTh1 的数值对齐(差多少?为什么差?写进报告)。
- **Step 2 · 代码精读**(2-3 天 · 最重要)
  - 在仓库里定位 **cross-modality alignment module** 的具体文件和函数,**逐行注释**:每行在干嘛、Q/K/V 怎么来的、相似度怎么算的。
  - 把 L3 第 9 题里"相似度到底用哪个公式"的答案,用代码实证。
- **Step 3 · Ablation(1天)**
  - ① **去掉 alignment 模块**(直接拿 prompt embedding 进预测头),看性能掉多少 → 量化 alignment 的贡献。
  - ② 换一个不同的 LLM backbone,看对齐效果对 backbone 的敏感度。
- **Step 4 · 可视化**(1 天)
  - 复现论文 **Figure 7 的注意力图**:看 TS token 和 word token 在 LLM 内部是怎么"纠缠"的。这张图是你理解"为什么要对齐"最直观的证据。
s
---

## 4. 你要交付什么

| 交付物 | 说明 | 暴露的能力 |
|---|---|---|
| **复现报告** | 环境、命令、复现数值 vs 论文数值(对齐度 + 差距原因) | 工程复现 |
| **代码批注** | alignment 模块逐行中文注释 | 代码阅读 |
| **Ablation 结果表** | Step 3 至少 2 组对比 + 结论 | 实验设计 |
| **L1/L2/L4 问题书面回答** | 用自己的话,带例子 | 论文理解 + 批判 |

---

## 5. 延伸

**"TimeCMA 给了我们哪些可以复用到三模态对齐的零件?它的对齐范式在加进图像模态时,会在哪一步首先失效?"**

把答案写进报告最后一节。这是你下一篇(I-JEPA)和视野篇(LatentMAS)的起点。

---

##一星期差不多了
