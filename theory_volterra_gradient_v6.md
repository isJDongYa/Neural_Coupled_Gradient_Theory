# Transformer 的耦合梯度定理（第六版）：
# 注意力机制的双重 Volterra 分解与训练动力学

> **与前五版的关系**：
> - 第一版：两层多项式激活，精确等式（单步梯度 = 各阶核误差之和）
> - 第二版：两层任意 $L^2(\gamma)$ 激活，Hermite→单项式近似
> - 第三版：$L$ 层 plain 网络 + 各阶学习速率 $\eta_{\text{eff}}(n)$ + SGD 隐式正则化
> - 第四版：残差连接 + 归一化层 + 泛化界 $n^* = O(\log m/\log p)$
> - 第五版：多步训练动力学 ODE + 阶间相变 + 双重瓶颈定理
> - **本版（第六版）**：将 NCGT 从 FFN 推广到完整 Transformer——建立注意力机制的**双重 Volterra 分解**（特征阶 $\times$ 交互阶），推导 Transformer block 的耦合梯度动力学

---

## 1. 动机

### 1.1 v1–v5 的共同局限

前五版建立了从单步梯度到多步动力学的完整理论链条，但分析对象始终限于 **FFN 类网络**：

- v1–v2：两层 MLP $f(\mathbf{x}) = \mathbf{w}_2^\top \sigma(W_1 \mathbf{x})$
- v3：$L$ 层 plain 全连接网络
- v4：残差连接 + 归一化层（Pre-Norm ResNet）
- v5：上述网络的多步训练动力学

所有版本中，**输入 $\mathbf{x} \in \mathbb{R}^p$ 是单个向量**，网络的非线性完全来自激活函数 $\sigma$，Volterra 分解的"阶"只有一个维度：对 $\mathbf{x}$ 的多项式次数 $n$。

然而，现代深度学习的核心架构——**Transformer**——有一个根本性的新结构：**注意力机制**（self-attention）。它引入了两个 v1–v5 未曾处理的要素：

1. **输入是序列**：$\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_T]^\top \in \mathbb{R}^{T \times p}$，不是单个向量
2. **Token 间的双线性交互**：$\text{Attn}(\mathbf{X}) = \text{softmax}\!\left(\frac{\mathbf{X} W_Q W_K^\top \mathbf{X}^\top}{\sqrt{d_k}}\right) \mathbf{X} W_V$

这意味着 Transformer 的输出**既依赖于每个 token 的特征（来自 FFN），也依赖于 token 之间的关系（来自 Attention）**。v1–v5 的分析只覆盖了前者。

### 1.2 为什么注意力需要新理论

将注意力纳入 Volterra 框架绝非简单的推广。具体困难：

**困难 1：双重非线性指标**

FFN 中，$\sigma(z) = \sum_n a_n z^n$ 的阶 $n$ 是唯一的非线性度量——它表示对输入 $\mathbf{x}$ 的多项式次数。

Attention 引入了一个**新的独立维度**：

$$\underbrace{Q_t^\top K_s}_{\text{token } t \text{ 与 token } s \text{ 的交互}} = \mathbf{x}_t^\top W_Q W_K^\top \mathbf{x}_s$$

这是关于**两个不同 token** $\mathbf{x}_t, \mathbf{x}_s$ 的双线性函数。在 softmax 中，$\exp(Q_t^\top K_s / \sqrt{d_k})$ 的 Taylor 展开每增加一阶，就增加一对 token 的交叉。因此需要区分：

- **特征阶**（feature order）$n$：对单个 token $\mathbf{x}_t$ 的多项式次数（FFN 贡献，v4 已覆盖）
- **交互阶**（interaction order）$r$：涉及多少个不同 token 位置的交叉项（Attention 贡献，本版新增）

一个 Transformer 的输出分量可以同时具有特征阶 $n$ 和交互阶 $r$。例如，$\mathbf{x}_t^\top A \mathbf{x}_s \cdot \sigma(\mathbf{x}_s^\top B \mathbf{x}_s)$ 具有特征阶 $n=2$（对 $\mathbf{x}_s$）和交互阶 $r=1$（涉及 token $t$ 和 $s$）。

**困难 2：Softmax 不是多项式**

FFN 的 $\sigma$ 可以精确地或近似地写为多项式 $\sum a_n z^n$。但 softmax 是**有理函数**：

$$\text{softmax}(\mathbf{s})_i = \frac{e^{s_i}}{\sum_j e^{s_j}}$$

分母的存在使得各交互阶**不可加地**混合在一起——类似 v4 中 Post-Norm 引起的阶混合问题，但程度更严重。

**困难 3：因果掩码**

语言模型中的因果掩码 $M_{ts} = \mathbf{1}[t \geq s]$ 破坏了注意力矩阵的对称性，使得 $(t,s)$ 和 $(s,t)$ 位置的交互不对称。这对 Volterra 核的对称性假设构成挑战。

### 1.3 v6 的核心贡献

本版将 NCGT 从 FFN 推广到完整的 Transformer block，建立以下结果：

1. **双重 Volterra 分解**（Section 3–4）：将 Transformer 输出分解为按 $(n, r)$——特征阶 $\times$ 交互阶——索引的 Volterra 分量
   - Linear Attention（Section 3）：精确分解，$r \leq 1$
   - Softmax Attention（Section 4）：Taylor 展开，$r = 0, 1, 2, \ldots$

2. **双重有效学习率**（Section 5）：推导 Transformer block 中每个 $(n, r)$ 分量的有效学习率 $\eta_{\text{eff}}(n, r)$，以及 FFN–Attention 之间的交叉耦合

3. **因果掩码的 Volterra 效应**（Section 6）：因果约束如何限制可表达的核类型，以及对各交互阶学习速率的影响

4. **与 CKHT 的连接**（Section 7）：CKHT（因果核层级理论）描述了序列模型**能表达什么**——核约束、状态介质、选择性三个轴上的层级。v6 描述 Transformer **如何学习这些核**——各阶的训练动力学。二者互补：CKHT 给出表达力的"地图"，v6 给出训练动力学的"导航"

### 1.4 章节导览

| Section | 内容 | 类比 v1–v5 |
|---------|------|-----------|
| 2 | 设置与符号 | v5 Section 2 |
| 3 | Linear Attention 的精确 Volterra 分解 | v1（精确等式，基准情形） |
| 4 | Softmax Attention 的 Taylor 展开与双重分解 | v2（近似展开） |
| 5 | Transformer Block 的耦合 ODE | v5 Section 4（耦合演化） |
| 6 | 因果掩码的效应 | 新（v1–v5 无对应） |
| 7 | 与 CKHT 的连接 | 新（v1–v5 无外部理论对接） |
| 8 | 讨论与展望 | v5 Section 9 |

---

## 2. 设置与符号

### 2.1 Transformer 架构

考虑 $L$ 层 Pre-Norm Transformer（decoder-only），输入序列 $\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_T]^\top \in \mathbb{R}^{T \times p}$。

**单层 Transformer block**（第 $l$ 层）：

$$\mathbf{H}^{(l)} = \mathbf{H}^{(l-1)} + \underbrace{\text{Attn}^{(l)}\!\left(\text{LN}_1^{(l)}(\mathbf{H}^{(l-1)})\right)}_{\text{token 间交互}} + \underbrace{\text{FFN}^{(l)}\!\left(\text{LN}_2^{(l)}(\mathbf{H}^{(l-1)} + \text{Attn}^{(l)})\right)}_{\text{token 内非线性}}$$

其中 $\mathbf{H}^{(0)} = \mathbf{X} W_E$（$W_E \in \mathbb{R}^{p \times d}$ 为嵌入矩阵），$d$ 为隐藏维度。

**注意力子层**（单头，简化表述）：

$$\text{Attn}(\mathbf{Z}) = \text{softmax}\!\left(\frac{\mathbf{Z} W_Q W_K^\top \mathbf{Z}^\top}{\sqrt{d_k}} + M\right) \mathbf{Z} W_V W_O$$

其中：
- $W_Q, W_K \in \mathbb{R}^{d \times d_k}$：Query 和 Key 投影
- $W_V \in \mathbb{R}^{d \times d_v}$，$W_O \in \mathbb{R}^{d_v \times d}$：Value 投影和输出投影
- $M \in \mathbb{R}^{T \times T}$：因果掩码，$M_{ts} = 0$ 若 $t \geq s$，$M_{ts} = -\infty$ 若 $t < s$
- $d_k = d / H$（$H$ 为注意力头数，单头时 $d_k = d$）

**FFN 子层**：

$$\text{FFN}(\mathbf{z}) = W_2\, \sigma(W_1\, \mathbf{z})$$

其中 $W_1 \in \mathbb{R}^{d_{ff} \times d}$，$W_2 \in \mathbb{R}^{d \times d_{ff}}$，$\sigma$ 为逐元素激活函数。

### 2.2 序列 Volterra 分解

v1–v5 中 Volterra 分解的对象是标量函数 $f: \mathbb{R}^p \to \mathbb{R}$：

$$f(\mathbf{x}) = \sum_{n=0}^{N_{\max}} f_n(\mathbf{x}), \quad f_n \text{ 为 } n \text{ 次齐次多项式}$$

对序列模型 $F: \mathbb{R}^{T \times p} \to \mathbb{R}^{T \times d}$，需要推广为**序列 Volterra 分解**。

**定义 2.1**（序列 Volterra 分解——双重索引）

Transformer 第 $t$ 个位置的输出 $\mathbf{h}_t = F(\mathbf{X})_t$ 分解为：

$$\mathbf{h}_t = \sum_{n=0}^{N_{\max}} \sum_{r=0}^{R_{\max}} \mathbf{h}_{t}^{(n,r)}$$

其中 $\mathbf{h}_t^{(n,r)}$ 满足：

**(i) 特征阶 $n$**：对每个参与的 token $\mathbf{x}_s$，$\mathbf{h}_t^{(n,r)}$ 关于 $\mathbf{x}_s$ 的多项式次数为 $n$（即特征空间中的 Volterra 阶，与 v1–v5 的定义一致）

**(ii) 交互阶 $r$**：$\mathbf{h}_t^{(n,r)}$ 涉及**恰好 $r+1$ 个不同位置** $s_0, s_1, \ldots, s_r$（包含 $t$ 自身）的 token 的交叉乘积。当 $r = 0$ 时只依赖 $\mathbf{x}_t$ 自身

形式地：

$$\mathbf{h}_t^{(n,r)} = \sum_{\substack{s_1, \ldots, s_r \\ s_i \leq t}} \mathcal{K}_{n,r}^{(t, s_1, \ldots, s_r)}\left[\mathbf{x}_t^{\otimes n_0}, \mathbf{x}_{s_1}^{\otimes n_1}, \ldots, \mathbf{x}_{s_r}^{\otimes n_r}\right]$$

其中 $n_0 + n_1 + \cdots + n_r = n \cdot (r+1)$（总多项式次数），$\mathcal{K}_{n,r}^{(\cdots)}$ 为**序列 Volterra 核**。

> **与 v1–v5 的关系**：FFN-only 网络的 Volterra 分解是 $r = 0$ 的特殊情形——只有 $\mathbf{h}_t^{(n,0)}$ 非零，退化为 $\mathbf{h}_t = \sum_n f_n(\mathbf{x}_t)$，即 v1–v5 的标准分解。

**定义 2.2**（序列 Volterra 核张量）

第 $(n,r)$ 阶的序列 Volterra 核是张量：

$$\mathcal{K}_{n,r} \in \mathbb{R}^{\underbrace{p \times \cdots \times p}_{n \cdot (r+1)} \times d}$$

它将 $r+1$ 个 token 各自的 $n$ 阶张量积映射到 $d$ 维输出空间。

### 2.3 各阶误差的定义

设目标序列函数 $F^*: \mathbb{R}^{T \times p} \to \mathbb{R}^{T \times d}$ 同样具有双重 Volterra 分解，核为 $\mathcal{K}_{n,r}^*$。

**各阶误差信号**：

$$\varepsilon_t^{(n,r)}(\mathbf{X}) = \mathbf{h}_t^{*(n,r)} - \mathbf{h}_t^{(n,r)}$$

$$\delta \mathcal{K}_{n,r} = \mathcal{K}_{n,r}^* - \mathcal{K}_{n,r}$$

**各阶误差能量**：

$$E_{n,r}(t) = \left\|\delta \mathcal{K}_{n,r}(t)\right\|_F^2$$

**总损失**（自回归交叉熵的二阶近似，或 MSE）：

$$\mathcal{L}(\theta) = \frac{1}{T} \sum_{t=1}^{T} \left\|F^*(\mathbf{X})_t - F(\mathbf{X}; \theta)_t\right\|^2$$

### 2.4 正交性结构

v1–v5 中，不同阶的齐次多项式在 $L^2(\gamma_p)$（高斯测度）下正交。对序列 Volterra 分解，有类似但更丰富的正交性。

**命题 2.3**（双重正交性）

设各 token $\mathbf{x}_t \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, I_p)$（或更一般地，token 间独立且同分布），则：

**(i) 特征阶正交性**（沿用 v1 引理 2.2）：

$$n \neq n' \implies \mathbb{E}_{\mathbf{x}}\!\left[\mathbf{h}_t^{(n,r)} \cdot \mathbf{h}_t^{(n',r')}\right] = 0$$

**(ii) 交互阶正交性**（新）：

$$r \neq r' \implies \mathbb{E}_{\mathbf{X}}\!\left[\mathbf{h}_t^{(n,r)} \cdot \mathbf{h}_t^{(n,r')}\right] = 0$$

**证明**：

(i) 固定参与的 token 集后，两个不同特征阶的齐次多项式在 $L^2(\gamma_p)$ 下正交，沿用 v1 引理 2.2。

(ii) 当 token 间独立时，$\mathbf{h}_t^{(n,r)}$ 涉及 $r$ 个额外 token 的求和。$r \neq r'$ 时，存在某个位置 $s$ 在一个项中出现但在另一个中不出现。对 $\mathbf{x}_s$ 取期望时，含奇数次 $\mathbf{x}_s$ 的项为零（零均值高斯）。形式验证需要对指标集做仔细的组合分析。$\square$

**推论 2.4**（总损失的双重分解）

在上述条件下，总损失精确分解为各阶误差能量之和（无交叉项）：

$$\mathcal{L}(\theta) = \sum_{n,r} c_{n,r}\, E_{n,r}(t)$$

其中 $c_{n,r}$ 为归一化常数。

> **关键意义**：与 v5 Section 2 相同——正交性保证总损失可精确分解为各 $(n,r)$ 阶的独立贡献。这使得我们可以**逐阶分析**训练动力学。

### 2.5 核心记号汇总

| 符号 | 含义 | 来源 |
|------|------|------|
| $\mathbf{X} \in \mathbb{R}^{T \times p}$ | 输入序列 | 本版 |
| $n$ | 特征阶（对单 token 的多项式次数） | v1–v5 |
| $r$ | 交互阶（涉及的 token 位置数 $- 1$） | 本版 |
| $\mathcal{K}_{n,r}$ | 第 $(n,r)$ 阶序列 Volterra 核 | 本版 |
| $E_{n,r}(t)$ | 第 $(n,r)$ 阶误差能量 | 本版 |
| $\eta_{\text{eff}}(n,r)$ | 第 $(n,r)$ 阶有效学习率 | 本版 |
| $\alpha_{n,r}$ | 第 $(n,r)$ 阶自衰减率 | 本版 |
| $W_Q, W_K, W_V, W_O$ | Attention 参数 | 标准 |
| $W_1, W_2$ | FFN 参数 | 标准 |
| $M$ | 因果掩码矩阵 | 本版 |
| $d_k$ | Key/Query 维度 | 标准 |
| $\epsilon = 1/\sqrt{d_k}$ | Attention 的自然小参数 | 本版 |

---

## 3. Linear Attention 的精确 Volterra 分解

Linear Attention 是 softmax attention 的简化版本，将 softmax 替换为恒等映射（或简单的特征映射 $\phi$）。它的关键优势是**关于输入的精确多项式结构**，类比 v1 中两层网络的精确分析。

### 3.1 Linear Attention 的定义

**定义 3.1**（Linear Attention）

$$\text{LinAttn}(\mathbf{Z})_t = \frac{\sum_{s \leq t} \phi(\mathbf{z}_t W_Q)^\top \phi(\mathbf{z}_s W_K) \cdot \mathbf{z}_s W_V}{\sum_{s \leq t} \phi(\mathbf{z}_t W_Q)^\top \phi(\mathbf{z}_s W_K)}$$

当 $\phi = \text{id}$（恒等映射）且忽略归一化分母时，简化为：

$$\text{LinAttn}(\mathbf{Z})_t = \sum_{s \leq t} (\mathbf{z}_t^\top W_{QK}\, \mathbf{z}_s) \cdot \mathbf{z}_s W_V W_O$$

其中 $W_{QK} = W_Q W_K^\top \in \mathbb{R}^{d \times d}$。

> **关键观察**：此式关于 $\mathbf{z}_t$ 是**线性**的（$n=1$ 对 query token），关于 $\mathbf{z}_s$ 是**二次**的（$n=2$ 对 key-value token），整体是关于输入序列的**三次多项式**（分别涉及 $\mathbf{z}_t$ 一次、$\mathbf{z}_s$ 两次）。且它涉及**两个不同位置** $t, s$，因此交互阶 $r = 1$。

### 3.2 无归一化 Linear Attention 的 Volterra 分解

我们先分析最简形式（无归一化分母），再在 3.4 中处理归一化。

**定理 3.2**（无归一化 Linear Attention 的精确 Volterra 分解）

设 $\mathbf{Z} = \text{LN}(\mathbf{X} W_E)$，无归一化的 Linear Attention 输出第 $t$ 个位置为：

$$\text{LinAttn}(\mathbf{Z})_t = \sum_{s \leq t} \underbrace{(\mathbf{z}_t^\top W_{QK}\, \mathbf{z}_s)}_{\text{attention score}} \cdot \underbrace{\mathbf{z}_s W_V W_O}_{\text{value}}$$

其双重 Volterra 分解为：

$$\text{LinAttn}(\mathbf{Z})_t = \mathbf{h}_t^{(1,1)}$$

**仅有一个非零分量**，具有：
- 特征阶 $n = 1$（对 $\mathbf{z}_t$）+ $n = 2$（对 $\mathbf{z}_s$），平均特征阶 $\bar{n} = 3/(1+1) = 1.5$
- 交互阶 $r = 1$（涉及 token $t$ 和 token $s$）

精确形式：

$$\mathbf{h}_t^{(1,1)} = \sum_{s=1}^{t} \mathcal{K}_{\text{lin}}^{(t,s)}\!\left[\mathbf{z}_t, \mathbf{z}_s \otimes \mathbf{z}_s\right]$$

其中序列 Volterra 核为：

$$\mathcal{K}_{\text{lin}}^{(t,s)}[\mathbf{u}, \mathbf{v} \otimes \mathbf{w}] = (\mathbf{u}^\top W_{QK}\, \mathbf{v}) \cdot \mathbf{w}^\top W_V W_O$$

> **与 v1 的类比**：v1 中两层网络 $f(\mathbf{x}) = \mathbf{w}_2^\top \sigma(W_1 \mathbf{x})$ 具有精确的多项式分解 $f = \sum_n f_n$。这里 Linear Attention 同样具有精确的多项式结构——不需要任何近似。这使得后续的梯度分析也是精确的。

### 3.3 Linear Attention 的梯度与有效学习率

**定理 3.3**（Linear Attention 参数的梯度分解）

考虑损失 $\mathcal{L} = \frac{1}{T}\sum_t \|\mathbf{h}_t^* - \mathbf{h}_t\|^2$。Linear Attention 各参数的梯度为：

**(i) 关于 $W_{QK}$ 的梯度**（控制 attention pattern）：

$$\nabla_{W_{QK}} \mathcal{L} = -\frac{2}{T} \sum_{t=1}^{T} \sum_{s \leq t} \mathbf{z}_t\, (\boldsymbol{\varepsilon}_t^\top W_O^\top W_V^\top \mathbf{z}_s) \cdot \mathbf{z}_s^\top$$

其中 $\boldsymbol{\varepsilon}_t = \mathbf{h}_t^* - \mathbf{h}_t$ 为第 $t$ 个位置的残差向量。

**(ii) 关于 $W_V W_O$ 的梯度**（控制 value 变换）：

$$\nabla_{W_V W_O} \mathcal{L} = -\frac{2}{T} \sum_{t=1}^{T} \sum_{s \leq t} (\mathbf{z}_t^\top W_{QK}\, \mathbf{z}_s) \cdot \mathbf{z}_s\, \boldsymbol{\varepsilon}_t^\top$$

**证明**：对 $\mathcal{L}$ 关于 $W_{QK}$ 和 $W_V W_O$ 直接求导。$\square$

**推论 3.4**（Linear Attention 的有效学习率）

在 $\mathbf{x}_t \sim \mathcal{N}(0, I_p)$、Pre-Norm 条件下，$(n,r) = (1,1)$ 分量的有效学习率为：

$$\eta_{\text{eff}}^{\text{LinAttn}}(1, 1) = \eta \cdot \frac{1}{d_k} \cdot \|W_V W_O\|_F^2 \cdot C_T$$

其中 $C_T = \frac{1}{T}\sum_{t=1}^T t$ 为因果掩码引起的**有效上下文长度**因子：

$$C_T = \frac{T+1}{2}$$

> **关键观察**：
> 1. 因子 $1/d_k$：来自 $Q^\top K$ 中的内积——$d_k$ 维随机向量的内积的方差为 $1/d_k$。这使得 attention 的有效学习率随 $d_k$ **衰减**。
> 2. 因子 $C_T$：因果掩码使得每个位置平均只看到 $(T+1)/2$ 个 token。更长的上下文 → 更大的学习率（更多的监督信号）。
> 3. 与 FFN 对比：FFN 的 $\eta_{\text{eff}}(n, 0) = \eta \cdot n^2 \tilde{a}_n^2 \cdot (2n-3)!!$（v5 推论 4.3），**不依赖上下文长度**。Attention 的学习率依赖 $T$，这是 token 间交互的本质特征。

### 3.4 归一化分母的效应

实际的 Linear Attention 包含归一化分母：

$$\text{LinAttn}(\mathbf{Z})_t = \frac{\sum_{s \leq t} (\mathbf{z}_t^\top W_{QK}\, \mathbf{z}_s) \cdot \mathbf{z}_s W_V W_O}{\sum_{s \leq t} \mathbf{z}_t^\top W_{QK}\, \mathbf{z}_s}$$

**命题 3.5**（归一化引起的阶混合）

归一化分母将纯粹的 $(1,1)$ 分量变为**有理函数**。展开为幂级数：

$$\frac{1}{\sum_s a_{ts}} = \frac{1}{\bar{a}_t \cdot t} \cdot \frac{1}{1 + \delta_t / (\bar{a}_t \cdot t)}
= \frac{1}{\bar{a}_t \cdot t} \sum_{k=0}^{\infty} (-1)^k \left(\frac{\delta_t}{\bar{a}_t \cdot t}\right)^k$$

其中 $a_{ts} = \mathbf{z}_t^\top W_{QK} \mathbf{z}_s$，$\bar{a}_t = \frac{1}{t}\sum_{s \leq t} a_{ts}$，$\delta_t = \sum_{s \leq t}(a_{ts} - \bar{a}_t)$。

每增加一阶 $k$，交互阶增加 $k$（因为 $\delta_t$ 涉及额外的 token）。因此归一化分母的 Taylor 展开生成**所有交互阶 $r \geq 1$** 的分量。

**命题 3.6**（归一化分母的影响量级）

设 attention scores 的变异系数为 $\text{CV} = \text{std}(a_{ts}) / \text{mean}(a_{ts})$，则：

- 第 $k$ 阶修正项的量级为 $O(\text{CV}^k)$
- 当 $\text{CV} \ll 1$（attention 近似均匀）时，归一化的影响可忽略
- 当 $\text{CV} \sim 1$（attention 高度集中）时，高阶修正不可忽略

> **实际意义**：训练初期 attention 近似均匀（$\text{CV}$ 小），归一化影响弱；训练收敛后 attention 变得稀疏（$\text{CV}$ 大），归一化的阶混合效应增强。这意味着 **Linear Attention 的训练动力学在早期近似为纯 $(1,1)$ 阶，晚期逐渐混入高交互阶**。

### 3.5 Linear Attention vs FFN 的学习率对比

将 Linear Attention 的有效学习率与 FFN（v5 推论 4.3）放在一起比较：

| 分量 | 来源 | 有效学习率 | 依赖 $T$？ |
|------|------|-----------|-----------|
| $(n, 0)$ | FFN | $\eta \cdot n^2 \tilde{a}_n^2 \cdot (2n-3)!!$ | 否 |
| $(1, 1)$ | LinAttn | $\eta \cdot \frac{1}{d_k} \cdot \|W_V W_O\|_F^2 \cdot \frac{T+1}{2}$ | **是** |

**推论 3.7**（Linear Attention 与 FFN 的竞争条件）

Attention 的 $(1,1)$ 分量比 FFN 的 $(n,0)$ 分量学得更快，当且仅当：

$$\frac{T+1}{2} \cdot \frac{\|W_V W_O\|_F^2}{d_k} > n^2 \tilde{a}_n^2 \cdot (2n-3)!!$$

对 $n = 2$（FFN 最快学习的低阶分量），ReLU 下 $\tilde{a}_2^2 \cdot 1!! \approx 0.24$，故条件为：

$$T > \frac{2 \times 4 \times 0.24 \times d_k}{\|W_V W_O\|_F^2} - 1 \approx \frac{1.9\, d_k}{\|W_V W_O\|_F^2}$$

在 Kaiming 初始化下 $\|W_V W_O\|_F^2 \approx d$，$d_k = d/H$，故：

$$T > \frac{1.9}{H} \approx 1 \quad \text{（多头时 $H \geq 2$）}$$

> **结论**：**在几乎所有实际情况下（$T \geq 2$），Linear Attention 学习 token 间交互的速度超过 FFN 学习二阶特征的速度**。这解释了为什么注意力机制是序列建模中不可或缺的——它为 token 间交互提供了一条比 FFN "更快"的梯度通道。

---

## 4. Softmax Attention 的 Taylor 展开与双重 Volterra 分解

### 4.1 预备：Softmax 的有理函数结构

Softmax attention 的第 $t$ 个位置输出为：

$$\text{Attn}(\mathbf{Z})_t = \sum_{s=1}^{t} \frac{\exp\!\left(\frac{\mathbf{z}_t^\top W_{QK}\, \mathbf{z}_s}{\sqrt{d_k}}\right)}{\sum_{j=1}^{t} \exp\!\left(\frac{\mathbf{z}_t^\top W_{QK}\, \mathbf{z}_j}{\sqrt{d_k}}\right)} \mathbf{z}_s W_V W_O$$

定义 attention logit $\ell_{ts} = \mathbf{z}_t^\top W_{QK}\, \mathbf{z}_s / \sqrt{d_k}$ 和 attention weight $\alpha_{ts} = \text{softmax}(\boldsymbol{\ell}_t)_s$。

**两个非线性来源**：

1. **指数函数** $\exp(\ell_{ts})$：将双线性 logit 变为超越函数
2. **归一化分母** $\sum_j \exp(\ell_{tj})$：引入所有 key 位置之间的**竞争**

两者结合使 softmax attention 成为关于 $\mathbf{Z}$ 的**有理超越函数**——既不是多项式也不是有理函数，需要 Taylor 展开。

### 4.2 指数函数的 Taylor 展开

**引理 4.1**（Attention logit 的指数展开）

$$\exp(\ell_{ts}) = \exp\!\left(\frac{\mathbf{z}_t^\top W_{QK}\, \mathbf{z}_s}{\sqrt{d_k}}\right) = \sum_{r=0}^{\infty} \frac{1}{r!} \left(\frac{\mathbf{z}_t^\top W_{QK}\, \mathbf{z}_s}{\sqrt{d_k}}\right)^r$$

第 $r$ 阶项为：

$$\exp^{(r)}(\ell_{ts}) = \frac{1}{r!\, d_k^{r/2}} \left(\mathbf{z}_t^\top W_{QK}\, \mathbf{z}_s\right)^r$$

**Volterra 分析**：

- $r = 0$：常数 $1$——均匀注意力，交互阶 $= 0$（不依赖 key content）
- $r = 1$：$\mathbf{z}_t^\top W_{QK} \mathbf{z}_s / \sqrt{d_k}$——双线性，交互阶 $= 1$，即 Linear Attention
- $r = 2$：$(\mathbf{z}_t^\top W_{QK} \mathbf{z}_s)^2 / (2 d_k)$——四次多项式，交互阶 $= 2$
- 一般 $r$：$2r$ 次多项式（$\mathbf{z}_t$ 出现 $r$ 次，$\mathbf{z}_s$ 出现 $r$ 次），交互阶 $= r$

**自然小参数**：$\epsilon = 1/\sqrt{d_k}$。第 $r$ 阶项的系数包含 $\epsilon^r$，因此：

$$\exp(\ell_{ts}) = 1 + \epsilon \cdot (\mathbf{z}_t^\top W_{QK}\, \mathbf{z}_s) + \frac{\epsilon^2}{2} (\mathbf{z}_t^\top W_{QK}\, \mathbf{z}_s)^2 + O(\epsilon^3)$$

> **关键洞察**：$d_k$ 不仅是一个计算上的 scaling（防止梯度过大），更是一个**自然的展开参数**。$d_k$ 越大，高交互阶的贡献越弱——softmax attention 越接近 Linear Attention。

### 4.3 Softmax 的完整 Taylor 展开

将 $\alpha_{ts}$ 展开：

$$\alpha_{ts} = \frac{e^{\ell_{ts}}}{\sum_{j} e^{\ell_{tj}}} = \frac{\sum_r \frac{\ell_{ts}^r}{r!}}{\sum_j \sum_r \frac{\ell_{tj}^r}{r!}}$$

**定理 4.2**（Softmax Attention 的 Taylor 展开——按交互阶排列）

定义 $\epsilon = 1/\sqrt{d_k}$，$g_{ts} = \mathbf{z}_t^\top W_{QK}\, \mathbf{z}_s$（无缩放的 logit）。Softmax attention weight 展开为：

$$\alpha_{ts} = \alpha_{ts}^{(0)} + \epsilon \cdot \alpha_{ts}^{(1)} + \epsilon^2 \cdot \alpha_{ts}^{(2)} + O(\epsilon^3)$$

其中：

**零阶**（$r = 0$，均匀注意力）：

$$\alpha_{ts}^{(0)} = \frac{1}{t}$$

**一阶**（$r = 1$，线性偏差）：

$$\alpha_{ts}^{(1)} = \frac{1}{t}\left(g_{ts} - \frac{1}{t}\sum_{j=1}^{t} g_{tj}\right) = \frac{1}{t}\left(g_{ts} - \bar{g}_t\right)$$

**二阶**（$r = 2$，二次修正）：

$$\alpha_{ts}^{(2)} = \frac{1}{t}\left[\frac{g_{ts}^2}{2} - \frac{1}{t}\sum_j \frac{g_{tj}^2}{2} - g_{ts}\left(\bar{g}_t - \frac{1}{t}\sum_j g_{tj}\right) + \left(\bar{g}_t\right)^2 - \frac{1}{t}\sum_j g_{tj}^2 / 2\right]$$

简化后：

$$\alpha_{ts}^{(2)} = \frac{1}{2t}\left[(g_{ts} - \bar{g}_t)^2 - \frac{1}{t}\sum_j (g_{tj} - \bar{g}_t)^2\right]$$

**证明**：设 $\ell_{ts} = \epsilon g_{ts}$。分子 $e^{\epsilon g_{ts}} = 1 + \epsilon g_{ts} + \epsilon^2 g_{ts}^2/2 + \cdots$，分母 $\sum_j e^{\epsilon g_{tj}} = t + \epsilon \sum_j g_{tj} + \epsilon^2 \sum_j g_{tj}^2/2 + \cdots$。

商的 Taylor 展开 $\frac{a_0 + \epsilon a_1 + \cdots}{b_0 + \epsilon b_1 + \cdots} = \frac{a_0}{b_0} + \epsilon \frac{a_1 b_0 - a_0 b_1}{b_0^2} + \cdots$，代入 $a_0 = 1$, $b_0 = t$, $a_1 = g_{ts}$, $b_1 = \sum_j g_{tj}$ 即得。$\square$

### 4.4 双重 Volterra 分解

将 Taylor 展开代入 attention 输出 $\text{Attn}(\mathbf{Z})_t = \sum_s \alpha_{ts} \mathbf{z}_s W_V W_O$：

**定理 4.3**（Softmax Attention 的双重 Volterra 分解）

$$\text{Attn}(\mathbf{Z})_t = \sum_{r=0}^{\infty} \epsilon^r\, \mathbf{A}_t^{(r)}$$

其中各交互阶分量为：

**$r = 0$（均匀聚合）**：

$$\mathbf{A}_t^{(0)} = \frac{1}{t} \sum_{s=1}^{t} \mathbf{z}_s W_V W_O$$

- 特征阶：$n = 1$（对 $\mathbf{z}_s$ 线性）
- 交互阶：$r = 0$（不依赖 query-key 相似度，只做均值池化）
- 对应 CKHT 中的"完全非选择性"核——所有 token 权重相等

**$r = 1$（线性选择性）**：

$$\mathbf{A}_t^{(1)} = \frac{1}{t} \sum_{s=1}^{t} (g_{ts} - \bar{g}_t) \cdot \mathbf{z}_s W_V W_O$$

- 特征阶：$n = 1$（对 $\mathbf{z}_s$）+ $n = 1$（对 $\mathbf{z}_t$ 通过 $g_{ts}$）+ $n = 1$（对 $\mathbf{z}_s$ 通过 $g_{ts}$），总特征阶 $n = 3$
- 交互阶：$r = 1$（$\mathbf{z}_t$ 和 $\mathbf{z}_s$ 的交叉）
- **这就是 Linear Attention 的核心贡献**——减去 $\bar{g}_t$ 的竞争效应

**$r = 2$（二次修正）**：

$$\mathbf{A}_t^{(2)} = \frac{1}{t} \sum_{s=1}^{t} \alpha_{ts}^{(2)} \cdot \mathbf{z}_s W_V W_O$$

- 总特征阶 $n = 5$（$g_{ts}^2$ 贡献 4 次 + value 的 1 次）
- 交互阶：$r = 2$（$g_{ts}^2$ 中 $\mathbf{z}_t$ 出现 2 次与 $\mathbf{z}_s$ 的 2 次交叉）

**一般阶**：第 $r$ 阶分量：
- 总特征阶：$n = 2r + 1$（logit 的 $r$ 次幂贡献 $2r$ 次 + value 的 1 次）
- 交互阶：$r$
- 幅度：$O(\epsilon^r) = O(d_k^{-r/2})$

**推论 4.4**（交互阶的谱衰减）

$$\frac{\|\mathbf{A}_t^{(r)}\|}{\|\mathbf{A}_t^{(0)}\|} = O\!\left(\frac{\text{Var}(g_{ts})^{r/2}}{d_k^{r/2}}\right) = O\!\left(\frac{\|W_{QK}\|_F^{2r}}{d_k^{r/2}}\right)$$

在 Kaiming 初始化下 $\|W_{QK}\|_F^2 \approx d_k$，故比值约为 $O(d_k^{r/2} / d_k^{r/2}) = O(1)$——不衰减！

> **修正**：上述估计在 Pre-Norm 条件下需要更仔细的分析。由于 $\text{LN}$ 将 $\|\mathbf{z}\| = \sqrt{d}$，实际上 $g_{ts} = \mathbf{z}_t^\top W_{QK} \mathbf{z}_s / \sqrt{d_k}$ 的方差为 $\text{Var}(g_{ts}) = \|W_{QK}\|_F^2 / d_k \approx 1$（因为 Kaiming 初始化使 $W_{QK}$ 的 Frobenius 范数 $\approx \sqrt{d_k}$）。因此：

$$\frac{\|\mathbf{A}_t^{(r)}\|}{\|\mathbf{A}_t^{(0)}\|} \approx \frac{1}{r!}$$

即**交互阶的衰减由 $1/r!$（指数 Taylor 系数）控制**。$r = 3$ 以上的交互阶在初始化附近通常可忽略。

### 4.5 各交互阶的有效学习率

**定理 4.5**（Softmax Attention 各交互阶的有效学习率）

在 Pre-Norm 条件下，Softmax Attention 的第 $(n_r, r)$ 阶分量（其中 $n_r = 2r+1$）的有效学习率为：

$$\eta_{\text{eff}}^{\text{softmax}}(n_r, r) = \eta \cdot \frac{1}{(r!)^2} \cdot \frac{1}{d_k^r} \cdot \|W_V W_O\|_F^2 \cdot C_T^{(r)}$$

其中 $C_T^{(r)}$ 为第 $r$ 阶的因果注意力因子：

$$C_T^{(r)} = \frac{1}{T} \sum_{t=1}^{T} \frac{1}{t^{2r-1}} \sum_{\substack{s_1, \ldots, s_r \leq t \\ \text{distinct}}} 1 \approx \frac{1}{T} \sum_t \binom{t}{r} / t^{2r-1}$$

在 $T \gg r$ 时，$C_T^{(r)} \approx T / (2(r+1))$。

**推论 4.6**（各交互阶学习率的衰减）

$$\frac{\eta_{\text{eff}}(n_r, r)}{\eta_{\text{eff}}(n_1, 1)} = \frac{1}{(r!)^2} \cdot \frac{1}{d_k^{r-1}} \cdot \frac{C_T^{(r)}}{C_T^{(1)}}$$

以 $d_k = 64$（GPT-2 级别）为例：

| 交互阶 $r$ | $(r!)^{-2}$ | $d_k^{-(r-1)}$ | 总衰减因子 | 相对学习率 |
|------------|-------------|----------------|-----------|-----------|
| 0 | 1 | $d_k$ | $d_k$ | 特殊（均匀聚合） |
| 1 | 1 | 1 | 1 | **基准** |
| 2 | 1/4 | 1/64 | $\sim 4 \times 10^{-3}$ | $0.4\%$ |
| 3 | 1/36 | $1/64^2$ | $\sim 7 \times 10^{-6}$ | $7 \times 10^{-4}\%$ |

> **核心结论**：**Softmax Attention 的训练动力学在实际维度下被 $r = 0$（均匀聚合）和 $r = 1$（线性选择性）主导**。$r \geq 2$ 的高交互阶虽然存在于表达力中（CKHT 保证 Transformer 能表达任意核），但其学习速率以 $1/((r!)^2 d_k^{r-1})$ 超指数衰减——在标准训练中几乎学不到。

> **这解释了一个长期困扰的现象**：为什么 Linear Attention 在许多任务上与 Softmax Attention 性能接近——因为 Softmax 的高阶交互（$r \geq 2$）虽然理论上可学，但在实际训练中的有效学习率太低，基本没有被利用。

### 4.6 与 v5 对角化理论的对接

v5 定理 4.1 给出了 FFN 各阶误差能量的耦合 ODE：

$$\dot{E}_n = -2\alpha_n E_n + \sum_{m \neq n} \beta_{nm} \sqrt{E_n E_m}$$

现在对 Attention 的各交互阶有类似结构：

$$\dot{E}_{n_r, r} = -2\alpha_{n_r, r}\, E_{n_r, r} + \sum_{r' \neq r} \beta_{rr'}^{\text{attn}} \sqrt{E_{n_r, r}\, E_{n_{r'}, r'}}$$

其中 $\alpha_{n_r, r} = \eta_{\text{eff}}^{\text{softmax}}(n_r, r)$，交互阶间的耦合系数 $\beta_{rr'}^{\text{attn}}$ 来自 softmax 归一化分母的阶混合效应（命题 3.5 的推广）。

**命题 4.7**（交互阶间耦合的量级）

$$\beta_{rr'}^{\text{attn}} = O\!\left(\frac{1}{d_k^{|r-r'|/2}}\right)$$

即**相邻交互阶之间的耦合最强**（$|r - r'| = 1$），远距离交互阶之间的耦合以 $d_k$ 的幂次衰减。

> **类比 v5**：v5 中特征阶之间的耦合为 $O(1/\sqrt{d})$（$d$ = 网络宽度）。这里交互阶之间的耦合为 $O(1/\sqrt{d_k})$（$d_k$ = key 维度）。**$d_k$ 在 Attention 中扮演的角色类似 $d$ 在 FFN 中的角色——都是控制各阶解耦程度的"宽度"参数**。

---

## 5. 完整 Transformer Block 的耦合梯度动力学

前两节分别分析了 Attention 和 FFN 的 Volterra 分解。本节将它们组合为完整的 Transformer block，建立 $(n, r)$ 双重索引下的耦合演化方程。

### 5.1 Transformer Block 的组合结构

Pre-Norm Transformer block 的完整结构（简化为单层）：

$$\mathbf{H} = \mathbf{X} + \underbrace{\text{Attn}(\text{LN}_1(\mathbf{X}))}_{\text{贡献交互阶 } r \geq 0} + \underbrace{\text{FFN}(\text{LN}_2(\mathbf{X} + \text{Attn}))}_{\text{贡献特征阶 } n \geq 1}$$

**三条信号路径**:

| 路径 | 表达式 | 特征阶 $n$ | 交互阶 $r$ |
|------|--------|-----------|-----------|
| **恒等路径** | $\mathbf{x}_t$ | 1 | 0 |
| **Attention 路径** | $\text{Attn}(\text{LN}_1(\mathbf{x}))_t$ | $2r+1$ | $r = 0, 1, 2, \ldots$ |
| **FFN 路径** | $\text{FFN}(\text{LN}_2(\mathbf{x}_t + \text{Attn}_t))$ | $n \geq 1$ | 继承自输入 |

关键新现象：**FFN 的输入包含 Attention 的输出**。因此 FFN 路径不仅产生纯特征阶（$r = 0$），还会产生**混合阶**——通过 $\sigma(\cdots \text{Attn}_t \cdots)$ 中的非线性，Attention 引入的交互阶会被 FFN 的特征阶**提升**。

### 5.2 FFN-Attention 交叉项的 Volterra 分析

**定理 5.1**（FFN 对 Attention 输出的非线性提升）

设 FFN 的输入为 $\mathbf{y}_t = \text{LN}_2(\mathbf{x}_t + \text{Attn}_t)$。FFN 的输出分解为：

$$\text{FFN}(\mathbf{y}_t) = W_2\, \sigma(W_1\, \mathbf{y}_t) = \sum_{n=1}^{N} a_n\, W_2\, (W_1\, \mathbf{y}_t)^{\circ n}$$

其中 $(\ )^{\circ n}$ 为逐元素 $n$ 次幂。将 $\mathbf{y}_t = \mathbf{y}_t^{(0)} + \mathbf{y}_t^{(\text{attn})}$（其中 $\mathbf{y}_t^{(0)}$ 为来自 $\mathbf{x}_t$ 的分量，$\mathbf{y}_t^{(\text{attn})}$ 为来自 Attention 的分量）代入，二项式展开：

$$(W_1 \mathbf{y}_t)^{\circ n} = \sum_{k=0}^{n} \binom{n}{k} (W_1 \mathbf{y}_t^{(0)})^{\circ (n-k)} \circ (W_1 \mathbf{y}_t^{(\text{attn})})^{\circ k}$$

第 $k$ 项来自 $k$ 份 Attention 输出和 $(n-k)$ 份原始输入的交叉：

- 特征阶：$n$（来自 $\sigma$ 的 $n$ 次幂）
- 交互阶：$k \cdot r_{\text{attn}}$（$k$ 份 Attention 输出，每份携带交互阶 $r_{\text{attn}}$）

**推论 5.2**（FFN-Attention 交叉的可学条件）

FFN 中 $k$ 份 Attention 输出的交叉项的有效学习率为：

$$\eta_{\text{eff}}^{\text{cross}}(n, k, r) \sim \eta \cdot n^2 \tilde{a}_n^2 \cdot (2n-3)!! \cdot \binom{n}{k}^2 \cdot \left(\frac{1}{d_k}\right)^{kr} \cdot \|W_V W_O\|_F^{2k}$$

比纯 FFN 的 $\eta_{\text{eff}}(n, 0)$ 多了因子 $\binom{n}{k}^2 / d_k^{kr} \cdot \|W_V W_O\|_F^{2k}$。

在 Kaiming 初始化下 $\|W_V W_O\|_F^2 \approx d$，对 $k \geq 1$、$r \geq 1$：

$$\frac{\eta_{\text{eff}}^{\text{cross}}}{\eta_{\text{eff}}^{\text{FFN}}} \sim \binom{n}{k}^2 \cdot \left(\frac{d}{d_k}\right)^k \cdot d_k^{-kr}$$

以 $d = 768$, $d_k = 64$, $n = 2$, $k = 1$, $r = 1$ 为例：

$$\text{ratio} = 4 \times 12 \times (1/64) \approx 0.75$$

> **这说明 FFN 的二阶非线性对 Attention 输出的一阶交叉项的学习速率与纯 FFN 路径相当**——交叉项不可忽略。

### 5.3 完整的双重耦合 ODE

**主定理 5.3**（Transformer Block 的双重耦合演化方程）

在 Pre-Norm 单层 Transformer 中，第 $(n, r)$ 阶误差能量 $E_{n,r}(t)$ 满足：

$$\boxed{\dot{E}_{n,r}(t) = -2\alpha_{n,r}\, E_{n,r}(t) + \sum_{(n',r') \neq (n,r)} \beta_{(n,r),(n',r')}(t)\, \sqrt{E_{n,r}\, E_{n',r'}} + \gamma_{n,r}(t)}$$

其中自衰减率 $\alpha_{n,r}$ 按分量来源分解为：

$$\alpha_{n,r} = \begin{cases}
\alpha_n^{\text{FFN}} & \text{若 } r = 0 \text{（纯 FFN 分量）} \\
\alpha_r^{\text{Attn}} & \text{若 } n = 2r+1 \text{（纯 Attention 分量）} \\
\alpha_{n,k,r}^{\text{cross}} & \text{（FFN-Attention 交叉分量）}
\end{cases}$$

耦合系数 $\beta_{(n,r),(n',r')}$ 有**四种来源**：

**(i) 特征阶耦合**（$r = r'$，$n \neq n'$）：与 v5 定理 4.1 相同

$$\beta_{(n,r),(n',r)} = O\!\left(\frac{1}{\sqrt{d}}\right)$$

**(ii) 交互阶耦合**（$n = n'$，$r \neq r'$）：来自 softmax 归一化

$$\beta_{(n,r),(n,r')} = O\!\left(\frac{1}{\sqrt{d_k}}\right)$$

**(iii) Attention-FFN 交叉耦合**（$r = 0 \leftrightarrow r \geq 1$）：来自 FFN 作用于 Attention 输出

$$\beta_{(n,0),(n',r')} = O\!\left(\frac{\|W_V W_O\|_F}{\sqrt{d \cdot d_k^{r'}}}\right)$$

**(iv) 多层路径耦合**（$L$ 层网络）：来自不同层路径的干涉

$$\beta^{\text{path}} = O\!\left(\frac{1}{2^L \sqrt{d}}\right) \quad \text{（沿用 v4 路径分析）}$$

### 5.4 解耦条件与有效宽度

**推论 5.4**（Transformer 的双重解耦定理）

在 Pre-Norm 条件下，当 $d, d_k \to \infty$ 且 $d_k / d = 1/H$（$H$ 为头数）保持固定时，各 $(n, r)$ 阶**近似独立演化**：

$$E_{n,r}(t) \approx E_{n,r}(0)\, e^{-2\alpha_{n,r} t}$$

**有效宽度**：控制解耦程度的参数为 $\min(d, d_k)$——即 FFN 宽度和 key 维度中较小的那个。

在实际模型中（$d = 768$, $d_k = 64$）：

$$\text{FFN 各阶耦合} = O(1/\sqrt{768}) \approx 0.036$$
$$\text{Attention 各阶耦合} = O(1/\sqrt{64}) \approx 0.125$$

> **实际含义**：**Attention 的交互阶之间的耦合（$\sim 12.5\%$）显著强于 FFN 的特征阶之间的耦合（$\sim 3.6\%$）**。这是因为 $d_k = d/H \ll d$。多头注意力虽然提高了表达力，但通过减小每个头的 $d_k$，也**加强了各交互阶之间的耦合**。

### 5.5 多层 Transformer 的路径分析

**命题 5.5**（$L$ 层 Transformer 的路径计数与阶结构）

$L$ 层 Transformer 中，从输入到输出的路径可表示为长度 $\leq L$ 的序列 $\pi = (c_1, c_2, \ldots, c_L)$，其中每个 $c_l \in \{\text{id}, \text{attn}, \text{ffn}\}$。

| 路径类型 | 路径数 | 最大特征阶 | 最大交互阶 |
|---------|-------|-----------|-----------|
| 纯恒等 | 1 | 1 | 0 |
| 纯 Attention | $L$ | $2L+1$ | $L$ |
| 纯 FFN | $L$ | $N^L$ | 0 |
| 混合 | $3^L - 2L - 1$ | $N^{L_{\text{ffn}}} \cdot (2L_{\text{attn}}+1)$ | $L_{\text{attn}}$ |
| 总计 | $3^L$ | — | — |

对比 v4 的 $2^L$ 条路径（$\{\text{id}, \text{ffn}\}$），Transformer 多了 Attention 节点，路径数从 $2^L$ 增加到 $3^L$。

**推论 5.6**（多层 Transformer 的交互阶上限）

$L$ 层 Transformer 能表达的最大交互阶为 $r_{\max} = L$（每层贡献一阶交互）。但由推论 4.6 的衰减估计，实际可学的交互阶上限为：

$$r^*(L) = \min\!\left(L, \; \left\lfloor \frac{\log(1/\delta)}{2\log d_k - 2\log r!} \right\rfloor\right)$$

以 $d_k = 64$, $L = 12$（GPT-2）为例，$r^* \approx 2$-$3$——**即使网络有 12 层，实际可学的交互阶也只有 2-3**。

> **深层含义**：这解释了为什么增加 Transformer 层数的边际收益递减——高层主要用于精炼已学到的 $r \leq 2$ 的交互模式，而非学习新的高阶 token 交互。

---

## 6. 因果掩码的 Volterra 效应

### 6.1 因果约束对核结构的限制

因果掩码 $M_{ts} = \mathbf{1}[t \geq s]$ 要求 token $t$ 的输出只依赖 $\{x_1, \ldots, x_t\}$。这在 Volterra 核上施加了结构约束。

**定义 6.1**（因果 Volterra 核）

序列 Volterra 核 $\mathcal{K}_{n,r}^{(t, s_1, \ldots, s_r)}$ 称为**因果的**，若仅当 $s_1, \ldots, s_r \leq t$ 时非零。

**命题 6.2**（因果约束减少核参数量）

无因果约束时，第 $r$ 阶交互的核参数量为 $T^{r+1} \cdot p^{n(r+1)} \cdot d$（所有 $(r+1)$-元组 $(t, s_1, \ldots, s_r)$ 都有自由核）。

因果约束将位置索引限制为 $s_i \leq t$，有效参数量减为：

$$\sum_{t=1}^T \binom{t}{r} \cdot p^{n(r+1)} \cdot d = \frac{1}{(r+1)!} \binom{T+1}{r+1} \cdot p^{n(r+1)} \cdot d$$

约为无约束时的 $\frac{1}{(r+1)!}$ 倍。

> **含义**：因果约束不仅限制了可表达的函数类，还**减少了需要学习的参数量**。高交互阶 $r$ 的参数量减少更多——这进一步支持了推论 4.6 的结论：高交互阶在因果设定下更难利用。

### 6.2 位置依赖性与平移不变核

**定义 6.3**（平移不变核与位置依赖核）

序列 Volterra 核 $\mathcal{K}_{n,r}^{(t, s)}$ 称为**平移不变的**，若 $\mathcal{K}_{n,r}^{(t, s)}$ 仅依赖 $t - s$（相对位置）。否则称为**位置依赖的**。

**命题 6.4**（因果掩码引入的位置不对称性）

在因果 Softmax Attention 中，位置 $t$ 的 attention 权重归一化因子为 $\sum_{s=1}^t \exp(\ell_{ts})$——求和上限 $t$ 取决于当前位置。因此：

(i) **均匀聚合项** $\mathbf{A}_t^{(0)} = \frac{1}{t}\sum_{s \leq t} \mathbf{z}_s W_V W_O$——权重 $1/t$ 是**位置依赖的**（前面的 token 获得更高权重，因为 $t$ 更小）

(ii) **线性选择性项** $\mathbf{A}_t^{(1)}$ 中的偏差减去项 $\bar{g}_t = \frac{1}{t}\sum_{j \leq t} g_{tj}$ 也与 $t$ 有关

(iii) 因此，**因果 Transformer 的 Volterra 核天然是位置依赖的**，即使参数 $W_Q, W_K, W_V$ 不含位置信息

> **与位置编码的关系**：RoPE、ALiBi 等位置编码方案为 $g_{ts}$ 添加了显式的位置偏差项 $b(t-s)$。在 Volterra 框架中，这等价于在 $r = 1$ 的核上添加了一个**先验的位置衰减模式**，使模型不必从数据中学习最基本的位置结构。

### 6.3 因果约束对各阶学习率的修正

**定理 6.5**（因果掩码对有效学习率的位置依赖修正）

在因果 Transformer 中，位置 $t$ 上第 $(n, r)$ 阶的有效学习率为：

$$\alpha_{n,r}(t) = \alpha_{n,r}^{\text{full}} \cdot \left(\frac{t}{T}\right)^r \cdot \left(1 + O(r/t)\right)$$

其中 $\alpha_{n,r}^{\text{full}}$ 为无因果掩码时（双向 attention）的自衰减率。

**证明要点**：因果掩码使位置 $t$ 只能注意 $t$ 个 token（而非全部 $T$ 个）。交互阶 $r$ 需要从 $t$ 个可用 token 中选择 $r$ 个产生交互，其组合数为 $\binom{t}{r} \approx (t/r)^r / r!$，与全注意力的 $\binom{T}{r}$ 之比约为 $(t/T)^r$。

**推论 6.6**（因果约束的不均匀效应）

对序列中不同位置的影响极不均匀：

| 位置 | $t/T$ | $r=1$ 修正 | $r=2$ 修正 | $r=3$ 修正 |
|------|-------|-----------|-----------|-----------|
| 开头 ($t=1$) | $1/T$ | $1/T$ | $1/T^2$ | $1/T^3$ |
| 中间 ($t=T/2$) | $1/2$ | $1/2$ | $1/4$ | $1/8$ |
| 末尾 ($t=T$) | $1$ | 1 | 1 | 1 |

> **核心发现**：**序列开头位置的高交互阶学习率被因果掩码极度压缩**。位置 $t=1$（第一个 token）完全没有交互对象，$r \geq 1$ 的所有交互阶学习率为零——它只能依赖 FFN 的 $(n, 0)$ 分量。

> **这解释了为什么 LLM 的第一个 token 位置（通常是 BOS/CLS）表现特殊**——它在 Volterra 分解中只有 $r = 0$ 的信息，是序列中 Volterra 谱最贫乏的位置。

### 6.4 因果约束与 CKHT 的因果核公理

CKHT 从公理化角度定义了因果核：

$$K(t, s) = 0 \quad \text{when } t < s$$

并证明了因果核的核心分类定理：在正定性约束下，**完全单调因果核**（对应 SSM 类模型）是因果核空间的一个严格子集。

v6 的梯度动力学分析将 CKHT 的**静态分类**动态化：

**命题 6.7**（因果核的梯度可达性）

在有限训练步数 $T_{\text{train}}$ 内，梯度下降从随机初始化出发能实际学到的因果核子集为：

$$\mathcal{K}_{\text{reachable}} = \left\{\mathcal{K}_{n,r} : \alpha_{n,r} \cdot T_{\text{train}} \gg 1,\; r \leq r^*\right\}$$

其中 $r^* \approx 2$-$3$（推论 5.6）。

CKHT 告诉我们 Transformer **能表达**完全任意的因果核（包括高度振荡、长程依赖等），但 v6 的动力学分析告诉我们：**实际只有低交互阶（$r \leq 2$-$3$）、低特征阶（$n \leq n^*$）的因果核子空间是梯度可达的**。

---

## 7. NCGT 与 CKHT 的统一框架

### 7.1 两个理论的互补性

**CKHT**（因果核层级理论）和 **NCGT**（神经耦合梯度理论）从两个正交的方向分析序列模型：

| 维度 | CKHT | NCGT (v1–v6) |
|------|------|-------------|
| 核心问题 | **能表达什么？** | **能学到什么？** |
| 分析对象 | 模型的函数空间 | 训练的梯度动力学 |
| 核心工具 | 泛函分析、Hausdorff 矩定理 | Volterra 分解、耦合 ODE |
| 关键结论 | 核约束层级 | 各阶有效学习率 |
| 回答 | 可能性边界 | 可行性边界 |

**类比**：CKHT 是"地图"（告诉你哪里有路），NCGT 是"导航"（告诉你哪条路走得通）。一个模型可能理论上能表达任意复杂的因果核（CKHT 保证），但在有限训练下只能学到其中简单的子集（NCGT 限制）。

### 7.2 CKHT 三轴在 NCGT 中的对应

CKHT 建立了三维坐标系来分类序列模型的表达力：

**轴 1：核约束**（完全单调核 ⊊ 衰减振荡核 ⊊ 任意正因果核）

在 NCGT v6 中，这对应**交互阶的可达性**：

- 完全单调核 → 仅需 $r = 0$（无选择性的均匀衰减聚合）+ 简单的 $r = 1$（线性选择性）
- 衰减振荡核 → 需要 $r = 1$ 的相位选择性（attention 的线性项可以实现正负交替）
- 任意正因果核 → 需要 $r \geq 2$（高阶交互捕捉复杂的非单调依赖模式）

由推论 4.6，$r \geq 2$ 的学习率以 $d_k^{-(r-1)}$ 衰减。因此：

> **NCGT 预测**：在标准训练条件下，Transformer 实际学到的核**以完全单调核和简单衰减振荡核为主**，即使其架构能表达任意因果核。这与 CKHT 的层级是一致的——**越简单的核类型越容易学习**。

**轴 2：状态介质**（向量状态 → 矩阵状态 → 显式历史）

- SSM（向量状态 $\mathbf{h} \in \mathbb{R}^d$）：v6 中对应 Linear Attention 的 KV 缓存形式 $S_t = \sum_{s \leq t} \mathbf{k}_s \mathbf{v}_s^\top \in \mathbb{R}^{d_k \times d_v}$——实质上是矩阵状态
- Full Attention（显式历史 $\{\mathbf{x}_1, \ldots, \mathbf{x}_t\}$）：允许任意交互阶 $r$

NCGT v6 的动力学分析揭示了一个新视角：**状态介质的"容量"决定了可学交互阶的上限**：

$$r_{\max}^{\text{learnable}}(\text{vector state}) = 1, \quad r_{\max}^{\text{learnable}}(\text{matrix state}) = 2, \quad r_{\max}^{\text{learnable}}(\text{full history}) \approx 2\text{-}3$$

矩阵状态（如 Mamba-2 的 SSM）虽然理论上限制了表达力，但在实际训练中**可学交互阶的差距远小于表达力的差距**。

**轴 3：选择性**（固定 LTI → 输入依赖门控 → 显式内容路由）

在 NCGT v6 中，选择性对应 $r = 1$ 项的结构：

- LTI（$g_{ts}$ 仅依赖 $t-s$）：$\alpha_{ts}^{(1)} = \frac{1}{t}\hat{g}(t-s)$，平移不变核
- 输入依赖选择性（$g_{ts}$ 依赖内容）：$\alpha_{ts}^{(1)} = \frac{1}{t}(\mathbf{z}_t^\top W_{QK} \mathbf{z}_s - \bar{g}_t)$，内容选择核
- 显式路由：需要 $r = 2$ 以上的交互实现 token 间的间接路由

NCGT v6 给出了选择性的**学习速率**：输入依赖的选择性学习率与 LTI 相同（都是 $r = 1$），但具有更多的核参数自由度（$d_k^2$ vs $T$）。

### 7.3 统一图景：表达力 × 可学性 = 实际能力

**定义 7.1**（实际能力空间）

模型 $\mathcal{M}$ 在训练时间 $T_{\text{train}}$ 和数据量 $m$ 下的**实际能力空间**为：

$$\mathcal{C}_{\text{actual}}(\mathcal{M}, T_{\text{train}}, m) = \underbrace{\mathcal{F}_{\text{express}}(\mathcal{M})}_{\text{CKHT: 可表达}} \cap \underbrace{\mathcal{F}_{\text{learn}}(T_{\text{train}}, m)}_{\text{NCGT: 可学习}}$$

**定理 7.2**（实际能力的双重瓶颈——序列模型版本）

$$\mathcal{C}_{\text{actual}} = \left\{(n, r) : n \leq n^*_{\text{express}} \cap n \leq n^*_{\text{learn}},\; r \leq r^*_{\text{express}} \cap r \leq r^*_{\text{learn}}\right\}$$

其中：
- $n^*_{\text{express}}$, $r^*_{\text{express}}$ 由 CKHT 的架构分析确定（如 SSM 的 $r^*_{\text{express}} = 1$）
- $n^*_{\text{learn}}$, $r^*_{\text{learn}}$ 由 NCGT v6 的动力学分析确定（如 $r^*_{\text{learn}} \approx 2$-$3$）
- 实际瓶颈取两者的 $\min$

**三种典型场景**：

| 模型 | 表达瓶颈 | 学习瓶颈 | 实际瓶颈 |
|------|---------|---------|---------|
| Linear RNN (SSM) | $r^*_{\text{express}} = 1$ | $r^*_{\text{learn}} \approx 2$ | 表达力 |
| Transformer ($d_k$ 小) | $r^*_{\text{express}} = \infty$ | $r^*_{\text{learn}} \approx 2$ | 可学性 |
| Transformer ($d_k$ 大) | $r^*_{\text{express}} = \infty$ | $r^*_{\text{learn}} \approx 3$ | 可学性（但更宽松） |

> **核心洞察**：对 SSM 类模型，瓶颈在表达力（CKHT 管辖）；对 Transformer，瓶颈在可学性（NCGT 管辖）。**两种理论各自主导了不同架构类型的性能分析**。

### 7.4 预测：SSM 与 Transformer 的性能差距

结合两个理论，可以精确刻画 SSM（如 Mamba）与 Transformer 的性能差距：

$$\Delta\text{Loss} = \sum_{r > r^*_{\text{SSM}}} E_{n,r}^*$$

即目标函数中**需要 $r > 1$ 的因果核分量的能量**决定了 SSM 相对于 Transformer 的性能损失。

**推论 7.3**（SSM 的充分条件）

SSM 与 Transformer 性能相当，当且仅当目标任务的 Volterra 分解中，$r \geq 2$ 的分量能量**可忽略**。

这与经验观察一致：在语言建模等需要复杂 token 交互的任务上，Transformer 优于 SSM；在主要依赖局部模式的任务（如审核、分类）上，两者表现相似。

---

## 8. 与前五版的统一对比

| 项目 | v1–v2 | v3 | v4 | v5 | **v6** |
|------|-------|-----|-----|-----|--------|
| 网络类型 | 两层 MLP | $L$ 层 plain | ResNet + Norm | 同 v4 | **Transformer** |
| Volterra 维度 | $n$ | $n$ | $n$ | $n$ | **$(n, r)$** |
| 时间尺度 | 单步 | 单步 | 单步 | 多步 | **多步** |
| Attention | — | — | — | — | **Linear + Softmax** |
| 因果约束 | 无 | 无 | 无 | 无 | **因果掩码** |
| 与 CKHT | 无 | 无 | 无 | 无 | **统一框架** |

---

## 9. 讨论与开放问题

### 9.1 本版核心贡献总结

1. **双重 Volterra 分解**：建立了 $(n, r)$——特征阶 $\times$ 交互阶——的分解框架，将 v1–v5 的单维 Volterra 理论推广到序列模型

2. **Linear Attention 的精确分析**：作为基准情形，Linear Attention 的 Volterra 结构是精确可分析的（类比 v1），为 softmax 分析提供校准

3. **Softmax 的自然展开参数**：$\epsilon = 1/\sqrt{d_k}$ 是 softmax attention 的 Taylor 展开参数。各交互阶以 $1/((r!)^2 d_k^{r-1})$ 超指数衰减

4. **FFN-Attention 交叉耦合**：FFN 对 Attention 输出的非线性提升产生混合阶分量，且在标准参数化下不可忽略

5. **因果掩码的位置不均匀效应**：序列开头位置的高交互阶学习率被极度压缩，解释了 BOS token 的特殊性

6. **NCGT × CKHT 统一框架**：表达力（CKHT）∩ 可学性（NCGT）= 实际能力。SSM 受限于表达，Transformer 受限于可学性

### 9.2 开放问题

**问题 1**（多头注意力的 Volterra 结构）

本版分析了单头注意力。$H$ 个头的多头注意力为 $\text{MHA}(\mathbf{Z}) = \sum_{h=1}^H \text{head}_h(\mathbf{Z}) W_O^{(h)}$。不同头的 Volterra 分解如何**叠加**？是否存在头间的"分工"——某些头专注低交互阶、某些头专注高交互阶？

**问题 2**（位置编码的 Volterra 效应）

RoPE 在 logit 中引入 $g_{ts} = \text{Re}[(\mathbf{z}_t \circ e^{i\theta t})^\top W_{QK} (\mathbf{z}_s \circ e^{i\theta s})]$。这个旋转操作如何改变各交互阶的系数？RoPE 是否在 Volterra 分解的层面上具有可解释的效应？

**问题 3**（Flash Attention 等计算等价变换）

Flash Attention 不改变数学计算，只改变内存访问模式。但某些近似方案（如局部窗口 attention、稀疏 attention）**截断了交互范围**。在 Volterra 框架中，这等价于将高交互阶的核投影到低秩子空间。这种近似的信息损失是否可以用 $E_{n,r}$ 的截断误差精确刻画？

**问题 4**（MoE 的 Volterra 分解）

Mixture-of-Experts 在 FFN 层引入了**条件计算**——不同 token 使用不同的专家。在 Volterra 框架中，这意味着 $(n, 0)$ 分量的核对不同 token 可以不同。这是否等价于引入了一种新的"专家交互阶"？

**问题 5**（在实际 LLM 上的实验验证）

v5 的理论在二层网络上有完整的实验验证。v6 的核心预测——交互阶的 $1/((r!)^2 d_k^{r-1})$ 衰减——是否可以在训练中的 attention pattern 演化中观测到？一个可能的实验方案：追踪训练过程中各 attention head 的"有效交互阶"（通过 attention weight 矩阵的低秩分解来估计）。

**问题 6**（KV cache 压缩的 Volterra 界）

推理时的 KV cache 压缩（如 token 丢弃、量化）在 Volterra 框架中等价于截断或近似序列 Volterra 核。低交互阶核（$r \leq 1$）可以被高效压缩（因为它们是低秩的），而高交互阶核需要完整历史。由于实际可学阶 $r^* \leq 3$，这为 KV cache 压缩的理论上限提供了依据。

---

## 10. 总览图

```
v1: ∇L = Σ_n (各阶核误差贡献)          ← 单步，两层，多项式
     |
v2: 推广到任意激活 (Hermite→单项式)      ← 单步，两层，任意 σ
     |
v3: 推广到 L 层 + η_eff(n) + SGD 正则   ← 单步，L 层 plain
     |
v4: + ResNet 路径 + Pre-Norm + 泛化界    ← 单步，现代架构
     |
v5: E_n(t) = E_n(0) exp(-2α_n t)        ← 多步动力学
    + 耦合 ODE + 相变时间 + 双重瓶颈      ← 训练全过程
     |
v6: (n,r) 双重 Volterra 分解             ← Transformer
    + Softmax Taylor (ε = 1/√d_k)        ← 注意力机制
    + 因果掩码效应 + CKHT 统一            ← 序列模型全景
```

核心等式链：

$$\underbrace{\nabla_W \mathcal{L} = \sum_n (\cdots)_n}_{\text{v1: 梯度分解}} \xrightarrow{\text{v3}} \underbrace{\eta_{\text{eff}}(n)}_{\text{各阶学习率}} \xrightarrow{\text{v5}} \underbrace{\dot{E}_n = -2\alpha_n E_n}_{\text{各阶动力学}} \xrightarrow{\text{v6}} \underbrace{\dot{E}_{n,r} = -2\alpha_{n,r} E_{n,r}}_{\text{双重动力学}}$$

新增等式链（Attention 特有）：

$$\underbrace{\exp(\ell/\sqrt{d_k}) = \sum_r \frac{\ell^r}{r! d_k^{r/2}}}_{\text{Softmax Taylor}} \xrightarrow{} \underbrace{\alpha_{n_r,r} \propto \frac{1}{(r!)^2 d_k^{r-1}}}_{\text{交互阶衰减}} \xrightarrow{} \underbrace{r^* \leq 3}_{\text{可学交互阶上限}}$$

> **一句话总结**：v6 将 NCGT 从 FFN 推广到 Transformer，建立了特征阶 $n$ × 交互阶 $r$ 的双重 Volterra 分解。Softmax attention 的交互阶以 $\epsilon = 1/\sqrt{d_k}$ 为小参数做 Taylor 展开，各交互阶学习率以 $1/((r!)^2 d_k^{r-1})$ 超指数衰减，使得标准训练下可学交互阶仅为 $r^* \approx 2$-$3$。结合 CKHT 的表达力分析，建立了"实际能力 = 表达力 ∩ 可学性"的统一图景：SSM 受限于表达力，Transformer 受限于可学性。
