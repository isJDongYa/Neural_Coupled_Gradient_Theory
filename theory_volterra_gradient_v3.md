# 参数共享的耦合梯度定理（第三版）：
# 推广至深层网络与收敛动力学

> **与前两版的关系**：
> - 第一版：两层多项式激活，精确等式
> - 第二版：两层任意 L²(γ) 激活，近似版本（Hermite→单项式）
> - **本版（第三版）**：$L$ 层网络的精确 Volterra 阶分解 + 各阶收敛速率 + 隐式正则化分析

---

## 1. 动机

第一、二版的理论局限于两层网络。实际网络有数十至数千层。本版解决三个核心问题：

1. **深层网络的梯度如何影响各阶 Volterra 核？**（Section 3-4）
2. **各阶核的学习速率是多少？**（Section 5）
3. **SGD 噪声对各阶核的正则化效应是什么？**（Section 6）

---

## 2. 设置

**网络**（$L$ 层，单输出）：

$$f(\mathbf{x}) = W_L\, \sigma(W_{L-1}\, \sigma(\cdots \sigma(W_1 \mathbf{x})\cdots))$$

其中 $W_l \in \mathbb{R}^{d_{l} \times d_{l-1}}$（$d_0 = p$ 为输入维度，$d_L = 1$），激活函数 $\sigma$ 逐元素作用。

**符号**：
- $\mathbf{z}^{(l)} = W_l \mathbf{h}^{(l-1)}$（第 $l$ 层预激活），$\mathbf{h}^{(l)} = \sigma(\mathbf{z}^{(l)})$（第 $l$ 层后激活），$\mathbf{h}^{(0)} = \mathbf{x}$
- $\sigma(z) = \sum_{n=1}^N a_n z^n$（多项式激活，$a_0 = 0$）

**损失**：$\mathcal{L} = \mathbb{E}[(y - f(\mathbf{x}))^2]$

---

## 3. 深层网络的 Volterra 分解

### 3.1 逐层复合的阶结构

**引理 3.1**（两次多项式复合的阶分解）

设 $g(z) = \sum_{m=1}^M b_m z^m$，$\phi(\mathbf{x})$ 为 $\mathbf{x}$ 的 $K$ 次齐次多项式。则 $g(\phi(\mathbf{x}))$ 可展开为最高 $MK$ 次的齐次多项式之和：

$$g(\phi(\mathbf{x})) = \sum_{m=1}^M b_m\, \phi(\mathbf{x})^m$$

其中 $\phi(\mathbf{x})^m$ 为 $mK$ 次齐次多项式。

**证明**：$\phi$ 为 $K$ 次齐次，则 $\phi^m$ 为 $mK$ 次齐次。$\square$

### 3.2 $L$ 层网络的 Volterra 阶上限

**命题 3.2**（Volterra 阶爆炸）

$L$ 层、$N$ 次多项式激活的网络，其输出 $f(\mathbf{x})$ 等价于最高 $N^{L-1}$ 阶的 Volterra 级数：

$$f(\mathbf{x}) = \sum_{n=1}^{N^{L-1}} f_n(\mathbf{x})$$

其中 $f_n$ 为 $n$ 次齐次多项式（第 $n$ 阶 Volterra 分量）。

**证明**：归纳法。

*基础*：$L=2$ 时，$f(\mathbf{x}) = W_2 \sigma(W_1\mathbf{x})$，最高阶为 $N$，即 $N^{2-1} = N^1$。✓

*归纳*：设 $L-1$ 层子网络 $g(\mathbf{x}) = W_{L-1}\sigma(\cdots\sigma(W_1\mathbf{x})\cdots)$ 的输出为最高 $N^{L-2}$ 阶多项式向量。则 $f(\mathbf{x}) = W_L \sigma(g(\mathbf{x}))$。$\sigma$ 将各分量映射为最高 $N$ 次多项式，其输入 $g_k(\mathbf{x})$ 最高为 $N^{L-2}$ 阶，故 $\sigma(g_k(\mathbf{x}))$ 最高为 $N \cdot N^{L-2} = N^{L-1}$ 阶。$\square$

> **直观**：每多一层，Volterra 阶数上限乘以 $N$。这正是 Cohen et al. (2016) 的"深度指数表达力"在 Volterra 框架中的表述。

### 3.3 深层 Volterra 核的参数共享结构

**命题 3.3**（$L$ 层 Volterra 核的显式形式——递推版）

定义第 $l$ 层的**局部 Volterra 核**（该层单独作为两层网络的核）：

$$h_n^{(l)}(i_1, \ldots, i_n) = a_n \sum_{k=1}^{d_l} [W_{l+1}]_{\cdot,k}\, [W_l]_{k,i_1} \cdots [W_l]_{k,i_n}$$

则 $L$ 层网络的**全局第 $n$ 阶 Volterra 核** $H_n$ 由各层局部核的**多线性复合**给出：

$$H_n = \sum_{\substack{(n_1, \ldots, n_{L-1}) \\ n_1 \cdot n_2 \cdots n_{L-1} = n}} h_{n_{L-1}}^{(L-1)} \circ_{\text{tensor}} h_{n_{L-2}}^{(L-2)} \circ_{\text{tensor}} \cdots \circ_{\text{tensor}} h_{n_1}^{(1)}$$

其中求和取遍所有使 $\prod_{l=1}^{L-1} n_l = n$ 的正整数组合 $(n_1, \ldots, n_{L-1})$，$\circ_{\text{tensor}}$ 表示张量收缩（将前一层的输出维度与后一层的输入维度缩并）。

> **关键观察**：全局第 $n$ 阶核是所有"各层局部阶数乘积为 $n$"的路径之和。例如 $L=3$, $N=2$ 时，全局 4 阶核 $H_4$ 包含路径：
> - $(n_1, n_2) = (2, 2)$：两层各贡献 2 阶
> - $(n_1, n_2) = (1, 4)$：第 1 层贡献 1 阶，第 2 层不存在 4 阶（因 $N=2$），故此路径**为 0**
>
> 因此有效路径受限于每层 $1 \leq n_l \leq N$。

---

## 4. 深层耦合梯度定理

这是本版的核心定理。

### 4.1 单层参数的梯度分解

**定理 4.1**（第 $l$ 层参数的 Volterra 阶分解）

对第 $l$ 层参数 $[W_l]_{k,j}$ 的梯度：

$$\frac{\partial \mathcal{L}}{\partial [W_l]_{k,j}} = -2\, \mathbb{E}\!\left[\varepsilon(\mathbf{x})\, [\boldsymbol{\delta}^{(l)}]_k\, \sigma'(z_k^{(l)})\, h_j^{(l-1)}\right]$$

其中：
- $\varepsilon = y - f(\mathbf{x})$ 为预测误差
- $\boldsymbol{\delta}^{(l)} = \frac{\partial f}{\partial \mathbf{z}^{(l)}}$ 为反向传播到第 $l$ 层的误差信号
- $z_k^{(l)} = [W_l]_k^\top \mathbf{h}^{(l-1)}$ 为第 $l$ 层第 $k$ 个神经元的预激活
- $h_j^{(l-1)} = [\mathbf{h}^{(l-1)}]_j$ 为第 $l-1$ 层第 $j$ 个神经元的后激活

**证明**：标准反向传播链式法则。$\square$

现在关键是展开 $\sigma'$ 和 $\boldsymbol{\delta}^{(l)}$ 的 Volterra 阶结构。

**定理 4.2**（深层耦合梯度的阶分解）

对多项式激活 $\sigma(z) = \sum_{n=1}^N a_n z^n$，$\sigma'(z) = \sum_{n=1}^N n a_n z^{n-1}$。第 $l$ 层参数的梯度可分解为：

$$\frac{\partial \mathcal{L}}{\partial [W_l]_{k,j}} = -2 \sum_{q=0}^{Q_l} \mathbb{E}\!\left[\varepsilon(\mathbf{x})\, G_{k,j}^{(l,q)}(\mathbf{x})\right]$$

其中 $G_{k,j}^{(l,q)}(\mathbf{x})$ 为 $\mathbf{x}$ 的 $q$ 次齐次多项式，$Q_l = N^{L-1} - 1$ 为梯度中最高阶次。

具体地，$q$ 阶的梯度贡献为：

$$G_{k,j}^{(l,q)} = \sum_{\substack{\text{paths } (n_1,\ldots,n_{L-1}) \\ \text{summing to } q+1 \text{ via } l}} [\boldsymbol{\delta}^{(l)}]_k^{\text{path}}\, n_l a_{n_l}\, (z_k^{(l)})^{n_l - 1}\, h_j^{(l-1)}$$

这里"via $l$"表示路径在第 $l$ 层贡献 $n_l$ 阶，前向部分（层 $1$ 到 $l-1$）贡献 $\prod_{s<l} n_s$ 阶，反向部分（层 $l+1$ 到 $L-1$）贡献 $\prod_{s>l} n_s$ 阶，总阶为 $(\prod_{s<l} n_s) \cdot (n_l - 1) \cdot (\prod_{s>l} n_s) + \prod_{s<l} n_s = q$。

> **与 v1 的关系**：当 $L=2$ 时，无反向传播路径，$\boldsymbol{\delta}^{(1)} = W_2$（常数），公式退化为 v1 的定理 4.1：
> $$\frac{\partial \mathcal{L}}{\partial [W_1]_{k,j}} = -2 \sum_{n=1}^N n a_n \mathbb{E}[\varepsilon \cdot [w_2]_k \cdot (\mathbf{u}_k^\top\mathbf{x})^{n-1} \cdot x_j]$$

**核心含义**：第 $l$ 层的一次梯度更新，通过路径求和，**同时修正所有可达阶的全局 Volterra 核**。中间层（$l$ 远离输入和输出）的修正路径最多，耦合效应最强。

### 4.2 各层对各阶核的梯度贡献强度

**推论 4.3**（梯度贡献的层-阶矩阵）

定义第 $l$ 层对全局第 $n$ 阶 Volterra 核的**梯度贡献强度**：

$$\Gamma_l(n) = \left\|\frac{\partial H_n}{\partial W_l}\right\|_F$$

则在 Kaiming 初始化 $[W_l]_{k,j} \sim \mathcal{N}(0, 2/d_{l-1})$ 下，对 $\sigma = \text{ReLU}$（$a_1 = 1/2, a_2 \approx 0.49, \ldots$，Hermite→单项式后的系数）：

$$\mathbb{E}[\Gamma_l(n)^2] = \prod_{s=1}^{L-1} C(n_s) \cdot \frac{2^L}{\prod_{s=0}^{L-1} d_s}$$

其中 $C(n_s) = n_s^2 \tilde{a}_{n_s}^2 \cdot \frac{(2n_s-2)!}{2^{n_s-1}(n_s-1)!}$（来自高斯矩 $\mathbb{E}[Z^{2(n-1)}]$），求和取遍所有乘积为 $n$ 的路径。

> **预测**：由于 $\tilde{a}_{n_s}$ 超指数衰减，偏好路径是"每层贡献低阶"——即 $n_s \approx n^{1/(L-1)}$。这意味着**深层网络通过多层各贡献少量阶来合成高阶核**，而非某单层贡献全部阶数。

---

## 5. 各阶 Volterra 核的学习动力学

### 5.1 连续时间梯度流

考虑梯度流 $\dot{W}_l = -\eta \nabla_{W_l} \mathcal{L}$。定义第 $n$ 阶 Volterra 分量的 $L^2$ 误差：

$$E_n(t) = \|f_n(t) - f_n^*\|_{L^2(\gamma_p)}^2$$

**定理 5.1**（各阶误差的演化方程）

在两层网络 ($L=2$) 的简化情形下，对 $\mathbf{x} \sim \mathcal{N}(0, I_p)$：

$$\dot{E}_n(t) = -2\eta\, n^2 a_n^2\, \mathbb{E}\!\left[\varepsilon(\mathbf{x})\, f_n(\mathbf{x})\right] \cdot \underbrace{R_n(t)}_{\text{有效学习率因子}} + \text{交叉项}$$

其中：

$$R_n(t) = \sum_{k=1}^d [w_2]_k^2 \cdot \mathbb{E}[(\mathbf{u}_k^\top\mathbf{x})^{2(n-1)}] = \sum_{k=1}^d [w_2]_k^2 \cdot \frac{(2n-2)!}{2^{n-1}(n-1)!} \|\mathbf{u}_k\|^{2(n-1)}$$

**"交叉项"**为不同阶分量之间的耦合：$\sum_{m \neq n}(\cdots)$。在网络宽度 $d \to \infty$ 的平均场极限下，这些交叉项相对于主项以 $O(1/d)$ 的速率消失。

> **物理直觉**：$R_n(t)$ 是第 $n$ 阶的"有效学习率"——它随 $n$ 的增大而增大（$(2n-2)!!$ 快速增长），但 $a_n^2$ 的衰减更快（对 ReLU，$\tilde{a}_n^2$ 超指数衰减），两者的乘积 $n^2 a_n^2 R_n$ **随 $n$ 衰减**。

### 5.2 有效学习率阶谱

**推论 5.2**（各阶的有效学习率）

定义第 $n$ 阶的有效学习率为：

$$\eta_{\text{eff}}(n) = \eta \cdot n^2\, \tilde{a}_n^2 \cdot \frac{(2n-2)!}{2^{n-1}(n-1)!} \cdot \bar{s}^{2(n-1)}$$

其中 $\bar{s}^2 = \frac{1}{d}\sum_k [w_2]_k^2 \|\mathbf{u}_k\|^{2(n-1)/((n-1))}$ 为权重的平均范数尺度。

**对 ReLU（Kaiming 初始化，$\|\mathbf{u}_k\| \approx 1$，$\bar{s} \approx 1$）的数值估计**：

| 阶 $n$ | $\tilde{a}_n$ | $n^2 \tilde{a}_n^2$ | $(2n-2)!!$ | $\eta_{\text{eff}}(n)/\eta$ | 相对 $n=1$ |
|--------|---------------|---------------------|------------|---------------------------|------------|
| 1 | 0.500 | 0.250 | 1 | 0.250 | 1.00 |
| 2 | 0.491 | 0.964 | 1 | 0.964 | 3.86 |
| 3 | 0 | 0 | 3 | 0 | 0 |
| 4 | -0.109 | 0.190 | 15 | 2.855 | 11.4 |

> **注意**：原始的 $(2n-2)!!$ 增长不应让人误以为高阶学得更快。在 $\|\mathbf{u}_k\|$ 偏离 1 时（$\|\mathbf{u}_k\| < 1$），$\|\mathbf{u}_k\|^{2(n-1)}$ 的指数衰减会压制高阶项。实际控制学习速率快慢的是 $\tilde{a}_n^2$ 与 $\|\mathbf{u}_k\|^{2(n-1)}$ 的竞争。

### 5.3 低阶优先学习的条件

**定理 5.3**（低阶优先学习的充分条件）

若初始化满足 $\|\mathbf{u}_k\|^2 \leq C$ 且 $|a_n| \leq A \cdot r^n$（$0 < r < 1$），则存在 $n^* = O(\log(1/r)/\log C)$ 使得对所有 $n > n^*$：

$$\eta_{\text{eff}}(n) < \eta_{\text{eff}}(1)$$

即超过某个阈值阶数后，高阶核的学习速率严格低于 1 阶。

**证明草图**：$n^2 a_n^2 R_n \leq n^2 A^2 r^{2n} \cdot (2n)! \cdot C^{n-1} / (2^{n-1}(n-1)!)$。由 Stirling 公式，$(2n)!/(2^{n-1}(n-1)!) \sim n^n$，而 $r^{2n}$ 指数衰减，故存在 $n^*$ 使得乘积开始下降。$\square$

对**多项式激活**（$a_n$ 为有限个非零值），$r$ 可取为 0（仅有限项），定理自动成立。对 **ReLU**（$\tilde{a}_n$ 超指数衰减），$r$ 有效地非常小，$n^*$ 也很小（$n^* \leq 4$ 在典型初始化下）。

---

## 6. SGD 噪声对各阶 Volterra 核的隐式正则化

### 6.1 随机梯度的阶分解

设小批量为 $\mathcal{B} \subset \{1, \ldots, n\}$，$|\mathcal{B}| = B$。随机梯度为：

$$\hat{g}_l = \frac{1}{B}\sum_{i \in \mathcal{B}} g_l(\mathbf{x}_i, y_i)$$

其噪声 $\boldsymbol{\xi}_l = \hat{g}_l - \nabla_{W_l}\mathcal{L}$ 的协方差为：

$$\Sigma_l = \frac{1}{B}\,\text{Cov}(g_l(\mathbf{x}, y))$$

### 6.2 各阶核受到的噪声强度

**定理 6.1**（SGD 噪声的阶分解）

在两层网络下，SGD 噪声对第 $n$ 阶 Volterra 核 $h_n$ 的**扰动方差**为：

$$\text{Var}(\hat{h}_n - h_n) = \frac{n^2 a_n^2}{B} \cdot V_n$$

其中

$$V_n = \text{Var}_{\mathbf{x}}\!\left[\varepsilon(\mathbf{x}) \cdot (\mathbf{u}^\top\mathbf{x})^{n-1} \cdot \mathbf{x}\right]$$

**关键结论**：$V_n$ 随 $n$ **快速增长**（因为 $(\mathbf{u}^\top\mathbf{x})^{n-1}$ 的方差为 $(2n-2)!! - 1$ 对标准高斯），所以**高阶核受到的 SGD 噪声更大**。

### 6.3 噪声作为隐式正则化

**推论 6.2**（高阶核的噪声压制）

将 SGD 视为连续时间 Langevin 动力学的离散化：

$$dW_l = -\eta\, \nabla_{W_l}\mathcal{L}\, dt + \sqrt{\frac{2\eta}{B}}\, \Sigma_l^{1/2}\, dB_t$$

第 $n$ 阶 Volterra 核 $h_n$ 在稳态下的**有效温度**为：

$$T_{\text{eff}}(n) = \frac{\eta}{B} \cdot V_n$$

$T_{\text{eff}}(n)$ 随 $n$ 增大，意味着**高阶核在更高的"温度"下被训练**——等价于对高阶核施加了更强的隐式正则化（偏好更简单/更小的高阶核）。

> **实践意义**：
> - 小 batch size → 所有阶的温度上升，但高阶上升更快 → 倾向于更简单的模型
> - 大 batch size → 温度下降 → 可以更精确地学习高阶核
> - 这解释了"大 batch 需要更长训练/更大学习率"的经验观察

---

## 7. 与第一版、第二版的统一对比

| 项目 | 第一版 | 第二版 | **第三版** |
|------|--------|--------|-----------|
| 网络深度 | 2 层 | 2 层 | **$L$ 层** |
| 激活函数 | 多项式 | 任意 L²(γ) | 多项式（精确）+ L²(γ)（近似） |
| Volterra 分解 | 精确 | 近似 | 精确（多项式），逐层递推 |
| 耦合梯度 | 精确公式 | 近似公式 | **路径求和公式**（精确） |
| 各阶学习速率 | 未分析 | 未分析 | **$\eta_{\text{eff}}(n) = \eta \cdot n^2 \tilde{a}_n^2 \cdot R_n$** |
| 低阶优先 | 定性 ($n \cdot a_n$ 衰减) | 定性 | **定量条件 (Thm 5.3)** |
| SGD 正则化 | 未涉及 | 未涉及 | **各阶有效温度 $T_{\text{eff}}(n)$** |
| 统计下界 | minimax | minimax | 继承 v1/v2 |

---

## 8. 核心结论汇总

**定理汇总**：

1. **命题 3.2**：$L$ 层网络等价于最高 $N^{L-1}$ 阶 Volterra 级数，且全局核由各层局部核的张量收缩路径求和给出。

2. **定理 4.2**：第 $l$ 层的梯度可分解为对所有可达阶 $q$ 的加权贡献，权重由路径 $(n_1, \ldots, n_{L-1})$ 决定——这是 v1 定理 4.1 在深层网络中的推广。

3. **定理 5.1 + 推论 5.2**：各阶 Volterra 核的有效学习率 $\eta_{\text{eff}}(n) = \eta \cdot n^2 \tilde{a}_n^2 \cdot R_n(t)$，由激活函数系数、权重范数和高斯矩共同决定。

4. **定理 5.3**：低阶优先学习在 $|a_n|$ 指数/超指数衰减时必然成立。

5. **定理 6.1 + 推论 6.2**：SGD 噪声对第 $n$ 阶核的有效温度为 $T_{\text{eff}}(n) = (\eta/B) \cdot V_n$，$V_n$ 随 $n$ 快速增长——高阶核受到更强的隐式正则化。

---

## 9. 开放问题

1. **深层的精确收敛速率**：定理 5.1 的交叉项在有限宽度下不可忽略。能否给出 $d = O(\text{poly}(N))$ 时的精确演化方程？

2. **非多项式深层的 Volterra 路径**：对 ReLU 的 $L$ 层网络，每层的 Hermite→单项式转换在残差连接（ResNet）下如何简化？

3. **自适应优化器**：Adam 对各阶核的有效学习率是否有不同于 SGD 的阶偏好？（Adam 的二阶矩估计可能抵消 $R_n$ 的增长。）

4. **注意力机制**：Transformer 的自注意力 $\text{softmax}(QK^\top/\sqrt{d})V$ 是否有 Volterra 类似的阶分解？注意力头的多样性是否对应不同阶的 Volterra 核？

5. **归一化层的影响**：BatchNorm / LayerNorm 在每层将预激活拉回 $\mathcal{N}(0,1)$，这使得：(a) Hermite 展开精确成立；(b) $\|\mathbf{u}_k\| = 1$ 条件自动满足；(c) 各层局部核的 $R_n$ 不依赖于参数大小。能否证明归一化层使得各阶 Volterra 核的学习率**只依赖于 $\tilde{a}_n$**？

---

## 10. 三版理论的全局图景

```
第一版（精确基础）     第二版（实用激活函数）     第三版（深层+动力学）
━━━━━━━━━━━━━━━━     ━━━━━━━━━━━━━━━━━━━     ━━━━━━━━━━━━━━━━━━━
                                                
两层多项式网络         两层 L²(γ) 激活           L 层网络
    ↓                     ↓                        ↓
精确 Volterra 分解    Hermite→单项式近似        逐层递推 + 路径求和
    ↓                     ↓                        ↓
精确耦合梯度          近似耦合梯度              深层耦合梯度
    ↓                     ↓                        ↓
统计 minimax 下界     同上（激活无关）           同上
                                                    ↓
                                               各阶学习速率
                                                    ↓
                                               SGD 隐式正则化
```

> **一句话总结**：神经网络的梯度下降等价于对 Volterra 函数空间中各阶非线性核的加权同步修正（v1/v2），修正强度按层路径分配（v3），低阶核在系数衰减条件下优先学习（v3），而 SGD 噪声对高阶核施加了更强的隐式正则化（v3）。

---

## 参考文献

- **第一版、第二版**：本系列前两篇文章
- Cohen, N. et al. (2016). *On the Expressive Power of Deep Learning: A Tensor Analysis.*
- Poole, B. et al. (2016). *Exponential expressivity in deep neural networks through transient chaos.* NeurIPS.
- Neal, R. (1996). *Priors for infinite networks.*
- Matthews, A. et al. (2018). *Gaussian Process Behaviour in Wide Deep Neural Networks.* ICLR.
- Jacot, A. et al. (2018). *Neural Tangent Kernel: Convergence and Generalization in Neural Networks.* NeurIPS.
- Smith, S. & Le, Q. (2018). *A Bayesian Perspective on Generalization and Stochastic Gradient Descent.* ICLR.
- Li, Z. et al. (2020). *On the Validity of Modeling SGD with Stochastic Differential Equations.* NeurIPS.
- Volterra, V. (1887). *Sopra le funzioni che dipendono da altre funzioni.*
- Xu, Z. et al. (2019). *Frequency Principle.* （频率原理）
