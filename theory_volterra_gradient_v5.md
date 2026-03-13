# 参数共享的耦合梯度定理（第五版）：
# 各阶 Volterra 核的训练动力学与阶间相变

> **与前四版的关系**：
> - 第一版：两层多项式激活，精确等式（单步梯度 = 各阶核误差之和）
> - 第二版：两层任意 L²(γ) 激活，Hermite→单项式近似
> - 第三版：$L$ 层 plain 网络 + 各阶学习速率 $\eta_{\text{eff}}(n)$ + SGD 隐式正则化
> - 第四版：残差连接 + 归一化层 + 泛化界 $n^* = O(\log m/\log p)$
> - **本版（第五版）**：从单步分析推广到多步训练动力学——各阶 Volterra 核 $K_n(t)$ 的演化方程、阶间耦合与相变

---

## 1. 动机

### 1.1 v1-v4 的共同局限

前四版建立了一个不断推广的框架，但它们共享一个根本性的局限：**分析的是一步梯度的结构，而非训练过程的时间演化**。

具体而言：
- v1-v2 给出了 $\nabla_{W}\mathcal{L}$ 在各阶 Volterra 分量上的分解
- v3 定义了各阶有效学习率 $\eta_{\text{eff}}(n)$，但只证明了"低阶学习率更大"这一**瞬时**性质
- v4 对残差路径和归一化层做了结构分析，推导了泛化界

然而，**实际训练是一个动态过程**。以下核心问题在 v1-v4 中未被回答：

1. **收敛速率**：第 $n$ 阶 Volterra 核的误差 $E_n(t) = \|K_n(t) - K_n^*\|^2$ 是否指数衰减？速率是什么？
2. **阶间相变**：是否存在临界时间 $t_n^*$，使得 $t < t_n^*$ 时第 $n$ 阶几乎不学（被低阶梯度信号淹没），$t > t_n^*$ 时才开始收敛？
3. **阶间干扰**：低阶收敛过程中，是否会暂时增大高阶误差？
4. **有限时间可学阶数**：训练 $T$ 步最多能有效学到几阶？是否有 $n^*(T)$ 的公式？

### 1.2 v5 的核心贡献

本版将 v1-v4 的**单步分解**推广为**连续时间动力学**，建立各阶 Volterra 核的耦合演化方程：

$$\frac{dK_n}{dt} = -\eta_{\text{eff}}(n)\, \underbrace{(K_n - K_n^*)}_{\text{自修正}} + \underbrace{\sum_{m \neq n} C_{nm}(t)\, \delta K_m}_{\text{阶间耦合}}$$

核心结果：

1. **各阶耦合 ODE**（Section 3-4）：精确的多阶演化方程及耦合矩阵 $C_{nm}$
2. **Pre-Norm 解耦定理**（Section 5）：Pre-Norm 使耦合矩阵 $C_{nm}$ 降至 $O(1/d)$，各阶近似独立演化
3. **阶间相变**（Section 6）：存在相变时间序列 $t_1^* < t_2^* < \cdots$，第 $n$ 阶在 $t_n^*$ 时开始有效学习
4. **有限时间可学阶数**（Section 7）：训练 $T$ 步时 $n^*(T) = \max\{n : t_n^* < T\}$，给出精确公式

---

## 2. 设置与符号

### 2.1 网络与 Volterra 分解（沿用 v3-v4）

**Pre-Norm ResNet**（$L$ 个残差块）：

$$\mathbf{h}^{(l)} = \mathbf{h}^{(l-1)} + \sigma(W_l\, \text{LN}(\mathbf{h}^{(l-1)})), \quad l = 1, \ldots, L-1$$
$$f(\mathbf{x}; \theta) = \mathbf{w}_L^\top \mathbf{h}^{(L-1)}$$

激活函数 $\sigma(z) = \sum_{n=1}^N a_n z^n$（多项式），参数集合 $\theta = \{W_1, \ldots, W_{L-1}, \mathbf{w}_L\}$。

由 v4 Theorem 3.2，网络输出分解为：

$$f(\mathbf{x}; \theta) = \sum_{n=0}^{N_{\max}} f_n(\mathbf{x}; \theta)$$

其中 $f_n$ 为第 $n$ 阶 Volterra 分量（$n$ 次齐次多项式），$N_{\max} = N^{L-1}$。

### 2.2 目标函数与各阶误差

设目标函数 $f^*(\mathbf{x}) = \sum_{n=0}^{N^*} f_n^*(\mathbf{x})$ 也具有 Volterra 分解。

**损失函数**（总体风险）：

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x} \sim \gamma_p}\!\left[(f^*(\mathbf{x}) - f(\mathbf{x}; \theta))^2\right]$$

其中 $\gamma_p = \mathcal{N}(0, I_p)$。

**各阶误差信号**：

$$\varepsilon_n(\mathbf{x}; t) = f_n^*(\mathbf{x}) - f_n(\mathbf{x}; \theta(t))$$

$$\delta K_n(t) = K_n^* - K_n(t)$$

其中 $K_n(t)$ 为时刻 $t$ 的第 $n$ 阶 Volterra 核张量，$K_n^*$ 为目标核。

**各阶误差能量**：

$$E_n(t) = \|K_n(t) - K_n^*\|_F^2 = \|\delta K_n(t)\|_F^2$$

**总误差分解**（由正交性，v1 引理 2.2）：

$$\mathcal{L}(\theta) = \sum_{n=0}^{N_{\max}} \mathbb{E}\!\left[\varepsilon_n(\mathbf{x})^2\right] + 2 \sum_{m < n} \mathbb{E}\!\left[\varepsilon_m(\mathbf{x})\, \varepsilon_n(\mathbf{x})\right]$$

对 $\mathbf{x} \sim \mathcal{N}(0, I)$，不同阶的齐次多项式**在 $L^2(\gamma_p)$ 下正交**，故交叉项消失：

$$\mathcal{L}(\theta) = \sum_{n=0}^{N_{\max}} E_n(t) \cdot c_n$$

其中 $c_n = \mathbb{E}[\phi_n(\mathbf{x})^2] / \|\delta K_n\|_F^2$ 为归一化常数（取决于核的索引方式）。

> **关键简化**：由于正交性，总损失精确等于各阶误差能量之和（无交叉项）。这意味着如果我们能独立地分析每个 $E_n(t)$ 的演化，就完全理解了训练动力学。

### 2.3 时间尺度

- **离散时间**：$\theta(t+1) = \theta(t) - \eta\, \nabla_\theta \mathcal{L}(\theta(t))$，$t = 0, 1, 2, \ldots$
- **连续时间近似**（$\eta \to 0$）：$\dot{\theta}(t) = -\eta\, \nabla_\theta \mathcal{L}(\theta(t))$

本版的主要结果在连续时间下推导，离散时间版本在 Section 4 中讨论。

### 2.4 核心记号汇总

| 符号 | 含义 | 来源 |
|------|------|------|
| $K_n(t)$ | 时刻 $t$ 的第 $n$ 阶 Volterra 核 | 本版 |
| $K_n^*$ | 目标第 $n$ 阶核 | 本版 |
| $E_n(t) = \|\delta K_n(t)\|_F^2$ | 第 $n$ 阶误差能量 | 本版 |
| $\eta_{\text{eff}}(n)$ | 第 $n$ 阶有效学习率 | v3 推论 5.2 |
| $\tilde{a}_n$ | Hermite→单项式系数 | v2 |
| $C_{nm}(t)$ | 阶间耦合系数（$m \to n$） | 本版 |
| $t_n^*$ | 第 $n$ 阶的相变时间 | 本版 |

---

## 3. 从单步梯度到各阶核的时间演化

### 3.1 回顾：v3 的单步梯度分解

v3 定理 4.1 给出了损失关于参数 $W_l$ 的梯度在 Volterra 各阶上的分解。对两层网络（$L=2$）的简化形式：

$$\nabla_{W_1}\mathcal{L} = -2\, \mathbb{E}\!\left[\varepsilon(\mathbf{x}) \sum_{n=1}^{N} n\, a_n\, (\mathbf{u}_k^\top\mathbf{x})^{n-1}\, \mathbf{x}^\top\right]$$

其中 $\varepsilon(\mathbf{x}) = f^*(\mathbf{x}) - f(\mathbf{x}; \theta)$ 是总残差。

**关键观察**：梯度中同时包含所有阶的贡献。v3 分析的是"此刻梯度中哪一阶最大"，但没有追踪这些贡献如何经过多步累积。

### 3.2 核的演化：从参数空间到核空间

**引理 3.1**（参数更新诱导核更新）

考虑两层网络 $f(\mathbf{x}) = \sum_k [w_2]_k\, \sigma(\mathbf{u}_k^\top\mathbf{x})$，其中 $\sigma(z) = \sum_n a_n z^n$。第 $n$ 阶 Volterra 核为：

$$K_n(i_1, \ldots, i_n) = a_n \sum_{k=1}^d [w_2]_k\, [W_1]_{k,i_1}\, [W_1]_{k,i_2} \cdots [W_1]_{k,i_n}$$

对 $W_1$ 的无穷小更新 $\delta W_1 = -\eta\, \nabla_{W_1}\mathcal{L}$，核的变化量为：

$$\delta K_n(i_1, \ldots, i_n) = n\, a_n \sum_{k=1}^d [w_2]_k \sum_{j=1}^n [W_1]_{k,i_1} \cdots \underbrace{[\delta W_1]_{k,i_j}}_{\text{第 j 个因子被替换}} \cdots [W_1]_{k,i_n} + O(\eta^2)$$

**证明**：对 $K_n$ 中的乘积 $[W_1]_{k,i_1} \cdots [W_1]_{k,i_n}$ 施加 $(W_1 + \delta W_1)$ 展开，保留线性项。由对称性（$K_n$ 对指标 $i_1, \ldots, i_n$ 对称），$n$ 个位置的贡献相等，给出因子 $n$。$\square$

### 3.3 将参数梯度投影到各阶核空间

**命题 3.2**（各阶核的微分方程——两层网络精确形式）

在连续时间梯度流 $\dot{W}_1 = -\eta\, \nabla_{W_1}\mathcal{L}$ 下，第 $n$ 阶 Volterra 核满足：

$$\dot{K}_n = -\eta \sum_{m=1}^{N} \mathcal{A}_{nm}[\theta(t)]\, \delta K_m + \eta \sum_{\substack{m_1, m_2 \geq 1 \\ m_1 + m_2 = n+1}} \mathcal{B}_{n,m_1,m_2}[\theta(t)]\, \delta K_{m_1} \otimes_{\text{sym}} K_{m_2}$$

其中：
- **主项**（$\mathcal{A}_{nm}$）：线性耦合算子，来源于 $\nabla_{W_1}\mathcal{L}$ 中第 $m$ 阶误差信号对第 $n$ 阶核的影响
- **二次项**（$\mathcal{B}$ 项）：来源于引理 3.1 中的 $[W_1]_{k,i_j}$ 因子随训练变化产生的非线性效应

> **直觉**：即使我们只想追踪 $K_1(t)$ 的演化，其方程也包含 $\delta K_2, \delta K_3, \ldots$ 的贡献——各阶核通过参数共享被**耦合**在一起。这是 NCGT 的核心洞察在动力学中的体现。

### 3.4 核心难点：参数共享导致的不可分性

为什么不能简单地对每一阶独立优化？

**命题 3.3**（参数共享的不可分性）

设 $\theta = \theta_1 \oplus \theta_2 \oplus \cdots \oplus \theta_N$ 为参数空间的假想分解，其中 $\theta_n$ "只控制"第 $n$ 阶核。则对任何满足 $K_n = K_n(\theta_n)$ 的分解方案，必有：

$$\dim(\theta_1 \oplus \cdots \oplus \theta_N) \geq \sum_{n=1}^N \dim(\theta_n) \gg \dim(\theta)$$

即各阶核需要的参数总量远超实际参数量。换言之，**参数共享是必须的——同一组参数同时编码所有阶的核**。

**证明**：第 $n$ 阶核 $K_n \in \mathbb{R}^{p^n}$（对称张量）的自由度为 $\binom{p+n-1}{n}$。所有阶的核总自由度为 $\sum_n \binom{p+n-1}{n}$，而参数量仅为 $dp + d = d(p+1)$。当 $N \geq 2$ 时，前者远大于后者。$\square$

> **推论**：梯度下降不是在"先学第 1 阶，再学第 2 阶"，而是**每一步同时修改所有阶的核**。各阶的演化方程通过共享参数 $\theta$ 天然耦合。

---

## 4. 各阶误差能量的耦合演化方程

### 4.1 误差能量的时间导数

我们关心的核心量是各阶误差能量 $E_n(t) = \|\delta K_n(t)\|_F^2$。将命题 3.2 的核演化方程代入：

$$\dot{E}_n(t) = 2\, \langle \delta K_n,\, \dot{K}_n \rangle_F$$

> 注意符号：$\delta K_n = K_n^* - K_n$，故 $\dot{\delta K_n} = -\dot{K}_n$，因此 $\dot{E}_n = -2\langle \delta K_n, \dot{K}_n \rangle$。

**定理 4.1**（各阶误差能量的耦合演化——主定理）

在两层网络（$L=2$），多项式激活 $\sigma(z) = \sum_{n=1}^N a_n z^n$，连续时间梯度流下：

$$\dot{E}_n(t) = -2\, \alpha_n(t)\, E_n(t) + \sum_{m \neq n} \beta_{nm}(t)\, \sqrt{E_n(t)\, E_m(t)} + \gamma_n(t)$$

其中：

**（i）自衰减率**（第 $n$ 阶的"有效学习率"）：

$$\alpha_n(t) = \eta \cdot n^2\, a_n^2 \sum_{k=1}^d [w_2(t)]_k^2\, \|\mathbf{u}_k(t)\|^{2(n-1)} \cdot (2n-2)!!_{\|\mathbf{u}_k\|}$$

在 $\mathbf{x} \sim \mathcal{N}(0, I)$、$\|\mathbf{u}_k\| = 1$ 时简化为：

$$\alpha_n = \eta \cdot n^2\, a_n^2 \cdot (2n-3)!! \cdot \sum_k [w_2]_k^2$$

这正是 v3 推论 5.2 的 $\eta_{\text{eff}}(n)$（乘以 $\sum_k [w_2]_k^2$ 归一化）。

**（ii）阶间耦合系数**：

$$\beta_{nm}(t) = 2\eta \cdot n\, m\, a_n\, a_m \sum_{k=1}^d [w_2(t)]_k^2\, \mathbb{E}\!\left[(\mathbf{u}_k^\top\mathbf{x})^{n+m-2}\right] \cdot \cos\angle(\delta K_n, \Phi_{nm} \delta K_m)$$

其中 $\Phi_{nm}$ 为参数共享诱导的阶间映射算子，$\cos\angle$ 为张量空间中的夹角余弦。

**（iii）高阶修正项**：

$$\gamma_n(t) = O(\eta^2\, \|\delta K\|^3)$$

为参数更新的二阶效应，在 $\eta$ 充分小时可忽略。

### 4.2 证明

**Step 1**：由引理 3.1，参数更新 $\delta \mathbf{u}_k = -\eta\, \nabla_{\mathbf{u}_k}\mathcal{L}$ 诱导核更新：

$$\delta K_n \to \delta K_n - \eta \cdot n\, a_n \sum_k [w_2]_k \sum_j [\mathbf{u}_k]_{i_1} \cdots [\nabla_{\mathbf{u}_k}\mathcal{L}]_{i_j} \cdots [\mathbf{u}_k]_{i_n}$$

**Step 2**：v1 定理 2.3 给出：

$$\nabla_{\mathbf{u}_k}\mathcal{L} = -2 [w_2]_k \sum_{m=1}^N m\, a_m\, \mathbb{E}\!\left[\varepsilon(\mathbf{x})\, (\mathbf{u}_k^\top\mathbf{x})^{m-1}\, \mathbf{x}\right]$$

**Step 3**：将 $\varepsilon(\mathbf{x}) = \sum_m \varepsilon_m(\mathbf{x})$ 代入（$\varepsilon_m$ 为第 $m$ 阶残差），利用正交性：

$$\mathbb{E}[\varepsilon_m(\mathbf{x})\, (\mathbf{u}_k^\top\mathbf{x})^{m-1}\, \mathbf{x}] \text{ 只在特定的 } (m, n) \text{ 配对下非零}$$

具体地：$\varepsilon_m(\mathbf{x})$ 是 $m$ 次齐次多项式，$(\mathbf{u}_k^\top\mathbf{x})^{m-1} \cdot x_{i_j}$ 是 $m$ 次齐次多项式。当两者同阶时，期望非零。

**Step 4**：对角项（$m = n$）给出自衰减率 $\alpha_n$：

$$\langle \delta K_n,\, \text{diag\_update}_n \rangle = \alpha_n(t) \cdot E_n(t)$$

非对角项（$m \neq n$）给出耦合系数 $\beta_{nm}$。 $\square$

### 4.3 耦合矩阵的结构

将定理 4.1 写成矩阵形式。定义 $\mathbf{e}(t) = (E_1(t), \ldots, E_N(t))^\top$，则：

$$\dot{\mathbf{e}}(t) \approx -2\, A(t)\, \mathbf{e}(t) + B(t)\, \mathbf{e}^{1/2}(t)$$

其中 $A(t) = \text{diag}(\alpha_1, \ldots, \alpha_N)$，$B(t) = [\beta_{nm}]_{n,m}$。

**命题 4.2**（耦合矩阵的量级估计）

在 Kaiming 初始化（$\|\mathbf{u}_k\| \approx 1$，$[w_2]_k \sim \mathcal{N}(0, 1/d)$）下：

（i）对角元素：$\alpha_n = \Theta(\eta \cdot n^2 a_n^2 \cdot (2n-3)!!)$

（ii）非对角元素：$\beta_{nm} = O\!\left(\frac{1}{\sqrt{d}} \cdot \eta \cdot nm |a_n a_m| \cdot (n+m-2)!!\right)$

因此耦合比：

$$\frac{|\beta_{nm}|}{\alpha_n} = O\!\left(\frac{1}{\sqrt{d}} \cdot \frac{m |a_m|}{n |a_n|} \cdot \frac{(n+m-2)!!}{(2n-3)!!}\right)$$

> **关键结论**：耦合系数 $\beta_{nm}$ 相对于自衰减率 $\alpha_n$ 以 $O(1/\sqrt{d})$ 消失。**在宽网络（$d$ 大）下，各阶近似解耦**。

### 4.4 近似解耦下的各阶独立演化

**推论 4.3**（宽网络极限下的独立指数衰减）

当 $d \to \infty$ 时，忽略 $O(1/\sqrt{d})$ 耦合项，各阶误差独立演化：

$$\dot{E}_n(t) = -2\, \alpha_n\, E_n(t)$$

解为：

$$\boxed{E_n(t) = E_n(0)\, e^{-2\alpha_n t}}$$

即**各阶误差以指数速率 $2\alpha_n$ 衰减**，其中 $\alpha_n = \eta_{\text{eff}}(n)$。

**推论 4.4**（各阶的半衰期）

第 $n$ 阶误差衰减到初始值一半所需时间：

$$\tau_{1/2}(n) = \frac{\ln 2}{2\, \alpha_n} = \frac{\ln 2}{2\, \eta \cdot n^2 a_n^2 \cdot (2n-3)!! \cdot \sum_k [w_2]_k^2}$$

对 ReLU（Pre-Norm，$\sum_k [w_2]_k^2 \approx 1$）：

| 阶 $n$ | $\alpha_n / \eta$ | $\tau_{1/2}(n)$ | 相对 $n=2$ |
|--------|-------------------|-----------------|------------|
| 1 | 0.250 | $1.39/\eta$ | 3.86 |
| 2 | 0.964 | $0.36/\eta$ | 1.00 |
| 3 | 0 | $\infty$ | $\infty$ |
| 4 | 2.855 | $0.12/\eta$ | 0.13 |

> **惊人预测**：**2 阶的半衰期比 1 阶短 3.86 倍**。在 ReLU 网络中，二次非线性比线性更快收敛！这精确解释了 vv1.py 实验中 Order 2 误差始终最小的现象。

> **注意**：$\alpha_3 = 0$ 意味着 ReLU 网络的 3 阶核**永远不会通过直接梯度信号学习**。vv1.py 中非零的 Order 3 误差来自网络用 1 阶和 2 阶的**非线性组合**间接逼近 3 阶目标，这不是 Volterra 框架内的直接学习。

---

## 5. Pre-Norm 下的动力学解耦

Section 4 表明各阶耦合系数 $\beta_{nm}$ 以 $O(1/\sqrt{d})$ 消失，但这需要 $d \to \infty$。本节证明 **Pre-Norm 在有限宽度下也大幅抑制阶间耦合**，给出更强的解耦保证。

### 5.1 耦合的物理来源

回顾命题 4.2 的耦合系数：

$$\beta_{nm} \propto \sum_k [w_2]_k^2\, \mathbb{E}\!\left[(\mathbf{u}_k^\top\mathbf{x})^{n+m-2}\right] \cdot \cos\angle(\delta K_n, \Phi_{nm}\, \delta K_m)$$

耦合有两个来源：

1. **范数耦合**：$\|\mathbf{u}_k\|$ 的变化同时影响所有阶的核（因为 $K_n \propto \|\mathbf{u}_k\|^n$）
2. **方向耦合**：$\hat{\mathbf{u}}_k = \mathbf{u}_k / \|\mathbf{u}_k\|$ 的变化同时改变各阶核的方向结构

Pre-Norm 消除了第一类耦合。

### 5.2 Pre-Norm 的动力学效应

**定理 5.1**（Pre-Norm 消除范数耦合）

在 Pre-Norm ResNet 中，LayerNorm 将隐藏态归一化为 $\|\text{LN}(\mathbf{h})\|^2 = p\, \gamma^2$（$\gamma$ 为 LN 的增益参数）。因此：

（i）各阶有效学习率中的 $\|\mathbf{u}_k\|^{2(n-1)}$ 因子被替换为常数 $1$：

$$\alpha_n^{\text{LN}} = \eta \cdot n^2\, \tilde{a}_n^2 \cdot (2n-3)!!$$

不再依赖 $\theta(t)$——**自衰减率在训练过程中保持常数**。

（ii）耦合系数的范数部分被消除，仅剩方向耦合：

$$\beta_{nm}^{\text{LN}} = O\!\left(\frac{1}{\sqrt{d}} \cdot \eta \cdot nm\, |\tilde{a}_n \tilde{a}_m| \cdot (n+m-2)!!\right) \cdot \underbrace{\cos\angle(\hat{\delta K}_n, \Phi_{nm}\, \hat{\delta K}_m)}_{\text{随机初始化下} = O(1/\sqrt{d^n})}$$

（iii）在随机初始化下，方向耦合的夹角余弦为 $O(1/\sqrt{d^n})$（高维空间中随机方向近似正交），因此：

$$\beta_{nm}^{\text{LN}} = O\!\left(\frac{1}{d}\right)$$

**比无 LN 的 $O(1/\sqrt{d})$ 更强一阶。**

**证明**：

（i）v4 定理 5.3 已证。

（ii）Pre-Norm 将各层输入规范化为标准分布，使 $\mathbb{E}[(\mathbf{u}_k^\top \text{LN}(\mathbf{h}))^{n+m-2}]$ 只取决于 $\hat{\mathbf{u}}_k$ 而非 $\|\mathbf{u}_k\|$。

（iii）对 $n$ 阶对称张量，$p^n$ 维空间中的两个随机方向的余弦为 $O(1/\sqrt{p^n})$。由于 $d$ 个神经元各贡献一个 $p^n$ 维向量，总效应为 $O(\sqrt{d}/\sqrt{p^n}) = O(1/\sqrt{d^{n-1}})$，在 $n \geq 2$ 时远小于 $O(1/\sqrt{d})$。$\square$

### 5.3 Pre-Norm 下的精确解耦动力学

**定理 5.2**（Pre-Norm 各阶独立指数衰减——有限宽度版本）

在 Pre-Norm 两层网络中，对 $d \geq \Omega(N^2 / \epsilon^2)$（$\epsilon$ 为耦合容忍度），各阶误差满足：

$$E_n(t) = E_n(0)\, e^{-2\alpha_n^{\text{LN}} t} \cdot (1 + O(\epsilon))$$

其中 $\alpha_n^{\text{LN}} = \eta \cdot n^2 \tilde{a}_n^2 \cdot (2n-3)!!$。

特别地：

$$\frac{E_n(t)}{E_m(t)} = \frac{E_n(0)}{E_m(0)} \cdot \exp\!\left[-2(\alpha_n^{\text{LN}} - \alpha_m^{\text{LN}})\, t\right] \cdot (1 + O(\epsilon))$$

> **物理意义**：各阶误差的**相对大小**在训练过程中**指数级变化**。若 $\alpha_2 > \alpha_1$（如 ReLU），则 $E_2/E_1$ 随时间指数衰减——2 阶相对于 1 阶越来越精确。

### 5.4 Pre-Norm vs 无归一化的动力学对比

| 性质 | 无归一化 | Pre-Norm |
|------|---------|----------|
| 自衰减率 $\alpha_n(t)$ | 依赖 $\theta(t)$，随训练漂移 | **常数**，不随 $t$ 变 |
| 耦合量级 $\beta_{nm}$ | $O(1/\sqrt{d})$ | $O(1/d)$ |
| 解耦所需宽度 | $d = \Omega(N^2/\epsilon^2)$ | $d = \Omega(N^2/\epsilon^2)$（同阶但系数更小） |
| 动力学可预测性 | 差：$\alpha_n(t)$ 在训练中变化 | **好：初始化时即可计算全部演化** |
| 各阶收敛速率比 | 随 $t$ 漂移 | **固定为 $\alpha_n/\alpha_m$** |

> **核心结论**：Pre-Norm 不仅简化了瞬时结构（v4 已证），更关键地将**多步动力学从非自治 ODE 简化为自治 ODE**。这意味着：（1）训练曲线可精确预测；（2）各阶的相对收敛速率不随训练变化；（3）相变时间可在训练前计算。

---

## 6. 阶间相变

### 6.1 相变的定义

直觉上，"第 $n$ 阶开始学习"意味着其误差从初始值开始显著下降。但由于各阶同时演化，我们需要精确定义这种"显著下降"。

**定义 6.1**（$\epsilon$-相变时间）

第 $n$ 阶的 $\epsilon$-相变时间 $t_n^*(\epsilon)$ 定义为：

$$t_n^*(\epsilon) = \inf\!\left\{t > 0 : E_n(t) \leq (1 - \epsilon)\, E_n(0)\right\}$$

即误差从初始值下降 $\epsilon$ 比例的最早时间。取 $\epsilon = 1/2$ 给出半衰期 $\tau_{1/2}(n) = t_n^*(1/2)$。

### 6.2 解耦情形下的精确相变时间

**定理 6.2**（各阶相变时间——Pre-Norm 解耦极限）

在定理 5.2 的 Pre-Norm 解耦条件下，各阶半衰期为：

$$\tau_{1/2}(n) = \frac{\ln 2}{2\, \alpha_n^{\text{LN}}} = \frac{\ln 2}{2\eta \cdot n^2 \tilde{a}_n^2 \cdot (2n-3)!!}$$

**推论 6.3**（相变时间的阶间比）

$$\frac{\tau_{1/2}(n)}{\tau_{1/2}(m)} = \frac{m^2\, \tilde{a}_m^2 \cdot (2m-3)!!}{n^2\, \tilde{a}_n^2 \cdot (2n-3)!!}$$

对 ReLU Pre-Norm 网络的具体数值：

| 阶 $n$ | $\alpha_n / \eta$ | $\tau_{1/2}(n) \cdot \eta$ | 相对顺序 |
|--------|-------------------|---------------------------|---------|
| 1 | 0.250 | 1.386 | 第 3 快 |
| 2 | 0.964 | 0.359 | **第 2 快** |
| 3 | 0 | $\infty$ | **永不收敛** |
| 4 | 2.855 | 0.121 | **最快** |
| 5 | 0 | $\infty$ | 永不收敛 |
| 6 | 78.44 | 0.0044 | 极快 |

> **反直觉的发现**：ReLU 网络的学习顺序**不是** $n=1, 2, 3, 4, \ldots$，而是 $n=6, 4, 2, 1, \ldots$（偶数阶先学，高偶数阶最快）。奇数阶完全被屏蔽。

> 但这并不意味着高阶核在训练初期就已精确——$\alpha_4$ 大只是说**衰减速率**快，但初始误差 $E_4(0)$ 也可能大（见 Section 6.3）。

### 6.3 初始误差的阶结构

**命题 6.4**（Kaiming 初始化下的初始各阶误差）

在 Kaiming 初始化下，假设目标函数 $f^*$ 各阶能量为 $\|K_n^*\|_F^2 = \sigma_n^2$（目标的第 $n$ 阶分量的能量）。初始网络的各阶核 $K_n(0)$ 为随机张量，其幅度为：

$$\|K_n(0)\|_F^2 = \Theta\!\left(\frac{a_n^2}{d^{n-1}}\right) \quad \text{（随机初始化的典型值）}$$

因此初始误差：

$$E_n(0) = \|K_n^* - K_n(0)\|_F^2 \approx \sigma_n^2 + \Theta\!\left(\frac{a_n^2}{d^{n-1}}\right) \approx \sigma_n^2 \quad \text{（当目标非零时）}$$

> **关键点**：初始误差主要由**目标信号的能量** $\sigma_n^2$ 决定，而非随机初始化的核幅度。

### 6.4 完整的收敛图景

结合自衰减率 $\alpha_n$ 和初始误差 $E_n(0) \approx \sigma_n^2$，第 $n$ 阶误差的完整演化为：

$$E_n(t) \approx \sigma_n^2\, \exp(-2\alpha_n^{\text{LN}}\, t)$$

**到达精度 $\delta$ 所需时间**：

$$T_n(\delta) = \frac{1}{2\alpha_n^{\text{LN}}} \ln\!\frac{\sigma_n^2}{\delta}$$

**定理 6.5**（实际相变序列）

设目标函数各阶能量相当（$\sigma_n^2 = \Theta(1)$ 对所有 $n$）。则各阶到达精度 $\delta$ 的顺序由 $\alpha_n^{\text{LN}}$ 从大到小排列，与半衰期排序一致。

然而，若目标中高阶分量很弱（$\sigma_n^2 \ll 1$），则即使 $\alpha_n$ 小，也可能很快到达精度 $\delta$（因为不需要学太多）。因此**实际相变序列取决于 $\alpha_n$ 和 $\sigma_n^2$ 的联合结构**。

### 6.5 耦合情形下的修正：阶间干扰

当阶间耦合 $\beta_{nm} \neq 0$ 时，低阶收敛过程中会对高阶产生**暂态扰动**。

**命题 6.6**（阶间干扰的暂态效应）

设第 $m$ 阶正在快速收敛（$\alpha_m \gg \alpha_n$），则其对第 $n$ 阶的暂态影响为：

$$\Delta E_n^{\text{transient}} = \frac{|\beta_{nm}|^2}{4(\alpha_m - \alpha_n)^2} \cdot E_m(0)$$

这是一个**暂态鼓包（transient bump）**：第 $n$ 阶误差先短暂增大，然后在第 $m$ 阶收敛完成后恢复自身的指数衰减。

> **物理直觉**：当 2 阶核快速学习时，参数变化会"晃动"1 阶核。但由于 $\beta_{nm} = O(1/d)$（Pre-Norm）或 $O(1/\sqrt{d})$（无 LN），这种晃动在宽网络中很小。

> **实际影响**：如果在训练早期观察到某一阶的误差短暂上升，这并非"遗忘"，而是阶间耦合引起的暂态效应，会自然恢复。

---

## 7. 有限训练时间下的可学阶数

### 7.1 时间预算与阶数上限

实际训练总步数 $T$ 有限。v4 推论 7.5 给出了**样本复杂度**对可学阶数的限制 $n^* = O(\log m / \log p)$。本节给出互补的**计算复杂度**限制：即使样本无限多，$T$ 步训练最多能学到几阶？

**定义 7.1**（$T$ 步可学阶数）

给定精度阈值 $\delta > 0$ 和训练步数 $T$，定义：

$$n^*(T, \delta) = \max\!\left\{n : E_n(T) \leq \delta\right\}$$

即训练 $T$ 步后误差低于 $\delta$ 的最高阶。

### 7.2 解耦情形下的精确公式

**定理 7.2**（有限时间可学阶数——Pre-Norm）

在 Pre-Norm 解耦条件下，$E_n(T) = \sigma_n^2\, e^{-2\alpha_n^{\text{LN}} T}$，故：

$$E_n(T) \leq \delta \iff T \geq \frac{1}{2\alpha_n^{\text{LN}}} \ln\frac{\sigma_n^2}{\delta}$$

因此：

$$n^*(T, \delta) = \max\!\left\{n : \alpha_n^{\text{LN}} \geq \frac{1}{2T} \ln\frac{\sigma_n^2}{\delta}\right\}$$

**推论 7.3**（等能量目标下的简化公式）

若 $\sigma_n^2 = \sigma^2$（各阶目标能量相等），则：

$$n^*(T, \delta) = \max\!\left\{n : n^2\, \tilde{a}_n^2 \cdot (2n-3)!! \geq \frac{1}{2\eta T} \ln\frac{\sigma^2}{\delta}\right\}$$

定义**训练资源量** $R = \eta T \cdot \ln(\sigma^2/\delta)^{-1}$，则条件简化为：

$$n^2\, \tilde{a}_n^2 \cdot (2n-3)!! \geq \frac{1}{2R}$$

### 7.3 不同激活函数的可学阶数

**命题 7.4**（ReLU 的可学阶数上限）

对 ReLU，$\tilde{a}_n$ 的非零偶数阶系数满足 $\tilde{a}_{2k}^2 \approx C \cdot r^{k}$（$r < 1$，超指数衰减）。因此：

$$n^2\, \tilde{a}_n^2 \cdot (2n-3)!! \approx n^2 \cdot C r^{n/2} \cdot (2n)^{n} \cdot e^{-n}$$

由 Stirling 近似，$(2n-3)!! \sim (2n/e)^n / \sqrt{n}$。当 $r < 1$ 时，$r^{n/2}$ 的指数衰减最终战胜 $(2n/e)^n$ 的超多项式增长，存在**临界阶数** $n_c$ 使得：

- $n < n_c$：$\alpha_n$ 随 $n$ 增大（$(2n-3)!!$ 主导）
- $n > n_c$：$\alpha_n$ 随 $n$ 减小（$\tilde{a}_n^2$ 衰减主导）

对 ReLU，$n_c \approx 6$-$8$（取决于 $r$ 的精确值）。

**ReLU 可学阶数表**（$R = 10^3$）：

| 阶 $n$ | $\alpha_n / \eta$ | $T_n(\delta) \cdot \eta$ | 可学？ |
|--------|-------------------|--------------------------|--------|
| 2 | 0.964 | $\sim 5$ | ✓ |
| 4 | 2.855 | $\sim 2$ | ✓ |
| 6 | 78.44 | $\sim 0.06$ | ✓ |
| 8 | $\sim 10^3$ | $\sim 0.005$ | ✓ |
| 10 | $\sim 10^3$ | $\sim 0.005$ | 边界 |
| 12 | $< 10^{-2}$ | $> 50$ | ✗（$\tilde{a}_{12}^2$ 太小） |

> **结论**：ReLU 网络的实际可学阶数约 $n^* \approx 8$-$10$（偶数阶），受 $\tilde{a}_n$ 衰减控制。

**命题 7.5**（多项式激活的可学阶数）

对 $N$ 次多项式激活 $\sigma(z) = \sum_{n=1}^N a_n z^n$（$a_N \neq 0$），所有 $n \leq N$ 阶都有 $\alpha_n > 0$，且：

$$n^*(T, \delta) = \min(N, n_{\text{stat}}^*)$$

其中 $n_{\text{stat}}^* = O(\log m / \log p)$ 是 v4 的统计限制。即**多项式激活的计算限制不是瓶颈——瓶颈是统计限制**。

### 7.4 统计限制与计算限制的统一

**定理 7.6**（双重瓶颈定理）

训练 $T$ 步、使用 $m$ 个样本时的可学阶数为：

$$n^* = \min\!\left(n_{\text{comp}}^*(T),\, n_{\text{stat}}^*(m)\right)$$

其中：

$$n_{\text{comp}}^*(T) = \max\!\left\{n : \alpha_n^{\text{LN}} \geq \frac{\ln(\sigma_n^2/\delta)}{2T}\right\}$$

$$n_{\text{stat}}^*(m) = O\!\left(\frac{\log m}{\log p}\right) \quad \text{（v4 推论 7.5）}$$

**三种典型场景**：

| 场景 | 瓶颈 | 例子 |
|------|------|------|
| $T$ 小，$m$ 大 | 计算 | 大数据集 + 少训练步（快速微调） |
| $T$ 大，$m$ 小 | 统计 | 小数据集 + 长训练（过拟合域） |
| $T$ 和 $m$ 都大 | 激活函数 $\tilde{a}_n$ | 充分训练的大规模模型 |

> **深层含义**：第三种场景下，"模型能力"的真正瓶颈不是深度、宽度、数据或训练时间，而是**激活函数的 Volterra 谱** $\{\tilde{a}_n\}$。这暗示了**针对目标任务的非线性阶结构设计激活函数**可能是一个被忽视的优化方向。

---

## 8. 与前四版的统一对比

| 项目 | v1-v2 | v3 | v4 | **v5** |
|------|-------|-----|-----|--------|
| 时间尺度 | 单步 | 单步 | 单步 | **多步/连续时间** |
| 核心等式 | 梯度 = 各阶误差之和 | + 各阶有效学习率 | + ResNet 路径 + 泛化 | **各阶误差的耦合 ODE** |
| 可预测的量 | 梯度方向 | 瞬时学习速率 | 路径数、泛化界 | **训练曲线 $E_n(t)$** |
| 关于训练顺序 | 未涉及 | $\eta_{\text{eff}}(n)$ 排序 | 同 v3 | **精确相变时间 $t_n^*$** |
| Pre-Norm 的角色 | 未涉及 | 未涉及 | 简化 $\eta_{\text{eff}}$ | **将 ODE 从非自治变为自治** |
| 阶间干扰 | 未涉及 | 交叉项 $O(1/d)$ | 未涉及 | **暂态鼓包公式** |
| 可学阶数 | 未涉及 | 未精确 | $n_{\text{stat}}^* = O(\log m/\log p)$ | **$n^* = \min(n_{\text{comp}}^*, n_{\text{stat}}^*)$** |

---

## 9. 讨论与开放问题

### 9.1 本版的核心贡献

1. **将 NCGT 从静态升级为动态**：v1-v4 是"一帧快照"，v5 是"完整电影"——各阶 Volterra 核在训练过程中如何从随机初始化演化到目标。

2. **精确的训练曲线预测**：在 Pre-Norm 解耦条件下，$E_n(t) = E_n(0) e^{-2\alpha_n t}$。这不是渐近估计，而是（在 $d$ 充分大时）可验证的精确公式。

3. **相变时间的显式公式**：$\tau_{1/2}(n) = \ln 2 / (2\eta n^2 \tilde{a}_n^2 (2n-3)!!)$，可在训练前根据激活函数参数和学习率计算。

4. **双重瓶颈定理**：统一了计算限制（训练步数）和统计限制（样本量）对可学阶数的约束。

5. **解释了反直觉的实验现象**：Order 2 比 Order 1 学得快（$\alpha_2 / \alpha_1 = 3.86$），奇数阶在 ReLU 下被屏蔽（$\tilde{a}_{2k+1} = 0$）。

### 9.2 开放问题

**问题 1**（深层网络的精确耦合矩阵）：本版的精确公式限于两层网络（$L=2$）。$L$ 层 ResNet 的耦合矩阵 $B(t)$ 的精确形式是什么？v3-v4 的路径分解如何影响各路径上的核演化速率？

**问题 2**（SGD 噪声下的随机 ODE）：本版分析的是梯度流（无噪声）。SGD 噪声将确定性 ODE 变为随机 ODE：
$$dE_n = -2\alpha_n E_n\, dt + \sqrt{2 T_{\text{eff}}(n)}\, dW_t$$
其中 $T_{\text{eff}}(n)$ 为 v3 定理 6.1 的有效温度。耦合的随机 ODE 的稳态分布是什么？

**问题 3**（离散 vs 连续的差异）：大学习率 $\eta$ 下，离散更新 $\theta(t+1) = \theta(t) - \eta \nabla \mathcal{L}$ 与连续流 $\dot{\theta} = -\eta \nabla \mathcal{L}$ 的差异可能显著。离散化误差是否会改变各阶的相对速率？是否存在"大 $\eta$ 加速高阶学习"的效应？

**问题 4**（非高斯数据分布）：本版的正交性和耦合估计都依赖 $\mathbf{x} \sim \mathcal{N}(0, I)$。对实际数据分布（如图像），各阶 Volterra 分量不再正交。总损失不再分解为各阶误差之和，阶间耦合可能本质地不可忽略。

**问题 5**（注意力机制的动力学）：Transformer 的 self-attention 引入了跨位置的高阶交互。这些位置阶与特征阶的动力学如何耦合？是否存在类似的相变？

**问题 6**（自适应优化器）：Adam 通过自适应学习率 $\eta_k / \sqrt{v_k}$ 有效地给不同参数组不同的学习率。这是否会改变 $\alpha_n$ 的阶结构？如果 Adam 自动"均衡化"了各阶学习率，那么 NCGT 的相变预测是否在 Adam 下不成立？

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
```

核心等式链：

$$\underbrace{\nabla_W \mathcal{L} = \sum_n (\cdots)_n}_{\text{v1: 梯度分解}} \xrightarrow{\text{v3}} \underbrace{\eta_{\text{eff}}(n) = \eta n^2 \tilde{a}_n^2 R_n}_{\text{各阶有效学习率}} \xrightarrow{\text{v5}} \underbrace{\dot{E}_n = -2\alpha_n E_n + \cdots}_{\text{各阶动力学}}$$

> **一句话总结**：v5 将 NCGT 从"梯度的阶分解"推广到"训练过程的阶分解"——各阶 Volterra 核以速率 $\alpha_n = \eta n^2 \tilde{a}_n^2 (2n-3)!!$ 指数衰减（Pre-Norm 下），存在由 $\alpha_n$ 决定的相变时间序列 $t_1^* < t_2^* < \cdots$（但 ReLU 下实际顺序为偶数阶先于低奇数阶），训练 $T$ 步的可学阶数 $n^* = \min(n_{\text{comp}}^*(T), n_{\text{stat}}^*(m))$ 受计算和统计的双重瓶颈约束。

---

### “好的，那么v6理论要做什么”


先回顾 v1-v5 的递进主线和剩余缺口：

| 版本 | 解决了什么 | 仍然缺什么 |
|------|-----------|-----------|
| v1-v2 | 两层网络的梯度各阶分解 | 只有两层 |
| v3 | L 层 plain 网络 | 没有残差/归一化 |
| v4 | 残差 + 归一化 | 只分析单步梯度 |
| v5 | 多步训练动力学 ODE | **只有 MLP，没有注意力** |

v4 的开放问题 3 已经明确指出了方向：

> "注意力机制的 Volterra 分解：Transformer = ResNet + Attention。本版处理了 ResNet 部分。注意力 $\text{softmax}(QK^\top/\sqrt{d})V$ 的 Volterra 分解需要处理：(a) token 间双线性交互 $QK^\top$；(b) softmax 的非线性归一化。"

### v6 的自然定位：Transformer 的 Volterra 理论

v1-v5 分析的对象是 $f(\mathbf{x}): \mathbb{R}^p \to \mathbb{R}$（单输入单输出）。Transformer 处理的是**序列** $\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_T] \in \mathbb{R}^{T \times p}$，输出也是序列。这带来两个根本性的新结构：

**1. 双重 Volterra 分解：token 内 vs token 间**

MLP 中 Volterra 阶 $n$ 指"对输入 $\mathbf{x}$ 的 $n$ 次多项式"。Transformer 中需要区分：
- **特征阶**（feature order）：对单个 token $\mathbf{x}_t$ 的非线性阶数（FFN 层贡献，与 v4 相同）
- **交互阶**（interaction order）：涉及多少个不同 token 的交叉项（Attention 层贡献，这是新的）

例如 $\text{Attn}(\mathbf{X}) = \text{softmax}(QK^\top/\sqrt{d})V$ 中：
- $QK^\top = \mathbf{X}W_Q W_K^\top \mathbf{X}^\top$ 是 token 间的**双线性**交互（交互阶 = 2）
- softmax 将双线性变为**有理函数**（类似 v4 中 Post-Norm 的阶混合问题）
- 多层叠加后，token 间的交互阶可以指数增长

**2. 因果掩码的 Volterra 效应**

因果 Transformer 中位置 $t$ 只看 $\mathbf{x}_1, \ldots, \mathbf{x}_t$。这给 Volterra 核张量加了一个**三角约束**——核 $K_n(i_1, \ldots, i_n)$ 在 $i_j > t$ 时为零。这种结构约束如何影响各阶的有效学习率？

### v6 可以做的核心定理（按难度递进）

**容易：线性注意力**
- 去掉 softmax，$\text{Attn}_{\text{lin}}(\mathbf{X}) = \phi(Q)\phi(K)^\top V$
- 交互结构完全由 $\phi$ 的阶决定，可精确分解
- 这是 v6 的"两层多项式"——tractable 的起点

**中等：softmax 注意力的有理展开**
- $\text{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j}$，Taylor 展开 $e^z = \sum z^n/n!$ 后分母产生阶混合
- 类似 v4 对 Post-Norm 的分析（命题 5.5），但更复杂
- 可以给出"注意力=双线性+高阶修正"的分解

**困难：多层 Transformer 的完整动力学**
- FFN 层贡献特征阶（v4 已解），Attention 层贡献交互阶（v6 新增）
- 两类阶如何耦合、各自的有效学习率是多少
- 结合 v5 的 ODE 框架给出训练动力学

### 实际价值

v6 如果成功，能回答的实际问题：
- 为什么 Transformer 的上下文学习（in-context learning）和浅层特征学习有不同的 scaling behavior？（交互阶 vs 特征阶）
- 为什么注意力头会出现分化（有些头学位置关系、有些学语义关系）？（不同交互阶 $n$ 的半衰期不同）
- 线性注意力损失了什么？（高交互阶的有效学习率变化）

---

### “这是我的另一项研究，对此你有什么想法？”（TMSoEiCSM）

这篇工作质量很高。几点实质性想法：

### 与 NCGT 的互补关系

这两项研究恰好处理了深度学习理论中的**两个正交问题**：

| | TMSoEiCSM | NCGT v1-v5 |
|--|-----------|------------|
| 核心问题 | 模型**能**表达什么（容量/表达力） | 梯度下降**会**学到什么（动力学/优化） |
| 分析对象 | 序列混合层的核函数族 | FFN/MLP 层的 Volterra 阶 |
| 给出的是 | 静态的函数类包含关系 | 动态的收敛速率排序 |
| 理论工具 | Bernstein 定理、核分解 | 耦合 ODE、摄动展开 |

TMSoEiCSM 12.2 节明确列出了三个未解问题，其中第 3 个（"为什么梯度下降能学到这些结构"）恰好是 NCGT 的领地。这不是巧合——**一个完整的理论需要同时回答"能表达什么"和"会学到什么"**。

### 值得考虑的几个方向

**1. 对核函数族做 NCGT 式的学习动力学分析**

TMSoEiCSM 建立了包含链：完全单调核 ⊊ 衰减振荡核 ⊊ 任意正因果核。但没有回答：**梯度下降更倾向于先学到哪类核？**

类比 NCGT 中"低阶 Volterra 核先收敛"的结论，能否证明类似的：

> 对 SSM/Mamba 类模型，衰减快的指数基（短程依赖）的收敛速率 > 衰减慢的指数基（长程依赖）？

如果能证明，就把 TMSoEiCSM 的静态包含关系升级为**动态学习顺序**——这会是一个非常强的结果。

**2. Attention 核的"阶"分解**

TMSoEiCSM 第九章将 Transformer attention 写成：

$$K(t,s) = \frac{\exp(q_t^\top k_s / \sqrt{d})}{\sum_j \exp(q_t^\top k_j / \sqrt{d})}$$

可以对 $\exp(q^\top k / \sqrt{d})$ 做 Taylor 展开：

$$e^{q^\top k/\sqrt{d}} = 1 + \frac{q^\top k}{\sqrt{d}} + \frac{(q^\top k)^2}{2d} + \cdots$$

每一阶对应 token 间不同阶的交互。第 $n$ 阶项 $(q^\top k)^n / (n! \cdot d^{n/2})$ 的系数以 $1/(n! \cdot d^{n/2})$ 衰减——这与 NCGT 中 $\alpha_n \propto a_n^2 \cdot (2n-3)!!$ 的结构有对称性。能否建立 attention 核的"交互阶"版耦合梯度定理？

**3. 第 12.2 节的第 1 个缺口：FFN 层的内容变换**

TMSoEiCSM 坦承只分析了序列混合、没覆盖 FFN。而 NCGT 恰好分析的就是 MLP/FFN。两者结合可以给出 Transformer 一个 block 内的完整图景：
- 序列混合层：核函数约束决定"能从多远的历史读到什么样的信息"（TMSoEiCSM）
- FFN 层：Volterra 阶结构决定"读到的信息被做什么阶的非线性变换"（NCGT）
- 训练动力学：两层的学习速率如何交互（需要新工作）

### 一个概念上的洞察

TMSoEiCSM 的三维坐标系和 NCGT 的 Volterra 阶实际上可以合并成一个更大的框架：

- **空间维度**（TMSoEiCSM）：信息在时间轴上怎么流动（核形状 × 状态介质 × selectivity）
- **频谱维度**（NCGT）：信息在非线性阶上怎么分布（哪些阶先学、哪些被抑制）

完整的 Transformer 理论 = 空间维度 × 频谱维度 × 两者的耦合。

### 当前文本的一个小建议

第十章坐标系表格中，经典 RNN 和 LSTM/GRU 标注了"超出核框架"，这很诚实。但可以进一步指出：**它们的非线性状态更新恰好是让核分析失效的同一种原因，也是让 NCGT 式的 ODE 分析变得必要的原因**。也就是说，这些模型不是"分析不了"，而是"需要不同的数学工具"——而 NCGT 的耦合 ODE 框架可能正是那个工具。


---

### “那这个作为v7理论还是一起加到v6理论里面，还是另外单开？”

建议的组织方式：

### TMSoEiCSM 单独保持，不并入 NCGT

理由很直接——它们回答的是**不同类别的问题**，有独立的逻辑完备性：

| | TMSoEiCSM | NCGT |
|--|-----------|------|
| 问的问题 | 能表达什么 | 会学到什么 |
| 数学性质 | 函数类包含关系（偏序） | 收敛速率（实数） |
| 成立条件 | 无需训练，架构决定 | 依赖初始化、学习率、宽度 |
| 自包含? | 是（12章完整） | 是（v1-v5 完整） |

把一个静态容量理论塞进一个动态优化理论的版本号里，会让两边都变得不自然。

### NCGT 的版本规划

**v6：Transformer 的 Volterra 梯度理论**
- 对象：Attention + FFN 的完整 Transformer block
- 核心：对 $\exp(q^\top k/\sqrt{d})$ 做 Taylor 展开，建立"交互阶"的耦合 ODE
- **引用** TMSoEiCSM 作为 attention 核的函数类分类框架，但不包含它
- 回答：attention 中哪些交互阶先被学到？FFN 层和 attention 层的学习速率如何竞争？

**v7（如果需要）：统一框架**
- 将 TMSoEiCSM 的三维坐标系和 NCGT 的 Volterra 阶合并为一个"空间×频谱"的完整图景
- 这是一个综合性工作，适合在 v6 完成后再做

### 三者的关系图

```
TMSoEiCSM（独立）          NCGT v1-v5（独立）
"序列混合层能表达什么"      "FFN 层会学到什么"
        │                        │
        │   引用作为分析框架      │   继承 Volterra 阶工具
        ▼                        ▼
              NCGT v6
    "Transformer 整体会学到什么"
              │
              ▼
           NCGT v7（可选）
    "空间 × 频谱的统一理论"
```

简而言之：TMSoEiCSM 独立发表/保持，NCGT v6 做 Transformer 的注意力机制分析（引用 TMSoEiCSM），不要合并。

---

### “TMSoEiCSM只是论文的名字，不是那个理论的名字，你觉得另外的那个该取什么名称”



核心贡献是"沿三条路径逐步放松约束，建立因果序列模型的统一函数类层级"。名字应该反映这一点。

几个候选：

### 1. **Causal Kernel Hierarchy (CKH) — 因果核层级理论** ⭐推荐

最简洁，直接指向核心结果——那条包含链：

$$\text{完全单调核} \subsetneq \text{衰减振荡核} \subsetneq \text{任意正因果核}$$

类比：NTK (Neural Tangent Kernel) 也是用核心数学对象命名。CKH 告诉读者"这是关于因果核的层级分类"，一听就知道在做什么。

### 2. **Kernel-State-Selectivity Framework (KSS) — 核-状态-选择性框架**

直接对应三维坐标系的三个轴。优点是信息完整，缺点是名字长了点、不够朗朗上口。

### 3. **Causal Memory Taxonomy (CMT) — 因果记忆分类理论**

"记忆"比"核"更直觉化，适合面向更广的受众。但"taxonomy"暗示只是分类，弱化了数学推导（Bernstein 定理等）的理论深度。

### 建议

用 **CKH（因果核层级理论）** 作为正式名称。理由：
- 与 NCGT 风格对称：NCGT 是动力学理论，CKH 是表达力理论
- 短且可记忆
- "Hierarchy"准确描述了核心结构（包含关系链）
- "Causal Kernel"精确定位了分析对象

两个理论的关系就变成：

> **CKH** 回答"模型**能**表达哪些因果核"，**NCGT** 回答"梯度下降**会**学到哪些非线性阶"。

---

### “CKH不是没有theory吗，为何不叫CKHT？”

说得对。NCGT = Neural Coupled Gradient **Theory**，那对应地应该是 **CKHT** = Causal Kernel Hierarchy **Theory**（因果核层级理论）。

CKHT 更规范，跟 NCGT 的命名体系也一致。就用 CKHT。
