# 参数共享的耦合梯度定理：  
# 梯度下降对 Volterra 核各阶的同步更新

---

## 1. 动机

神经网络为何能用 $O(d^2)$ 个参数高效学习高阶非线性函数？  
本文证明：参数共享不只是降低了表达复杂度，更使梯度下降在每一步中**同时更新所有阶 Volterra 核**——这是深度学习高效性的一个基础机制。

---

## 2. 设置与符号

**网络**（两层，单输出，方便推导）：

$$f(\mathbf{x}) = \mathbf{w}_2^\top \sigma(W_1 \mathbf{x})$$

其中 $W_1 \in \mathbb{R}^{d \times p}$，$\mathbf{w}_2 \in \mathbb{R}^d$，$\mathbf{x} \in \mathbb{R}^p$。

**激活函数**（多项式，逐元素作用，**无常数项**）：

$$\sigma(z) = \sum_{n=1}^{N} a_n z^n$$

> **假设说明**：令 $a_0 = 0$（无常数项），使 $\sigma(0)=0$。这不影响模型的表达能力——常数偏置可由输出层 $\mathbf{w}_2$ 的偏置项吸收——但简化了 Volterra 分解：所有核阶数从 $n=1$ 开始，避免引入零阶（常数）核。

> 注：ReLU 等非多项式激活可用 Hermite 多项式在 $\mathcal{N}(0,1)$ 下展开得到近似版本，详见第二版。

**损失**：

$$\mathcal{L} = \mathbb{E}\left[(y - f(\mathbf{x}))^2\right]$$

---

## 3. Volterra 分解定理

**命题 3.1**（网络的 Volterra 分解）

$f(\mathbf{x})$ 可分解为有限阶 Volterra 级数：

$$f(\mathbf{x}) = \sum_{n=1}^{N} f_n(\mathbf{x})$$

其中第 $n$ 阶项为：

$$f_n(\mathbf{x}) = \sum_{i_1, \ldots, i_n=1}^{p} h_n(i_1, \ldots, i_n)\, x_{i_1} \cdots x_{i_n}$$

第 $n$ 阶 Volterra 核由**共享参数** $W_1, \mathbf{w}_2$ 显式给出：

$$\boxed{h_n(i_1, \ldots, i_n) = a_n \sum_{k=1}^{d} [\mathbf{w}_2]_k\, [W_1]_{k, i_1} \cdots [W_1]_{k, i_n}}$$

**证明**：

$$f(\mathbf{x}) = \mathbf{w}_2^\top \sigma(W_1 \mathbf{x}) = \sum_{k=1}^{d} [\mathbf{w}_2]_k\, \sigma\!\left(\sum_{j=1}^{p} [W_1]_{k,j} x_j\right)$$

代入 $\sigma(z) = \sum_{n=1}^N a_n z^n$：

$$= \sum_{k=1}^{d} [\mathbf{w}_2]_k \sum_{n=1}^{N} a_n \left(\sum_{j=1}^{p} [W_1]_{k,j} x_j\right)^n$$

展开 $n$ 次幂，交换求和顺序即得。 $\square$

**推论**：$h_n$ 是一个**秩至多为 $d$ 的对称张量**，而其存储量若直接表示则为 $O(p^n)$。参数共享将存储压缩到 $O(dp + d)$，**与阶数 $n$ 无关**。

---

## 4. 耦合梯度定理

这是本文的核心结论。

**定理 4.1**（梯度的 Volterra 阶分解）

对参数 $[W_1]_{k,j}$ 的梯度可分解为各阶 Volterra 核误差的加权和：

$$\frac{\partial \mathcal{L}}{\partial [W_1]_{k,j}} = -2 \sum_{n=1}^{N} n\, a_n\, \mathbb{E}\!\left[\varepsilon(\mathbf{x})\, [\mathbf{w}_2]_k\, (\mathbf{u}_k^\top \mathbf{x})^{n-1}\, x_j\right]$$

其中 $\varepsilon(\mathbf{x}) = y - f(\mathbf{x})$ 为预测误差，$\mathbf{u}_k = W_1[k, :]^\top$ 为 $W_1$ 的第 $k$ 行。

**证明**：

$$\frac{\partial \mathcal{L}}{\partial [W_1]_{k,j}} = -2\, \mathbb{E}\!\left[\varepsilon(\mathbf{x})\, \frac{\partial f}{\partial [W_1]_{k,j}}\right]$$

$$\frac{\partial f}{\partial [W_1]_{k,j}} = [\mathbf{w}_2]_k\, \sigma'\!(\mathbf{u}_k^\top \mathbf{x})\, x_j$$

代入 $\sigma'(z) = \sum_{n=1}^N n\,a_n\,z^{n-1}$，即得定理。 $\square$

---

## 5. 高阶 Volterra 核的统计估计下界

在建立统计下界之前，先证明一个关键引理，用于解决各阶 Volterra 核的**可识别性**问题。

**引理 5.0**（不同阶 Volterra 分量的 $L^2$ 正交性）

设 $\mathbf{x} \sim \mathcal{N}(0, I_p)$。对任意 $m \neq n$，有：

$$\mathbb{E}_{\mathbf{x}}\!\left[f_m(\mathbf{x})\, f_n(\mathbf{x})\right] = 0$$

**证明**：$f_m$ 为 $m$ 次齐次多项式，$f_n$ 为 $n$ 次齐次多项式，分别可展开为 $|\boldsymbol{\alpha}|=m$ 和 $|\boldsymbol{\beta}|=n$ 阶的 Hermite 多项式之和。由 Hermite 正交性，不同次数的基函数内积为零，故 $\mathbb{E}[f_m f_n]=0$。$\square$

**推论**：设 $f = \sum_{n=1}^N f_n$ 为已知函数（或其估计量），则第 $k$ 阶分量可通过 $L^2$ 投影唯一恢复：$f_k = P_k[f]$，其中 $P_k$ 为到 $k$ 次 Hermite 子空间的正交投影。这保证了即使网络以耦合方式联合学习所有 $f_n$，各阶分量仍是**独立可识别**的。

---

**定理 5.1**（极小化极大下界）

设观测模型为 $y_i = f^*(\mathbf{x}_i) + \xi_i$，其中 $\xi_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$，$\mathbf{x}_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, I_p)$，$i = 1, \ldots, n$。由引理 5.0，$f_k^*$ 由数据唯一可识别。对**任意**基于 $n$ 个样本的估计量 $\hat{f}_k$（包括有偏估计量），有：

$$\boxed{\mathbb{E}\left[\|\hat{f}_k - f_k^*\|_{L^2}^2\right] \geq \frac{\sigma^2 \cdot d_k}{n}}$$

其中 $\|g\|_{L^2}^2 = \mathbb{E}_{\mathbf{x}\sim\mathcal{N}(0,I_p)}[g(\mathbf{x})^2]$，$d_k = \binom{p + k - 1}{k}$。

> **范数说明**：此处使用 $L^2(\mathcal{N}(0,I_p))$ 范数。Hermite 基到单项式基的变换非正交，故不能将下界直接写成张量 Frobenius 范数的形式。

**证明**：

**Step 1：正交基重参数化。** 令 $\mathcal{M}_k = \{\boldsymbol{\alpha} \in \mathbb{Z}_{\geq 0}^p : |\boldsymbol{\alpha}| = k\}$（$|\mathcal{M}_k| = d_k$）。将 $f_k$ 展开为：

$$f_k(\mathbf{x}) = \sum_{\boldsymbol{\alpha} \in \mathcal{M}_k} \theta_{\boldsymbol{\alpha}}\, \mathsf{He}_{\boldsymbol{\alpha}}(\mathbf{x}), \quad \mathsf{He}_{\boldsymbol{\alpha}}(\mathbf{x}) = \prod_{j=1}^p \mathsf{He}_{\alpha_j}(x_j)$$

此展开与单项式展开之间为可逆线性变换，参数个数 $d_k$ 不变。

**Step 2：充分统计量。** 固定 $\boldsymbol{\alpha} \in \mathcal{M}_k$，由引理 5.0，不同阶及同阶不同多指标的 Hermite 展开项互相正交，因此：

$$T_{\boldsymbol{\alpha}} := \frac{1}{n\,\boldsymbol{\alpha}!}\sum_{i=1}^n y_i\, \mathsf{He}_{\boldsymbol{\alpha}}(\mathbf{x}_i)$$

是 $\theta_{\boldsymbol{\alpha}}$ 的无偏估计量，且 $T_{\boldsymbol{\alpha}}$ 是 $\theta_{\boldsymbol{\alpha}}$ 的**充分统计量**（高斯线性模型，其余参数已从正交子空间分离）。

**Step 3：充分统计量的方差。** 直接计算：

$$\mathrm{Var}(T_{\boldsymbol{\alpha}}) = \frac{1}{n^2 (\boldsymbol{\alpha}!)^2}\sum_{i=1}^n \sigma^2\,\mathbb{E}[\mathsf{He}_{\boldsymbol{\alpha}}(\mathbf{x})^2] = \frac{\sigma^2\,\boldsymbol{\alpha}!}{n(\boldsymbol{\alpha}!)^2} = \frac{\sigma^2}{n\,\boldsymbol{\alpha}!}$$

**Step 4：对任意估计量的下界。** $T_{\boldsymbol{\alpha}}$ 是 $\theta_{\boldsymbol{\alpha}}$ 充分统计量的线性函数。对高斯位置族参数 $\theta_{\boldsymbol{\alpha}} \in \mathbb{R}$，任意估计量的极小化极大均方误差等于充分统计量的方差（偏估计量无法在所有 $\theta_{\boldsymbol{\alpha}}$ 上统一降低 MSE）：

$$\inf_{\hat{\theta}_{\boldsymbol{\alpha}}} \sup_{\theta_{\boldsymbol{\alpha}} \in \mathbb{R}}\, \mathbb{E}\!\left[(\hat{\theta}_{\boldsymbol{\alpha}} - \theta_{\boldsymbol{\alpha}})^2\right] = \mathrm{Var}(T_{\boldsymbol{\alpha}}) = \frac{\sigma^2}{n\,\boldsymbol{\alpha}!}$$

**Step 5：对 $L^2$ 范数取迹。** 由 $\|f_k\|_{L^2}^2 = \sum_{\boldsymbol{\alpha}} \boldsymbol{\alpha}!\,\theta_{\boldsymbol{\alpha}}^2$（Hermite 正交性），加权求和：

$$\mathbb{E}\!\left[\|\hat{f}_k - f_k^*\|_{L^2}^2\right] = \sum_{\boldsymbol{\alpha}\in\mathcal{M}_k} \boldsymbol{\alpha}!\,\mathbb{E}\!\left[(\hat{\theta}_{\boldsymbol{\alpha}}-\theta_{\boldsymbol{\alpha}}^*)^2\right] \geq \sum_{\boldsymbol{\alpha}\in\mathcal{M}_k} \boldsymbol{\alpha}!\cdot\frac{\sigma^2}{n\,\boldsymbol{\alpha}!} = \frac{\sigma^2\,d_k}{n} \qquad \square$$

---

**推论 5.2**（维数诅咒的统计版）

由于 $d_k = \binom{p+k-1}{k} \sim \frac{p^k}{k!}$，达到精度 $\epsilon$ 所需样本量为：

$$n = \Omega\!\left(\frac{\sigma^2\, p^k}{k!\,\epsilon^2}\right)$$

随阶数 $k$ **指数增长**，与是否使用梯度下降无关。

---

**推论 5.3**（任意学习算法无法突破统计下界）

定理 5.1 的极小化极大下界对**任意**估计量成立（包括梯度下降产生的有偏估计），因此：无论训练多少步 $T$，在有限 $n$ 个样本下，第 $k$ 阶核的 $L^2$ 误差存在不可消除的下界：

$$\mathbb{E}\!\left[\|\hat{f}_k^{(T)} - f_k^*\|_{L^2}^2\right] \geq \frac{\sigma^2\, d_k}{n}$$

**这与训练步数 $T$ 无关**——这不是优化问题（梯度下降收不收敛），而是**统计不可分辨性**问题：$n$ 个样本本身携带的信息不足以确定高阶核。

这正是"神经网络为何无法收敛到测试损失全局最优"的一个信息论根源：高阶 Volterra 核的 $L^2$ 估计误差对**任意学习算法**均有 $\sigma^2 d_k / n$ 的精确下界，随阶数 $k$ 以 $d_k \sim p^k/k!$ 的速度增长。

---

## 6. 核心含义：单步梯度同时减小所有阶误差

将上式按阶拆开，第 $n$ 阶分量为：

$$\Delta_n W_1 \propto \mathbb{E}\!\left[\varepsilon(\mathbf{x})\, (\mathbf{w}_2 \odot (\mathbf{u}^\top \mathbf{x})^{n-1})\, \mathbf{x}^\top\right]$$

**注意**：$\varepsilon(\mathbf{x})$ 包含所有阶的拟合误差，$(\mathbf{u}^\top \mathbf{x})^{n-1}$ 提取了第 $n$ 阶交互模式。

因此 $W_1$ 的一次梯度更新等价于对 $h_1, h_2, \ldots, h_N$ **同时施加一次残差驱动的修正**，修正强度由 $n\,a_n$ 和当前误差 $\varepsilon$ 共同决定。

与线性情形（Wiener 最优解）的对比：

| | 线性（Wiener） | 非线性（本框架） |
|--|--------------|----------------|
| 最优解 | $H^* = R_{xx}^{-1} r_{xy}$（闭合式） | 无闭合式 |
| 梯度含义 | 减小一阶核误差 | **同时减小所有阶核误差** |
| 参数量 | $O(p^2)$（$p$ 阶 FIR 滤波器） | $O(dp)$（与 $n$ 无关） |
| 收敛保证 | 全局（凸） | 局部（非凸） |

---

## 7. 深度 $L$ 层的推广

对 $L$ 层网络 $f = W_L \sigma(W_{L-1} \cdots \sigma(W_1 x)\cdots)$，Volterra 阶数上限为 $N^{L-1}$（复合激活的幂次累积），而参数量仍为 $O(L d^2)$。

**推论 7.1**（非正式草图，文献支持见下）：深度每增加一层，可表达的 Volterra 阶数**指数增长**，参数量**线性增长**。梯度信号通过链式法则反向传播，每一层的参数更新均携带来自所有有效 Volterra 阶的误差信号。

> **文献支撑**：Cohen et al. (2016) 用张量网络（Tensor Networks）严格证明了：深度 $L$、宽度 $d$ 的多项式激活网络，其等效多项式展开的张量秩随深度呈**指数增长**，而参数量仅**线性增长** $O(Ld^2)$。本推论是其结论在梯度动力学框架下的表述形式，严格证明直接引用 Cohen et al. (2016) 定理 1-2。

> 这从理论上解释了"深度比宽度更重要"：深度实现了 Volterra 阶数的指数级扩展，而参数代价仅线性增长。

---

## 8. 与频率原理的联系

频率原理（Xu et al. 2019）观察到梯度下降先拟合低频。本节探讨其与 Volterra 框架的可能联系，但需注意**多项式阶数与 Fourier 频率并非简单对应关系**。

> **重要区分**：Volterra 阶数衡量的是变量间的**交互阶数**（多少个输入分量相乘），而 Fourier 频率衡量的是函数在空间中的**振荡速率**。两者是不同的"复杂度"概念。例如 $\cos(\omega x)$ 对任意大 $\omega$ 都是高频函数，但其 Taylor 展开涉及所有阶多项式；反之，$x^{100}$ 是 100 阶多项式但并非直觉上的"高频"函数。

尽管如此，在本框架中存在一个间接联系：

权重 $n\,a_n$ 控制各阶在梯度中的贡献强度。对大多数常用激活函数（Sigmoid、Tanh、Softplus），$|a_n|$ 随 $n$ 衰减，因此**低阶核在梯度中占主导，低阶交互先被拟合**。

这并不直接等价于"低频先被拟合"，但提供了一个互补视角：**激活函数系数 $a_n$ 的衰减结构**导致梯度下降优先学习低阶交互，而低阶交互在许多自然数据分布上倾向于对应较低频的变化模式。将此联系严格化——即精确刻画 Volterra 阶数与 Fourier 频率在何种数据分布下具有单调对应关系——是一个开放问题。

---

## 9. 开放问题

1. **非多项式激活**：ReLU 的 Volterra 展开是无限阶发散级数。能否用截断误差分析给出有限阶近似的误差界？

2. **收敛速度**：各阶核的估计误差 $\|h_n^* - \hat{h}_n(t)\|$ 随训练步数 $t$ 如何衰减？

3. **批量大小的影响**：SGD 的随机性是否对各阶核有不同的正则化效果（高阶核更容易被噪声抹去）？

4. **与 NTK 的关系**：NTK 固定了 1 阶核的学习，本框架显式描述高阶核的演化。能否给出"特征学习发生" iff "高阶核误差主导梯度"的精确判据？

---

## 参考文献

- Cohen, N. et al. (2016). *On the Expressive Power of Deep Learning: A Tensor Analysis.*
- Volterra, V. (1887). *Sopra le funzioni che dipendono da altre funzioni.*
- Xu, Z. et al. (2019). *Frequency Principle: Fourier Analysis Sheds Light on Implicit Regularization of Deep Neural Networks.*
- Chrysos, G. et al. (2021). *Deep Polynomial Neural Networks.*
- Widrow, B. & Hoff, M. (1960). *Adaptive Switching Circuits.* (LMS 算法)
