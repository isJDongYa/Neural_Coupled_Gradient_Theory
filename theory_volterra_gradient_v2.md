# 参数共享的耦合梯度定理（第二版）：  
# 推广至任意 L²(γ) 激活函数

> **与第一版的关系**：第一版在多项式激活下建立精确等式。本版将框架推广到 ReLU 等非光滑激活函数，核心工具从 Taylor 展开替换为 L²(γ) Hermite 展开 + 均方收敛 + 平均场近似。代价是定理从精确等式变为近似版本，但近似误差有明确的收敛率（ReLU：函数近似 $O(K^{-3/2})$，梯度偏差 $O(K^{-1/4})$）。

---

## 1. 动机与问题设置

第一版的核心假设——激活函数有收敛的幂级数展开——对 ReLU、LeakyReLU、GELU 等现代激活函数均不满足。本版通过将函数空间从"多项式代数"替换为"高斯测度下的 L² 空间"，绕过这一障碍，使耦合梯度框架适用于实践中最常用的激活函数。

**网络设置**（两层，单输出）：

$$f(\mathbf{x}) = \mathbf{w}_2^\top \sigma(W_1 \mathbf{x})$$

其中 $W_1 \in \mathbb{R}^{d \times p}$，$\mathbf{w}_2 \in \mathbb{R}^d$，$\mathbf{x} \in \mathbb{R}^p$，激活函数 $\sigma : \mathbb{R} \to \mathbb{R}$ 逐元素作用。

**对激活函数的要求**（比第一版弱得多）：

$$\sigma \in L^2(\gamma), \quad \text{即 } \mathbb{E}_{Z \sim \mathcal{N}(0,1)}[\sigma(Z)^2] < \infty$$

ReLU、LeakyReLU、GELU、Sigmoid、Tanh 均满足此条件。

---

## 2. 工具准备：Hermite 展开

### 2.1 Hermite 多项式

物理学家 Hermite 多项式定义为：

$$H_n(z) = (-1)^n e^{z^2/2} \frac{d^n}{dz^n} e^{-z^2/2}$$

前几项：$H_0=1,\ H_1=z,\ H_2=z^2-1,\ H_3=z^3-3z$。

关键性质（在高斯测度 $\gamma = \mathcal{N}(0,1)$ 下）：

$$\mathbb{E}[H_m(Z) H_n(Z)] = n!\, \delta_{mn} \quad \Longrightarrow \quad \left\{\frac{H_n}{\sqrt{n!}}\right\}_{n=0}^\infty \text{ 构成 } L^2(\gamma) \text{ 的标准正交基}$$

### 2.2 任意函数的 Hermite 展开

对任意 $\sigma \in L^2(\gamma)$：

$$\sigma(z) = \sum_{n=0}^{\infty} \hat{a}_n H_n(z), \quad \hat{a}_n = \frac{1}{n!}\mathbb{E}_{Z \sim \mathcal{N}(0,1)}[\sigma(Z) H_n(Z)]$$

此展开在 **L² 意义下收敛**：$\sum_{n=0}^\infty n!\, \hat{a}_n^2 = \mathbb{E}[\sigma(Z)^2] < \infty$。

**ReLU 的系数**（显式计算，利用 $\mathrm{ReLU}(z)=\frac{z+|z|}{2}$ 及高斯半矩 $\mathbb{E}[Z^{2k+1}\mathbf{1}_{Z>0}]=2^k k!/\sqrt{2\pi}$）：

$$\hat{a}_0 = \frac{1}{\sqrt{2\pi}}, \quad \hat{a}_1 = \frac{1}{2}, \quad \hat{a}_{2m} = \frac{(-1)^{m-1}}{(2m-1)\cdot 2^m \cdot m! \cdot \sqrt{2\pi}} \text{ (偶数 } n=2m \geq 2\text{)}$$

$$\hat{a}_{2m+1} = 0 \text{ (奇数 } n \geq 3 \text{，因 ReLU} - \tfrac{1}{2}z \text{ 为偶函数)}$$

> 注：$\hat{a}_1 = 1/2$ 是奇数阶中唯一非零项，因为 $\mathrm{ReLU}(z) - \frac{1}{2}z = \frac{|z|}{2}$ 是偶函数，其展开只含偶数阶 Hermite 多项式。

衰减速率：由 Stirling 公式，$|\hat{a}_{2m}| \sim \frac{e^m}{m^{m+3/2} \cdot 2^m \cdot \sqrt{2\pi}}$（**超指数衰减**）。但截断误差涉及 $n!\hat{a}_n^2$ 的求和，由 $\binom{2m}{m} \sim 4^m/\sqrt{\pi m}$ 可得每项 $(2m)!\hat{a}_{2m}^2 = O(m^{-5/2})$，故：

$$\left\|\sigma - \sum_{n=0}^K \hat{a}_n H_n\right\|_{L^2(\gamma)}^2 = \sum_{n>K} n!\,\hat{a}_n^2 = O(K^{-3/2})$$

---

## 3. 近似 Volterra 分解定理

### 3.1 关键步骤：Hermite → 单项式基转换

Hermite 截断 $\sigma^{(K)}(z) = \sum_{n=0}^K \hat{a}_n H_n(z)$ 是一个 $K$ 次**多项式**。但 $H_n(z) \neq z^n$——每个 Hermite 多项式包含低阶单项式（例如 $H_2(z) = z^2 - 1$, $H_3(z) = z^3 - 3z$）。因此**Hermite 阶 $n$ ≠ Volterra 阶 $n$**。

要获得正确的 Volterra 核，必须将 $\sigma^{(K)}$ 重新展开为单项式形式：

$$\sigma^{(K)}(z) = \sum_{j=0}^K \tilde{a}_j\, z^j, \quad \tilde{a}_j = \sum_{\substack{n = j,\, j+2,\, j+4, \ldots}}^{K} \hat{a}_n \cdot [z^j]H_n(z)$$

其中 $[z^j]H_n(z)$ 为 $H_n(z)$ 中 $z^j$ 的系数。例如：
- $[z^2]H_2 = 1,\ [z^0]H_2 = -1$
- $[z^3]H_3 = 1,\ [z^1]H_3 = -3$
- $[z^4]H_4 = 1,\ [z^2]H_4 = -6,\ [z^0]H_4 = 3$

> **关键区别**：$\tilde{a}_j \neq \hat{a}_j$。每个 Hermite 阶 $n$ 将其单项式成分分散到 Volterra 阶 $n, n{-}2, n{-}4, \ldots$。例如 $\hat{a}_4 H_4(z) = \hat{a}_4(z^4 - 6z^2 + 3)$ 同时贡献了 $\tilde{a}_4$ 中的 $+\hat{a}_4$、$\tilde{a}_2$ 中的 $-6\hat{a}_4$、和 $\tilde{a}_0$ 中的 $+3\hat{a}_4$。

### 3.2 Volterra 分解

**命题 3.1**（网络的截断 Volterra 分解，L² 版）

设 $\sigma^{(K)}(z) = \sum_{j=0}^K \tilde{a}_j z^j$ 为 Hermite 截断的单项式形式。相应网络 $f^{(K)}(\mathbf{x}) = \mathbf{w}_2^\top \sigma^{(K)}(W_1 \mathbf{x})$ 可精确分解为 $K$ 阶 Volterra 级数。

第 $j$ 阶 Volterra 核为（直接由第一版命题 3.1 推出，$\tilde{a}_j$ 替换 $a_j$）：

$$\boxed{h_j^{(K)}(i_1, \ldots, i_j) = \tilde{a}_j \sum_{k=1}^{d} [\mathbf{w}_2]_k\, [W_1]_{k,i_1} \cdots [W_1]_{k,i_j}}$$

> **证明**：$\sigma^{(K)}$ 是 $K$ 次多项式，可直接应用第一版框架（将 $\tilde{a}_j$ 视为 Taylor 系数）。$\square$

### 3.3 近似误差

**近似误差**（在 $\mathbf{x} \sim \mathcal{N}(0, I_p)$ 下的均方误差）：

$$\mathbb{E}\!\left[(f(\mathbf{x}) - f^{(K)}(\mathbf{x}))^2\right] \leq \|\mathbf{w}_2\|^2 \sum_{k=1}^d \mathbb{E}\!\left[|\sigma(\mathbf{u}_k^\top\mathbf{x}) - \sigma^{(K)}(\mathbf{u}_k^\top\mathbf{x})|^2\right]$$

右端每项为 $\sigma - \sigma^{(K)}$ 在 $\mathcal{N}(0, \|\mathbf{u}_k\|^2)$ 下的 $L^2$ 范数，**注意不等于** $\|\mathbf{u}_k\|^2 \cdot \|\sigma-\sigma^{(K)}\|_{L^2(\gamma)}^2$（Hermite 正交性仅在标准高斯 $\gamma = \mathcal{N}(0,1)$ 下成立，方差不为 1 时失效）。

**特殊情形**：

- 若 $\|\mathbf{u}_k\| = 1$（如权重归一化或 Kaiming 初始化的期望值），则 $\mathbf{u}_k^\top\mathbf{x} \sim \mathcal{N}(0,1)$，误差精确等于 $\|\sigma-\sigma^{(K)}\|_{L^2(\gamma)}^2$，对 ReLU 为 $O(K^{-3/2})$。

- 若 $\sigma$ 是正齐次函数（如 ReLU：$\sigma(\alpha z) = \alpha\sigma(z)$ for $\alpha > 0$），则 $\sigma(s_k z) = s_k \sigma(z)$。但注意 $\sigma^{(K)}(s_k z) \neq s_k \sigma^{(K)}(z)$（因为 $H_n(s_k z) \neq s_k H_n(z)$ for $n \geq 2$），所以即使对 ReLU，非单位方差的误差也不能简单通过缩放得到。

**证明**：由 Cauchy-Schwarz 不等式, $|f - f^{(K)}|^2 = |\sum_k [w_2]_k (\sigma(u_k) - \sigma^{(K)}(u_k))|^2 \leq \|\mathbf{w}_2\|^2 \sum_k |\sigma(u_k) - \sigma^{(K)}(u_k)|^2$，取期望即得。 $\square$

---

## 4. 耦合梯度定理（近似版）

**定理 4.1**（非多项式激活的近似耦合梯度）

设 $\sigma^{(K)}(z) = \sum_{j=0}^K \tilde{a}_j z^j$ 为 Hermite 截断的单项式形式（见 Section 3.1）。对应网络 $f^{(K)}$ 的参数梯度满足：

$$\frac{\partial \mathcal{L}^{(K)}}{\partial [W_1]_{k,j}} = -2 \sum_{n=1}^{K} n\, \tilde{a}_n\, \mathbb{E}\!\left[\varepsilon^{(K)}(\mathbf{x})\, [\mathbf{w}_2]_k\, (\mathbf{u}_k^\top \mathbf{x})^{n-1}\, x_j\right]$$

其中 $\varepsilon^{(K)} = y - f^{(K)}(\mathbf{x})$。

> **与第一版的关系**：此公式与第一版定理 4.1 **结构完全相同**——只是 Taylor 系数 $a_n$ 被替换为单项式系数 $\tilde{a}_n$。这是因为 $\sigma^{(K)}$ 是 $K$ 次多项式，第一版框架原封不动适用。

**证明**：$(\sigma^{(K)})'(z) = \sum_{n=1}^K n\tilde{a}_n z^{n-1}$，链式法则直接给出（无需标准化或归一化因子）：

$$\frac{\partial f^{(K)}}{\partial [W_1]_{k,j}} = [\mathbf{w}_2]_k \cdot (\sigma^{(K)})'(\mathbf{u}_k^\top\mathbf{x}) \cdot x_j = [\mathbf{w}_2]_k \sum_{n=1}^K n\tilde{a}_n (\mathbf{u}_k^\top\mathbf{x})^{n-1} x_j$$

代入 $\partial\mathcal{L}^{(K)}/\partial [W_1]_{k,j} = -2\mathbb{E}[\varepsilon^{(K)} \cdot \partial f^{(K)}/\partial [W_1]_{k,j}]$ 即得。$\square$

> **注**：文中**不出现** $\|\mathbf{u}_k\|^{-1}$ 归一化因子。虽然 Hermite 展开在 $\mathcal{N}(0,1)$ 下定义，但 $\sigma^{(K)}$ 作为多项式直接作用于原始预激活 $\mathbf{u}_k^\top\mathbf{x}$，求导时不涉及标准化变换。

**真实梯度与近似梯度的偏差**：

梯度偏差来自两个来源：(1) 前向传播误差导致 $\varepsilon \neq \varepsilon^{(K)}$；(2) 导数近似误差 $\sigma' \neq (\sigma^{(K)})'$。总偏差依赖于 $\sigma$ 的具体形式和预激活方差 $\|\mathbf{u}_k\|^2$。

在 $\|\mathbf{u}_k\| = 1$（标准高斯预激活）的特殊情形下，对 ReLU：

$$\left\|\nabla_{W_1} \mathcal{L} - \nabla_{W_1} \mathcal{L}^{(K)}\right\|_F = O(K^{-1/4})$$

其中 $O(K^{-1/4})$ 来源于 ReLU 导数 $\sigma' = \mathbf{1}_{z>0}$ 的 Hermite 截断误差的 $L^2(\gamma)$ 收敛速率，该收敛比函数本身慢（$\sigma$ 截断误差为 $O(K^{-3/4})$，$\sigma'$ 截断误差为 $O(K^{-1/4})$）。

---

## 5. 高斯输入假设的合理性：平均场论

上述推导要求预激活 $\mathbf{u}_k^\top \mathbf{x}$ 近似高斯。对第一层，若 $\mathbf{x} \sim \mathcal{N}(0,I)$，这精确成立。对更深网络，需要**平均场理论**（Poole et al. 2016）的支持。

**定理 5.1**（宽网络的逐层高斯近似）

设网络宽度 $d \to \infty$，权重 $[W_l]_{k,j} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_w^2/d)$，偏置 $[b_l]_k \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_b^2)$。则对任意固定输入 $\mathbf{x}$，第 $l$ 层预激活 $z_k^{(l)}$ 的各分量在 $d \to \infty$ 时**边缘分布**收敛于 $\mathcal{N}(0, q^l)$（Neal 1996，GP 极限），其中方差递推为：

$$q^{l+1} = \sigma_w^2\, \mathbb{E}_{Z \sim \mathcal{N}(0, q^l)}[\sigma(Z)^2] + \sigma_b^2$$

> **引用说明**：此结果在无限宽极限下由 Neal (1996) 首先建立（将网络视为高斯过程）。Poole et al. (2016) 的贡献是在此基础上研究**信号传播**（混沌边界），而非逐层高斯性本身。Matthews et al. (2018) 对有限宽情形给出了收敛速率 $O(1/d)$。

**推论**：在宽网络（$d$ 足够大）极限下，Hermite 展开在每一层都**近似**成立（误差 $O(1/d)$），定理 4.1 逐层近似适用。

---

## 6. 统计下界（与第一版同步，无需假设激活函数）

**引理 6.0**（可辨识性）

不同阶的 Wiener–Hermite 分量在 $L^2(\gamma_p)$ 下正交：对 $j \neq k$，

$$\langle f_j, f_k \rangle_{L^2(\gamma_p)} = 0$$

这直接来自 Hermite 多项式的正交性：$\mathbb{E}[H_j(Z)H_k(Z)] = j!\,\delta_{jk}$。因此各阶分量是统计可辨识的（可通过投影唯一恢复）。

**定理 6.1**（极小极大下界）

对 $k$ 阶分量的 $L^2$ 估计误差：

$$\inf_{\hat{f}_k} \sup_{f_k^* \in \mathcal{F}_k} \mathbb{E}\!\left[\|\hat{f}_k - f_k^*\|_{L^2(\gamma_p)}^2\right] \geq \frac{\sigma_\varepsilon^2 \cdot d_k}{n}, \quad d_k = \binom{p+k-1}{k}$$

其中下确界取遍所有基于 $n$ 个样本的估计器，$\sigma_\varepsilon^2$ 为噪声方差。

> **与第一版的区别**：第一版最初使用 CRB（Cramér-Rao 下界），但 CRB 仅适用于无偏估计器。修正版改用极小极大框架：高斯位置族的极小极大风险 = 充分统计量的方差，无需假设无偏性。此处完全沿用该修正。

**推论 6.2**（学习算法无关的下界）

无论使用何种激活函数、何种优化算法，有限样本 $n$ 下第 $k$ 阶分量的估计误差均有不可消除的下界 $\Omega(p^k / (k! \cdot n))$。

---

## 7. 与第一版的对比

| 项目 | 第一版（多项式激活） | 第二版（任意 L²(γ) 激活） |
|------|-------------------|------------------------|
| Volterra 分解 | 精确等式 | L² 近似（Hermite→单项式），误差 $O(K^{-3/2})$（ReLU） |
| 耦合梯度公式 | 精确等式 | 近似，偏差 $O(K^{-1/4})$（ReLU） |
| 适用激活函数 | 多项式 | ReLU、GELU、Tanh 等 |
| 深层网络 | 直接推广 | 需要平均场近似（宽网络极限） |
| 统计下界 | $L^2(\gamma_p)$ 极小极大下界 | 相同（与激活函数无关） |
| 关键步骤 | Taylor → 单项式 | Hermite → 单项式 → v1 框架 |
| 主要额外假设 | 无 | 1) $\sigma \in L^2(\gamma)$；2) 预激活方差需控制 |

---

## 8. 开放问题

1. **有限宽网络**：平均场近似在有限 $d$ 时的误差如何？能否给出宽度 $d$ 的有限样本修正项？

2. **截断阶数的最优选择**：对给定任务精度 $\epsilon$，最优截断 $K^* = K^*(\epsilon, p, n)$ 的表达式是什么？

3. **批归一化的影响**：BatchNorm 强制每层激活接近 $\mathcal{N}(0,1)$，这应当使平均场近似更精确——能否严格量化这一改善？

4. **各阶 Hermite 系数的训练动力学**：ReLU 的 Hermite 系数 $|\hat{a}_{2m}| = \frac{1}{(2m-1)\cdot 2^m \cdot m!\cdot\sqrt{2\pi}}$ 呈超指数衰减，梯度在高阶分量上的信号更弱。这是否直接导致高阶特征交互更难学习（需要更多样本/更长训练）？

---

## 参考文献

- **第一版框架**：本系列前一篇文章（多项式激活下的精确版本）
- Poole, B. et al. (2016). *Exponential expressivity in deep neural networks through transient chaos.* NeurIPS. （信号传播与混沌边界）
- Neal, R. (1996). *Priors for infinite networks.* （宽网络高斯过程极限——逐层高斯性的原始证明）
- Matthews, A. et al. (2018). *Gaussian Process Behaviour in Wide Deep Neural Networks.* ICLR. （有限宽收敛速率 $O(1/d)$）
- Rahimi, A. & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines.* (L²(γ) 特征展开在 ML 中的应用)
- Cho, Y. & Saul, L. (2009). *Kernel Methods for Deep Learning.* (弧余弦核 = ReLU 网络的无限宽极限)
- Cohen, N. et al. (2016). *On the Expressive Power of Deep Learning: A Tensor Analysis.*
- Volterra, V. (1887). *Sopra le funzioni che dipendono da altre funzioni.*
- Widrow, B. & Hoff, M. (1960). *Adaptive Switching Circuits.* (LMS 算法)
·