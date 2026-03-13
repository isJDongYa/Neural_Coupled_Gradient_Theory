# NCGT — Neural Coupled Gradient Theory

**神经耦合梯度理论（NCGT）** 建立了神经网络参数空间梯度下降与函数空间各阶非线性核同步修正之间的精确数学对应。一步参数更新如何同时改变网络输出中的线性分量、二次分量、三次分量……直至最高阶——NCGT 给出了精确等式，并通过实验验证到 $10^{-15}$ 机器精度。

## 解决的核心问题

| 问题 | NCGT 的回答 |
|------|------------|
| 参数共享为什么高效？ | $O(dp)$ 参数一步更新同时修正指数级多个阶的核 |
| 为什么低阶特征先学会？ | 有效学习率 $\eta_{\text{eff}}(n)$ 随阶数 $n$ 衰减 |
| SGD 的正则化从何而来？ | 高阶核噪声方差超阶乘增长，等效施加更强惩罚 |
| ResNet 为什么能极深？ | 恒等路径梯度不经过中间非线性层，不随深度衰减 |
| Pre-Norm 为什么优于 Post-Norm？ | Pre-Norm 各阶解耦，Post-Norm 各阶耦合 |
| 网络最多能学到多高阶？ | $n^* = O(\log m / \log p)$，由样本量和维度决定 |

## 理论版本

NCGT 由四个逐步推广的版本构成：

| 版本 | 网络结构 | 激活函数 | 核心推进 |
|------|---------|---------|---------|
| **NCGT-I** | 两层 | 多项式 | 建立精确等式：梯度 = 各阶核误差的加权和 |
| **NCGT-II** | 两层 | 任意（ReLU 等） | Hermite→单项式转换，推广到实际激活函数 |
| **NCGT-III** | L 层 plain | 多项式 | 路径求和分解 + 各阶学习速率 + SGD 隐式正则化 |
| **NCGT-IV** | L 层 ResNet + LayerNorm | 多项式 | 残差路径计数 + 恒等路径梯度 + Pre-Norm 稳定性 + 泛化界 |

## 与主流理论的定位

| 理论 | 适用范围 | 局限 | NCGT 的不同 |
|------|---------|------|------------|
| NTK (Jacot 2018) | 无限宽极限 | 核不变，无特征学习 | 有限宽精确等式，核随训练演化 |
| 均场理论 (Mei 2018) | 无限宽粒子分布 | 不区分各阶非线性 | 精确到每一阶的独立动力学 |
| Tensor Programs (Yang 2021) | 渐近极限 | 工具而非物理图景 | 有限宽精确（非渐近） |
| 频率偏好 (Rahaman 2019) | 实验观察 | 无精确机制 | 给出精确的有效学习率公式 |

## 实验验证

每个版本均附有完整的数值验证代码，核心结果均达到 $10^{-15}$ 机器精度。

| 版本 | 实验数 | 关键验证 | 精度 |
|------|-------|---------|------|
| NCGT-I | 6 | 耦合梯度等式 | $10^{-15}$ |
| NCGT-II | 6 | Hermite 近似 + 梯度等式 | $10^{-15}$ |
| NCGT-III | 5 | 深层路径求和 + 各阶学习速率 + SGD 噪声 | $10^{-15}$ |
| NCGT-IV | 6 | ResNet 梯度分解 + 恒等路径 + Pre-Norm + 泛化 | $10^{-15}$ |

## 文件结构

### 理论文档

| 文件 | 内容 |
|------|------|
| `theory_volterra_gradient_v1.md` | NCGT-I：两层多项式网络 |
| `theory_volterra_gradient_v2.md` | NCGT-II：两层任意激活 |
| `theory_volterra_gradient_v3.md` | NCGT-III：L 层 plain 网络 |
| `theory_volterra_gradient_v4.md` | NCGT-IV：ResNet + 归一化 + 泛化 |
| `theory_volterra_gradient_v5.md` |  |
| `theory_volterra_gradient_v6.md` | |

### 验证代码（最终版本）

| 文件 | 对应理论 | 实验数 |
|------|---------|-------|
| `verify_volterra_gradient_v1.py` | NCGT-I | 6 |
| `verify_volterra_gradient_v2.py` | NCGT-II | 6 |
| `verify_volterra_gradient_v3.py` | NCGT-III | 5 |
| `verify_volterra_gradient_v4.py` | NCGT-IV | 6 |

### 可视化

| 文件 | 内容 |
|------|------|
| `volterra_v1_verification.png` | NCGT-I 实验汇总图 |
| `volterra_v2_verification.png` | NCGT-II 实验汇总图 |
| `volterra_v3_verification.png` | NCGT-III 实验汇总图 |
| `volterra_v4_verification.png` | NCGT-IV 实验汇总图 |

## 当前局限

- **注意力机制**：softmax(QK^T)V 的非线性归一化不在当前框架内
- **特征语义**：理论回答"学了几阶"，不回答"学了什么语义"
- **非多项式激活**：ReLU/GELU 需 Hermite 截断近似，非精确等式
- **定理 3.2（v4）的修正**：子网络分解 $f = \sum_S f_S$ 对非线性 σ 不成立（已在文档中修正为多项式单项式层面的路径分类）

## 运行

```bash
# 需要 PyTorch + NumPy + Matplotlib
pip install torch numpy matplotlib

# 运行某个版本的实验验证
python verify_volterra_gradient_v1.py      # NCGT-I
python verify_volterra_gradient_v2.py   # NCGT-II
python verify_volterra_gradient_v3.py   # NCGT-III
python verify_volterra_gradient_v4.py      # NCGT-IV
```

所有实验使用 `float64` 双精度计算，输出格式化的验证结果和 PASS/FAIL 判定。
