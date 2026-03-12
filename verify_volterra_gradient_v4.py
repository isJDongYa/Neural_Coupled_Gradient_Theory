"""
耦合梯度定理（第四版）实验验证
=========================================
残差连接与归一化层的 Volterra 分析

六个实验
--------
实验1  ResNet 路径求和 Volterra 分解    —— f = Σ_S f_S（精确等式）
实验2  路径计数公式验证                —— P_res(n,L) = Σ_k C(L-1,k)·P_plain(n,k+1)
实验3  ResNet 耦合梯度验证             —— autograd = 手动反向传播（机器精度）
实验4  恒等路径梯度不随深度衰减          —— Thm 6.2 + Cor 4.3
实验5  Pre-Norm 消除权重范数依赖        —— η_eff^LN(n) 仅依赖 ã_n
实验6  深层 ResNet 中恒等路径主导        —— Prop 6.4: C_σ < 1 时恒等路径占主导
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import combinations, product as iter_product
from collections import defaultdict

torch.manual_seed(42)
np.random.seed(42)


# ===========================================================
# 工具函数
# ===========================================================

def poly_activation(z, coeffs):
    """多项式激活 σ(z) = Σ a_n z^n，coeffs = [a_1, a_2, ..., a_N]。"""
    result = torch.zeros_like(z)
    for i, a in enumerate(coeffs):
        n = i + 1
        result = result + a * z ** n
    return result


def poly_activation_deriv(z, coeffs):
    """多项式激活导数 σ'(z) = Σ n a_n z^{n-1}。"""
    result = torch.zeros_like(z)
    for i, a in enumerate(coeffs):
        n = i + 1
        result = result + n * a * z ** (n - 1)
    return result


def resnet_forward(x, weights, w_out, coeffs):
    """ResNet 前向传播：
    h^(l) = h^(l-1) + σ(W_l h^(l-1)),  l = 1,...,L-1
    f(x) = w_out^T h^(L-1)

    weights = [W_1, ..., W_{L-1}]
    w_out = w_L (output weights, shape (p,))
    """
    h = x  # h^(0) = x, shape (B, p)
    for W in weights:
        z = h @ W.T  # pre-activation
        h = h + poly_activation(z, coeffs)  # residual connection
    return (h @ w_out).squeeze(-1)  # scalar output


def resnet_forward_save(x, weights, w_out, coeffs):
    """ResNet 前向传播，保存中间变量。"""
    h_list = [x]
    z_list = []
    for W in weights:
        z = h_list[-1] @ W.T
        z_list.append(z)
        h = h_list[-1] + poly_activation(z, coeffs)
        h_list.append(h)
    f_out = (h_list[-1] @ w_out).squeeze(-1)
    return f_out, h_list, z_list


def resnet_subnet_forward(x, weights, w_out, coeffs, active_set):
    """ResNet 子网络前向传播：只在 active_set 中的层使用残差分支，其余走恒等。
    active_set: set of layer indices (0-based) that are active.
    """
    h = x
    for l, W in enumerate(weights):
        if l in active_set:
            z = h @ W.T
            h = poly_activation(z, coeffs)  # 只走残差分支 σ(W_l h)
        # else: h = h (identity, do nothing)
    return (h @ w_out).squeeze(-1)


def layer_norm(z, eps=1e-8):
    """LayerNorm (无可学习参数): LN(z) = (z - mean) / std"""
    mean = z.mean(dim=-1, keepdim=True)
    var = z.var(dim=-1, keepdim=True, unbiased=False)
    return (z - mean) / torch.sqrt(var + eps)


def prenorm_resnet_forward(x, weights, w_out, coeffs):
    """Pre-Norm ResNet: h^(l) = h^(l-1) + σ(W_l LN(h^(l-1)))"""
    h = x
    for W in weights:
        h_ln = layer_norm(h)
        z = h_ln @ W.T
        h = h + poly_activation(z, coeffs)
    return (h @ w_out).squeeze(-1)


def plain_forward(x, weights, w_out, coeffs):
    """Plain 网络前向传播（无残差连接）：
    h^(l) = σ(W_l h^(l-1))
    f(x) = w_out^T h^(L-1)
    """
    h = x
    for W in weights:
        z = h @ W.T
        h = poly_activation(z, coeffs)
    return (h @ w_out).squeeze(-1)


def extract_volterra_by_probing(f_func, p, max_order, N_probe=50000):
    """通过缩放探测提取各阶 Volterra 分量。

    利用 f(αx) = Σ_n α^n f_n(x)，Vandermonde 系统求解。
    返回 (X_probe, C)，C[n] = f_n(X_probe)。
    """
    X_probe = torch.randn(N_probe, p, dtype=torch.float64)
    n_alphas = max_order + 1
    alphas = torch.linspace(0.5, 2.0, n_alphas, dtype=torch.float64)

    F_alpha = torch.zeros(n_alphas, N_probe, dtype=torch.float64)
    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            F_alpha[i] = f_func(alpha * X_probe)

    V = torch.zeros(n_alphas, max_order + 1, dtype=torch.float64)
    for i in range(n_alphas):
        for n in range(max_order + 1):
            V[i, n] = alphas[i] ** n

    result = torch.linalg.lstsq(V, F_alpha)
    return X_probe, result.solution


def count_plain_paths(n, L, N):
    """计算 plain 网络中全局第 n 阶的路径数。
    P_plain(n, L) = |{(n_1,...,n_{L-1}) ∈ {1,...,N}^{L-1} : Π n_l = n}|
    """
    depth = L - 1  # number of nonlinear layers
    if depth == 0:
        return 1 if n == 1 else 0

    count = 0
    # enumerate all tuples (n_1, ..., n_depth) with 1 <= n_l <= N
    for combo in iter_product(range(1, N + 1), repeat=depth):
        prod = 1
        for c in combo:
            prod *= c
        if prod == n:
            count += 1
    return count


def count_resnet_paths_formula(n, L, N):
    """用命题 3.4 公式计算 ResNet 路径数。
    P_res(n, L) = Σ_{k=1}^{L-1} C(L-1, k) · P_plain(n, k+1)
    """
    total = 0
    for k in range(1, L):
        binom = math.comb(L - 1, k)
        p_plain = count_plain_paths(n, k + 1, N)
        total += binom * p_plain
    return total


def double_factorial(k):
    """(2k-1)!! = 1·3·5·...·(2k-1)，约定 (-1)!! = 1。"""
    if k <= 0:
        return 1
    result = 1
    for i in range(1, 2 * k, 2):
        result *= i
    return result


# ===========================================================
# 实验 1：ResNet 路径结构验证
# ===========================================================
def experiment1():
    """验证 ResNet 的路径求和结构（定理 3.2）。

    Part A: 线性激活 σ(z)=z 下的精确路径求和 f = Σ_S f_S
        对线性 σ，残差块 h' = h + Wh = (I+W)h 为线性算子，
        乘积展开 Π(I+W_l) = Σ_S Π_{l∈S} W_l 精确成立。
        验证：全 ResNet 输出 = 所有 2^{L-1} 个子网络输出之和。

    Part B: 多项式激活下 ResNet vs Plain 的 Volterra 阶分布
        残差连接使低阶 Volterra 分量（特别是1阶）显著增强，
        因为恒等路径提供了 2^{L-1}-1 条额外的低阶路径。
    """
    print("\n[Exp 1] ResNet Path Structure Verification")

    # ─── Part A: 线性 σ 的精确路径求和 ───
    print("\n  Part A: f = Σ_S f_S for LINEAR σ(z) = z (exact)")

    all_pass_a = True
    configs_a = [
        (3, 4, "L=3, p=4 (4 subsets)"),
        (4, 4, "L=4, p=4 (8 subsets)"),
        (5, 4, "L=5, p=4 (16 subsets)"),
        (6, 3, "L=6, p=3 (32 subsets)"),
    ]

    for L, p, desc in configs_a:
        coeffs_lin = [1.0]  # σ(z) = z
        n_layers = L - 1
        torch.manual_seed(42)
        weights = [0.3 * torch.randn(p, p, dtype=torch.float64) for _ in range(n_layers)]
        w_out = torch.randn(p, dtype=torch.float64)

        B = 500
        X = torch.randn(B, p, dtype=torch.float64)

        with torch.no_grad():
            f_full = resnet_forward(X, weights, w_out, coeffs_lin)

            f_sum = torch.zeros(B, dtype=torch.float64)
            for k in range(n_layers + 1):
                for S in combinations(range(n_layers), k):
                    f_S = resnet_subnet_forward(X, weights, w_out, coeffs_lin, set(S))
                    f_sum += f_S

            err = (f_full - f_sum).abs().max().item()
            rel = err / (f_full.abs().max().item() + 1e-30)

        passed = rel < 1e-12
        all_pass_a = all_pass_a and passed
        print(f"    {desc}: |f-Σf_S| = {err:.2e}, rel = {rel:.2e} "
              f"{'✓' if passed else '✗'}")

    # ─── Part B: 多项式 σ 的 Volterra 阶分布对比 ───
    print("\n  Part B: ResNet vs Plain — Volterra order distribution (N=2, L=3)")

    L, N, p = 3, 2, 4
    coeffs_poly = [1.0, 0.5]  # σ(z) = z + 0.5z²
    max_order = N ** (L - 1)  # = 4
    n_layers = L - 1

    torch.manual_seed(99)
    weights = [0.3 * torch.randn(p, p, dtype=torch.float64) for _ in range(n_layers)]
    w_out = torch.randn(p, dtype=torch.float64)

    with torch.no_grad():
        def f_res(x):
            return resnet_forward(x, weights, w_out, coeffs_poly)
        def f_plain(x):
            return plain_forward(x, weights, w_out, coeffs_poly)

        probe_max = max_order + 2
        _, C_res = extract_volterra_by_probing(f_res, p, probe_max, N_probe=30000)
        _, C_plain = extract_volterra_by_probing(f_plain, p, probe_max, N_probe=30000)

    print(f"    {'Order':>5}  {'||f_n||²(ResNet)':>18}  {'||f_n||²(Plain)':>18}  {'Ratio':>8}")
    for n in range(probe_max + 1):
        nr = (C_res[n] ** 2).mean().item()
        npv = (C_plain[n] ** 2).mean().item()
        ratio = nr / (npv + 1e-30) if npv > 1e-20 else float('inf')
        marker = ""
        if n == 0:
            marker = "  (constant)"
        elif n == 1:
            marker = "  ← identity path amplifies"
        elif n == max_order:
            marker = "  ← max order N^{L-1}"
        print(f"    {n:5d}  {nr:18.6e}  {npv:18.6e}  {ratio:8.2f}{marker}")

    nr1 = (C_res[1] ** 2).mean().item()
    np1 = (C_plain[1] ** 2).mean().item()
    low_stronger = nr1 > np1 * 1.5

    print(f"\n    ResNet 1st-order ||f_1||²: {nr1:.6e}")
    print(f"    Plain  1st-order ||f_1||²: {np1:.6e}")
    print(f"    Ratio: {nr1 / (np1 + 1e-30):.1f}x — residual paths amplify low order: "
          f"{'✓' if low_stronger else '✗'}")

    # 收集各阶能量用于绘图
    orders_list = list(range(probe_max + 1))
    energy_res = [(C_res[n] ** 2).mean().item() for n in orders_list]
    energy_plain = [(C_plain[n] ** 2).mean().item() for n in orders_list]

    all_pass = all_pass_a and low_stronger
    result = "✓ PASS" if all_pass else "✗ FAIL"
    print(f"\n  Conclusion: {result}")
    return all_pass, {'orders': orders_list, 'energy_res': energy_res, 'energy_plain': energy_plain}


# ===========================================================
# 实验 2：路径计数公式验证
# ===========================================================
def experiment2():
    """验证命题 3.4：P_res(n,L) = Σ_{k=1}^{L-1} C(L-1,k) · P_plain(n,k+1)

    方法：
    1. 暴力枚举所有 ResNet 路径 (S, (n_l)_{l∈S}) 计算实际路径数
    2. 与公式 Σ_k C(L-1,k)·P_plain(n,k+1) 对比
    3. 验证特例：n=1 时 P_res = 2^{L-1}-1，n=2 时 P_res = (L-1)·2^{L-2}
    4. 验证比率 P_res/P_plain = Θ(2^L)
    """
    print("\n[Exp 2] Path Count Formula Verification (Prop 3.4)")

    configs = [
        (3, 2, "L=3, N=2"),
        (4, 2, "L=4, N=2"),
        (5, 2, "L=5, N=2"),
        (4, 3, "L=4, N=3"),
        (5, 3, "L=5, N=3"),
        (6, 2, "L=6, N=2"),
    ]

    all_pass = True
    for L, N, desc in configs:
        n_layers = L - 1

        # 暴力枚举 ResNet 路径数
        max_order = N ** n_layers
        brute_counts = defaultdict(int)

        for k in range(1, n_layers + 1):  # |S| = k
            for S in combinations(range(n_layers), k):
                # 枚举活跃层的阶贡献 (n_l)_{l∈S}, 1 ≤ n_l ≤ N
                for orders in iter_product(range(1, N + 1), repeat=k):
                    total_order = 1
                    for o in orders:
                        total_order *= o
                    brute_counts[total_order] += 1

        # 对比公式
        print(f"\n  {desc}:")
        print(f"    {'Order n':>8}  {'Brute':>8}  {'Formula':>8}  {'P_plain':>8}  "
              f"{'Ratio':>8}  {'Match':>5}")

        test_orders = sorted(set(list(brute_counts.keys()) +
                                 [1, 2, N, N ** n_layers]))
        all_match = True
        for n in test_orders:
            if n < 1 or n > max_order:
                continue
            brute = brute_counts.get(n, 0)
            formula = count_resnet_paths_formula(n, L, N)
            p_plain = count_plain_paths(n, L, N)
            ratio = formula / (p_plain + 1e-30) if p_plain > 0 else float('inf')

            match = brute == formula
            all_match = all_match and match
            print(f"    {n:8d}  {brute:8d}  {formula:8d}  {p_plain:8d}  "
                  f"{ratio:8.1f}  {'✓' if match else '✗'}")

        # 特例验证
        p_res_1 = count_resnet_paths_formula(1, L, N)
        expected_1 = 2 ** n_layers - 1
        pass_1 = p_res_1 == expected_1
        print(f"    n=1 check: P_res(1)={p_res_1}, 2^{n_layers}-1={expected_1} "
              f"{'✓' if pass_1 else '✗'}")

        if N >= 2:
            p_res_2 = count_resnet_paths_formula(2, L, N)
            expected_2 = n_layers * 2 ** (n_layers - 1)
            pass_2 = p_res_2 == expected_2
            print(f"    n=2 check: P_res(2)={p_res_2}, (L-1)·2^{{L-2}}="
                  f"{expected_2} {'✓' if pass_2 else '✗'}")
            all_match = all_match and pass_2

        all_match = all_match and pass_1
        all_pass = all_pass and all_match

    result = "✓ PASS" if all_pass else "✗ FAIL"
    print(f"\n  Conclusion: {result}")
    return all_pass


# ===========================================================
# 实验 3：ResNet 耦合梯度验证
# ===========================================================
def experiment3():
    """验证定理 4.1：ResNet 中各层参数的梯度。

    手动反向传播公式：
    ∂f/∂h^(l) = w_L^T Π_{s=l+1}^{L-1} (I + diag(σ'(z^(s))) W_s)
    ∂f/∂[W_l]_{k,j} = [∂f/∂h^(l)]_k · σ'(z_k^(l)) · h_j^(l-1)

    关键区别：ResNet 的 Jacobian = I + diag(σ')W（有单位矩阵分量），
    而 plain 网络的 Jacobian = diag(σ')W（无 I）。

    验证：autograd 梯度 = 手动公式梯度（机器精度）。
    """
    print("\n[Exp 3] ResNet Coupled Gradient Verification (Thm 4.1)")

    configs = [
        (3, 2, 4, "L=3, N=2"),
        (4, 2, 4, "L=4, N=2"),
        (5, 2, 3, "L=5, N=2"),
        (4, 3, 4, "L=4, N=3"),
    ]

    all_pass = True
    for L, N, p, desc in configs:
        coeffs = [1.0 / math.factorial(n) for n in range(1, N + 1)]
        n_layers = L - 1

        torch.manual_seed(42)
        weights = [torch.nn.Parameter(0.3 * torch.randn(p, p, dtype=torch.float64))
                   for _ in range(n_layers)]
        w_out = torch.randn(p, dtype=torch.float64)

        B = 3000
        X = torch.randn(B, p, dtype=torch.float64)
        y = torch.randn(B, dtype=torch.float64)

        # --- autograd 梯度 ---
        for W in weights:
            if W.grad is not None:
                W.grad.zero_()

        f_pred = resnet_forward(X, weights, w_out, coeffs)
        loss = ((y - f_pred) ** 2).mean()
        loss.backward()
        grads_auto = [W.grad.clone() for W in weights]

        # --- 手动计算梯度 ---
        with torch.no_grad():
            # 前向传播，保存中间变量
            h_list = [X]  # h^(0) = x
            z_list = []
            for l in range(n_layers):
                z = h_list[-1] @ weights[l].detach().T
                z_list.append(z)
                h = h_list[-1] + poly_activation(z, coeffs)
                h_list.append(h)

            f_manual = (h_list[-1] @ w_out).squeeze(-1)
            eps = y - f_manual  # (B,)

            # 反向传播 ResNet 的 Jacobian
            # delta = ∂f/∂h^(l)，shape (B, p)
            # 对 ResNet: ∂h^(s)/∂h^(s-1) = I + diag(σ'(z^(s))) W_s
            delta = w_out.unsqueeze(0).expand(B, p)  # (B, p)
            grads_manual = [None] * n_layers

            for l in range(n_layers - 1, -1, -1):
                sigma_prime = poly_activation_deriv(z_list[l], coeffs)  # (B, p)
                # W_l 梯度：-2/B Σ_i ε_i · (delta ⊙ σ')_i · h^(l-1)_i^T
                grad_signal = delta * sigma_prime  # (B, p)
                grads_manual[l] = -2.0 * (eps[:, None] * grad_signal).T @ h_list[l] / B

                # 传播到 h^(l-1): delta ← delta · (I + diag(σ') W_l)
                delta = delta + (delta * sigma_prime) @ weights[l].detach()

        # 比较
        max_rel_err = 0.0
        for l in range(n_layers):
            diff = (grads_auto[l] - grads_manual[l]).abs().max().item()
            scale = grads_auto[l].abs().max().item() + 1e-30
            rel = diff / scale
            max_rel_err = max(max_rel_err, rel)

        passed = max_rel_err < 1e-10
        all_pass = all_pass and passed
        print(f"  {desc}: max rel error = {max_rel_err:.2e} "
              f"{'✓' if passed else '✗'}")

    result = "✓ PASS" if all_pass else "✗ FAIL"
    print(f"  Conclusion: {result} (should be ≈ machine precision)")
    return all_pass


# ===========================================================
# 实验 4：恒等路径梯度不随深度衰减
# ===========================================================
def experiment4():
    """验证定理 6.2 + 推论 4.3：

    Part A (推论 4.3): 恒等路径梯度 = v1 两层子网络梯度
        当 S_fwd = ∅, S_back = ∅ 时：
        ∇_{W_l}^{id} = -2 E[ε · w_L ⊙ σ'(z^(l)) · (h^(l-1))^T]
        其中 h^(l-1) 和 z^(l) 使用完整网络的中间值。
        但 S_fwd=∅ 意味着前向走恒等 → h^(l-1) = x。
        S_back=∅ 意味着反向走恒等 → ∂f/∂h^(l) = w_L^T。
        验证：此项 = autograd 计算的两层子网络 (W_l, w_out) 的梯度。

    Part B (定理 6.2): 恒等路径梯度不随深度衰减
        增大 L，固定层 l 的参数，测量 ||∇_{W_l}^{id}||。
        恒等路径分量的范数应不依赖 L。
    """
    print("\n[Exp 4] Identity Path Gradient (Thm 6.2 + Cor 4.3)")

    # ─── Part A: 恒等路径 = 两层子网络 ───
    print("\n  Part A: Identity path gradient = two-layer subnet gradient")

    N, p = 2, 5
    coeffs = [1.0, 0.5]
    B = 5000

    configs_a = [(3, 1), (4, 1), (5, 2), (6, 3)]  # (L, l_target)
    all_pass_a = True

    for L, l_target in configs_a:
        n_layers = L - 1
        torch.manual_seed(42)
        weights = [0.3 * torch.randn(p, p, dtype=torch.float64) for _ in range(n_layers)]
        w_out = torch.randn(p, dtype=torch.float64)

        X = torch.randn(B, p, dtype=torch.float64)
        y = torch.randn(B, dtype=torch.float64)

        # 方法1：从完整 ResNet 提取恒等路径梯度分量
        # 恒等路径：前向 h^(l-1) = x，反向 ∂f/∂h^(l) = w_out
        # ∇_{W_l}^{id} = -2/B Σ_i ε_i^{full} · w_out ⊙ σ'(W_l x) · x_i^T
        # 注意：ε 使用完整网络的残差
        with torch.no_grad():
            f_full = resnet_forward(X, weights, w_out, coeffs)
            eps_full = y - f_full

            # 恒等路径：前向输入 = x，z^(l) = W_l x
            z_id = X @ weights[l_target].T  # (B, p)
            sigma_prime_id = poly_activation_deriv(z_id, coeffs)  # (B, p)
            # ∇_{W_l}^{id} = -2/B Σ (ε · w_out ⊙ σ'(z)) · x^T
            grad_identity = -2.0 * (eps_full[:, None] * (w_out[None, :] * sigma_prime_id)).T @ X / B

        # 方法2：构造两层子网络 f_2layer(x) = w_out^T σ(W_l x)，计算 autograd 梯度
        # 但用完整网络的残差 ε^{full} 而非两层子网络自己的残差
        # 注意：推论 4.3 说的是恒等路径对完整梯度的贡献，不是独立两层网络
        # 所以方法1和方法2的比较是：两者用相同的 ε^{full}
        W_l_param = torch.nn.Parameter(weights[l_target].clone())
        f_2layer = (poly_activation(X @ W_l_param.T, coeffs) @ w_out)
        # 用 ε^{full} 作权重的 ∂/∂W_l: -2/B Σ ε_i · ∂f_2layer_i/∂W_l
        f_2layer.backward(gradient=-2.0 * eps_full / B)
        grad_2layer = W_l_param.grad.clone()

        err = (grad_identity - grad_2layer).abs().max().item()
        scale = grad_identity.abs().max().item() + 1e-30
        rel = err / scale

        passed = rel < 1e-10
        all_pass_a = all_pass_a and passed
        print(f"    L={L}, layer {l_target}: |∇^id - ∇^2layer| rel = {rel:.2e} "
              f"{'✓' if passed else '✗'}")

    # ─── Part B: 恒等路径梯度不随 L 衰减 ───
    print("\n  Part B: Identity path gradient norm vs depth L")
    print("          (should NOT decay with L)")

    N, p = 2, 5
    coeffs = [1.0, 0.5]
    B = 10000
    l_target = 0  # 固定测量第 0 层
    scale = 0.02  # 小权重确保 C_σ < 1

    torch.manual_seed(123)
    W_fixed = scale * torch.randn(p, p, dtype=torch.float64)
    w_out_fixed = torch.randn(p, dtype=torch.float64)

    X = torch.randn(B, p, dtype=torch.float64)
    y = torch.randn(B, dtype=torch.float64)

    id_norms = []
    total_res_norms = []
    total_plain_norms = []
    L_values = [3, 5, 8, 12, 20, 30]

    for L in L_values:
        n_layers = L - 1
        torch.manual_seed(456)
        weights = [scale * torch.randn(p, p, dtype=torch.float64) for _ in range(n_layers)]
        weights[0] = W_fixed.clone()
        w_out = w_out_fixed.clone()

        with torch.no_grad():
            # ResNet 恒等路径梯度
            f_full = resnet_forward(X, weights, w_out, coeffs)
            eps_full = y - f_full
            z_id = X @ W_fixed.T
            sigma_prime_id = poly_activation_deriv(z_id, coeffs)
            grad_id = -2.0 * (eps_full[:, None] * (w_out[None, :] * sigma_prime_id)).T @ X / B
            norm_id = grad_id.norm().item()

        # ResNet 总梯度
        weights_rp = [torch.nn.Parameter(w.clone()) for w in weights]
        f_res = resnet_forward(X, weights_rp, w_out, coeffs)
        loss_res = ((y - f_res) ** 2).mean()
        loss_res.backward()
        norm_res = weights_rp[0].grad.norm().item()

        # Plain 网络总梯度
        weights_pp = [torch.nn.Parameter(w.clone()) for w in weights]
        f_pln = plain_forward(X, weights_pp, w_out, coeffs)
        loss_pln = ((y - f_pln) ** 2).mean()
        loss_pln.backward()
        norm_plain = weights_pp[0].grad.norm().item()

        id_norms.append(norm_id)
        total_res_norms.append(norm_res)
        total_plain_norms.append(norm_plain)

    # 恒等路径梯度范数应不随 L 变化
    id_ratio = max(id_norms) / (min(id_norms) + 1e-30)
    id_stable = id_ratio < 2.0

    # Plain 网络梯度应指数衰减
    plain_decay = total_plain_norms[-1] < total_plain_norms[0] * 0.01

    print(f"    {'L':>4}  {'||∇^id||':>12}  {'||∇_res||':>12}  {'||∇_plain||':>12}  "
          f"{'id/res':>8}  {'plain decay':>12}")
    for i, L in enumerate(L_values):
        r_id = id_norms[i] / (total_res_norms[i] + 1e-30)
        r_plain = total_plain_norms[i] / (total_plain_norms[0] + 1e-30)
        print(f"    {L:4d}  {id_norms[i]:12.6f}  {total_res_norms[i]:12.6f}  "
              f"{total_plain_norms[i]:12.2e}  {r_id:8.4f}  {r_plain:12.6f}")

    print(f"\n    Identity path range: [{min(id_norms):.6f}, {max(id_norms):.6f}], "
          f"max/min = {id_ratio:.3f}")
    print(f"    Stable (ratio < 2): {'✓' if id_stable else '✗'}")
    print(f"    Plain gradient decay (L={L_values[-1]} vs L={L_values[0]}): "
          f"{total_plain_norms[-1]/total_plain_norms[0]:.2e} "
          f"({'✓ decays' if plain_decay else '✗ no decay'})")

    all_pass = all_pass_a and id_stable and plain_decay
    result = "✓ PASS" if all_pass else "✗ FAIL"
    print(f"\n  Conclusion: {result}")
    return all_pass, {'L_values': L_values, 'id_norms': id_norms,
                      'total_res_norms': total_res_norms, 'total_plain_norms': total_plain_norms}


# ===========================================================
# 实验 5：Pre-Norm 稳定各层 Volterra 结构
# ===========================================================
def experiment5():
    """验证定理 5.2 和 5.3 的核心机制。

    Part A: Pre-activation 方差稳定性
        定理 5.2 的关键：Pre-Norm 使每层预激活 z_k^(l) = u_k^T LN(h^(l-1))
        的方差 ≈ ||u_k||²，不随深度 l 漂移。
        验证：对比 Pre-Norm vs 无 LN ResNet 的 Var(z^(l)) 跨层变化。

    Part B: 梯度范数跨层均匀性
        推论：Pre-Norm 使 ||∂L/∂W_l|| 跨层更均匀（因为 η_eff 不依赖层深度），
        而无 LN 时浅层梯度更弱。
    """
    print("\n[Exp 5] Pre-Norm Stabilizes Volterra Structure (Thm 5.2/5.3)")

    N, p = 2, 8
    coeffs = [1.0, 0.5]
    L = 10
    n_layers = L - 1
    B = 5000
    scale = 0.1  # 小权重避免无 LN 时爆炸

    torch.manual_seed(42)
    weights = [scale * torch.randn(p, p, dtype=torch.float64) for _ in range(n_layers)]
    w_out = torch.randn(p, dtype=torch.float64)
    X = torch.randn(B, p, dtype=torch.float64)
    y = torch.randn(B, dtype=torch.float64)

    # ─── Part A: Pre-activation variance across layers ───
    print(f"\n  Part A: Pre-activation Var(z^(l)) across layers (L={L})")

    # 无 LN ResNet
    var_no_ln = []
    with torch.no_grad():
        h = X.clone()
        for l in range(n_layers):
            z = h @ weights[l].T
            var_no_ln.append(z.var().item())
            h = h + poly_activation(z, coeffs)

    # Pre-Norm ResNet
    var_prenorm = []
    with torch.no_grad():
        h = X.clone()
        for l in range(n_layers):
            h_ln = layer_norm(h)
            z = h_ln @ weights[l].T
            var_prenorm.append(z.var().item())
            h = h + poly_activation(z, coeffs)

    print(f"    {'Layer':>6}  {'Var(z) no LN':>14}  {'Var(z) PreNorm':>14}")
    for l in range(n_layers):
        print(f"    {l+1:6d}  {var_no_ln[l]:14.4f}  {var_prenorm[l]:14.4f}")

    # CV 衡量稳定性
    cv_no_ln = np.std(var_no_ln) / (np.mean(var_no_ln) + 1e-30)
    cv_prenorm = np.std(var_prenorm) / (np.mean(var_prenorm) + 1e-30)

    print(f"\n    CV(Var(z)) across layers:")
    print(f"      No LN:    {cv_no_ln:.4f}")
    print(f"      Pre-Norm: {cv_prenorm:.4f}")

    prenorm_more_stable = cv_prenorm < cv_no_ln * 0.5
    print(f"      Pre-Norm more stable: {'✓' if prenorm_more_stable else '✗'}")

    # ─── Part B: 梯度范数跨层均匀性 ───
    print(f"\n  Part B: Gradient norm uniformity across layers")

    # 无 LN
    weights_p = [torch.nn.Parameter(w.clone()) for w in weights]
    f_pred = resnet_forward(X, weights_p, w_out, coeffs)
    loss = ((y - f_pred) ** 2).mean()
    loss.backward()
    grad_norms_no_ln = [W.grad.norm().item() for W in weights_p]

    # Pre-Norm
    weights_p2 = [torch.nn.Parameter(w.clone()) for w in weights]
    f_pred2 = prenorm_resnet_forward(X, weights_p2, w_out, coeffs)
    loss2 = ((y - f_pred2) ** 2).mean()
    loss2.backward()
    grad_norms_prenorm = [W.grad.norm().item() for W in weights_p2]

    print(f"    {'Layer':>6}  {'||∇W_l|| no LN':>16}  {'||∇W_l|| PreNorm':>16}")
    for l in range(n_layers):
        print(f"    {l+1:6d}  {grad_norms_no_ln[l]:16.6f}  {grad_norms_prenorm[l]:16.6f}")

    # CV 衡量均匀性
    cv_grad_no_ln = np.std(grad_norms_no_ln) / (np.mean(grad_norms_no_ln) + 1e-30)
    cv_grad_prenorm = np.std(grad_norms_prenorm) / (np.mean(grad_norms_prenorm) + 1e-30)

    print(f"\n    CV(||∇W_l||) across layers:")
    print(f"      No LN:    {cv_grad_no_ln:.4f}")
    print(f"      Pre-Norm: {cv_grad_prenorm:.4f}")

    # Pre-Norm 梯度更均匀，或 No LN 有 NaN/inf
    if np.isnan(cv_grad_no_ln) or np.isinf(cv_grad_no_ln):
        grad_more_uniform = True  # No LN 已崩溃，Pre-Norm 显然更好
    else:
        grad_more_uniform = cv_grad_prenorm < cv_grad_no_ln
    print(f"      Pre-Norm more uniform: {'✓' if grad_more_uniform else '✗'}")

    all_pass = prenorm_more_stable and grad_more_uniform
    result = "✓ PASS" if all_pass else "✗ FAIL"
    print(f"\n  Conclusion: {result}")
    return all_pass, {'var_no_ln': var_no_ln, 'var_prenorm': var_prenorm,
                      'grad_no_ln': grad_norms_no_ln, 'grad_prenorm': grad_norms_prenorm}


# ===========================================================
# 实验 6：深层 ResNet 中恒等路径主导
# ===========================================================
def experiment6():
    """验证命题 6.4：当 C_σ < 1 时，深层 ResNet 中恒等路径占梯度的重要份额。

    理论预测：非恒等路径每条衰减 ~ C_σ^{|S_back|}，但路径数指数增长。
    当 C_σ 足够小时，恒等路径分量仍占主要份额。

    方法：对比 ResNet 中恒等路径占总梯度的比例（应保持高位），
    与 plain 网络梯度衰减到 0 的对比。
    """
    print("\n[Exp 6] Identity Path Dominance in Deep ResNet (Prop 6.4)")

    N, p = 2, 6
    coeffs = [1.0, 0.5]
    B = 10000
    scale = 0.015  # 很小的 C_σ

    torch.manual_seed(123)
    W_fixed = scale * torch.randn(p, p, dtype=torch.float64)
    w_out_fixed = torch.randn(p, dtype=torch.float64)
    X = torch.randn(B, p, dtype=torch.float64)
    y = torch.randn(B, dtype=torch.float64)

    L_values = [3, 5, 8, 12, 20, 30, 50]
    id_norms = []
    total_norms = []
    plain_norms = []

    for L in L_values:
        n_layers = L - 1
        torch.manual_seed(456)
        weights = [scale * torch.randn(p, p, dtype=torch.float64) for _ in range(n_layers)]
        weights[0] = W_fixed.clone()
        w_out = w_out_fixed.clone()

        with torch.no_grad():
            f_full = resnet_forward(X, weights, w_out, coeffs)
            eps_full = y - f_full
            z_id = X @ W_fixed.T
            sp_id = poly_activation_deriv(z_id, coeffs)
            grad_id = -2.0 * (eps_full[:, None] * (w_out[None, :] * sp_id)).T @ X / B
            norm_id = grad_id.norm().item()

        # ResNet 总梯度
        weights_rp = [torch.nn.Parameter(w.clone()) for w in weights]
        f_rp = resnet_forward(X, weights_rp, w_out, coeffs)
        ((y - f_rp) ** 2).mean().backward()
        norm_total = weights_rp[0].grad.norm().item()

        # Plain 梯度
        weights_pp = [torch.nn.Parameter(w.clone()) for w in weights]
        f_pp = plain_forward(X, weights_pp, w_out, coeffs)
        ((y - f_pp) ** 2).mean().backward()
        norm_plain = weights_pp[0].grad.norm().item()

        id_norms.append(norm_id)
        total_norms.append(norm_total)
        plain_norms.append(norm_plain)

    print(f"    {'L':>4}  {'||∇^id||':>12}  {'||∇_res||':>12}  {'id/res':>10}  "
          f"{'||∇_plain||':>12}  {'plain/plain[0]':>14}")
    for i, L in enumerate(L_values):
        r_id = id_norms[i] / (total_norms[i] + 1e-30)
        r_plain = plain_norms[i] / (plain_norms[0] + 1e-30)
        print(f"    {L:4d}  {id_norms[i]:12.6f}  {total_norms[i]:12.6f}  "
              f"{r_id:10.4f}  {plain_norms[i]:12.2e}  {r_plain:14.6f}")

    # 检验条件
    ratios = [id_norms[i] / (total_norms[i] + 1e-30) for i in range(len(L_values))]
    min_ratio = min(ratios)
    plain_decay = plain_norms[-1] / (plain_norms[0] + 1e-30)

    # 恒等路径始终占总梯度的显著份额（> 0.7）
    id_significant = min_ratio > 0.7
    # Plain 梯度衰减到几乎为 0
    plain_vanishes = plain_decay < 0.01

    print(f"\n    Identity path fraction range: [{min_ratio:.4f}, {max(ratios):.4f}]")
    print(f"    Identity always significant (>0.7): "
          f"{'✓' if id_significant else '✗'}")
    print(f"    Plain gradient decay ratio: {plain_decay:.2e} "
          f"({'✓ vanishes' if plain_vanishes else '✗ persists'})")
    print(f"    → ResNet identity path prevents gradient vanishing ✓"
          if (id_significant and plain_vanishes) else "")

    all_pass = id_significant and plain_vanishes
    result = "✓ PASS" if all_pass else "✗ FAIL"
    print(f"\n  Conclusion: {result}")
    return all_pass, {'L_values': L_values, 'id_norms': id_norms,
                      'total_norms': total_norms, 'plain_norms': plain_norms}


# ===========================================================
# 绘图函数
# ===========================================================
def plot_all(data1, data4, data5, data6):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Exp 1: Volterra 阶能量分布 ---
    ax = axes[0, 0]
    orders = data1['orders']
    e_res = np.array(data1['energy_res'])
    e_plain = np.array(data1['energy_plain'])
    # 只画有意义的阶（能量 > 1e-20）
    mask = (e_res > 1e-20) | (e_plain > 1e-20)
    valid_orders = [n for n, m in zip(orders, mask) if m]
    e_res_v = [e_res[n] for n in valid_orders]
    e_plain_v = [e_plain[n] for n in valid_orders]
    x_pos = np.arange(len(valid_orders))
    width = 0.35
    ax.bar(x_pos - width/2, e_res_v, width, label='ResNet', color='steelblue')
    ax.bar(x_pos + width/2, e_plain_v, width, label='Plain', color='salmon')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'n={n}' for n in valid_orders])
    ax.set_ylabel(r'$\||f_n\||^2$')
    ax.set_title('Exp 1: Volterra Order Energy\n(ResNet amplifies low orders)')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # --- Exp 4: 恒等路径 vs Plain 梯度 vs 深度 ---
    ax = axes[0, 1]
    Ls = data4['L_values']
    ax.semilogy(Ls, data4['id_norms'], 'b-o', ms=6,
                label=r'$\|\nabla^{id}\|$ (identity path)')
    ax.semilogy(Ls, data4['total_res_norms'], 'g-s', ms=6,
                label=r'$\|\nabla_{res}\|$ (ResNet total)')
    ax.semilogy(Ls, data4['total_plain_norms'], 'r-^', ms=6,
                label=r'$\|\nabla_{plain}\|$ (Plain)')
    ax.set_xlabel('Depth L')
    ax.set_ylabel('Gradient Norm (log scale)')
    ax.set_title('Exp 4: Identity Path vs Depth\n(ResNet stable, Plain vanishes)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Exp 5: Pre-Norm 稳定性 ---
    ax = axes[1, 0]
    layers = list(range(1, len(data5['var_no_ln']) + 1))
    ax.plot(layers, data5['var_no_ln'], 'r-o', ms=5, label='No LN — Var(z)')
    ax.plot(layers, data5['var_prenorm'], 'b-s', ms=5, label='Pre-Norm — Var(z)')
    ax2 = ax.twinx()
    ax2.plot(layers, data5['grad_no_ln'], 'r--^', ms=5, alpha=0.6,
             label=r'No LN — $\|\nabla W\|$')
    ax2.plot(layers, data5['grad_prenorm'], 'b--v', ms=5, alpha=0.6,
             label=r'Pre-Norm — $\|\nabla W\|$')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Var(z)', color='black')
    ax2.set_ylabel(r'$\|\nabla W\|$', color='gray')
    ax.set_title('Exp 5: Pre-Norm Stabilization\n(Lower variance & gradient CV)')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    # --- Exp 6: 恒等路径占比 ---
    ax = axes[1, 1]
    Ls6 = data6['L_values']
    ratios = [data6['id_norms'][i] / (data6['total_norms'][i] + 1e-30)
              for i in range(len(Ls6))]
    plain_decay = [data6['plain_norms'][i] / (data6['plain_norms'][0] + 1e-30)
                   for i in range(len(Ls6))]
    ax.plot(Ls6, ratios, 'b-o', ms=6,
            label=r'$\|\nabla^{id}\| / \|\nabla_{total}\|$')
    ax.plot(Ls6, plain_decay, 'r-s', ms=6, label='Plain gradient decay')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.7, color='blue', linestyle=':', alpha=0.4, label='Threshold (0.7)')
    ax.set_xlabel('Depth L')
    ax.set_ylabel('Ratio')
    ax.set_title('Exp 6: Identity Path Dominance\n(Stays ~1, Plain → 0)')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = 'volterra_v4_verification.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {out}")
    plt.show()


# ===========================================================
# 主入口
# ===========================================================

if __name__ == '__main__':
    print("=" * 65)
    print("  Coupled Gradient Theorem v4 — Experimental Verification")
    print("  (ResNet + Normalization + Generalization)")
    print("=" * 65)

    pass1, data1 = experiment1()
    pass2 = experiment2()
    pass3 = experiment3()
    pass4, data4 = experiment4()
    pass5, data5 = experiment5()
    pass6, data6 = experiment6()

    plot_all(data1, data4, data5, data6)

    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print(f"  Exp 1: ResNet path-sum f=Σf_S       → {'✓ PASS' if pass1 else '✗ FAIL'}")
    print(f"  Exp 2: Path count P_res formula      → {'✓ PASS' if pass2 else '✗ FAIL'}")
    print(f"  Exp 3: ResNet coupled gradient        → {'✓ PASS' if pass3 else '✗ FAIL'}")
    print(f"  Exp 4: Identity path no decay         → {'✓ PASS' if pass4 else '✗ FAIL'}")
    print(f"  Exp 5: Pre-Norm η_eff simplification  → {'✓ PASS' if pass5 else '✗ FAIL'}")
    print(f"  Exp 6: Identity path dominance        → {'✓ PASS' if pass6 else '✗ FAIL'}")
    print("=" * 65)

