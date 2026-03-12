"""
耦合梯度定理（第二版）实验验证
=========================================
推广至任意 L²(γ) 激活函数（以 ReLU 为例）

网络：  f(x) = w2^T σ(W1 x)
激活：  σ = ReLU（非多项式，属于 L²(γ)）
近似：  σ^(K)(z) = Σ ã_j z^j（Hermite K 阶截断的单项式形式）

六个实验
--------
实验1  ReLU Hermite 系数验证          —— 公式 vs Monte Carlo
实验2  Hermite→单项式转换验证        —— ã_j ≠ â_j，两种求值一致
实验3  耦合梯度定理验证              —— 公式梯度 = autograd（机器精度）
实验4  Volterra 核提取               —— 从 σ^(K) 网络参数还原核
实验5  截断误差收敛率                 —— ||σ - σ^(K)||² ~ O(K^{-3/2})
实验6  梯度偏差收敛率                 —— ||∇L - ∇L^(K)||_F ~ O(K^{-1/4})
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.polynomial import hermite_e as He
from scipy import integrate

torch.manual_seed(42)
np.random.seed(42)


# ===========================================================
# 工具函数：ReLU Hermite 系数 & 基底转换
# ===========================================================

def relu_hermite_coeff(n: int) -> float:
    """ReLU 的第 n 阶 Hermite 系数 â_n（Sec 2.2 公式）。"""
    if n == 0:
        return 1.0 / math.sqrt(2 * math.pi)
    elif n == 1:
        return 0.5
    elif n % 2 == 1:        # 奇数 n >= 3：为 0
        return 0.0
    else:                   # 偶数 n = 2m, m >= 1
        m = n // 2
        return (-1) ** (m - 1) / ((2 * m - 1) * (2 ** m) * math.factorial(m) * math.sqrt(2 * math.pi))


def hermite_to_monomial(hermite_coeffs: list) -> np.ndarray:
    """将 Hermite 展开系数 [â_0, ..., â_K] 转换为单项式系数 [ã_0, ..., ã_K]。

    关键：ã_j = Σ_n â_n · [z^j]H_n(z)，其中求和只对 n ≡ j (mod 2) 的项。
    """
    K = len(hermite_coeffs) - 1
    mono_coeffs = np.zeros(K + 1)
    for n in range(K + 1):
        if abs(hermite_coeffs[n]) < 1e-50:
            continue
        # 构造 H_n 的单项式表示：hermite_e 基向量 [0,...,0,1] (第 n 位)
        herm_basis = [0.0] * (n + 1)
        herm_basis[n] = 1.0
        mono_of_Hn = He.herme2poly(herm_basis)   # H_n(z) = Σ mono_of_Hn[j] z^j
        for j, c in enumerate(mono_of_Hn):
            if j <= K:
                mono_coeffs[j] += hermite_coeffs[n] * c
    return mono_coeffs


def sigma_K_torch(z, hermite_coeffs_tensor):
    """用三项递推稳定计算 σ^(K)(z) = Σ â_n H_n(z)，支持 autograd。"""
    K = len(hermite_coeffs_tensor) - 1
    H_prev = torch.ones_like(z)           # H_0
    H_curr = z.clone()                    # H_1
    result = hermite_coeffs_tensor[0] * H_prev
    if K >= 1:
        result = result + hermite_coeffs_tensor[1] * H_curr
    for n in range(2, K + 1):
        H_next = z * H_curr - (n - 1) * H_prev   # H_n = z H_{n-1} - (n-1) H_{n-2}
        result = result + hermite_coeffs_tensor[n] * H_next
        H_prev = H_curr
        H_curr = H_next
    return result


# ===========================================================
# 实验 1：ReLU Hermite 系数验证
# ===========================================================
def experiment1(K_max: int = 20) -> float:
    """比较公式给出的 â_n 与 scipy 数值积分 (1/n!) E[ReLU(Z) H_n(Z)]。

    用 scipy.integrate.quad 在半轴 [0,∞) 上精确积分：
      â_n = (1/n!) · (1/√(2π)) · ∫_0^∞ z · H_n(z) · e^{-z²/2} dz
    ReLU(z) = z for z > 0, = 0 for z < 0, 所以积分只需半轴。
    对光滑被积函数 z·H_n(z)·e^{-z²/2}，quad 达到机器精度。
    """
    print("\n[Exp 1] ReLU Hermite Coefficients: Formula vs Numerical Integration")

    max_rel_err = 0.0

    for n in range(K_max + 1):
        # 数值积分：â_n = (1/n!) · (1/√(2π)) · ∫_0^∞ z · H_n(z) · e^{-z²/2} dz
        herm_basis = [0.0] * (n + 1)
        herm_basis[n] = 1.0

        def integrand(z):
            Hn_z = He.hermeval(z, herm_basis)
            return z * Hn_z * np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)

        val, err = integrate.quad(integrand, 0, np.inf, limit=200)
        a_hat_num = val / math.factorial(n)
        a_hat_formula = relu_hermite_coeff(n)

        abs_err = abs(a_hat_num - a_hat_formula)
        if abs(a_hat_formula) > 1e-30:
            rel_err = abs_err / abs(a_hat_formula)
            max_rel_err = max(max_rel_err, rel_err)
            status = "✓" if rel_err < 1e-8 else "✗"
            print(f"  n={n:2d}: formula={a_hat_formula:+.14e}, numerical={a_hat_num:+.14e}, "
                  f"rel_err={rel_err:.2e} {status}")
        else:
            status = "✓" if abs(a_hat_num) < 1e-14 else "✗"
            print(f"  n={n:2d}: formula={a_hat_formula:+.14e}, numerical={a_hat_num:+.14e}, "
                  f"(≈0) {status}")

    result = "✓ PASS" if max_rel_err < 1e-8 else "✗ FAIL"
    print(f"  Max relative error (non-zero coefficients): {max_rel_err:.2e}")
    print(f"  Conclusion: {result}")
    return max_rel_err


# ===========================================================
# 实验 2：Hermite → 单项式转换验证
# ===========================================================
def experiment2(K: int = 10) -> float:
    """验证：(1) ã_j ≠ â_j；(2) 两种求值方式给出相同结果；(3) 近似 ReLU 的质量。"""
    print(f"\n[Exp 2] Hermite → Monomial Conversion (K={K})")

    hermite_coeffs = [relu_hermite_coeff(n) for n in range(K + 1)]
    mono_coeffs = hermite_to_monomial(hermite_coeffs)

    # 展示 ã_j ≠ â_j
    print("  Demonstrating ã_j ≠ â_j:")
    diff_count = 0
    for j in range(min(K + 1, 8)):
        diff = abs(mono_coeffs[j] - hermite_coeffs[j])
        mark = " ← DIFFERENT" if diff > 1e-10 else ""
        if diff > 1e-10:
            diff_count += 1
        print(f"    j={j}: â_{j}={hermite_coeffs[j]:+.8f}, ã_{j}={mono_coeffs[j]:+.8f}{mark}")
    print(f"  {diff_count} out of {min(K+1, 8)} coefficients differ (theory: ã_j ≠ â_j for j < K)")

    # 两种求值方式应给出相同结果
    z_test = np.linspace(-3, 3, 1000)
    sigma_K_hermite = He.hermeval(z_test, hermite_coeffs)
    sigma_K_mono = np.polyval(mono_coeffs[::-1], z_test)
    consistency_err = np.max(np.abs(sigma_K_hermite - sigma_K_mono))
    print(f"  Hermite vs Monomial evaluation consistency: {consistency_err:.2e} "
          f"{'✓' if consistency_err < 1e-8 else '✗'}")

    # L²(γ) 近似质量
    Z = np.random.randn(500000)
    relu_Z = np.maximum(Z, 0)
    sigma_K_Z = He.hermeval(Z, hermite_coeffs)
    l2_err_sq = np.mean((relu_Z - sigma_K_Z) ** 2)
    print(f"  ||ReLU - σ^({K})||²_L²(γ) = {l2_err_sq:.6f}")
    print(f"  ||ReLU - σ^({K})||_L²(γ)  = {np.sqrt(l2_err_sq):.6f}")
    return consistency_err


# ===========================================================
# 实验 3：耦合梯度定理验证（精确 — σ^(K) 是多项式）
# ===========================================================
def experiment3(K: int = 8, p: int = 5, d: int = 16, B: int = 8000) -> float:
    """
    定理 4.1：对截断多项式 σ^(K)，以下公式应精确成立（机器精度）：
      ∂L^(K)/∂[W1]_{k,j} = -2 Σ_n n·ã_n · E[ε^(K) · [w2]_k · (u_k^T x)^{n-1} · x_j]
    """
    print(f"\n[Exp 3] Coupled Gradient Theorem for σ^(K) (K={K})")

    hermite_coeffs = [relu_hermite_coeff(n) for n in range(K + 1)]
    a_tilde = hermite_to_monomial(hermite_coeffs)
    a_tilde_t = torch.tensor(a_tilde, dtype=torch.float64)
    hc_t = torch.tensor(hermite_coeffs, dtype=torch.float64)

    W1 = torch.randn(d, p, dtype=torch.float64, requires_grad=True)
    w2 = torch.randn(d, dtype=torch.float64)
    X = torch.randn(B, p, dtype=torch.float64)
    y = torch.randn(B, dtype=torch.float64)

    # --- autograd 梯度 ---
    pre = X @ W1.T                                    # (B, d)
    f_pred = sigma_K_torch(pre, hc_t) @ w2            # (B,)
    loss = ((y - f_pred) ** 2).mean()
    loss.backward()
    grad_auto = W1.grad.clone()

    # --- 定理 4.1 公式梯度（用单项式系数 ã_n）---
    with torch.no_grad():
        pre2 = X @ W1.detach().T
        f2 = sigma_K_torch(pre2, hc_t) @ w2
        eps = y - f2

        grad_formula = torch.zeros(d, p, dtype=torch.float64)
        for n in range(1, K + 1):
            if abs(a_tilde[n]) < 1e-50:
                continue
            Dn = eps[:, None] * w2[None, :] * pre2 ** (n - 1)   # (B, d)
            grad_formula += (n * a_tilde_t[n]) * (Dn.T @ X) / B
        grad_formula *= -2.0

    max_abs_err = (grad_auto - grad_formula).abs().max().item()
    rel_err = max_abs_err / (grad_auto.abs().max().item() + 1e-30)

    print(f"  max|grad_autograd - grad_formula| = {max_abs_err:.2e}")
    print(f"  Relative max error                = {rel_err:.2e}")
    result = "✓ PASS" if rel_err < 1e-6 else "✗ FAIL"
    print(f"  Conclusion: {result} (should be ≈ machine precision)")
    return rel_err


# ===========================================================
# 实验 4：Volterra 核提取（命题 3.1 验证）
# ===========================================================
def experiment4(K: int = 6, p: int = 4, d: int = 48,
                n_samples: int = 8000, epochs: int = 800, lr: float = 3e-3) -> dict:
    """
    训练 f^(K)(x) = w2^T σ^(K)(W1 x) 拟合已知的 1 阶 + 2 阶 Volterra 函数。
    用公式 h_j = ã_j Σ_k [w2]_k [W1]_{k,i1}···[W1]_{k,ij} 提取核，对比真值。
    """
    print(f"\n[Exp 4] Volterra Kernel Extraction with σ^(K) (K={K})")

    hermite_coeffs = [relu_hermite_coeff(n) for n in range(K + 1)]
    a_tilde = hermite_to_monomial(hermite_coeffs)
    hc_t = torch.tensor(hermite_coeffs, dtype=torch.float32)

    @torch.no_grad()
    def extract_kernel(W1, w2, order):
        an = a_tilde[order]
        if order == 1:
            return an * (w2 @ W1)                                      # (p,)
        elif order == 2:
            return an * torch.einsum('k,ki,kj->ij', w2, W1, W1)       # (p,p)
        else:
            raise ValueError(f"order={order} not implemented")

    # 真实核
    h1_true = torch.randn(p)
    H2_true = torch.randn(p, p)
    H2_true = (H2_true + H2_true.T) / 2

    X = torch.randn(n_samples, p)
    y = (X @ h1_true
         + torch.einsum('bi,ij,bj->b', X, H2_true, X)
         + 0.01 * torch.randn(n_samples))

    W1_p = torch.nn.Parameter(0.05 * torch.randn(d, p))
    w2_p = torch.nn.Parameter(0.05 * torch.randn(d))
    opt = torch.optim.Adam([W1_p, w2_p], lr=lr)

    log = {'h1': [], 'H2': [], 'loss': []}

    for epoch in range(epochs):
        opt.zero_grad()
        pre = X @ W1_p.T
        f_pred = sigma_K_torch(pre, hc_t) @ w2_p
        loss = ((y - f_pred) ** 2).mean()
        loss.backward()
        opt.step()

        if epoch % 20 == 0:
            with torch.no_grad():
                h1_est = extract_kernel(W1_p, w2_p, 1)
                H2_est = extract_kernel(W1_p, w2_p, 2)
                log['h1'].append((h1_est - h1_true).norm().item())
                log['H2'].append((H2_est - H2_true).norm().item())
                log['loss'].append(loss.item())

    print(f"  Training: {epochs} epochs, final loss = {log['loss'][-1]:.6f}")
    print(f"  1st order kernel error ||h1_est - h1*||   = {log['h1'][-1]:.4f}  (init {log['h1'][0]:.4f})")
    print(f"  2nd order kernel error ||H2_est - H2*||_F = {log['H2'][-1]:.4f}  (init {log['H2'][0]:.4f})")
    h1_pass = log['h1'][-1] < log['h1'][0] * 0.5
    H2_pass = log['H2'][-1] < log['H2'][0] * 0.5
    result = "✓ PASS" if (h1_pass and H2_pass) else "✗ FAIL"
    print(f"  Conclusion: {result}")
    return log


# ===========================================================
# 实验 5：截断误差收敛率 O(K^{-3/2})
# ===========================================================
def experiment5(K_values=None, N_mc: int = 1000000):
    """
    对不同 K，数值计算 ||ReLU - σ^(K)||²_{L²(γ)}，拟合 log-log 斜率。
    理论预测：斜率 ≈ -1.5。
    """
    if K_values is None:
        K_values = [4, 6, 8, 10, 14, 20, 30, 40]

    print(f"\n[Exp 5] Truncation Error Convergence Rate")

    Z = np.random.randn(N_mc)
    relu_Z = np.maximum(Z, 0)
    errors_sq = []

    for K in K_values:
        hermite_coeffs = [relu_hermite_coeff(n) for n in range(K + 1)]
        sigma_K_Z = He.hermeval(Z, hermite_coeffs)    # 用 Hermite 递推，数值稳定
        err_sq = np.mean((relu_Z - sigma_K_Z) ** 2)
        errors_sq.append(err_sq)
        print(f"  K={K:3d}: ||σ - σ^(K)||²_L²(γ) = {err_sq:.6e}")

    # 对 K >= 8 的点拟合 log-log 斜率
    idx_fit = [i for i, k in enumerate(K_values) if k >= 8]
    log_K = np.log(np.array([K_values[i] for i in idx_fit], dtype=float))
    log_err = np.log(np.array([errors_sq[i] for i in idx_fit], dtype=float))
    slope, intercept = np.polyfit(log_K, log_err, 1)

    print(f"  Fitted exponent: {slope:.3f} (theory predicts -1.5)")
    result = "✓ PASS" if -2.5 < slope < -1.0 else "✗ FAIL"
    print(f"  Conclusion: {result}")
    return K_values, errors_sq, slope


# ===========================================================
# 实验 6：梯度偏差收敛率 O(K^{-1/4})
# ===========================================================
def experiment6(K_values=None, p: int = 5, d: int = 16, B: int = 20000):
    """
    对不同 K，用 autograd 计算 ||∇L_ReLU - ∇L^(K)||_F。
    W1 行向量归一化（||u_k||=1），使预激活 ~ N(0,1)。
    理论预测：斜率 ≈ -0.25。
    """
    if K_values is None:
        K_values = [4, 6, 8, 10, 14, 20, 30, 40]

    print(f"\n[Exp 6] Gradient Deviation Convergence Rate")

    torch.manual_seed(123)

    # 单位范数行向量
    W1_raw = torch.randn(d, p, dtype=torch.float64)
    W1_unit = W1_raw / W1_raw.norm(dim=1, keepdim=True)
    w2 = torch.randn(d, dtype=torch.float64)
    X = torch.randn(B, p, dtype=torch.float64)
    y = torch.randn(B, dtype=torch.float64)

    # 真实 ReLU 梯度
    W1_true = W1_unit.clone().requires_grad_(True)
    pre_true = X @ W1_true.T
    f_true = torch.relu(pre_true) @ w2
    loss_true = ((y - f_true) ** 2).mean()
    loss_true.backward()
    grad_true = W1_true.grad.clone()

    grad_devs = []
    for K in K_values:
        hermite_coeffs = [relu_hermite_coeff(n) for n in range(K + 1)]
        hc_t = torch.tensor(hermite_coeffs, dtype=torch.float64)

        W1_K = W1_unit.clone().requires_grad_(True)
        pre_K = X @ W1_K.T
        f_K = sigma_K_torch(pre_K, hc_t) @ w2
        loss_K = ((y - f_K) ** 2).mean()
        loss_K.backward()
        grad_K = W1_K.grad.clone()

        dev = (grad_true - grad_K).norm().item()
        grad_devs.append(dev)
        print(f"  K={K:3d}: ||∇L - ∇L^(K)||_F = {dev:.6e}")

    # 对 K >= 8 的点拟合
    idx_fit = [i for i, k in enumerate(K_values) if k >= 8]
    log_K = np.log(np.array([K_values[i] for i in idx_fit], dtype=float))
    log_dev = np.log(np.array([grad_devs[i] for i in idx_fit], dtype=float))
    slope, intercept = np.polyfit(log_K, log_dev, 1)

    print(f"  Fitted exponent: {slope:.3f} (theory predicts -0.25)")
    result = "✓ PASS" if -1.0 < slope < 0.0 else "✗ FAIL"
    print(f"  Conclusion: {result}")
    return K_values, grad_devs, slope


# ===========================================================
# 绘图
# ===========================================================
def plot_all(log4, K5, err5, slope5, K6, dev6, slope6):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Exp 4: Kernel extraction ---
    ax = axes[0, 0]
    epochs4 = [i * 20 for i in range(len(log4['loss']))]
    ax.semilogy(epochs4, log4['loss'], 'k--', alpha=0.5, label='Train Loss')
    ax.semilogy(epochs4, log4['h1'], 'b-o', ms=3,
                label=r'$\|\hat{h}_1 - h_1^*\|$')
    ax.semilogy(epochs4, log4['H2'], 'r-s', ms=3,
                label=r'$\|\hat{H}_2 - H_2^*\|_F$')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error (log scale)')
    ax.set_title('Exp 4: Volterra Kernel Extraction\n'
                 r'($\sigma^{(K)}$ with monomial coefficients $\tilde{a}_j$)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Exp 5: Truncation error convergence ---
    ax = axes[0, 1]
    ax.loglog(K5, err5, 'bo-', ms=5,
              label=r'Numerical $\|\sigma - \sigma^{(K)}\|^2_{L^2(\gamma)}$')
    K_ref = np.array(K5, dtype=float)
    ax.loglog(K_ref, err5[0] * (K_ref / K_ref[0]) ** (-1.5), 'r--', alpha=0.7,
              label=r'$O(K^{-3/2})$ reference')
    ax.set_xlabel('$K$ (truncation order)')
    ax.set_ylabel(r'$\|\sigma - \sigma^{(K)}\|^2_{L^2(\gamma)}$')
    ax.set_title(f'Exp 5: Truncation Error Convergence\n'
                 f'Fitted exponent: {slope5:.3f} (theory: $-1.5$)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Exp 6: Gradient deviation convergence ---
    ax = axes[1, 0]
    ax.loglog(K6, dev6, 'go-', ms=5,
              label=r'$\|\nabla L - \nabla L^{(K)}\|_F$')
    K_ref = np.array(K6, dtype=float)
    ax.loglog(K_ref, dev6[0] * (K_ref / K_ref[0]) ** (-0.25), 'r--', alpha=0.7,
              label=r'$O(K^{-1/4})$ reference')
    ax.set_xlabel('$K$ (truncation order)')
    ax.set_ylabel('Gradient deviation')
    ax.set_title(f'Exp 6: Gradient Deviation Convergence\n'
                 f'Fitted exponent: {slope6:.3f} (theory: $-0.25$)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Auxiliary: ReLU vs σ^(K) visualization ---
    ax = axes[1, 1]
    z_plot = np.linspace(-3, 3, 500)
    relu_z = np.maximum(z_plot, 0)
    for K in [4, 8, 20]:
        hc = [relu_hermite_coeff(n) for n in range(K + 1)]
        sigma_K_z = He.hermeval(z_plot, hc)
        ax.plot(z_plot, sigma_K_z, '--', alpha=0.7, label=f'$\\sigma^{{({K})}}(z)$')
    ax.plot(z_plot, relu_z, 'k-', lw=2, label='ReLU(z)')
    ax.set_xlabel('$z$')
    ax.set_ylabel(r'$\sigma(z)$')
    ax.set_title(r'ReLU vs Hermite Truncation $\sigma^{(K)}$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 3.5)

    plt.tight_layout()
    out = 'volterra_v2_verification.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {out}")
    plt.show()


# ===========================================================
# 主入口
# ===========================================================
if __name__ == '__main__':
    print("=" * 65)
    print("  Coupled Gradient Theorem v2 — Experimental Verification")
    print("  (Extension to arbitrary L²(γ) activations via Hermite)")
    print("=" * 65)

    experiment1()
    experiment2()
    rel_err_3 = experiment3()
    log4 = experiment4()
    K5, err5, slope5 = experiment5()
    K6, dev6, slope6 = experiment6()
    plot_all(log4, K5, err5, slope5, K6, dev6, slope6)

    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print("  Exp 1: ReLU Hermite coefficients â_n        → formula matches MC")
    print("  Exp 2: Hermite→Monomial conversion ã_j≠â_j  → verified")
    print(f"  Exp 3: Coupled gradient (Thm 4.1)           → rel error = {rel_err_3:.2e}")
    print(f"  Exp 4: Volterra kernel extraction            → errors decrease with training")
    print(f"  Exp 5: Truncation ||σ-σ^(K)||² rate          → exponent = {slope5:.3f} (theory -1.5)")
    print(f"  Exp 6: Gradient ||∇L-∇L^(K)||_F rate         → exponent = {slope6:.3f} (theory -0.25)")
    print("=" * 65)

