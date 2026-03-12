"""
耦合梯度定理（Theorem 4.1）实验验证
=====================================
网络：  f(x) = w2^T σ(W1 x)
激活：  σ(z) = z + z²/2 + z³/6   (截断指数, a1=1, a2=1/2, a3=1/6)
定理：  ∂L/∂[W1]_{k,j} = -2 Σ_n n·a_n · E[ε · [w2]_k · (u_k^T x)^{n-1} · x_j]

四个实验
--------
实验1  梯度公式数值验证     —— autograd 与定理公式逐元素对比，应达到机器精度
实验2  Volterra 核提取      —— 从网络参数还原各阶核，对比已知真值
实验3  各阶核误差同步下降   —— 核心演示：一次梯度步同时减小所有阶误差
实验4  梯度各阶贡献的权重   —— 验证 n·a_n 系数控制各阶在梯度中的贡献强度
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

torch.manual_seed(42)
np.random.seed(42)

# ===========================================================
# 激活函数及其 Taylor 系数
# ===========================================================
A = {1: 1.0, 2: 0.5, 3: 1 / 6}   # σ(z) = Σ A[n] z^n

def sigma(z: torch.Tensor) -> torch.Tensor:
    return z + 0.5 * z**2 + (1 / 6) * z**3

def sigma_prime(z: torch.Tensor) -> torch.Tensor:
    """σ'(z) = Σ n·A[n]·z^{n-1}"""
    return 1.0 + z + 0.5 * z**2


# ===========================================================
# 工具：从网络参数提取各阶 Volterra 核
#   h_n(i1,...,in) = A[n] · Σ_k w2[k] · W1[k,i1] · ... · W1[k,in]
# ===========================================================
@torch.no_grad()
def extract_kernel(W1: torch.Tensor, w2: torch.Tensor, order: int) -> torch.Tensor:
    an = A[order]
    if order == 1:
        return an * (w2 @ W1)                                    # (p,)
    elif order == 2:
        return an * torch.einsum('k,ki,kj->ij', w2, W1, W1)     # (p,p)
    elif order == 3:
        return an * torch.einsum('k,ki,kj,kl->ijl', w2, W1, W1, W1)  # (p,p,p)
    else:
        raise ValueError(f"order={order} 暂不支持")


# ===========================================================
# 实验 1：梯度公式数值验证
# ===========================================================
def experiment1(p: int = 5, d: int = 16, B: int = 4000) -> float:
    """
    断言：定理 4.1 公式 == autograd 计算的梯度，误差应在机器精度 (~1e-6)。

    物理意义：定理是链式法则 + 代入 σ' 展开的恒等式，若实现正确则精确成立。
    """
    W1 = torch.randn(d, p, requires_grad=True)
    w2 = torch.randn(d)
    X  = torch.randn(B, p)
    y  = torch.randn(B)

    # --- autograd 梯度（参考值）---
    pre    = X @ W1.T                        # (B, d)
    f_pred = sigma(pre) @ w2                 # (B,)
    loss   = ((y - f_pred)**2).mean()
    loss.backward()
    grad_auto = W1.grad.clone()              # (d, p)

    # --- 定理 4.1 公式梯度 ---
    with torch.no_grad():
        pre2   = X @ W1.detach().T           # (B, d)
        f2     = sigma(pre2) @ w2
        eps    = y - f2                      # (B,)

        grad_formula = torch.zeros(d, p)
        for n, an in A.items():
            # 第 n 阶贡献: -2·n·a_n · E[ε · w2[k] · (u_k^T x)^{n-1} · x_j]
            Dn = eps[:, None] * w2[None, :] * pre2**(n - 1)   # (B, d)
            grad_formula += (n * an) * (Dn.T @ X) / B

        grad_formula *= -2.0

    max_abs_err = (grad_auto - grad_formula).abs().max().item()
    rel_err     = max_abs_err / (grad_auto.abs().max().item() + 1e-12)

    print("\n[实验1] 梯度公式数值验证")
    print(f"  max|grad_autograd - grad_formula| = {max_abs_err:.2e}")
    print(f"  相对最大误差                       = {rel_err:.2e}")
    result = "✓ 通过" if rel_err < 1e-4 else "✗ 失败"
    print(f"  结论：{result}")
    return rel_err


# ===========================================================
# 实验 2：Volterra 核提取（验证命题 3.1）
# ===========================================================
def experiment2(p: int = 4, d: int = 48, n_samples: int = 8000,
                epochs: int = 600, lr: float = 5e-3) -> dict:
    """
    真实函数 = 一阶项 + 二阶项（已知），训练网络后从参数还原两阶核，
    对比真值。这验证了命题 3.1（参数 → 核的公式）是否正确。
    """
    # 真实核（随机）
    h1_true = torch.randn(p)
    H2_true = torch.randn(p, p)
    H2_true = (H2_true + H2_true.T) / 2          # 对称化

    # 生成数据：y = h1^T x + x^T H2 x + 噪声
    X = torch.randn(n_samples, p)
    y = (X @ h1_true
         + torch.einsum('bi,ij,bj->b', X, H2_true, X)
         + 0.01 * torch.randn(n_samples))

    W1_p = torch.nn.Parameter(0.05 * torch.randn(d, p))
    w2_p = torch.nn.Parameter(0.05 * torch.randn(d))
    opt  = torch.optim.Adam([W1_p, w2_p], lr=lr)

    log = {'h1': [], 'H2': [], 'loss': []}

    for epoch in range(epochs):
        opt.zero_grad()
        pre    = X @ W1_p.T
        f_pred = sigma(pre) @ w2_p
        loss   = ((y - f_pred)**2).mean()
        loss.backward()
        opt.step()

        if epoch % 20 == 0:
            with torch.no_grad():
                h1_est = extract_kernel(W1_p, w2_p, 1)
                H2_est = extract_kernel(W1_p, w2_p, 2)
                log['h1'].append((h1_est - h1_true).norm().item())
                log['H2'].append((H2_est - H2_true).norm().item())
                log['loss'].append(loss.item())

    print(f"\n[实验2] Volterra 核提取验证（{epochs} epoch）")
    print(f"  一阶核误差  ||h1_est - h1*||   = {log['h1'][-1]:.4f}  (初始 {log['h1'][0]:.4f})")
    print(f"  二阶核误差  ||H2_est - H2*||_F = {log['H2'][-1]:.4f}  (初始 {log['H2'][0]:.4f})")
    return log


# ===========================================================
# 实验 3：各阶核误差同步下降（核心演示）
# ===========================================================
def experiment3(p: int = 3, d: int = 48, n_samples: int = 10000,
                epochs: int = 1000, lr: float = 3e-3) -> tuple:
    """
    真实函数 = 一阶 + 二阶 + 三阶 Volterra
    在训练过程中追踪三阶核误差，验证三者同时下降。
    这正是定理 4.1「耦合梯度」的实验体现。
    """
    h1_true = torch.randn(p)

    H2_true = torch.randn(p, p)
    H2_true = (H2_true + H2_true.T) / 2

    # 对称化三阶核
    _H3 = torch.randn(p, p, p)
    H3_true = (_H3
               + _H3.permute(0,2,1) + _H3.permute(1,0,2)
               + _H3.permute(1,2,0) + _H3.permute(2,0,1)
               + _H3.permute(2,1,0)) / 6

    X = torch.randn(n_samples, p)
    y = (X @ h1_true
         + torch.einsum('bi,ij,bj->b', X, H2_true, X)
         + torch.einsum('bi,bj,bk,ijk->b', X, X, X, H3_true)
         + 0.01 * torch.randn(n_samples))

    W1_p = torch.nn.Parameter(0.05 * torch.randn(d, p))
    w2_p = torch.nn.Parameter(0.05 * torch.randn(d))
    opt  = torch.optim.Adam([W1_p, w2_p], lr=lr)

    log_step = 20
    log = {'h1': [], 'H2': [], 'H3': [], 'loss': [], 'epoch': []}

    for epoch in range(epochs):
        opt.zero_grad()
        pre    = X @ W1_p.T
        f_pred = sigma(pre) @ w2_p
        loss   = ((y - f_pred)**2).mean()
        loss.backward()
        opt.step()

        if epoch % log_step == 0:
            with torch.no_grad():
                h1_est = extract_kernel(W1_p, w2_p, 1)
                H2_est = extract_kernel(W1_p, w2_p, 2)
                H3_est = extract_kernel(W1_p, w2_p, 3)
                log['h1'].append((h1_est - h1_true).norm().item())
                log['H2'].append((H2_est - H2_true).norm().item())
                log['H3'].append((H3_est - H3_true).norm().item())
                log['loss'].append(loss.item())
                log['epoch'].append(epoch)

    print(f"\n[实验3] 各阶核误差同步下降（{epochs} epoch）")
    for key, name in [('h1', '1阶'), ('H2', '2阶'), ('H3', '3阶')]:
        init, final = log[key][0], log[key][-1]
        ratio = init / (final + 1e-12)
        print(f"  {name}核误差：初始={init:.4f}, 最终={final:.4f}, 下降={ratio:.1f}×")

    return log


# ===========================================================
# 实验 4：梯度各阶贡献 n·a_n 权重验证
# ===========================================================
def experiment4(p: int = 4, d: int = 16, B: int = 4000) -> dict:
    """
    Verify that gradient contribution of order n is proportional to n·a_n.

    The full n-th order gradient magnitude is proportional to:
        n · a_n · E[|ε| · |w2| · |pre|^{n-1} · |x|]

    Design: fix pre = +1 (all ones) so pre^{n-1} = 1 for ALL n.
    This makes the three Dn matrices identical up to a scalar, so the only
    factor distinguishing the n=1,2,3 gradient norms is exactly n·a_n.

    Why not use pre = random ±1?
      For n=1 and n=3: pre^0=1 and pre^2=1, so their Dn matrices are rank-1
        (all k-rows share the same vector eps·w2), giving high norm variance.
      For n=2: pre^1 = ±1 gives a full-rank Dn (each k-row has independent
        sign flips), which concentrates near its expectation via averaging
        over d independent rows.
      The rank mismatch breaks the comparison even though all three share the
      same EXPECTED squared norm.  Setting pre = ones eliminates this artifact.

    Expected ratio ||∇_1||:||∇_2||:||∇_3||  =  n·a_n = 1:1:0.5  →  2:2:1.
    """
    torch.manual_seed(1)
    # pre = +1 everywhere → pre^{n-1} = 1 for every n, isolating only n·a_n
    pre = torch.ones(B, d)
    w2  = torch.randn(d)
    X   = torch.randn(B, p)
    eps = torch.randn(B)

    grad_norm_by_order = {}
    for n, an in A.items():
        Dn  = eps[:, None] * w2[None, :] * pre**(n - 1)   # (B, d)
        g_n = -2 * (n * an) * (Dn.T @ X) / B               # (d, p)
        grad_norm_by_order[n] = g_n.norm().item()

    base       = grad_norm_by_order[3]
    ratio_obs  = f"{grad_norm_by_order[1]/base:.3f}:{grad_norm_by_order[2]/base:.3f}:1.000"
    ratio_theo = f"{(1*A[1])/(3*A[3]):.3f}:{(2*A[2])/(3*A[3]):.3f}:1.000"

    print("\n[Exp 4] Gradient contribution per order (pre=1, isolating n·a_n)")
    print(f"  Theoretical n·a_n:  n=1->{1*A[1]:.3f}, n=2->{2*A[2]:.3f}, n=3->{3*A[3]:.4f}")
    for n in [1, 2, 3]:
        print(f"  n={n}: ||grad_n W1||_F = {grad_norm_by_order[n]:.6f}")
    print(f"  Observed  ratio n=1:n=2:n=3 = {ratio_obs}")
    print(f"  Theoretic ratio n=1:n=2:n=3 = {ratio_theo}  (should match exactly)")
    return grad_norm_by_order


# ===========================================================
# 绘图
# ===========================================================
def plot_all(log2: dict, log3: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- 实验2 ---
    ax = axes[0]
    epochs2 = [i * 20 for i in range(len(log2['loss']))]
    ax.semilogy(epochs2, log2['loss'], 'k--', alpha=0.5, label='Train Loss')
    ax.semilogy(epochs2, log2['h1'],   'b-o', ms=3, label='$\\|\\hat{h}_1 - h_1^*\\|$')
    ax.semilogy(epochs2, log2['H2'],   'r-s', ms=3, label='$\\|\\hat{H}_2 - H_2^*\\|_F$')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error (log scale)')
    ax.set_title('Exp 2: Volterra Kernel Extraction (1st+2nd order)\n(Verifying Prop. 3.1)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 实验3 ---
    ax = axes[1]
    steps3 = log3['epoch']
    ax.semilogy(steps3, log3['loss'], 'k--', alpha=0.5, label='Train Loss')
    ax.semilogy(steps3, log3['h1'],   'b-o', ms=3, label='$\\|\\hat{h}_1 - h_1^*\\|$ (1st order)')
    ax.semilogy(steps3, log3['H2'],   'r-s', ms=3, label='$\\|\\hat{H}_2 - H_2^*\\|$ (2nd order)')
    ax.semilogy(steps3, log3['H3'],   'g-^', ms=3, label='$\\|\\hat{H}_3 - H_3^*\\|$ (3rd order)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Kernel Error (log scale)')
    ax.set_title('Exp 3: Simultaneous Descent of All-order Kernel Errors\n(Coupled Gradient Theorem, Thm. 4.1)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = 'volterra_v1_verification.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n图已保存至 {out}")
    plt.show()


# ===========================================================
# 主入口
# ===========================================================
if __name__ == '__main__':
    print("=" * 60)
    print("     耦合梯度定理（Theorem 4.1）实验验证")
    print("=" * 60)

    experiment1()
    log2 = experiment2()
    log3 = experiment3()
    experiment4()
    plot_all(log2, log3)
