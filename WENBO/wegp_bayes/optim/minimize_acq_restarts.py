import torch
from torch.quasirandom import SobolEngine
from scipy.optimize import minimize
from typing import Callable, Tuple

# ------------- 1) 初始化 & 起点挑选 -------------
def _sample_raw_x(bounds: torch.Tensor, n: int, seed: int = None, device=None, dtype=None):
    """Sobol 采样 n 个点，只在连续空间 x 上。"""
    d = bounds.shape[1]
    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    X = sobol.draw(n).to(device=device, dtype=dtype)
    X = bounds[0] + (bounds[1] - bounds[0]) * X
    return X

def _pick_best_starts(acq_vals: torch.Tensor, X_raw: torch.Tensor, k: int):
    """按采集函数值挑 top-k 起点。"""
    topk = torch.topk(acq_vals.flatten(), k=k).indices
    return X_raw[topk]


# ------------- 2) SciPy 局部优化（给定 h0）-------------
def _scipy_optimize_x(
    x0: torch.Tensor,
    acq_obj: Callable[[torch.Tensor, any], torch.Tensor],
    h0,
    bounds: torch.Tensor,
    maxiter: int = 200,
):
    """
    用 SciPy L-BFGS-B 优化 x（最小化 -acq_obj(x,h0)）。
    """
    device = x0.device
    dtype = x0.dtype
    lb = bounds[0].detach().cpu().numpy()
    ub = bounds[1].detach().cpu().numpy()
    scipy_bounds = list(zip(lb, ub))

    def obj_and_grad(x_np):
        x = torch.from_numpy(x_np).to(device=device, dtype=dtype).requires_grad_(True)
        val = acq_obj(x.unsqueeze(0), h0).sum()  # batch=1
        loss = -val                               # 最大化 → 最小化
        (grad,) = torch.autograd.grad(loss, x)
        return float(loss.detach().cpu()), grad.detach().cpu().numpy()

    res = minimize(
        fun=obj_and_grad,
        x0=x0.detach().cpu().numpy(),
        jac=True,
        bounds=scipy_bounds,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "disp": False},
    )
    x_opt = torch.from_numpy(res.x).to(device=device, dtype=dtype)
    v_opt = acq_obj(x_opt.unsqueeze(0), h0).detach().squeeze(0)
    return x_opt, v_opt


# ------------- 3) 主函数：固定 h0 优化 x -------------
def optimize_x_given_h(
    acq_obj: Callable[[torch.Tensor, any], torch.Tensor],
    h0,
    bounds_x: torch.Tensor,
    num_restarts: int = 10,
    raw_samples: int = 256,
    maxiter: int = 200,
    seed: int = 0,
    device=None,
    dtype=torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回：best_x, best_val
    """
    device = device or bounds_x.device
    dtype = dtype or bounds_x.dtype

    # 1) 采 raw samples
    X_raw = _sample_raw_x(bounds_x.to(device=device, dtype=dtype),
                          n=raw_samples,
                          seed=seed,
                          device=device,
                          dtype=dtype)

    # 2) 评估 EI/采集函数，挑 starts
    with torch.no_grad():
        vals_raw = acq_obj(X_raw, h0).view(-1)
    starts = _pick_best_starts(vals_raw, X_raw, num_restarts)

    # 3) 对每个 start 做 L-BFGS-B
    cands, cand_vals = [], []
    for x0 in starts:
        x_opt, v_opt = _scipy_optimize_x(x0, acq_obj, h0, bounds_x, maxiter=maxiter)
        cands.append(x_opt)
        cand_vals.append(v_opt)

    cand_vals = torch.stack(cand_vals)
    best_idx = torch.argmax(cand_vals)
    return cands[best_idx], cand_vals[best_idx]
