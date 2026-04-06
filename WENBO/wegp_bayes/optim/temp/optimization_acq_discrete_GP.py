import numpy as np
import scipy.linalg
import scipy.optimize
import random
import torch
from scipy.stats import norm
from wegp_bayes.optim.acquisition_functions import EI

# —— 1. 离散 GP（无噪声）—— #
class DiscreteGP:
    def __init__(self, H, kernel):
        self.H = H
        self.n = len(H)
        # 构造核矩阵 K
        self.K = np.array([[kernel(hi, hj) for hj in H] for hi in H])
        self.observed_idx = []
        self.A_obs = []

    def update(self, idx_list, A_list):
        self.observed_idx += idx_list
        self.A_obs += A_list

    def predict(self):
        o = self.observed_idx
        if not o:
            mu = np.zeros(self.n)
            var = np.diag(self.K)
            print("[DiscreteGP.predict] no observations yet; returning prior")
            return mu, var

        Koo = self.K[np.ix_(o, o)]
        Kof = self.K[np.ix_(o, range(self.n))]

        # Cholesky 分解
        L = scipy.linalg.cholesky(Koo, lower=True)
        alpha = scipy.linalg.cho_solve((L, True), np.array(self.A_obs))
        mu = Kof.T.dot(alpha)

        v = scipy.linalg.cho_solve((L, True), Kof)
        var = np.diag(self.K) - np.sum(Kof * v, axis=0)

        print(f"[DiscreteGP.predict] mu = {mu}")
        print(f"[DiscreteGP.predict] var = {var}")
        return mu, var

# # —— 2. Expected Improvement 采集函数 —— #
# class EI:
#     def __init__(self, model, best_f):
#         self.model = model
#         self.best_f = best_f

#     def evaluate(self, x_tensor):
#         self.model.eval()
#         with torch.no_grad():
#             mu, sigma = self.model.predict(x_tensor.unsqueeze(0), return_std=True)
#         mu = mu.numpy().flatten()
#         sigma = sigma.numpy().flatten()
#         Z = (mu - self.best_f) / sigma
#         ei = (mu - self.best_f) * norm.cdf(Z) + sigma * norm.pdf(Z)
#         return ei.item()

# —— 3. 工具函数：编码离散 + 连续输入 —— #
def encode_input(h_idx, x_cont, n_categories):
    """
    简单地将 h_onehot 和 x_cont 串在一起
    h_idx: 整数索引
    x_cont: np.array of shape (d,)
    """
    h_vec = np.zeros(n_categories)
    h_vec[h_idx] = 1
    return torch.from_numpy(np.concatenate([h_vec, x_cont])).float()

# —— 4. 优化单个 (h, x) 上的采集 —— #
def optimize_continuous_acq(h_idx, acq, d, n_cats):
    x0 = np.random.randn(d)
    print(f"[optimize] start optimizing for h={h_idx}, x0={x0}")
    def obj(x):
        xt = encode_input(h_idx, x, n_cats)
        return -acq.evaluate(xt)
    res = scipy.optimize.minimize(obj, x0, method='L-BFGS-B')
    x_opt = res.x
    val = -res.fun
    print(f"[optimize] result for h={h_idx}: x*={x_opt}, A={val}")
    return x_opt, val

# —— 5. 主流程 —— #
def bayes_opt_mixed(H, model, init_size, N_cand, T, d):
    n = len(H)
    kernel = lambda hi, hj: 1.0 if hi == hj else 0.5  # 举例
    dgps = DiscreteGP(H, kernel)

    # 当前全局 surrogate 的 best_f（假设从历史真实评估中得来；最开始可设为 -inf）
    best_f = -np.inf
    acq = EI(model, best_f)

    # 1) 生成 init_idxs
    init_idxs = random.sample(range(n), init_size)
    print(f"[main] init_idxs = {init_idxs}")

    # 2) 对 init_idxs 做一次采集优化，填充初始观测 A
    for idx in init_idxs:
        x_opt, A_val = optimize_continuous_acq(idx, acq, d, n)
        dgps.update([idx], [A_val])
        print(f"[main] initial A[{idx}] = {A_val}")

    # 3) 迭代
    for t in range(init_size, T):
        print(f"\n[main] === Iteration {t} ===")
        mu, var = dgps.predict()

        # 3.1) 选 top-N_cand
        cand_idxs = np.argsort(mu)[-N_cand:]
        print(f"[main] candidate idxs = {cand_idxs.tolist()} with mu = {mu[cand_idxs]}")

        # 3.2) 在候选上精炼
        refined = []
        for idx in cand_idxs:
            x_opt, A_val = optimize_continuous_acq(idx, acq, d, n)
            refined.append((idx, x_opt, A_val))
        dgps.update([i for i,_,_ in refined], [A for _,_,A in refined])

        # 3.3) 真实评估选出的最优
        best_local = max(refined, key=lambda x: x[2])
        best_idx, best_x, best_A = best_local
        print(f"[main] refined A's = {[r[2] for r in refined]}")
        print(f"[main] choosing h={best_idx}, x={best_x} with A={best_A} for true eval")

        # 真正调用黑盒 f_eval, 更新 global surrogate & best_f
        y = f_eval(H[best_idx], best_x)
        print(f"[main] true f({H[best_idx]}, {best_x}) = {y}")
        if y > best_f:
            best_f = y
            acq.best_f = best_f
            print(f"[main] new best_f = {best_f}")

    print("[main] Optimization finished.")
