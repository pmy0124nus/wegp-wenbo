import numpy as np
import scipy.linalg
import scipy.optimize

class DiscreteGP:
    """
    离散 GP，不考虑噪声：
      f(H) ~ N(0, K)  （零均值可按需改成非零m）
    后验直接用多元高斯条件分布，无噪声项。
    """
    def __init__(self, H, kernel):
        """
        H: list of 离散类别 h1...hn
        kernel: 函数 kernel(h_i, h_j) → K_ij
        """
        self.H = H
        self.n = len(H)
        # 构造完整的核矩阵 K
        self.K = np.zeros((self.n, self.n))
        for i, hi in enumerate(H):
            for j, hj in enumerate(H):
                self.K[i, j] = kernel(hi, hj)

        # 记录已观测的类别索引及对应的 A 值
        self.observed_idx = []   # e.g. [2, 5, 7]
        self.A_obs = []          # 对应的 [A_t(h2), A_t(h5), A_t(h7)]

    def update(self, idx_list, A_list):
        """在已有观测基础上，添加新观测（无噪声）。"""
        self.observed_idx += idx_list
        self.A_obs += A_list

    def predict(self):
        """
        对所有类别计算后验均值和方差：
          μ = K_fo K_oo^{-1} A_o
          Σ_diag = diag(K_ff - K_fo K_oo^{-1} K_of)

        返回：
          mu: shape (n,), 后验均值
          var: shape (n,), 后验方差（对角）
        """
        o = self.observed_idx
        if len(o) == 0:
            # 未观测时退回到先验
            return np.zeros(self.n), np.diag(self.K)

        # 抽取子块
        Koo = self.K[np.ix_(o, o)]       # (m×m)
        Kof = self.K[np.ix_(o, range(self.n))]  # (m×n)

        # 直接用 Cholesky 分解求逆
        L = scipy.linalg.cholesky(Koo, lower=True)
        alpha = scipy.linalg.cho_solve((L, True), np.array(self.A_obs))  # Koo^{-1} A_o
        mu = Kof.T.dot(alpha)     # (n,)

        # 计算方差
        v = scipy.linalg.cho_solve((L, True), Kof)  # (m×n)
        # Σ_ff_diag = diag(K_ff - Kof^T @ Koo^{-1} @ Kof)
        var = np.diag(self.K) - np.sum(Kof * v, axis=0)

        return mu, var


def optimize_continuous_acq(h, acq_func, x0):
    """
    对固定离散类别 h，最大化 α_t(h, x)：
      max_x α_t(h, x)
    返回：
      x_opt: 最优点
      A_val: α_t(h, x_opt)
    """
    # 用 L-BFGS-B 负目标最小化
    res = scipy.optimize.minimize(
        lambda x: -acq_func(h, x),
        x0,
        method='L-BFGS-B'
    )
    x_opt = res.x
    return x_opt, acq_func(h, x_opt)


def bayes_opt_mixed(H, init_idxs, acq_func, f_eval, N_cand, T):
    """
    H: 离散类别列表
    init_idxs: 初始随机评估的类别索引列表
    acq_func(h, x): 采集函数
    f_eval(h, x): 真实黑盒目标
    N_cand: 每轮从离散 GP 中挑前 N 个候选
    T: 总迭代次数
    """
    # —— 1. 初始化 DiscreteGP —— #
    dgps = DiscreteGP(H, kernel=kernel_d)

    # 对 init_idxs 做一次连续优化，填充初始观测
    for idx in init_idxs:
        x0 = np.random.randn(d)         # d 是连续输入维度
        x_opt, A_val = optimize_continuous_acq(H[idx], acq_func, x0)
        dgps.update([idx], [A_val])
        # （可选）把 (h,x_opt,f_eval(h,x_opt)) 反馈给全局 surrogate

    # —— 2. 主循环 —— #
    for t in range(len(init_idxs), T):
        # 2.1 离散 GP 预测
        mu, var = dgps.predict()

        # 2.2 选出均值最高的前 N_cand 个类别
        cand_idxs = np.argsort(mu)[-N_cand:]

        # 2.3 对这 N 个类别做精炼连续优化
        refined = []
        for idx in cand_idxs:
            x0 = np.random.randn(d)
            x_opt, A_val = optimize_continuous_acq(H[idx], acq_func, x0)
            refined.append((idx, x_opt, A_val))
        # 更新离散 GP 用最新的 A
        dgps.update([i for i,_,_ in refined],
                    [A for _,_,A in refined])

        # 2.4 从 refined 里取 A 最大的做真实评估
        best = max(refined, key=lambda item: item[2])
        best_idx, best_x, _ = best
        y = f_eval(H[best_idx], best_x)
        # 把 (H[best_idx], best_x, y) 送给全局 surrogate

    # 最后可返回最佳 (h*, x*, y*)
    return
