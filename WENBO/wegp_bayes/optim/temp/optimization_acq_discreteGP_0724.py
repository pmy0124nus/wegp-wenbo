"""
一个自包含的示例实现：
- DiscreteGP: 离散核高斯过程（用 Cholesky + jitter，检查 NaN/Inf）
- MixedBayesOptGPDiscrete: 混合离散+连续空间的 BO 框架
- EI: 期望改进（torch 实现）
- ConfigFun: 简单的 LHS 采样器 + 连续维度索引
- SimpleTorchModel: 演示用的 surrogate（真实项目请换成你的模型）
- RBF kernel 构造或任意 SPD kernel 构造示例

运行主程序时，会在一个简单 toy 问题上演示完整流程。
你可以根据自己项目把其中的 model/EI/config_fun/kernel 等替换掉。

改进版本新增功能：
1. _local_search_continuous(): 对连续变量进行局部搜索
2. _global_search_continuous(): 多种全局搜索策略
3. optimize_x_given_h_hybrid(): 混合优化策略
4. 在run()方法中可以选择使用原始策略或混合策略

使用示例：
```python
# 创建优化器
optimizer = MixedBayesOptGPDiscrete(
    acq_obj=your_acq_obj,
    rng=np.random.RandomState(42),
    config_fun=your_config_fun,
    combined_cat_index=your_cat_index,
    model=your_model
)

# 使用混合策略（推荐）
result = optimizer.run(init_size=5, N_cand=10, T=20, use_hybrid=True)

# 或使用原始策略
result = optimizer.run(init_size=5, N_cand=10, T=20, use_hybrid=False)
```
"""

import numpy as np
import scipy.linalg
import scipy.optimize
import random
import torch
from scipy.stats import norm
from wegp_bayes.optim.acquisition_functions import EI
import time

"""
一个自包含的示例实现：
- DiscreteGP: 离散核高斯过程（用 Cholesky + jitter，检查 NaN/Inf）
- MixedBayesOptGPDiscrete: 混合离散+连续空间的 BO 框架
- EI: 期望改进（torch 实现）
- ConfigFun: 简单的 LHS 采样器 + 连续维度索引
- SimpleTorchModel: 演示用的 surrogate（真实项目请换成你的模型）
- RBF kernel 构造或任意 SPD kernel 构造示例

运行主程序时，会在一个简单 toy 问题上演示完整流程。
你可以根据自己项目把其中的 model/EI/config_fun/kernel 等替换掉。
"""

import numpy as np
import torch
import scipy.linalg
import scipy.optimize
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, Optional


# ============================================================
# 工具函数
# ============================================================
def ensure_finite(name: str, arr: np.ndarray):
    """检查数组是否全为有限值"""
    if not np.all(np.isfinite(arr)):
        idx = np.where(~np.isfinite(arr))[0]
        raise ValueError(f"{name} 含有 NaN/Inf，位置 {idx}，值 {arr[idx]}")


def add_jitter(K: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """给核矩阵加抖动，确保可 Cholesky 分解"""
    return K + jitter * np.eye(K.shape[0], dtype=K.dtype)

class SimpleTorchModel:
    """
    一个非常简单的 surrogate：只返回 0 均值、固定方差。仅演示 EI 流程。
    你可以换成 GP、神经网络等，并确保实现 predict(x, return_std=True)。
    """

    def eval(self):
        pass

    def predict(self, x: torch.Tensor, return_std: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        # mu = 0, sigma = 0.1
        if x.dim() == 1:
            x = x.unsqueeze(0)
        N = x.shape[0]
        mu = torch.zeros(N, 1, dtype=torch.float32)
        sigma = torch.ones(N, 1, dtype=torch.float32) * 0.1
        return mu, sigma



# ============================================================
# Discrete GP
# ============================================================
class DiscreteGP:
    def __init__(self, index_number: int, kernel: np.ndarray):
        """
        index_number: 离散元素个数
        kernel: 形状 [n, n] 的 SPD 核矩阵
        """
        self.n = index_number
        self.K = np.asarray(kernel, dtype=float)
        if self.K.shape != (self.n, self.n):
            raise ValueError(f"kernel 维度应为 ({self.n},{self.n}), 但得到 {self.K.shape}")
        self.observed_idx: List[int] = []
        self.A_obs: List[float] = []

    def update(self, idx_list: Sequence[int], A_list: Sequence[float]):
        idx_list = list(map(int, idx_list))
        A_arr = np.asarray(A_list, dtype=float).reshape(-1)
        if len(idx_list) != len(A_arr):
            raise ValueError("update: idx_list 与 A_list 长度不一致")
        # 过滤 NaN/Inf
        finite_mask = np.isfinite(A_arr)
        if not np.all(finite_mask):
            print("Warning: 丟弃非有限 A 值:", A_arr[~finite_mask])
        idx_list = np.array(idx_list)[finite_mask].tolist()
        A_arr = A_arr[finite_mask]

        # 允许重复索引：策略是直接追加。也可以选择平均化或只保留最新一次
        self.observed_idx.extend(idx_list)
        self.A_obs.extend(A_arr.tolist())

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.observed_idx:
            mu = np.zeros(self.n, dtype=float)
            var = np.diag(self.K).copy()
            return mu, var

        o = self.observed_idx
        A = np.asarray(self.A_obs, dtype=float)
        assert len(o) == len(A), "observed_idx 与 A_obs 长度不相等"

        Koo = self.K[np.ix_(o, o)].astype(float)
        Kof = self.K[np.ix_(o, range(self.n))].astype(float)

        # 检查有限性
        ensure_finite("A_obs", A)
        ensure_finite("Koo", Koo)
        ensure_finite("Kof", Kof)

        # jitter
        Koo_j = add_jitter(Koo, jitter=1e-8)

        # Cholesky
        L = scipy.linalg.cholesky(Koo_j, lower=True, check_finite=True)
        alpha = scipy.linalg.cho_solve((L, True), A, check_finite=True)

        mu = Kof.T @ alpha
        v = scipy.linalg.cho_solve((L, True), Kof, check_finite=True)
        var = np.diag(self.K) - np.sum(Kof * v, axis=0)
        var = np.maximum(var, 0.0)  # 防止微小负数

        return mu, var


# ============================================================
# Mixed Bayesian Optimization with Discrete GP
# ============================================================
class MixedBayesOptGPDiscrete:
    def __init__(
        self,
        acq_obj,
        rng: np.random.RandomState,
        config_fun,
        combined_cat_index: np.ndarray,
        model,
        kernel: Optional[np.ndarray] = None,
        # initial_best_f: float = -np.inf,
        # repeat_policy: str = "append",  # or "skip" / "overwrite"
    ):
        """
        rng: numpy RandomState
        config_fun: 连续空间采样工具
        combined_cat_index: shape [n_cat, m] 的离散特征编码矩阵（每个离散元素一行）
        model: surrogate，用于 EI
        f_eval: 真正的目标函数 f(h, x_cont) -> float
        kernel: 离散核矩阵
        initial_best_f: 初始最优目标值
        maximize: True 表示目标是最大化
        repeat_policy: 对离散 idx 重复观测的策略；这里留钩子。当前实现里只做 append。
        """

        self.rng = rng
        self.config_fun = config_fun
        self.combined_cat_index = np.asarray(combined_cat_index, dtype=float)
        self.n = self.combined_cat_index.shape[0]
        self.model = model
        # self.best_f = initial_best_f
        self.acq_obj = acq_obj
        self.dgps = DiscreteGP(self.n, kernel)

        # 添加计数器
        self.evaluate_count = 0

        # 为了演示，也记录每一次评估
        self.history = []

    def _encode_input(self, h_idx: int, x_cont: np.ndarray) -> torch.Tensor:
        """
        将离散选择的 one-hot / embedding 与连续变量拼接成一个 torch tensor
        """
        h_vec = self.combined_cat_index[h_idx]  # shape [m]
        arr = np.concatenate([x_cont, h_vec]).astype(np.float32)
        return torch.from_numpy(arr)

    def _evaluate_with_counting(self, x_tensor):
        """
        包装evaluate方法，同时计数
        """
        self.evaluate_count += 1
        return self.acq_obj.evaluate(x_tensor, num_samples=100)

    def _optimize_continuous(self, h_idx: int, d: int) -> Tuple[np.ndarray, float]:
        """
        在固定的离散选项 h_idx 下优化连续变量 x，最大化 EI
        d: 连续变量维度
        返回： (x_opt, A_val) 其中 A_val = EI(x_opt)
        """
        # 初始点：拉丁超立方里随便取一个
        init_z = self.config_fun.latinhypercube_sample(np.random.RandomState(h_idx + int(time.time() * 1e6) % 1000000), 16)
        x0 = init_z[..., self.config_fun.quant_index][0]  # 取第一点作为起点

        # # 若需要 bounds：L-BFGS-B 要求 bounds 为列表
        # lower = self.config_fun.bounds[self.config_fun.quant_index, 0]
        # upper = self.config_fun.bounds[self.config_fun.quant_index, 1]
        # bounds = list(zip(lower, upper))

        # def obj(x_numpy: np.ndarray) -> float:
        #     z_t = self._encode_input(h_idx, x_numpy)
        #     val = self.acq_obj.evaluate(z_t)
        #     return -val  # scipy.minimize -> minimize, 我们最大化 EI

        # # res = scipy.optimize.minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
        # res = scipy.optimize.minimize(obj, x0, method="L-BFGS-B") #可能就是这里不对 优化的时候没有bounds

        # A_val = -res.fun
        # if not np.isfinite(A_val):
        #     # fallback：用 x0 评估一次
        #     A_val = -obj(x0)
        #     if not np.isfinite(A_val):
        #         raise ValueError("A_val 仍然 NaN/Inf，请检查 EI 或 model.predict")
        # x_opt = res.x
        # return x_opt, A_val

    def optimize_x_given_h(self, h_idx: int, n_starts: int = 100, top_k: int = 10, use_local_search: bool = True) -> Tuple[np.ndarray, float]:
        """
        在固定的离散选项 h_idx 下优化连续变量 x，最大化 EI
        - n_starts: 采样多少个初始点
        - top_k: 选出多少个最优点作为L-BFGS起点
        - use_local_search: 是否使用局部搜索来改进起始点
        返回： (x_opt, A_val) 其中 A_val = EI(x_opt)
        """
        # 用固定种子+ h_idx 采样 starter
        seed = 12345 + h_idx * 1000  # 12345 可换成你喜欢的基准种子
        rng = np.random.RandomState(seed)
        init_z = self.config_fun.latinhypercube_sample(rng, n_starts)
        x_candidates = init_z[..., self.config_fun.quant_index]

        # 2. 计算每个点的EI
        ei_list = []
        for x in x_candidates:
            z_t = self._encode_input(h_idx, x)
            val = self._evaluate_with_counting(z_t)
            ei_list.append(val)
        ei_list = np.array(ei_list)

        # 3. 选出 top_k 个EI最大的点作为起点
        if len(ei_list) < top_k:
            top_k = len(ei_list)
        topk_idx = np.argsort(ei_list)[-top_k:]
        starters = x_candidates[topk_idx]

        # 4. 如果启用局部搜索，对每个起始点进行局部搜索改进
        if use_local_search:
            improved_starters = []
            improved_eis = []
            
            for i, x0 in enumerate(starters):
                # 对每个起始点进行局部搜索
                improved_x, improved_ei = self._local_search_continuous(h_idx, x0)
                improved_starters.append(improved_x)
                improved_eis.append(improved_ei)
            
            # 选择局部搜索后最好的几个点作为L-BFGS的起始点
            best_indices = np.argsort(improved_eis)[-top_k:]
            starters = np.array([improved_starters[i] for i in best_indices])
            ei_list = np.array([improved_eis[i] for i in best_indices])

        # 5. 对每个starter做L-BFGS优化
        best_x = None
        best_ei = -np.inf
        for x0 in starters:
            def obj(x_numpy: np.ndarray) -> float:
                z_t = self._encode_input(h_idx, x_numpy)
                val = self._evaluate_with_counting(z_t)
                return -val  # scipy.minimize -> minimize, 我们最大化 EI

            n_quant = len(self.config_fun.quant_index)
            bounds = [(0.0, 1.0)] * n_quant
            res = scipy.optimize.minimize(obj, x0, method="L-BFGS-B", bounds=bounds)

            A_val = -res.fun
            if np.isfinite(A_val) and A_val > best_ei:
                best_ei = A_val
                best_x = res.x

        # fallback
        if best_x is None:
            best_x = starters[0]
            best_ei = ei_list[0]

        return best_x, best_ei

    def optimize_x_given_h_hybrid(self, h_idx: int, n_starts: int = 100, top_k: int = 10, 
                                  use_local_search: bool = True, use_global_search: bool = True) -> Tuple[np.ndarray, float]:
        """
        混合优化策略：结合局部搜索和全局搜索来找到更好的起始点
        - h_idx: 离散选项索引
        - n_starts: 采样多少个初始点
        - top_k: 选出多少个最优点作为L-BFGS起点
        - use_local_search: 是否使用局部搜索
        - use_global_search: 是否使用全局搜索
        返回： (x_opt, A_val) 其中 A_val = EI(x_opt)
        """
        all_starters = []
        all_eis = []
        
        # 1. 随机采样起始点
        seed = 12345 + h_idx * 1000
        rng = np.random.RandomState(seed)
        init_z = self.config_fun.latinhypercube_sample(rng, n_starts)
        x_candidates = init_z[..., self.config_fun.quant_index]
        
        # 评估随机起始点
        for x in x_candidates:
            z_t = self._encode_input(h_idx, x)
            val = self._evaluate_with_counting(z_t)
            all_starters.append(x)
            all_eis.append(val)
        
        # 2. 局部搜索改进（如果启用）
        if use_local_search:
            # 选择最好的几个随机点进行局部搜索
            best_random_idx = np.argsort(all_eis)[-min(5, len(all_eis)):]
            for idx in best_random_idx:
                improved_x, improved_ei = self._local_search_continuous(h_idx, all_starters[idx])
                all_starters.append(improved_x)
                all_eis.append(improved_ei)
        
        # 3. 全局搜索（如果启用）
        if use_global_search:
            # 使用更激进的采样策略
            global_candidates = self._global_search_continuous(h_idx, n_samples=20)
            for x in global_candidates:
                z_t = self._encode_input(h_idx, x)
                val = self._evaluate_with_counting(z_t)
                all_starters.append(x)
                all_eis.append(val)
        
        # 4. 选择最好的top_k个点进行L-BFGS优化
        all_eis = np.array(all_eis)
        best_indices = np.argsort(all_eis)[-top_k:]
        starters = np.array([all_starters[i] for i in best_indices])
        
        # 5. L-BFGS优化
        best_x = None
        best_ei = -np.inf
        for x0 in starters:
            def obj(x_numpy: np.ndarray) -> float:
                z_t = self._encode_input(h_idx, x_numpy)
                val = self._evaluate_with_counting(z_t)
                return -val

            n_quant = len(self.config_fun.quant_index)
            bounds = [(0.0, 1.0)] * n_quant
            res = scipy.optimize.minimize(obj, x0, method="L-BFGS-B", bounds=bounds)

            A_val = -res.fun
            if np.isfinite(A_val) and A_val > best_ei:
                best_ei = A_val
                best_x = res.x

        # fallback
        if best_x is None:
            best_x = starters[0]
            best_ei = all_eis[best_indices[0]]

        return best_x, best_ei

    def _local_search_continuous(self, h_idx: int, x0: np.ndarray, max_iter: int = 20, sigma: float = 0.1, delta: float = 0.3) -> Tuple[np.ndarray, float]:
        """
        对连续变量进行局部搜索，改进起始点
        - h_idx: 离散选项索引
        - x0: 初始连续变量
        - max_iter: 最大迭代次数
        - sigma: 高斯采样标准差
        - delta: 距离阈值
        返回： (best_x, best_ei)
        """
        x_t = x0.copy()
        z_t = self._encode_input(h_idx, x_t)
        best_ei = self._evaluate_with_counting(z_t)
        best_x = x_t.copy()
        
        no_improve = 0
        max_no_improve = 5
        
        for iter in range(max_iter):
            # 在x_t周围采样候选点
            candidates = []
            for _ in range(10):  # 每次采样10个候选
                cand = x_t + np.random.randn(*x_t.shape) * sigma
                # 确保在边界内
                cand = np.clip(cand, 0.0, 1.0)
                # 检查距离
                if np.linalg.norm(cand - x_t) <= delta:
                    candidates.append(cand)
            
            if not candidates:
                continue
                
            # 评估所有候选点
            best_cand_ei = -np.inf
            best_cand = None
            
            for cand in candidates:
                z_cand = self._encode_input(h_idx, cand)
                ei_cand = self._evaluate_with_counting(z_cand)
                if ei_cand > best_cand_ei:
                    best_cand_ei = ei_cand
                    best_cand = cand
            
            # 如果找到更好的点，更新
            if best_cand_ei > best_ei + 1e-6:
                x_t = best_cand
                best_ei = best_cand_ei
                best_x = best_cand.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # 如果连续多次没有改进，提前停止
            if no_improve >= max_no_improve:
                break
        
        return best_x, best_ei

    def _global_search_continuous(self, h_idx: int, n_samples: int = 20) -> List[np.ndarray]:
        """
        全局搜索策略：使用多种采样方法
        - h_idx: 离散选项索引
        - n_samples: 采样数量
        返回：候选点列表
        """
        candidates = []
        n_quant = len(self.config_fun.quant_index)
        
        # 1. 拉丁超立方采样
        lhd_samples = int(n_samples * 0.4)
        if lhd_samples > 0:
            rng = np.random.RandomState(42 + h_idx)
            lhd_z = self.config_fun.latinhypercube_sample(rng, lhd_samples)
            lhd_x = lhd_z[..., self.config_fun.quant_index]
            candidates.extend(lhd_x)
        
        # 2. 随机采样
        random_samples = int(n_samples * 0.3)
        if random_samples > 0:
            for _ in range(random_samples):
                x = np.random.uniform(0.0, 1.0, n_quant)
                candidates.append(x)
        
        # 3. 边界采样
        boundary_samples = int(n_samples * 0.2)
        if boundary_samples > 0:
            for _ in range(boundary_samples):
                x = np.random.choice([0.0, 1.0], size=n_quant)
                # 添加一些噪声
                x += np.random.normal(0, 0.1, n_quant)
                x = np.clip(x, 0.0, 1.0)
                candidates.append(x)
        
        # 4. 中心采样
        center_samples = n_samples - len(candidates)
        if center_samples > 0:
            for _ in range(center_samples):
                x = np.random.normal(0.5, 0.2, n_quant)
                x = np.clip(x, 0.0, 1.0)
                candidates.append(x)
        
        return candidates

    def run(self, init_size: int, N_cand: int, T: int, use_hybrid: bool = True):
        """
        init_size: 初始随机选择的离散点数
        N_cand: 每轮选取的离散候选个数
        T: 总迭代次数
        use_hybrid: 是否使用混合优化策略
        """

        # 1. 初始采样
        init_idxs = self.rng.choice(self.n, size=init_size, replace=False).tolist()
        for idx in init_idxs:
            if use_hybrid:
                x_opt, A_val = self.optimize_x_given_h_hybrid(idx)
            else:
                x_opt, A_val = self.optimize_x_given_h(idx)
            print(f"init_idxs: {idx}, x_opt: {x_opt}, A_val: {A_val}")
            self.dgps.update([idx], [A_val])
            # 真值评估
            # f_val = self.f_eval(idx, x_opt)
            # if self.maximize:
            #     if f_val > self.best_f:
            #         self.best_f = f_val
            # else:
            #     if f_val < self.best_f:
            #         self.best_f = f_val
            # self.acq.best_f = self.best_f
            # self.history.append((idx, x_opt, A_val, f_val))

        # 2. 迭代
        for t in range(init_size, T):
            mu, var = self.dgps.predict()
            # 取均值最大的 N_cand 作为候选（也可以用 EI on discrete）
            cand_idxs = np.argsort(mu)[-N_cand:]

            refined = []
            for idx in cand_idxs:
                if use_hybrid:
                    x_opt, A_val = self.optimize_x_given_h_hybrid(idx)
                else:
                    x_opt, A_val = self.optimize_x_given_h(idx)
                refined.append((idx, x_opt, A_val))
                self.dgps.update([idx], [A_val])

                # 真值评估
                # f_val = self.f_eval(idx, x_opt)
                # if self.maximize:
                #     if f_val > self.best_f:
                #         self.best_f = f_val
                # else:
                #     if f_val < self.best_f:
                #         self.best_f = f_val
                # self.acq.best_f = self.best_f
                # self.history.append((idx, x_opt, A_val, f_val))

        # 3. 最终选择（基于 GP 均值）
        mu, var = self.dgps.predict()
        best_idx = int(np.argmax(mu))
        print(f"mu: {mu}")
        
        print(f"best_idx: {best_idx}")
        if use_hybrid:
            best_x, best_acq = self.optimize_x_given_h_hybrid(best_idx)
        else:
            best_x, best_acq = self.optimize_x_given_h(best_idx)
        # best_val = self.f_eval(best_idx, best_x)

        print(f"GP方法总共调用了 {self.evaluate_count} 次 acquisition function evaluate")
        return best_acq


# ============================================================
# 测试函数
# ============================================================
def test_optimization_strategies():
    """
    测试不同优化策略的效果
    """
    import time
    
    # 模拟配置
    class MockConfigFun:
        def __init__(self):
            self.quant_index = [0, 1]  # 连续变量索引
            self.qual_index = [2, 3]   # 离散变量索引
            
        def latinhypercube_sample(self, rng, n):
            return np.random.uniform(0, 1, (n, 4))
    
    class MockAcqObj:
        def evaluate(self, x_tensor, num_samples=100):
            # 模拟一个简单的目标函数
            if hasattr(x_tensor, 'numpy'):
                x = x_tensor.numpy()
            else:
                x = x_tensor
            # 简单的二次函数，在(0.5, 0.5)处有最大值
            return -np.sum((x - 0.5)**2) + 0.1
    
    # 创建测试实例
    config_fun = MockConfigFun()
    acq_obj = MockAcqObj()
    combined_cat_index = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 4个离散选项
    model = SimpleTorchModel()
    rng = np.random.RandomState(42)
    
    # 创建优化器实例
    optimizer = MixedBayesOptGPDiscrete(
        acq_obj=acq_obj,
        rng=rng,
        config_fun=config_fun,
        combined_cat_index=combined_cat_index,
        model=model
    )
    
    # 测试不同策略
    strategies = [
        ("原始策略", False),
        ("混合策略", True)
    ]
    
    for name, use_hybrid in strategies:
        print(f"\n=== 测试 {name} ===")
        optimizer.evaluate_count = 0  # 重置计数器
        
        start_time = time.time()
        result = optimizer.run(init_size=2, N_cand=2, T=4, use_hybrid=use_hybrid)
        end_time = time.time()
        
        print(f"{name}结果: {result}")
        print(f"评估次数: {optimizer.evaluate_count}")
        print(f"耗时: {end_time - start_time:.4f}秒")

if __name__ == "__main__":
    test_optimization_strategies()



