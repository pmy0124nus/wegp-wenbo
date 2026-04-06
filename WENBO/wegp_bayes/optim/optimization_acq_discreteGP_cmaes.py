

import os
import gc
import numpy as np
import scipy.linalg
import random
import torch
from scipy.stats import norm
from wegp_bayes.optim.acquisition_functions import EI
import time
from typing import Callable, List, Sequence, Tuple, Optional

# Optional memory tracking (enabled via env var WEGP_MEMLOG=1)
_MEMLOG_ENABLED = os.environ.get("WEGP_MEMLOG", "0") == "1"
_MEMLOG_FREQ_GEN = int(os.environ.get("WEGP_MEMLOG_FREQ_GEN", "5"))  # CMA-ES generations
_MEMLOG_FREQ_KOF = int(os.environ.get("WEGP_MEMLOG_FREQ_KOF", "1"))  # DiscreteGP Kof chunks

def _rss_mb() -> float:
    """Return current process RSS in MB (best effort, Unix-friendly)."""
    # Try psutil if available (more accurate)
    try:
        import psutil  # type: ignore
        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 ** 2)
    except Exception:
        pass
    # Fallback to resource on Unix
    try:
        import resource  # type: ignore
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is KB on Linux, bytes on macOS; normalize to MB
        maxrss = float(rusage.ru_maxrss)
        # Heuristic: if value is too small, assume MB already
        if maxrss > 1e6:  # likely KB
            return maxrss / 1024.0
        return maxrss
    except Exception:
        return float('nan')

def _memlog(tag: str):
    if _MEMLOG_ENABLED:
        print(f"[MEM] {tag}: RSS={_rss_mb():.1f} MB", flush=True)

"""
使用CMA-ES优化的离散GP优化策略
- 使用CMA-ES替代简单局部搜索
- 自适应调整搜索参数
- 更高效的全局搜索能力
"""

# ============================================================
# CMA-ES优化器实现
# ============================================================
class CMAESOptimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) 优化器
    用于连续优化问题
    """
    def __init__(self, 
                 dim: int,
                 x0: np.ndarray = None,
                 sigma0: float = 0.3,
                 lambda_: int = None,
                 mu: int = None,
                 max_iter: int = 50,
                 tol: float = 1e-6):
        """
        初始化CMA-ES优化器
        
        Args:
            dim: 问题维度
            x0: 初始解，如果为None则随机生成
            sigma0: 初始步长
            lambda_: 种群大小，默认为4 + floor(3*ln(dim))
            mu: 父代大小，默认为lambda//2
            max_iter: 最大迭代次数
            tol: 收敛容差
        """
        self.dim = dim
        self.x0 = x0 if x0 is not None else np.random.uniform(0.1, 0.9, dim)
        self.sigma0 = sigma0
        self.lambda_ = lambda_ if lambda_ is not None else 4 + int(np.floor(3 * np.log(dim)))
        self.mu = mu if mu is not None else self.lambda_ // 2
        self.max_iter = max_iter
        self.tol = tol
        
        # 权重计算
        self.weights = np.log(self.lambda_ + 1) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        
        # 策略参数
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        
        # 初始化状态
        self.reset()
    
    def reset(self):
        """重置优化器状态"""
        self.x = self.x0.copy()
        self.sigma = self.sigma0
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.best_x = self.x.copy()
        self.best_f = -np.inf
    
    def optimize(self, objective_func: Callable[[np.ndarray], float], h_idx: int = None, encode_func=None, batch_eval_func=None) -> Tuple[np.ndarray, float]:
        """
        使用CMA-ES优化目标函数
        
        Args:
            objective_func: 目标函数，接受numpy数组，返回标量值
            h_idx: 离散选项索引，用于批量评估
            encode_func: 编码函数，用于将连续变量转换为tensor
            batch_eval_func: 批量评估函数
            
        Returns:
            (best_x, best_f): 最优解和最优值
        """
        self.reset()
        
        for generation in range(self.max_iter):
            if _MEMLOG_ENABLED and generation % max(1, _MEMLOG_FREQ_GEN) == 0:
                _memlog(f"CMAES gen={generation} start")
            # 生成候选解
            candidates = []
            for _ in range(self.lambda_):
                # 生成正态分布随机向量
                z = np.random.randn(self.dim)
                # 应用协方差矩阵
                x = self.x + self.sigma * (self.C @ z)
                t = np.mod(x, 2.0)
                x = 1.0 - np.abs(1.0 - t)   # 元素级镜像

                candidates.append(x)
            
            # 评估候选解
            fitness_values = []
            
            # 批量评估所有候选解
            if h_idx is not None and encode_func is not None and batch_eval_func is not None:
                try:
                    # 将候选解转换为tensor格式
                    candidate_tensors = []
                    for x in candidates:
                        x_tensor = encode_func(h_idx, x)
                        candidate_tensors.append(x_tensor)
                    
                    # 批量评估
                    batch_results = batch_eval_func(candidate_tensors, h_idx)
                    fitness_values = batch_results
                    
                except Exception as e:
                    print(f"批量评估失败，回退到单个评估: {e}")
                    # 回退到单个评估
                    for i, x in enumerate(candidates):
                        try:
                            f = objective_func(x)
                            fitness_values.append(f)
                        except Exception as e:
                            print(f"评估失败: {e}")
                            fitness_values.append(-np.inf)
            else:
                # 单个评估
                for i, x in enumerate(candidates):
                    try:
                        f = objective_func(x)
                        fitness_values.append(f)
                    except Exception as e:
                        print(f"评估失败: {e}")
                        fitness_values.append(-np.inf)
            
            # 选择最优个体
            sorted_indices = np.argsort(fitness_values)[::-1]  # 降序排列
            best_idx = sorted_indices[0]
            best_fitness = fitness_values[best_idx]
            
            # 更新最优解
            if best_fitness > self.best_f:
                self.best_f = best_fitness
                self.best_x = candidates[best_idx].copy()
            
            # 选择父代
            parents = [candidates[i] for i in sorted_indices[:self.mu]]
            parent_fitness = [fitness_values[i] for i in sorted_indices[:self.mu]]
            
            # 计算加权平均
            x_old = self.x.copy()
            self.x = np.average(parents, axis=0, weights=self.weights)
            
            # 更新进化路径
            y = (self.x - x_old) / self.sigma
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.C @ y)
            
            # 更新协方差矩阵
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (self.ps.reshape(-1, 1) @ self.ps.reshape(1, -1))
            
            # 更新步长
            self.sigma *= np.exp((np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1) / self.damps)
            
            # 检查收敛
            if self.sigma < self.tol:
                break
        
        return self.best_x, self.best_f

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

# ============================================================
# 惰性核 Provider（GPyTorch kernel + embeddings）
# ============================================================
class GPKernelProvider:
    """
    惰性子矩阵提供器：基于 (embeddings, gpytorch kernel) 现场计算核子块。
    仅支持 GPyTorch 风格的 kernel(Xr, Xc).evaluate()。
    """
    def __init__(self,
                 embeddings: torch.Tensor,   # [n, d]，建议 CPU + float32
                 gp_kernel,
                 dtype: torch.dtype = torch.float32):
        if embeddings.device.type != "cpu":
            embeddings = embeddings.to("cpu")
        self.X = embeddings.to("cpu", dtype=dtype).contiguous()
        self.kernel = gp_kernel
        self.n = int(self.X.shape[0])

    @torch.no_grad()
    def diag(self, idx: Optional[Sequence[int]] = None) -> np.ndarray:
        """
        Return diagonal of K(Xs, Xs) without materializing the full dense matrix.

        - Prefer LazyTensor .diag() when available to avoid NxN allocation.
        - Fallback to chunked evaluation of blocks to cap peak memory.
        """
        if idx is None:
            idx = np.arange(self.n, dtype=int)
        idx = np.asarray(idx, dtype=int)
        Xs = self.X[idx]

        try:
            # Try LazyTensor path: this should avoid building a full dense matrix
            K_lazy = self.kernel(Xs, Xs)                  # LazyTensor in GPyTorch
            if hasattr(K_lazy, "diag"):
                d = K_lazy.diag()
                return d.detach().cpu().numpy().astype(float)
        except Exception:
            # Fall back to chunked evaluation below
            pass

        # Chunked fallback: compute diag using smaller blocks
        b = Xs.shape[0]
        B = int(os.environ.get("WEGP_DIAG_CHUNK", "2048"))
        out = np.empty(b, dtype=float)
        for i in range(0, b, B):
            sl = slice(i, min(i + B, b))
            Xi = Xs[sl]
            K_block = self.kernel(Xi, Xi).evaluate()      # [bi, bi]
            d_block = torch.diagonal(K_block, dim1=-2, dim2=-1)
            out[sl] = d_block.detach().cpu().numpy().astype(float)
            # help GC
            del Xi, K_block, d_block
        return out

    @torch.no_grad()
    def submatrix(self, rows: Sequence[int], cols: Sequence[int]) -> np.ndarray:
        r = np.asarray(rows, dtype=int)
        c = np.asarray(cols, dtype=int)
        Xr = self.X[r]
        Xc = self.X[c]
        K = self.kernel(Xr, Xc).evaluate()                # [|r|, |c|]
        return K.cpu().numpy().astype(float)

# ============================================================
# Discrete GP（全量 n；Kof 分块；不接收稠密核）
# ============================================================
class DiscreteGP:
    def __init__(self,
                 index_number: int,
                 kernel_provider: GPKernelProvider,
                 kof_chunk: int = 2048):
        """
        index_number: 全部离散元素个数 n
        kernel_provider: GPKernelProvider
        kof_chunk: 计算 Kof(m×n) 时的列分块大小（越小越省内存）
        """
        self.n = int(index_number)
        self.Kp = kernel_provider
        if getattr(self.Kp, "n", None) != self.n:
            raise ValueError(f"kernel_provider.n 应为 {self.n}, 实际为 {getattr(self.Kp, 'n', None)}")
        self.observed_idx: List[int] = []
        self.A_obs: List[float] = []
        self.kof_chunk = int(kof_chunk)

    def update(self, idx_list: Sequence[int], A_list: Sequence[float]):
        idx_list = list(map(int, idx_list))
        A_arr = np.asarray(A_list, dtype=float).reshape(-1)
        if len(idx_list) != len(A_arr):
            raise ValueError("update: idx_list 与 A_list 长度不一致")
        finite_mask = np.isfinite(A_arr)
        if not np.all(finite_mask):
            print("Warning: 丢弃非有限 A 值:", A_arr[~finite_mask])
        idx_list = np.array(idx_list)[finite_mask].tolist()
        A_arr = A_arr[finite_mask]
        self.observed_idx.extend(idx_list)
        self.A_obs.extend(A_arr.tolist())

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.n
        if not self.observed_idx:
            mu = np.zeros(n, dtype=float)
            var = self.Kp.diag(None).astype(float)
            return mu, var

        o = np.asarray(self.observed_idx, dtype=int)
        A = np.asarray(self.A_obs, dtype=float)
        ensure_finite("A_obs", A)

        # Koo (m×m)
        _memlog(f"DiscreteGP.predict: before Koo (m={len(o)}, n={n})")
        Koo = self.Kp.submatrix(o, o).astype(float)
        ensure_finite("Koo", Koo)
        L = scipy.linalg.cholesky(add_jitter(Koo, 1e-8), lower=True, check_finite=True)
        alpha = scipy.linalg.cho_solve((L, True), A, check_finite=True)  # [m]
        if _MEMLOG_ENABLED:
            # help GC older temporaries
            gc.collect()
            _memlog("DiscreteGP.predict: after Koo chol/alpha")

        # 先取全对角
        var = self.Kp.diag(None).astype(float)  # [n]
        mu  = np.zeros(n, dtype=float)

        # 分块计算 Kof (m×n)
        B = self.kof_chunk
        all_cols = np.arange(n, dtype=int)
        for i in range(0, n, B):
            cols = all_cols[i:i+B]
            Kof = self.Kp.submatrix(o, cols).astype(float)    # [m, B]
            ensure_finite("Kof", Kof)
            mu[cols] = Kof.T @ alpha
            v = scipy.linalg.cho_solve((L, True), Kof, check_finite=True)  # [m, B]
            var[cols] -= np.sum(Kof * v, axis=0)
            if _MEMLOG_ENABLED and ((i // B) % max(1, _MEMLOG_FREQ_KOF) == 0):
                _memlog(f"DiscreteGP.predict: after Kof chunk i={i} size={len(cols)}")
                # Explicitly free chunk temporaries and collect
                del Kof, v
                gc.collect()

        var = np.maximum(var, 0.0)
        return mu, var

# ============================================================
# 使用CMA-ES的 Mixed Bayesian Optimization with Discrete GP
# （改：不再接收稠密 kernel；新增 gp_kernel / embeddings / kof_chunk）
# ============================================================
class MixedBayesOptGPDiscreteCMAES:
    def __init__(
        self,
        acq_obj,
        rng: np.random.RandomState,
        config_fun,
        combined_cat_index: np.ndarray,
        model,
        # ---- 改动开始：不再接收 dense kernel ----
        gp_kernel,                      # GPyTorch kernel，例如 qual_kern
        embeddings: torch.Tensor,       # [n, d] 的 embedding，建议 float32 + CPU
        kof_chunk: int = 512,          # Kof 分块大小（越小越省内存）
        # ----------------------------------------
        cmaes_max_iter: int = 50,
        cmaes_tol: float = 1e-6,
        cmaes_sigma0: float = 0.3,
        cmaes_lambda_: Optional[int] = None,
        cmaes_mu: Optional[int] = None,
        n_starts: int = 3,
        num_model_samples: int = 128,
    ):
        """
        rng: numpy RandomState
        config_fun: 连续空间采样工具
        combined_cat_index: shape [n_cat, m] 的离散特征编码矩阵（每个离散元素一行）
        model: surrogate，用于 EI
        gp_kernel: GPyTorch kernel（不再传稠密核矩阵）
        embeddings: torch.Tensor [n, d]，用于 kernel 计算
        kof_chunk: DiscreteGP 在全量 n 上做预测时 Kof 的分块大小
        cmaes_*: CMA-ES 相关参数
        """

        self.rng = rng
        self.config_fun = config_fun
        self.combined_cat_index = np.asarray(combined_cat_index, dtype=float)
        self.n = self.combined_cat_index.shape[0]
        self.model = model
        self.acq_obj = acq_obj

        # ---- 新：构建惰性核 provider + 全量 n 的 DiscreteGP ----
        # 强烈建议 float32 以省内存
        if embeddings.device.type != "cpu":
            embeddings = embeddings.to("cpu")
        embeddings = embeddings.to(dtype=torch.float32).contiguous()
        self.kernel_provider = GPKernelProvider(embeddings=embeddings, gp_kernel=gp_kernel, dtype=torch.float32)
        self.dgps = DiscreteGP(self.n, kernel_provider=self.kernel_provider, kof_chunk=kof_chunk)

        # CMA-ES参数
        self.cmaes_max_iter = cmaes_max_iter
        self.cmaes_tol = cmaes_tol
        self.cmaes_sigma0 = cmaes_sigma0
        self.cmaes_lambda_ = cmaes_lambda_
        self.cmaes_mu = cmaes_mu
        self.n_starts = n_starts
        self.num_model_samples = num_model_samples

        # 计数器
        self.evaluate_count = 0
        self.optimize_call_count = 0

        # 记录
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
        # 禁用梯度，避免不必要的 Autograd 内存
        with torch.no_grad():
            return self.acq_obj.evaluate(x_tensor, num_model_samples=self.num_model_samples)
    
    def _evaluate_batch_with_counting(self, x_tensors, h_idx):
        """
        批量评估EI，提高效率
        """
        if len(x_tensors) == 0:
            return []
        
        self.evaluate_count += len(x_tensors)
        
        # 将多个tensor堆叠成batch
        if isinstance(x_tensors[0], torch.Tensor):
            batch_tensor = torch.stack(x_tensors, dim=0)
        else:
            batch_tensor = torch.tensor(x_tensors, dtype=torch.float32)
        
        # 批量评估（禁用梯度，减少内存）
        with torch.no_grad():
            results = self.acq_obj.evaluate(batch_tensor, num_model_samples=self.num_model_samples)
        # 释放中间 batch tensor 引用，帮助及时回收
        del batch_tensor
        if _MEMLOG_ENABLED:
            gc.collect()
            _memlog(f"_evaluate_batch_with_counting: batch={len(x_tensors)} done")

        return results

    def _objective_function(self, h_idx: int, x: np.ndarray) -> float:
        """
        目标函数：将numpy数组转换为tensor并评估EI
        """
        try:
            x_tensor = self._encode_input(h_idx, x)
            return float(self._evaluate_with_counting(x_tensor))
        except Exception as e:
            print(f"目标函数评估失败: {e}")
            return -np.inf

    def optimize_x_given_h(self, h_idx: int, n_starts: int = None, iteration_count: int = 0) -> Tuple[np.ndarray, float]:
        """
        使用CMA-ES优化连续变量
        - h_idx: 离散选项索引
        - n_starts: 使用多少个不同的初始点，None时使用默认值
        - iteration_count: 当前迭代次数
        返回： (x_opt, A_val) 其中 A_val = EI(x_opt)
        """
        self.optimize_call_count += 1
        
        # 使用默认值或传入的值
        if n_starts is None:
            n_starts = self.n_starts
        
        # 获取连续变量维度
        cont_dim = len(self.config_fun.quant_index)
        
        best_x = None
        best_ei = -np.inf
        
        # 使用多个初始点运行CMA-ES
        _memlog(f"optimize_x_given_h start: h_idx={h_idx}, n_starts={n_starts}")
        for start_idx in range(n_starts):
            # 生成初始点
            if start_idx == 0:
                # 第一个初始点使用拉丁超立方采样
                try:
                    init_z = self.config_fun.latinhypercube_sample(self.rng, 1)
                    x0 = init_z[0, self.config_fun.quant_index]
                except ValueError:
                    x0 = self.rng.uniform(0.1, 0.9, cont_dim)
            else:
                # 其他初始点使用随机采样
                x0 = self.rng.uniform(0.1, 0.9, cont_dim)
            
            # 创建CMA-ES优化器
            cmaes = CMAESOptimizer(
                dim=cont_dim,
                x0=x0,
                sigma0=self.cmaes_sigma0,
                lambda_=self.cmaes_lambda_,
                mu=self.cmaes_mu,
                max_iter=self.cmaes_max_iter,
                tol=self.cmaes_tol
            )
            
            # 定义目标函数
            def objective_func(x):
                return self._objective_function(h_idx, x)
            
            # 运行CMA-ES优化
            try:
                x_opt, ei_val = cmaes.optimize(
                    objective_func, 
                    h_idx=h_idx,
                    encode_func=self._encode_input,
                    batch_eval_func=self._evaluate_batch_with_counting
                )
                
                # 更新最优解
                if ei_val > best_ei:
                    best_ei = ei_val
                    best_x = x_opt.copy()
                    
            except Exception as e:
                print(f"CMA-ES优化失败 (start {start_idx}): {e}")
                continue
            finally:
                if _MEMLOG_ENABLED:
                    gc.collect()
                    _memlog(f"optimize_x_given_h: after start_idx={start_idx}")
        
        # fallback: 如果所有CMA-ES运行都失败，使用随机采样
        if best_x is None:
            print("CMA-ES优化失败，使用随机采样作为fallback")
            x_random = self.rng.uniform(0.1, 0.9, cont_dim)
            best_x = x_random
            best_ei = self._objective_function(h_idx, x_random)

        _memlog(f"optimize_x_given_h end: h_idx={h_idx}, best_ei={best_ei}")
        return best_x, best_ei

    def run(self, init_size: int, N_cand: int, iteration_count: int = 0):
        """
        init_size: 初始随机选择的离散点数
        N_cand: 每轮选取的离散候选个数
        iteration_count: 当前迭代次数（从外部传入）
        """

        _memlog(f"run start: init_size={init_size}, N_cand={N_cand}, iter={iteration_count}")
        # 记录每个离散选项的优化结果
        self.optimized_results = {}  # {h_idx: (x_opt, A_val)}

        # 1. 初始采样
        # 使用传入的rng而不是重新生成种子
        temp_rng = np.random.RandomState(self.rng.get_state()[1][0] + iteration_count)
        # 确保init_size不超过离散选项总数
        actual_init_size = min(init_size, self.n)
        init_idxs = temp_rng.choice(self.n, size=actual_init_size, replace=False).tolist()
        
        # 优化初始离散选项
        for idx in init_idxs:
            x_opt, A_val = self.optimize_x_given_h(idx, iteration_count=iteration_count)
            self.optimized_results[idx] = (x_opt, A_val)
            self.dgps.update([idx], [A_val])

        # 根据GP的mu选择N_cand个候选点
        _memlog("run: before dgps.predict()")
        mu, var = self.dgps.predict()
        _memlog("run: after dgps.predict()")
        # 确保N_cand不超过离散选项总数
        actual_N_cand = min(N_cand, self.n)
        # 取mu最大的N_cand个作为候选
        cand_idxs = np.argsort(mu)[-actual_N_cand:]

        # 对每个候选点进行优化
        for idx in cand_idxs:
            if idx not in self.optimized_results:
                # 如果这个离散选项还没有优化过，就优化一下
                x_opt, A_val = self.optimize_x_given_h(idx, iteration_count=iteration_count)
                self.optimized_results[idx] = (x_opt, A_val)
                if _MEMLOG_ENABLED:
                    gc.collect()
                    _memlog(f"run: optimized cand h={idx}")

        # 从N个候选的observation中选择最大的
        best_idx = None
        best_acq = -np.inf
        best_x = None
        
        for idx in cand_idxs:
            x_opt, A_val = self.optimized_results[idx]
            if A_val > best_acq:
                best_acq = A_val
                best_x = x_opt
                best_idx = idx
        
        # 将best_idx转换为对应的离散变量h
        best_h = self.combined_cat_index[best_idx]
        
        # 将离散变量h和连续变量x拼接成完整的候选点
        best_candidate = np.concatenate([best_x, best_h])

        print(f"CMA-ES优化方法总共调用了 {self.evaluate_count} 次 acquisition function evaluate")
        _memlog("run end")
        return best_candidate, best_acq
