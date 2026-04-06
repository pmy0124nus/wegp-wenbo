import numpy as np
import torch
from scipy.stats import norm
from wegp_bayes.optim import acquisition_functions
import time


class MixedSpaceLocalSearchBO:
    def __init__(
        self,
        acq_obj,           # EI 实例
        model,
        rng,
        config_fun,
        embedding_matrix,
        combined_cat_index,
        sigma_cat=1.0,
        sigma_cont=0.1,
        delta=0.5,
        N_cand=20,
        K_restart=5,
        eps=1e-6,
        T_max=100,
    ):
        self.acq_obj = acq_obj
        # self.C = cat_space
        # self.bounds = np.array(cont_bounds)
        self.model =model
        # 下面是搜索用的超参
        self.sigma_cat = sigma_cat
        self.sigma_cont = sigma_cont
        self.delta = delta
        self.N_cand = N_cand
        self.K_restart = K_restart
        self.eps = eps
        self.T_max = T_max
        self.rng = rng
        self.config_fun=config_fun
        self.embedding_matrix=embedding_matrix
        self.combined_cat_index=torch.tensor(combined_cat_index,dtype=torch.float64)

        # 添加计数器
        self.evaluate_count = 0

    def _evaluate_with_counting(self, x_tensor):
        """
        包装evaluate方法，同时计数
        """
        self.evaluate_count += 1
        return self.acq_obj.evaluate(x_tensor, num_samples=100)

    # ---------- 内部工具 ----------
    # def _concat_design(self, k, xs):
    #     """
    #     把 N 个 (cat, x) → (N, D_total) numpy array。
    #     你可以根据自己的 GP 输入格式改这里：
    #       - 如果 GP 需要整数类别列：return np.hstack([cats[:,None], xs])
    #       - 如果 GP 需要 one‑hot：自己写 one‑hot 再 hstack
    #     下面假设 **第 0 列放整数类别 ID** ：
    #     """
    #     cats = [self.combined_cat_index[i] for i in k]
    #     return np.hstack([cats, xs])

    def _sample_categorical(self, cur_cat):
        #cur_cat是当前类别组合的索引
        """
        根据 P(h') ∝ exp(-d_cat(h',cur_cat)^2 / (2 σ_cat^2))
        对所有类别做加权采样，返回 N_cand 个 h'
        """
        # 1. 计算当前类别到所有候选类别的距离
        #cur_cat -> k
        arr = self.combined_cat_index.numpy()
        target = cur_cat.numpy()

        rows = np.where((arr == target).all(axis=1))[0]
        cur_cat_index = int(rows[0])
        dists = self._calculate_dists(cur_cat_index)
        # 从gp中读取类别latents,计算latents之间的距离，也就是当前是0-3中的i，比如i是0，就计算01 02 03 的距离，这个要怎么写代码比较合适
        # 2. 计算未归一化权重
        weights = np.exp(-dists**2 / (2 * self.sigma_cat**2))
        # 3. 归一化为概率分布
        probs = weights / weights.sum()
        # 4. 按照概率分布采样
        sample_size = np.prod(list(self.config_fun.num_levels.values()))
        all_idxs = np.arange(sample_size)
        other_idxs = all_idxs[all_idxs != cur_cat_index] 
        
        # 使用传入的rng
        return self.rng.choice(other_idxs, size=self.N_cand, replace=True, p=probs)

    def _sample_continuous(self, cur_x):
        """
        从 N(cur_x, σ_cont^2 I) 中采样，丢弃掉 ∥x' - cur_x∥ > δ 或超出边界的点，
        直到收集到 self.N_cand 个有效样本。
        返回 shape=(N_cand, D) 的二维 np.ndarray。
        """
        import numpy as np

        # 确保 cur_x 为一维数组
        cur_x = np.asarray(cur_x).ravel()
        dim = cur_x.shape[0]

        xs = []
        while len(xs) < self.N_cand:
            remaining = self.N_cand - len(xs)
            # 一次生成 2 * remaining 个候选，加速筛选
            # 使用传入的rng
            candidates = cur_x + self.rng.randn(2 * remaining, dim) * self.sigma_cont

            # 计算与 cur_x 的距离
            dists = np.linalg.norm(candidates - cur_x, axis=1)

            # 初步筛选：距离限制
            mask = dists <= self.delta

            # 可选：如果定义了边界，过滤超出范围的点
            if hasattr(self, 'x_min') and hasattr(self, 'x_max'):
                in_bounds = np.all(candidates >= self.x_min, axis=1) & np.all(candidates <= self.x_max, axis=1)
                mask &= in_bounds

            valid = candidates[mask]

            # 收集前 self.N_cand 个有效样本
            for v in valid:
                xs.append(v)
                if len(xs) >= self.N_cand:
                    break

        # 将列表堆叠为二维数组，shape=(N_cand, dim)
        return np.stack(xs, axis=0)
    def _calculate_dists(self,cat_index):
        points_k=self.embedding_matrix[cat_index]
        dists = np.linalg.norm(self.embedding_matrix - points_k,axis=1)
        other_idxs = [i for i in range(len(self.embedding_matrix)) if i != cat_index]
        dists_others = dists[other_idxs]
        return dists_others


    # ---------- 单起点局部搜索 ----------
    def _local_search_from(self, start_cat, start_x):
        cat_t, x_t = start_cat, start_x
        cat_t = torch.tensor(cat_t,dtype=torch.float64)
        # x_t = torch.tensor(x_t,dtype=torch.float64)


        out = torch.cat((x_t,cat_t),dim=0)
        best_af = self._evaluate_with_counting(out)

        no_improve = 0

        for iter_count in range(self.T_max):
            # 1. 生成候选
            cat_index = self._sample_categorical(cat_t)
            # print(f"cat_index shape: {cat_index.shape}, cat_index: {cat_index}")
            cats = self.combined_cat_index[cat_index]  # shape: (N_cand, num_cat_vars)
            # print(f"cats shape: {cats.shape}")
            # print(f"combined_cat_index shape: {self.combined_cat_index.shape}")
            # print(f"cats[0]: {cats[0]}")
            # 确保 cats 是 numpy array
            if hasattr(cats, "detach"):
                cats = cats.detach().cpu()
            # 确保 cats 是二维数组
            if cats.ndim == 1:
                cats = cats.reshape(1, -1)
            # print(f"cats shape: {cats.shape}")
            xs = self._sample_continuous(x_t)
            design_batch = np.concatenate([xs, cats], axis=1)
            design_batch = torch.tensor(design_batch, dtype=torch.float64)
            # 2. 批量算 acquisition
            # print(f"design_batch shape: {design_batch.shape}")
            afs = self._evaluate_with_counting(design_batch)

            # print(f"afs shape: {afs.shape}") #这里错了 afs没有返回值
            # print(f"xs: {xs}")
            # 3. 拿最优
            idx_best = np.argmax(afs)
            # print(f"idx_best: {idx_best}, cats type: {type(cats)}, cats shape: {cats.shape if hasattr(cats, 'shape') else 'scalar'}")
            cat_new, x_new, af_new = cats[idx_best], xs[idx_best], afs[idx_best]

            # 4. 改进判定
            if af_new - best_af < self.eps:
                no_improve += 1
            else:
                cat_t, x_t, best_af = cat_new, x_new, af_new
                no_improve = 0

            # 5. 自适应重启
            if no_improve >= self.K_restart:
                init_z = torch.from_numpy(self.config_fun.latinhypercube_sample(self.rng, 16))
                init_cat=[]
                for i,e in enumerate(self.model.lv_weighting_layers):
                    arr = init_z[...,self.config_fun.qual_index[i]].long()# 每一个i代表一个categorical variable，但是我们计算距离的时候，得用所有的categorical variable的距离，怎么说呢
                    # 如果它原本是 torch.Tensor，就先转成 numpy
                    if hasattr(arr, "detach"):
                        arr = arr.detach().cpu().numpy()
                    init_cat.append(arr)
                paired_cat = [[x, y] for x, y in zip(init_cat[0], init_cat[1])]
                #这里只取[0]是因为重启的话只需要找一个starter，我们相当于找了16个
                paired_cat=torch.tensor(paired_cat[0],dtype=torch.float64)
                x_t = init_z[...,self.config_fun.quant_index][0] 

                best_af = self.acq_obj.evaluate(
                    torch.cat((x_t,cat_t)), num_samples=100
                )[0]
                no_improve = 0

            # 6. 收敛
            #这里收敛是什么意思 会很久收敛不了吗
            # if no_improve == 0 and af_new - best_af < self.eps:
            #     break

        return cat_t, x_t, best_af

    # ---------- 多起点 ----------
    def optimize(self, M=1):
        results = []
        for m in range(M):
            t0 = time.perf_counter()
            #1. 找到多起点，一开始可以先random选取多起点 （TODO，之后要修改， random选取起始点不是一个很好的办法）
            #todo,你找到的 qual index里的 是 每个类别分开的，这样肯定不行，要把不同的类别变量聚合在一起，一起进行距离计算！！！要先对坐标进行concat，你觉得呢？
            init_z = torch.from_numpy(self.config_fun.latinhypercube_sample(self.rng, 16))
            init_cat=[]
            for i,e in enumerate(self.model.lv_weighting_layers):
                arr = init_z[...,self.config_fun.qual_index[i]].long()# 每一个i代表一个categorical variable，但是我们计算距离的时候，得用所有的categorical variable的距离，怎么说呢
                # 如果它原本是 torch.Tensor，就先转成 numpy
                if hasattr(arr, "detach"):
                    arr = arr.detach().cpu().numpy()
                init_cat.append(arr)
            paired_cat = [[x, y] for x, y in zip(init_cat[0], init_cat[1])]
            init_x = init_z[...,self.config_fun.quant_index]
            #2. 以每一个起点为local search的中心，用greedy search（结束条件：1.达到最大搜索上限 2.已经找到附近的最优点），找到local里的最优点
            for j in range(init_z.shape[0]):
                cat_star, x_star, af_star = self._local_search_from(paired_cat[j], init_x[j])
                results.append((cat_star, x_star, af_star))
            t1= time.perf_counter()
            print(f"Total optimize acquisition function time: {t1 - t0:.4f} seconds")
            print(f"Local方法总共调用了 {self.evaluate_count} 次 acquisition function evaluate")
        # 找到acquisition值最大的结果
        best_result = max(results, key=lambda r: r[2])
        best_cat, best_x, best_af = best_result
        
        # 将cat和x拼接成一个tensor
        best_cat_tensor = torch.tensor(best_cat, dtype=torch.float64)
        best_x_tensor = torch.tensor(best_x, dtype=torch.float64)
        best_combined = torch.cat((best_x_tensor, best_cat_tensor), dim=0)
        
        return best_combined, best_af
    

    # def _sample_continuous(self, cur_x):
        """
        # 从 N(cur_x, σ_cont^2 I) 中采样，丢弃掉 ∥x' - cur_x∥ > δ 的点，
        # 直到收集到 N_cand 个有效样本
        # """
        # xs = []
        # print("cur_x",cur_x)
        # # 只要还没凑够，就继续采
        # while len(xs) < self.N_cand:
        #     # 在 cur_x 周围 Gaussian 采样
        #     cand = cur_x + np.random.randn(*cur_x.shape) * self.sigma_cont
        #     print("cand",cand)
        #     # 丢弃过远的点
        #     #TODO,超出x的bound的点也要被丢掉哦
        #     if np.linalg.norm(cand - cur_x) <= self.delta:
        #         xs.append(cand)
        #     print("xs",xs)
        # return np.array(xs)