
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
torch.set_num_threads(1)

import random
import argparse
import numpy as np
from joblib import dump, Parallel, delayed
from scipy.stats import norm
from tqdm import tqdm
import configs
import functions
import configs
from wegp_bayes.models import WEGP
import itertools
from wegp_bayes.optim import run_hmc_numpyro_wegp
from wegp_bayes.optim import acquisition_functions
from wegp_bayes.optim import optimize_acq_local
from wegp_bayes.optim.optimization_acq_discreteGP_cmaes import MixedBayesOptGPDiscreteCMAES
import jax
import time
import datetime

def _ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _log(msg):
    print(f"[{_ts()}] {msg}", flush=True)

jax.config.update("jax_enable_x64", False)

jax.config.update("jax_platform_name", "cpu")

import numpyro
numpyro.set_host_device_count(6)  

parser = argparse.ArgumentParser('Engg functions fully Bayesian - CMA-ES Optimization Version')
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--which_func', type=str, required=True)
parser.add_argument('--train_factor', type=int, required=True)
parser.add_argument('--n_jobs', type=int, required=True)
parser.add_argument('--n_repeats', type=int, default=30)
parser.add_argument('--maxfun', type=int, default=500)
parser.add_argument('--noise', type=bool, default=False)
parser.add_argument('--budget', type=int, default=10)
parser.add_argument('--num_permutations',type=int,default=0)
parser.add_argument('--cmaes_max_iter', type=int, default=40)
parser.add_argument('--cmaes_tol', type=float, default=1e-4)
parser.add_argument('--cmaes_sigma0', type=float, default=0.4)
parser.add_argument('--cmaes_lambda', type=int, default=8)
parser.add_argument('--cmaes_mu', type=int, default=4)
parser.add_argument('--n_starts', type=int, default=2)
parser.add_argument('--init_size', type=int, default=5)
parser.add_argument('--N_cand', type=int, default=10)
parser.add_argument('--num_samples', type=int, default=1500)
parser.add_argument('--warmup_steps', type=int, default=1500)
parser.add_argument('--num_model_samples', type=int, default=200)
parser.add_argument('--seeds', type=str, default=None, help='Comma-separated list of seeds to use')

args = parser.parse_args()
func = args.which_func
save_dir = os.path.join(
    args.save_dir,
    '%s/train_factor_%d' % (args.which_func, args.train_factor),
)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

_log(f"启动；save_dir={save_dir}, func={func}")

config_fun = getattr(configs, func)()
obj = getattr(functions, func)

def _calculate_embedding(model):
    """
    兼容 1~N 个类别变量的 embedding 计算：
    返回形状 [num_combos, sum(dim_i)] 的 numpy 数组
    """
    latents = []
    for e in model.lv_weighting_layers:
        arr = e.weighted_latents
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr_mean = arr.mean(axis=0)
        latents.append(arr_mean.astype(np.float64))

    if len(latents) == 1:
        return latents[0]

    level_ranges = [range(L.shape[0]) for L in latents]
    rows = []
    for idx_tuple in itertools.product(*level_ranges):
        chunks = [latents[i][idx_tuple[i], :] for i in range(len(latents))]
        rows.append(np.concatenate(chunks, axis=0))
    combined = np.vstack(rows)
    return combined

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def generate_latents(num_levels, num_permutations):
    all_permutations = list(itertools.permutations(range(num_levels)))

    def sample_permutations(all_permutations, num_permutations):
        selected_indices = np.random.choice(len(all_permutations), num_permutations, replace=False)
        selected_permutations = [all_permutations[i] for i in selected_indices]
        return selected_permutations

    def is_full_rank(permutations, num_levels, num_permutations):
        num_distances = num_levels * (num_levels - 1) // 2
        distance_matrix = torch.zeros((num_distances, num_permutations))
        for i, perm in enumerate(permutations):
            distances = [abs(perm[j] - perm[k]) for j in range(num_levels) for k in range(j + 1, num_levels)]
            distance_matrix[:, i] = torch.tensor(distances)
        return torch.linalg.matrix_rank(distance_matrix) >= min(distance_matrix.size())

    while True:
        permutations = sample_permutations(all_permutations, num_permutations)
        if is_full_rank(permutations, num_levels, num_permutations):
            perm_tensor = torch.tensor(permutations).T.float()
            return perm_tensor

def build_combined_cat_index(num_levels_per_var):
    value_lists = [np.arange(n, dtype=int) for n in num_levels_per_var]
    if len(value_lists) == 1:
        return value_lists[0][:, None]
    combos = np.array(list(itertools.product(*value_lists)), dtype=int)
    return combos

def main_script(seed):
    _log(f"[seed {seed}] 任务开始")
    t_seed0 = time.perf_counter()

    save_dir_seed = os.path.join(save_dir, 'seed_%d' % seed)
    if not os.path.exists(save_dir_seed):
        os.makedirs(save_dir_seed)
    _log(f"[seed {seed}] save_dir_seed={save_dir_seed}")

    num_levels_per_var = list(config_fun.num_levels.values())
    n_train = args.train_factor
    for n in num_levels_per_var:
        n_train *= n
    n_train = 20
    rng = np.random.RandomState(seed)

    tx_path = os.path.join(save_dir_seed, 'train_x.pt')
    ty_path = os.path.join(save_dir_seed, 'train_y.pt')

    if os.path.exists(tx_path) and os.path.exists(ty_path):
        _log(f"[seed {seed}] [init] 读取已有训练数据: {tx_path}, {ty_path}")
        train_x = torch.load(tx_path)
        train_y = torch.load(ty_path)
    else:
        _log(f"[seed {seed}] [init] 生成初始拉丁超立方样本，共 {n_train} 个")
        rng = np.random.RandomState(seed)
        t0 = time.perf_counter()
        train_x = torch.from_numpy(config_fun.latinhypercube_sample(rng, n_train))
        _log(f"[seed {seed}] [init] 计算初始目标值 obj(...)")
        train_y = [obj(config_fun.get_dict_from_array(x.numpy())) for x in train_x]
        train_y = torch.tensor(train_y).to(train_x)
        torch.save(train_x, tx_path)
        torch.save(train_y, ty_path)
        t1 = time.perf_counter()
        _log(f"[seed {seed}] [init] 生成并保存训练数据完成，用时 {t1 - t0:.2f}s")

    latents_list = []

    set_seed(seed)
    combined_cat_index = build_combined_cat_index(num_levels_per_var)
    _log(f"[seed {seed}] combined_cat_index 生成完成，形状={combined_cat_index.shape}")

    def default_permutation_num(input_list):
        return [(n * (n - 1)) // 2 for n in input_list]

    if args.num_permutations==0:
        num_permutations = default_permutation_num(num_levels_per_var)
        _log(f"[seed {seed}] 使用默认 num_permutations={num_permutations}")
    else:
        num_permutations = [args.num_permutations]*len(num_levels_per_var)
        _log(f"[seed {seed}] 使用自定义 num_permutations={num_permutations}")

    _log(f"[seed {seed}] 生成 latents_list ...")
    t_lat0 = time.perf_counter()
    for num_levels, num_perms in zip(num_levels_per_var, num_permutations):
        latents = generate_latents(
            num_levels=num_levels,
            num_permutations=num_perms
            )
        latents_list.append(latents)
    t_lat1 = time.perf_counter()
    _log(f"[seed {seed}] 生成 latents_list 完成，用时 {t_lat1 - t_lat0:.2f}s")

    _log(f"[seed {seed}] 构建 WEGP 模型")
    model = WEGP(
        train_x=train_x,
        train_y=train_y,
        quant_correlation_class='Matern32Kernel',
        qual_index=config_fun.qual_index,
        quant_index=config_fun.quant_index,
        num_levels_per_var=num_levels_per_var,
        num_permutations = num_permutations,
        latents_list=latents_list,
        noise=torch.tensor(0.25).double() if args.noise else None,
        fix_noise=args.noise)

    jax.config.update("jax_enable_x64", True)
    _log(f"[seed {seed}] 第一次 MCMC 开始（warmup={args.warmup_steps}, samples={args.num_samples}）")
    t_mcmc0 = time.perf_counter()
    run_hmc_numpyro_wegp(
        model,
        latents_list,
        num_samples=args.num_samples,
        warmup_steps=args.warmup_steps,
        max_tree_depth=7,
        disable_progbar=True,
        num_chains=1,  # 使用多链提升混合效果（此处保持原逻辑）
        num_model_samples=args.num_model_samples,
        seed=seed
    )
    t_mcmc1 = time.perf_counter()
    _log(f"[seed {seed}] 第一次 MCMC 完成，用时 {t_mcmc1 - t_mcmc0:.2f}s")

    best_y_list = []
    best_af_list = []
    _log(f"[seed {seed}] 进入 BO 主循环，总预算={args.budget}")
    for iteration in tqdm(range(args.budget)):
        _log(f"[seed {seed}] [iter {iteration}] 开始")
        best_f = train_y.max().item()
        best_y_list.append(best_f)

        _log(f"[seed {seed}] [iter {iteration}] 构建 EI_NUTS")
        acq_object = acquisition_functions.EI_NUTS(model, best_f)

        t_iter0 = time.perf_counter()
        _log(f"[seed {seed}] [iter {iteration}] 计算 embedding_matrix")
        embedding_matrix = _calculate_embedding(model)
        print("embedding_matrix",embedding_matrix.shape)
        embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

        _log(f"[seed {seed}] [iter {iteration}] 计算 K_qual 核矩阵")
        base = model.covar_module.base_kernel
        qual_kern = base.kernels[0]
        # with torch.no_grad():
        #     K_qual = qual_kern(embedding_matrix, embedding_matrix).evaluate()

        rng = np.random.RandomState(seed)
        best_f = train_y.max().item()
        acq_object = acquisition_functions.EI_NUTS(model, best_f)

        _log(f"[seed {seed}] [iter {iteration}] 初始化离散GP的 CMA-ES 优化器")
        # mbo = MixedBayesOptGPDiscreteCMAES(
        #     acq_object,
        #     rng,
        #     config_fun,
        #     combined_cat_index,
        #     model,
        #     kernel=K_qual,
        #     cmaes_max_iter=args.cmaes_max_iter,
        #     cmaes_tol=args.cmaes_tol,
        #     cmaes_sigma0=args.cmaes_sigma0,
        #     cmaes_lambda_=args.cmaes_lambda,
        #     cmaes_mu=args.cmaes_mu,
        #     n_starts=args.n_starts,
        #     num_model_samples=args.num_model_samples
        # )

        mbo = MixedBayesOptGPDiscreteCMAES(
            acq_object, rng, config_fun, combined_cat_index, model,
            gp_kernel=qual_kern,
            embeddings=embedding_matrix,
            kof_chunk=512,                     # 需要更省内存可调小如 512/256
            cmaes_max_iter=args.cmaes_max_iter,
            cmaes_tol=args.cmaes_tol,
            cmaes_sigma0=args.cmaes_sigma0,
            cmaes_lambda_=args.cmaes_lambda,
            cmaes_mu=args.cmaes_mu,
            n_starts=args.n_starts,
            num_model_samples=args.num_model_samples,
            )

        _log(f"[seed {seed}] [iter {iteration}] 运行 CMA-ES 优化 (init_size={args.init_size}, N_cand={args.N_cand})")
        t_cma0 = time.perf_counter()
        best_candidate, best_acq = mbo.run(
            init_size=args.init_size,
            N_cand=args.N_cand,
            iteration_count=iteration
        )
        t_cma1 = time.perf_counter()
        best_af_list.append(best_acq)
        _log(f"[seed {seed}] [iter {iteration}] CMA-ES 完成，用时 {t_cma1 - t_cma0:.2f}s；best_acq={best_acq}")

        _log(f"[seed {seed}] [iter {iteration}] 评估目标函数并扩充训练集")
        next_y = obj(config_fun.get_dict_from_array(best_candidate))
        next_x_tensor = torch.tensor(best_candidate, dtype=torch.float64, device=train_x.device).unsqueeze(0)
        train_x = torch.cat([train_x, next_x_tensor], dim=0)
        train_y = torch.cat([train_y, torch.tensor([next_y], dtype=torch.float64, device=train_y.device)], dim=0)
        current_max = train_y.max().item()
        _log(f"[seed {seed}] [iter {iteration}] 当前最大值: {current_max:.6f}, next_x: {next_x_tensor.numpy()}")

        _log(f"[seed {seed}] [iter {iteration}] 重建 WEGP 模型（保持原逻辑）")
        model = WEGP(
            train_x=train_x,
            train_y=train_y,
            quant_correlation_class='Matern32Kernel',
            qual_index=config_fun.qual_index,
            quant_index=config_fun.quant_index,
            num_levels_per_var=num_levels_per_var,
            num_permutations = num_permutations,
            latents_list=latents_list,
            noise=torch.tensor(0.25).double() if args.noise else None,
            fix_noise=args.noise)

        _log(f"[seed {seed}] [iter {iteration}] 第二次（本轮）MCMC 开始")
        t_mcmc_it0 = time.perf_counter()
        run_hmc_numpyro_wegp(
            model,
            latents_list,
            num_samples=args.num_samples,
            warmup_steps=args.warmup_steps,
            max_tree_depth=7,
            disable_progbar=True,
            num_chains=1,  # 保持原逻辑
            num_model_samples=args.num_model_samples,
            seed=seed
        )
        t_mcmc_it1 = time.perf_counter()
        t_iter1 = time.perf_counter()
        _log(f"[seed {seed}] [iter {iteration}] 本轮 MCMC 完成，用时 {t_mcmc_it1 - t_mcmc_it0:.2f}s")
        _log(f"[seed {seed}] [iter {iteration}] 结束，总用时 {t_iter1 - t_iter0:.2f}s")

        print(f"iteration: {iteration}-best_acq{best_acq},当前最大值: {current_max:.6f},next_x: {next_x_tensor.numpy()},time: {t_iter1 - t_iter0:.2f}s", flush=True)

    _log(f"[seed {seed}] 保存统计与模型 state_dict")
    stats = {
        'best_y_list': np.array(best_y_list),
        'best_af_list': np.array(best_af_list),
    }
    torch.save(model.state_dict(), os.path.join(save_dir_seed, f'WEGP_cmaes_buget_{args.budget}_{args.num_permutations}_N_cand{args.N_cand}state.pth'))
    dump(stats, os.path.join(save_dir_seed, f'stats_WEGP_cmaes_buget_{args.budget}_optimization_num_permutation_{args.num_permutations}.pkl'))

    t_seed1 = time.perf_counter()
    _log(f"[seed {seed}] 任务完成，总用时 {t_seed1 - t_seed0:.2f}s")

# 处理seeds参数
if args.seeds is not None:
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    _log(f"使用自定义seeds: {seeds}")
else:
    all_seeds = np.linspace(100,1000,args.n_repeats).astype(int)
    seeds = all_seeds[all_seeds<=193]
    _log(f"使用默认seeds: {seeds}")

_log(f"准备并行启动，共 {len(seeds)} 个 seed；n_jobs={args.n_jobs}")
t_all0 = time.perf_counter()
Parallel(n_jobs=args.n_jobs,verbose=0)(
    delayed(main_script)(seed) for seed in seeds
)
t_all1 = time.perf_counter()
_log(f"全部 seed 运行结束，总用时 {t_all1 - t_all0:.2f}s")
