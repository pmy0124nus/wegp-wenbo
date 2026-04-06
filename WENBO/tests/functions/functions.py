import math
import numpy as np
from typing import Dict
import numpy as np


def borehole(params:Dict)->float:
    numerator = 2*math.pi*params['T_u']*(params['H_u']-params['H_l'])
    den_term1 = math.log(params['r']/params['r_w'])
    den_term2 = 1+ 2*params['L']*params['T_u']/(den_term1*params['r_w']**2*params['K_w']) + \
        params['T_u']/params['T_l']
    
    return numerator/den_term1/den_term2

def piston(params:Dict)->float:
    A = params['P_0']*params['S'] + 19.62*params['M'] - params['k']*params['V_0']/params['S']
    term1 = params['P_0']*params['V_0']/params['T_0']*params['T_a']
    V = params['S']/2/params['k']*(math.sqrt(A**2 + 4*term1)-A)
    term2 = params['k'] + (params['S']**2)*term1/(V**2)
    return 2*math.pi*math.sqrt(params['M']/term2)

def myrosenbrock(x1,x2):
    fx = 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2
    return fx / 300

# =============================================================================
#  Six-hump Camel Function (f_min = - 1.0316 )
#  https://www.sfu.ca/~ssurjano/camel6.html       
# =============================================================================
def mysixhumpcamp(x1,x2):
    
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
    fval = term1 + term2 + term3
    return fval/ 10

# =============================================================================
# Beale function (f_min = 0)
# https://www.sfu.ca/~ssurjano/beale.html
# =============================================================================
def mybeale(x1,x2):
    x1= x1*2
    x2= x2*2
    fval = (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (
            2.625 - x1 + x1 * x2 ** 3) ** 2
    return fval / 50

def func2C(params: Dict) -> float:
    if params['ht1'] == 0:  # rosenbrock
        f = myrosenbrock(params['x1'],params['x2'])
    elif params['ht1'] == 1:  # six hump
        f = mysixhumpcamp(params['x1'],params['x2'])
    elif params['ht1'] == 2:  # beale
        f = mybeale(params['x1'],params['x2'])

    if params['ht2'] == 0:  # rosenbrock
        f = f + myrosenbrock(params['x1'],params['x2'])
    elif params['ht2']== 1:  # six hump
        f = f + mysixhumpcamp(params['x1'],params['x2'])
    else:
        f = f + mybeale(params['x1'],params['x2'])

    # y = f + 1e-6 * np.random.rand(f.shape[0], f.shape[1])
    y=-f
    # print("y",y)
    # input()
    
    return y


def func3C(params):
    # X = np.array([params['x1'], params['x2']]) * 2

    if params['ht1'] == 0:
        f = myrosenbrock(params['x1'],params['x2'])

    elif params['ht1'] == 1:
        f = mysixhumpcamp(params['x1'],params['x2'])
    elif params['ht1'] == 2:
        f = mybeale(params['x1'],params['x2'])

    if params['ht2'] == 0:
        f = f + myrosenbrock(params['x1'],params['x2'])
    elif params['ht2'] == 1:
        f = f + mysixhumpcamp(params['x1'],params['x2'])
    else:
        f = f + mybeale(params['x1'],params['x2'])

    if params['ht3'] == 0:
        f = f + 5 * mysixhumpcamp(params['x1'],params['x2'])
    elif params['ht3'] == 1:
        f = f + 2 * myrosenbrock(params['x1'],params['x2'])
    else:
        f = f + params['ht3'] * mybeale(params['x1'],params['x2'])
    y=f
    return -y


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing
import os

def load_california_housing_dataset():
    """
    Load the California housing dataset with robust error handling.
    If the default cache is corrupted (e.g., joblib/pickle KeyError 192),
    fall back to a fresh local data_home inside the repo to force re-download.
    """
    try:
        return fetch_california_housing(return_X_y=True)
    except Exception as e:
        # Common with corrupted cache files: KeyError during unpickling
        # Use a clean, repo-local cache directory to force a fresh download
        data_dir = os.path.join(os.getcwd(), "sklearn_data_cache")
        try:
            os.makedirs(data_dir, exist_ok=True)
        except Exception:
            pass
        try:
            return fetch_california_housing(return_X_y=True, data_home=data_dir)
        except Exception as e2:
            # Surface a clear error so users can fix cache manually if needed
            raise RuntimeError(
                "Failed to load California housing dataset. "
                "Tried default cache and fresh data_home at 'sklearn_data_cache'. "
                "Consider deleting your '~/.scikit_learn_data' California dataset cache and retry."
            ) from e2

def mlp_mse(params):
    # Load dataset with proper error handling
    X, y = load_california_housing_dataset()
    from sklearn.preprocessing import StandardScaler    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # print("X",X)

    # Define hyperparameter options
    switch_act_fun = {0: 'logistic', 1: 'tanh', 2: 'relu'}
    switch_learning = {0: 'constant', 1: 'invscaling', 2: 'adaptive'}
    switch_solver = {0: 'sgd', 1: 'adam', 2: 'lbfgs'}
    switch_stopping = True

    # Map h_list to hyperparameters
    act = switch_act_fun[params['ht1']]
    lea = switch_learning[params['ht2']]
    sol = switch_solver[params['ht3']]
    sto = switch_stopping

    # Set MLPRegressor hyperparameters
    mlp_reg = MLPRegressor(random_state=0, activation=act, learning_rate=lea, solver=sol, early_stopping=sto,
                           hidden_layer_sizes=int(params['x1']), alpha=params['x2'], tol=params['x3'], max_iter=5000)
    
    # Evaluate the model using cross-validation
    score = np.mean(cross_val_score(mlp_reg, X, y, cv=5, n_jobs=4, scoring="neg_mean_squared_error"))
    log_score = np.log(np.absolute(score))
    
    return -np.absolute(log_score)



def svm_mse(params):
    """
    Evaluation function for SVR model using Mean Squared Error (MSE).

    Parameters:
    - params (dict): Dictionary containing hyperparameters:
        - 'h1': Categorical variable (0: 'poly', 1: 'rbf', 2: 'sigmoid')
        - 'C': Continuous variable (0.1 to 100)
        - 'epsilon': Continuous variable (0.01 to 1)

    Returns:
    - float: The absolute logarithm of the mean negative MSE from cross-validation.
    """
    # Load and preprocess the dataset with the same robust loader
    X, y = load_california_housing_dataset()
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Define hyperparameter options
    switch_kernel = {0: 'poly', 1: 'rbf', 2: 'sigmoid', 3:'linear'}

    # Map categorical hyperparameter to actual kernel
    kernel = switch_kernel.get(params['h1'], 'rbf')  # Default to 'rbf' if key not found
    C = params['C']
    epsilon = params['epsilon']

    # Initialize the SVR model with given hyperparameters
    svr = SVR(gamma='scale', C=C, kernel=kernel, epsilon=epsilon)

    # Evaluate the model using cross-validation
    # Using 5-fold CV for better stability
    cv_scores = cross_val_score(svr, X, y, cv=5, n_jobs=-1, scoring="neg_mean_squared_error")
    mean_neg_mse = np.mean(cv_scores)
    
    # To avoid log of zero or negative, take absolute and add a small epsilon if necessary
    absolute_score = np.abs(mean_neg_mse) + 1e-8
    log_score = np.log(absolute_score)

    return -np.abs(log_score)
def Ackley3_4(params):
    # minimum = 0
    a = 20
    b = 0.2
    c = 2 * np.pi
    
    # 提取 param 中的所有变量，按照 'x1' 到 'ht50' 的顺序
    keys = ['x1', 'x2', 'x3'] + [f'ht{i}' for i in range(1,4)]
    
    # 将这些变量的值组织成一个数组 X
    X = np.array([params[key] for key in keys])
    
    # 原先的公式保持不变
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(np.square(X)) / 6))
    cos_term = -1 * np.exp(np.sum(np.cos(c * np.copy(X)) / 6))
    result = a + np.exp(1) + sum_sq_term + cos_term
    
    return -result


####################### nn yacht ######################


from typing import Dict, Tuple
import os
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 允许通过环境变量改写存放路径和下载地址
DATA_PATH = os.environ.get("YACHT_DATA_PATH", "yacht_hydrodynamics.data")
DEFAULT_URLS = [
    # UCI ML Repository（常用、稳定）
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
]
# ======== 超参索引 -> 实际值的映射 ========
_ACTS = {0: nn.ReLU, 1: nn.Tanh, 2: nn.Sigmoid}
_OPTS = {
    0: torch.optim.SGD,
    1: torch.optim.Adam,
    2: torch.optim.RMSprop,
    3: torch.optim.Adagrad,
}
_DROP_LIST = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]  # h3 索引到 p_drop

# ======== 模型 ========
class OneHiddenDropoutRegressor(nn.Module):
    def __init__(self, in_dim: int, width: int, p_drop: float, act_cls):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, width)
        self.act = act_cls()
        self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(width, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)  # 训练&测试(MC)都生效（测试阶段会保持 train() 以启用dropout）
        x = self.fc2(x)
        return x.squeeze(-1)

# ======== 工具函数 ========
def _download_yacht_data(path: str) -> None:
    """下载 yacht_hydrodynamics.data 文件"""
    import urllib.request
    import os
    
    # 确保目录存在
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    # 尝试从多个URL下载
    for url in DEFAULT_URLS:
        try:
            print(f"正在从 {url} 下载 yacht_hydrodynamics.data...")
            urllib.request.urlretrieve(url, path)
            print(f"下载完成: {path}")
            return
        except Exception as e:
            print(f"从 {url} 下载失败: {e}")
            continue
    
    raise FileNotFoundError(f"无法从任何URL下载 yacht_hydrodynamics.data 文件")

def _load_yacht(path: str) -> Tuple[np.ndarray, np.ndarray]:
    # UCI Yacht Hydrodynamics：7 列，前6为特征，最后1列为目标
    if not os.path.exists(path):
        print(f"文件 {path} 不存在，尝试自动下载...")
        _download_yacht_data(path)
    
    try:
        df = pd.read_csv(path, header=None, sep=r"\s+")
    except Exception:
        df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = df.iloc[:, -1].to_numpy(dtype=np.float32)
    return X, y

def _denorm_lr(x1_norm: float) -> float:
    # [-1,1] -> 10^[−5,−1]
    exp_ = -5 + (x1_norm + 1.0) * 0.5 * ( -1 - (-5) )
    return float(10.0 ** exp_)

def _denorm_width(x2_norm: float) -> int:
    # [-1,1] -> 2^[3,5] （减少网络宽度范围，加快训练）
    k = 3 + (x2_norm + 1.0) * 0.5 * (5 - 3)
    return int(2 ** int(round(k)))

def _denorm_aleatoric(x3_norm: float) -> float:
    # [-1,1] -> [0.2, 0.8]
    return 0.2 + (x3_norm + 1.0) * 0.5 * (0.8 - 0.2)

# ======== 主测试函数（接口名保持 func2C） ========
def nnYacht(params: Dict) -> float:
    """
    需要的键：h1, h2, h3, x1, x2, x3
    返回值：-NLL（取负号便于使用“最大化”优化器）
    """
    # ---- 反归一化 / 解析离散超参 ----
    act_idx = int(params['h1'])
    opt_idx = int(params['h2'])
    drop_idx = int(params['h3'])
    lr = _denorm_lr(float(params['x1']))
    width = _denorm_width(float(params['x2']))
    alea_var = _denorm_aleatoric(float(params['x3']))

    act_cls = _ACTS[act_idx]
    opt_cls = _OPTS[opt_idx]
    p_drop = _DROP_LIST[drop_idx]

    # ---- 数据 ----
    X, y = _load_yacht(DATA_PATH)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.05, random_state=0)  # 减少测试集大小

    x_scaler = StandardScaler().fit(X_tr)
    y_scaler = StandardScaler().fit(y_tr.reshape(-1, 1))

    X_tr = x_scaler.transform(X_tr).astype(np.float32)
    X_te = x_scaler.transform(X_te).astype(np.float32)
    y_tr_z = y_scaler.transform(y_tr.reshape(-1, 1)).astype(np.float32).ravel()
    y_te_z = y_scaler.transform(y_te.reshape(-1, 1)).astype(np.float32).ravel()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr_z)),
        batch_size=256, shuffle=True, drop_last=False  # 增大batch size减少迭代次数
    )

    # ---- 训练（MSE, 10 epochs，减少训练轮数） ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OneHiddenDropoutRegressor(
        in_dim=X.shape[1], width=width, p_drop=p_drop, act_cls=act_cls
    ).to(device)
    optimizer = opt_cls(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    model.train()
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(10):  # 从20减少到10
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = mse(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # 简单早停机制
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # ---- MC Dropout 推断（10 次，大幅减少采样次数） ----
    Xte = torch.from_numpy(X_te).to(device)
    model.train()  # 关键：保持 dropout 开启
    preds = []
    with torch.no_grad():
        for _ in range(10):  # 从100减少到10
            preds.append(model(Xte).detach().cpu().numpy())
    preds = np.stack(preds, axis=0)           # [T,N]
    mu_z = preds.mean(axis=0)
    var_epistemic_z = preds.var(axis=0, ddof=1)

    # 把观测噪声方差换算到 z-score 空间
    std_y = float(y_scaler.scale_[0])
    alea_var_z = alea_var / (std_y ** 2)

    sigma2 = np.clip(var_epistemic_z + alea_var_z, 1e-9, None)
    yte_z = y_te_z

    # Gaussian NLL（逐点后取平均）
    nll = 0.5 * (np.log(2 * math.pi * sigma2) + (yte_z - mu_z) ** 2 / sigma2)
    nll_mean = float(nll.mean())

    # 与原例保持一致：返回负号，便于"最大化"优化器
    return -nll_mean

########################nas 101###########################
# function.py
# 定义 NASBench-101（CIFAR-10）目标函数：给定混合输入（5 类别 + 21 连续），
# 固定 top-k=9 生成架构并评估，返回 -mean_accuracy 便于“最小化”优化。

import os
from typing import Dict, List, Tuple
import numpy as np
# from nasbench import api  # pip install nasbench

# 如果需要：pip install git+https://github.com/google-research/nasbench

# ===== 常量（与 NASBench-101 定义一致）=====
# INPUT   = 'input'
# OUTPUT  = 'output'
# CONV1X1 = 'conv1x1-bn-relu'
# CONV3X3 = 'conv3x3-bn-relu'
# MAXPOOL3X3 = 'maxpool3x3'

# NUM_NODES = 7
# EDGE_COUNT_LIMIT = 9   # NASBench-101 允许的最大边数（本问题固定 top-k=9）

# # 评估设置
# EPOCHS = 108
# ASBENCH_PATH = '/home/longtao/PycharmProjects/WEGP/WEBO_0821/nasbench/nasbench_only108.tfrecord'
# # 目标可选 'val' 或 'test'，默认用验证集（更贴近搜索阶段）
# OBJECTIVE = 'val'

# # 类别名到 NASBench 运算符名映射
# OPS_MAP = {
#     'conv3x3': CONV3X3,
#     'conv1x1': CONV1X1,
#     'maxpool3x3': MAXPOOL3X3
# }

# # 全局缓存 NASBench 句柄，避免每次 IO 反复加载
# _NASBENCH = None
# def _get_nasbench() -> api.NASBench:
#     global _NASBENCH
#     if _NASBENCH is None:
#         _NASBENCH = api.NASBench(NASBENCH_PATH)
#     return _NASBENCH

# def _upper_tri_edges(n: int = NUM_NODES) -> List[Tuple[int, int]]:
#     """与 config 中变量命名顺序严格一致的边顺序：i<j 的上三角"""
#     return [(i, j) for i in range(n) for j in range(i + 1, n)]

# def _extract_ops_and_edge_probs(params: Dict) -> Tuple[List[str], np.ndarray]:
#     """从 BO 参数中读出 ops 和 edge 概率（顺序与 _upper_tri_edges 对齐）"""
#     # 五个中间节点算子：op1..op5
#     try:
#         middle_ops = [OPS_MAP[params[f'op{k}']] for k in range(1, 6)]
#     except KeyError as e:
#         raise ValueError(f"未知的算子类别：{e}. 合法取值：{list(OPS_MAP.keys())}")

#     ops = [INPUT] + middle_ops + [OUTPUT]

#     # 21 条边的概率：edge_p_i_j
#     edge_names = [f'edge_p_{i}_{j}' for (i, j) in _upper_tri_edges()]
#     try:
#         edge_probs = np.array([float(params[name]) for name in edge_names], dtype=float)
#     except KeyError as e:
#         raise ValueError(f"缺少边概率参数：{e}. 需要的键包括：{edge_names[:3]}... 共 21 个")
#     if np.any(edge_probs < 0.0) or np.any(edge_probs > 1.0):
#         raise ValueError("edge 概率必须在 [0,1] 范围内。")

#     return ops, edge_probs


# def _build_spec_from_probs(ops: List[str], edge_probs: np.ndarray, k: int = EDGE_COUNT_LIMIT) -> api.ModelSpec:
#     """
#     按概率取 Top-k 条边，随后做一次“IO 连通性修复”，确保至少存在 0->...->N-1 的路径；
#     若修复新增了若干“骨干边”，会用已选边中最低概率的边做置换，保证总边数不超过 k。
#     """
#     n = NUM_NODES
#     edges = _upper_tri_edges(n)
#     if edge_probs.shape[0] != len(edges):
#         raise ValueError(f"edge_probs 长度应为 {len(edges)}，当前为 {edge_probs.shape[0]}")
#     if not (0 <= k <= EDGE_COUNT_LIMIT):
#         raise ValueError(f"k 必须在 0..{EDGE_COUNT_LIMIT} 之间")

#     # 1) 选出分数最高的 k 条边
#     top_idx = np.argsort(edge_probs)[::-1][:k]
#     selected = set(top_idx.tolist())
#     print("selected",selected)

#     def _mat_from_selected(selected_idx: set) -> np.ndarray:
#         mat = np.zeros((n, n), dtype=int)
#         for idx in selected_idx:
#             i, j = edges[idx]
#             mat[i, j] = 1
#         return mat

#     def _has_path_io(mat: np.ndarray) -> bool:
#         # 简单 DFS/BFS 检查 0 -> n-1 是否存在路径（图本身是上三角，天然有向无环）
#         from collections import deque
#         q = deque([0])
#         seen = {0}
#         while q:
#             u = q.popleft()
#             if u == n - 1:
#                 return True
#             for v in range(u + 1, n):
#                 if mat[u, v] and v not in seen:
#                     seen.add(v)
#                     q.append(v)
#         return False

#     # 2) 初始矩阵
#     mat = _mat_from_selected(selected)

#     # 3) 若无 0->N-1 路径，补一条“骨干链” 0->1->2->...->N-1
#     if not _has_path_io(mat):
#         print("false")
#         backbone = [(i, i + 1) for i in range(n - 1)]  # 一共 n-1 条边（对 NASBench-101，n=7 => 6 条）
#         backbone_idx = []
#         for (i, j) in backbone:
#             # 找到该边在 edges 里的索引
#             try:
#                 idx = edges.index((i, j))
#             except ValueError:
#                 # 理论不会发生，因为 edges 是完整上三角列表
#                 continue
#             backbone_idx.append(idx)

#         # 需要新增的骨干边（当前没选中的）
#         need_add = [idx for idx in backbone_idx if idx not in selected]
#         if need_add:
#             # 若新增会超标，需要移除若干“当前已选中里，概率最低且不在骨干里的”边
#             room = max(0, k - len(selected))
#             lack = len(need_add) - room
#             if lack > 0:
#                 # 可移除集合：selected - backbone_idx，按概率从低到高移除 lack 条
#                 removable = [idx for idx in selected if idx not in backbone_idx]
#                 removable.sort(key=lambda ii: edge_probs[ii])  # 从低概率开始移除
#                 if len(removable) < lack:
#                     # 极端情况下（k 很小且大多已是骨干边），允许移除非骨干边不够时，移除一部分骨干中概率最低的已选边
#                     extra_need = lack - len(removable)
#                     backbone_selected = [ii for ii in selected if ii in backbone_idx]
#                     backbone_selected.sort(key=lambda ii: edge_probs[ii])
#                     removable += backbone_selected[:extra_need]
#                 # 执行移除
#                 for ii in removable[:lack]:
#                     selected.remove(ii)
#             # 加入骨干边
#             selected.update(need_add)
#             # 再次截断到 k（理论上已满足）
#             if len(selected) > k:
#                 # 非骨干里概率最低的多余边再裁掉
#                 overflow = len(selected) - k
#                 non_backbone_selected = [ii for ii in selected if ii not in backbone_idx]
#                 non_backbone_selected.sort(key=lambda ii: edge_probs[ii])
#                 for ii in non_backbone_selected[:overflow]:
#                     selected.remove(ii)

#         # 重建矩阵
#         mat = _mat_from_selected(selected)

#     # 4) 生成 ModelSpec（此时应满足：上三角、节点数合法、且存在至少一条 IO 路径）
#     spec = api.ModelSpec(matrix=mat.tolist(), ops=ops)
#     return spec

# def nas101(params: Dict) -> float:
#     """
#     目标函数：给定
#       - 类别：op1..op5 ∈ {'conv3x3','conv1x1','maxpool3x3'}
#       - 连续：edge_p_i_j ∈ [0,1]（21 条）
#     固定 top-k=9 生成架构，并返回 -mean_accuracy（默认验证集），便于“最小化”优化器使用。
#     遇到非法拓扑（不连通/挂点等）时返回惩罚 1.0（因为 -acc ∈ [-1,0]，1.0 明显更差）。
#     """
#     # 1) 解析参数并构图
#     try:
#         ops, edge_probs = _extract_ops_and_edge_probs(params)
#         spec = _build_spec_from_probs(ops, edge_probs, k=EDGE_COUNT_LIMIT)
#     except Exception:
#         print("fail")
#         # 非法配置直接给大惩罚，避免 BO 偏好它
#         return 1.0

#     # 2) 评估（优先用 get_metrics_from_spec 拿 3 次重复的统计，不计预算）
#     nasbench = _get_nasbench()
#     fixed, computed = nasbench.get_metrics_from_spec(spec)

#     if EPOCHS not in computed or len(computed[EPOCHS]) == 0:
#         # 兜底：若没有预计算指标，执行一次 query（会计入预算）
#         data = nasbench.query(spec, epochs=EPOCHS)
#         acc = data['validation_accuracy'] if OBJECTIVE == 'val' else data['test_accuracy']
#         return -float(acc)

#     runs = computed[EPOCHS]
#     if OBJECTIVE == 'val':
#         accs = [r['final_validation_accuracy'] for r in runs]
#     else:
#         accs = [r['final_test_accuracy'] for r in runs]

#     mean_acc = float(np.mean(accs))
#     print("mean_acc",mean_acc)
#     return -mean_acc



# def llm_gsm8k(params: dict) -> float:
#     """
#     仅使用 top_p 解码：
#     - 连续：temperature、top_p
#     - 类别：tpl/style
#     - 固定：language='zh'，max_tokens=256；few-shot=0
#     """
#     import os, re, time
#     from typing import List, Tuple
#     try:
#         from datasets import load_dataset
#     except Exception:
#         return 0.0
#     try:
#         from openai import OpenAI
#     except Exception:
#         return 0.0

#     STRATEGY_TEMPLATE_SET = ["rubric", "critique", "cot"]
#     STYLE_SET=[ "step_by_step", "formal", "creative"]
#     TEMPLATE_TO_SYSTEM = {
#     # "qa": "Answer the question directly with minimal reasoning.",
#     "rubric":   "Knowns → equation → compute → check → final number.",
#     "critique": "Draft briefly, self-check quickly, then corrected final number.",
#     "cot":      "Solve in 2–4 ultra-brief steps."
#     }
#     STYLE_TO_PREFIX = {
#         "step_by_step": "Show steps briefly.",
#         "formal":       "Be precise.",
#         "creative":     "Be engaging."
#     }
    
#     # TEMPLATE_TO_SYSTEM = {
#     #     "qa": "Answer the question directly with minimal reasoning.",
#     #     "rubric": "Solve according to this rubric: identify knowns, set up equation, compute carefully, verify units, then give the final answer.",
#     #     "critique": "Draft a solution, self-critique any arithmetic or logic mistakes, then give the corrected final answer.",
#     #     "cot": "Think step by step. Show concise reasoning, then produce the final numeric answer."
#     # }
#     LANG_TO_SUFFIX = {
#         "en": "Answer in English.",
#         "zh": "请使用中文回答。",
#         "zh_en_mix": "Answer bilingually: first Chinese, then English."
#     }
#     # STYLE_TO_PREFIX = {
#     #     "concise": "Be concise and direct.",
#     #     "step_by_step": "Explain the steps clearly before the final answer.",
#     #     "formal": "Use a formal and precise tone.",
#     #     "creative": "Use an engaging tone while staying accurate."
#     # }

#     def extract_answer(text: str):
#         if not text:
#             return None
#         m = re.findall(r'(?:final\s*answer|最终答案)[^0-9\-]*([-+]?\d*\.?\d+)', text, flags=re.IGNORECASE)
#         if m:
#             return m[-1].strip()
#         m = re.findall(r'#{2,}\s*([-+]?\d*\.?\d+)', text)
#         if m:
#             return m[-1].strip()
#         m = re.findall(r'[-+]?\d*\.?\d+', text)
#         return m[-1].strip() if m else None


#     def build_messages(question: str,  prompt_style: str, strategy_template: int, language: str ):
#         system_content = f"{STYLE_TO_PREFIX[prompt_style]}{TEMPLATE_TO_SYSTEM[strategy_template]} {LANG_TO_SUFFIX[language]}"
#         # print(system_content)
#         messages = [{"role": "system", "content": system_content}]
#         messages.append({"role": "user", "content": f"{question}\nLet's think step by step. Give your final answer in the format: #### <number>"})
#         return messages

#     strategy_template_idx = int(params.get('strategy_template', 0))
#     style_idx = int(params.get('prompt_style', 0))
#     strategy_template = STRATEGY_TEMPLATE_SET[strategy_template_idx % len(STRATEGY_TEMPLATE_SET)]
#     prompt_style = STYLE_SET[style_idx %len(STYLE_SET)]

#     temperature = float(params.get('temperature', 0.4))
#     top_p = float(params.get('top_p', 0.9))

#     try:
#         n_eval = int(os.getenv('LLM_EVAL_N', '10'))
#     except Exception:
#         n_eval = 10

#     try:
#         ds = load_dataset("gsm8k", "main")
#         test_set = ds["test"].select(range(min(n_eval, len(ds["test"]))))
#     except Exception:
#         return 0.0

#     items = list(test_set)
#     n = len(items)
#     if n == 0:
#         return 0.0

#     try:
#         client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
#     except Exception:
#         return 0.0

#     correct = 0
#     t_total_start = time.perf_counter()
#     for i in range(n):
#         q = items[i]["question"]
#         gt = extract_answer(items[i]["answer"])  # 提取数值答案

#         t_step_start = time.perf_counter()
#         msgs = build_messages(q, prompt_style, strategy_template,"en")
#         # msgs = build_messages(q, prompt_style, "rubric","en")
        
#         try:
#             out = client.chat.completions.create(
#                 model="qwen2.5-7b-instruct",
#                 messages=msgs,
#                 max_tokens=1024,
#                 temperature=float(temperature),
#                 top_p=float(top_p),
#             ).choices[0].message.content
#             pred = extract_answer(out)

#         except Exception:
#             pred = None

#         is_ok = (gt is not None and pred is not None and pred == gt)
#         correct += int(is_ok)
#         t_step = time.perf_counter() - t_step_start
#         # print(f"[LLM-top_p] step {i+1}/{n} time={t_step:.2f}s", flush=True)

#     acc = correct / n
#     t_total = time.perf_counter() - t_total_start
#     avg_t = t_total / max(1, n)
#     print(f"[time|acc] total time={t_total:.2f}s | final accuracy: {acc:.3f} ({correct}/{n})", flush=True)
#     print(
#         f"[params] params => strategy_template={strategy_template}, style={prompt_style}, temperature={float(temperature):.3f}, top_p={float(top_p):.3f}",
#         flush=True,
#     )
#     return float(acc)

# def llm_gsm8k(params: dict) -> float:
#     """
#     仅使用 top_p 解码（并发评测版本：ThreadPoolExecutor）：
#     - 连续：temperature、top_p
#     - 类别：tpl/style
#     - 固定：few-shot=0
#     """
#     import os, re, time
#     from typing import List, Tuple
#     try:
#         from datasets import load_dataset
#     except Exception:
#         return 0.0
#     try:
#         from openai import OpenAI
#     except Exception:
#         return 0.0

#     # ===== 配置集合与映射 =====
#     STRATEGY_TEMPLATE_SET = ["rubric", "critique", "cot"]
#     STYLE_SET = ["concise", "step_by_step", "formal", "creative"]

#     TEMPLATE_TO_SYSTEM = {
#         "rubric":   "Knowns → equation → compute → check → final number.",
#         "critique": "Draft briefly, self-check quickly, then corrected final number.",
#         "cot":      "Solve in 2–4 ultra-brief steps."
#     }
#     STYLE_TO_PREFIX = {

#         "step_by_step":  "Show steps briefly. ",
#         "formal":        "Be precise. ",
#         "creative":      "Be engaging. "
#     }
#     LANG_TO_SUFFIX = {
#         "en": "Answer in English.",
#         "zh": "请使用中文回答。",
#         "zh_en_mix": "Answer bilingually: first Chinese, then English."
#     }

#     def extract_answer(text: str):
#         if not text:
#             return None
#         m = re.findall(r'(?:final\s*answer|最终答案)[^0-9\-]*([-+]?\d*\.?\d+)', text, flags=re.IGNORECASE)
#         if m:
#             return m[-1].strip()
#         m = re.findall(r'#{2,}\s*([-+]?\d*\.?\d+)', text)
#         if m:
#             return m[-1].strip()
#         m = re.findall(r'[-+]?\d*\.?\d+', text)
#         return m[-1].strip() if m else None

#     def build_messages(question: str, prompt_style: str, strategy_template: str, language: str):
#         # 组装 system 指令；使用 .get 防止 KeyError
#         style_prefix = STYLE_TO_PREFIX.get(prompt_style, "")
#         strategy_text = TEMPLATE_TO_SYSTEM.get(strategy_template, "")
#         lang_suffix = LANG_TO_SUFFIX.get(language, "")
#         system_content = f"{style_prefix}{strategy_text} {lang_suffix}".strip()
#         messages = [{"role": "system", "content": system_content}]
#         messages.append({
#             "role": "user",
#             "content": f"{question}\nLet's think step by step. Give your final answer in the format: #### <number>"
#         })
#         return messages

#     # ===== 读取解码参数 =====
#     strategy_template_idx = int(params.get('strategy_template', 0))
#     style_idx = int(params.get('prompt_style', 0))
#     strategy_template = STRATEGY_TEMPLATE_SET[strategy_template_idx % len(STRATEGY_TEMPLATE_SET)]
#     prompt_style = STYLE_SET[style_idx % len(STYLE_SET)]
#     temperature = float(params.get('temperature', 0.4))
#     top_p = float(params.get('top_p', 0.9))

#     # 评测条数
#     try:
#         n_eval = int(os.getenv('LLM_EVAL_N', '10'))
#     except Exception:
#         n_eval = 10

#     # 限流/并发设置
#     try:
#         max_workers = int(os.getenv('LLM_EVAL_WORKERS', '8'))
#     except Exception:
#         max_workers = 8
#     max_workers = max(1, max_workers)

#     # 简单重试次数（例如 429/5xx）
#     try:
#         max_retries = int(os.getenv('LLM_EVAL_RETRIES', '2'))
#     except Exception:
#         max_retries = 2
#     base_backoff = 0.5  # 秒

#     # ===== 加载数据集 =====
#     try:
#         ds = load_dataset("gsm8k", "main")
#         test_set = ds["test"].select(range(min(n_eval, len(ds["test"]))))
#     except Exception:
#         return 0.0

#     items = list(test_set)
#     n = len(items)
#     if n == 0:
#         return 0.0

#     # ===== worker 函数（每个线程内各自创建 client 更省心） =====
#     def solve_one(i: int, item) -> Tuple[int, str, str, float]:
#         q = item["question"]
#         gt = extract_answer(item["answer"])  # 提取数值答案
#         msgs = build_messages(q, prompt_style, strategy_template, "en")  # 如果想固定中文可改为 "zh"
#         t0 = time.perf_counter()
#         pred = None

#         for attempt in range(max_retries + 1):
#             try:
#                 client = OpenAI(
#                     api_key=os.getenv("DASHSCOPE_API_KEY"),
#                     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
#                 )
#                 out = client.chat.completions.create(
#                     model="qwen2.5-7b-instruct",
#                     messages=msgs,
#                     max_tokens=1024,
#                     temperature=float(temperature),
#                     top_p=float(top_p),
#                 ).choices[0].message.content
#                 pred = extract_answer(out)
#                 break  # 成功则跳出重试
#             except Exception:
#                 if attempt < max_retries:
#                     # 指数退避
#                     time.sleep(base_backoff * (2 ** attempt))
#                 else:
#                     pred = None  # 最终失败

#         latency = time.perf_counter() - t0
#         return i, gt, pred, latency

#     # ===== 并发执行 =====
#     from concurrent.futures import ThreadPoolExecutor, as_completed
#     t_wall_start = time.perf_counter()

#     # 若样本数很少，避免过度开线程
#     workers = min(max_workers, n if n > 0 else 1)

#     futures = []
#     with ThreadPoolExecutor(max_workers=workers) as ex:
#         for i, it in enumerate(items):
#             futures.append(ex.submit(solve_one, i, it))

#         results = []
#         for f in as_completed(futures):
#             try:
#                 results.append(f.result())
#             except Exception:
#                 # 极端情况下线程内部异常，记为空结果
#                 results.append((-1, None, None, 0.0))

#     # 保持原顺序便于复现/对齐
#     results = [r for r in results if r[0] != -1]
#     results.sort(key=lambda x: x[0])

#     correct = sum(1 for _, gt, pred, _ in results if gt is not None and pred is not None and pred == gt)
#     acc = float(correct) / len(results) if results else 0.0
#     latencies = [t for *_, t in results]
#     t_wall = time.perf_counter() - t_wall_start
#     avg_t = (sum(latencies) / len(latencies)) if latencies else 0.0

#     # ===== 打印统计信息 =====
#     print(f"[time|acc] wall time={t_wall:.2f}s | avg req latency={avg_t:.2f}s | final accuracy: {acc:.3f} ({correct}/{len(results)})", flush=True)
#     print(
#         f"[params] strategy_template={strategy_template}, style={prompt_style}, "
#         f"temperature={float(temperature):.3f}, top_p={float(top_p):.3f}, "
#         f"workers={workers}, retries={max_retries}",
#         flush=True,
#     )
#     return float(acc)


# def llm_gsm8k(params: dict) -> float:
#     """
#     仅使用 top_p 解码（并发评测版，极简输出）：
#     - 评判标准：数值等价即正确（整数/小数/科学计数法/分数/混合数/百分号），容差= max(1e-8, 1e-6*max(1,|gt|))
#     - 输出：wall time / avg latency / final accuracy（numeric）
#     - 不保存、不打印错误样例
#     """
#     import os, re, time
#     from typing import Tuple, Optional

#     try:
#         from datasets import load_dataset
#     except Exception:
#         return 0.0
#     try:
#         from openai import OpenAI
#     except Exception:
#         return 0.0

#     # ===== 配置集合与映射（保持原有可调性） =====
#     STRATEGY_TEMPLATE_SET = ["rubric", "critique", "cot"]
#     STYLE_SET = ["concise", "step_by_step", "formal", "creative"]

#     TEMPLATE_TO_SYSTEM = {
#         "rubric":   "Knowns → equation → compute → check → final number.",
#         "critique": "Draft briefly, self-check quickly, then corrected final number.",
#         "cot":      "Solve in 2–4 ultra-brief steps."
#     }
#     STYLE_TO_PREFIX = {
#         "step_by_step":  "Show steps briefly. ",
#         "formal":        "Be precise. ",
#         "creative":      "Be engaging. "
#     }
#     LANG_TO_SUFFIX = {
#         "en": "Answer in English.",
#         "zh": "请使用中文回答。",
#         "zh_en_mix": "Answer bilingually: first Chinese, then English."
#     }

#     # ===== 数值提取与比较（极简但稳健） =====
#     from decimal import Decimal, InvalidOperation, getcontext
#     from fractions import Fraction
#     getcontext().prec = 50

#     _NUM_TOKEN = r'[-+]?(?:(?:\d[\d,]*)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
#     _FRAC      = r'\d+\s*/\s*\d+'
#     _MIXED     = r'[-+]?\d+\s+\d+\s*/\s*\d+'
#     _ANYNUM    = r'(?:' + _MIXED + r'|' + _FRAC + r'|' + _NUM_TOKEN + r')'

#     def _clean(s: str) -> str:
#         s = (s or "").strip()
#         s = s.replace("−", "-").replace("\u00a0", " ")
#         s = re.sub(r',', '', s)  # 去千分位
#         s = re.sub(r'^\$', '', s)  # 去货币符（前缀）
#         return s

#     def _percent_adjust(raw: str, x: Decimal) -> Decimal:
#         return (x / Decimal(100)) if ('%' in raw) else x

#     def parse_number(text: Optional[str]) -> Optional[Decimal]:
#         """解析为 Decimal；支持整数/小数/科学计数/分数/混合数；允许尾随非数字文本；处理百分号。"""
#         if not text:
#             return None
#         raw = text
#         s = _clean(text)

#         # 混合数 c a/b
#         m = re.fullmatch(_MIXED, s)
#         if m:
#             whole, a, b = re.match(r'([-+]?\d+)\s+(\d+)\s*/\s*(\d+)', s).groups()
#             val = Fraction(int(whole), 1) + Fraction(int(a), int(b))
#             return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))

#         # 分数 a/b
#         m = re.fullmatch(r'([-+]?\d+)\s*/\s*(\d+)', s)
#         if m:
#             a, b = m.groups()
#             val = Fraction(int(a), int(b))
#             return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))

#         # 纯数字（含科学计数法）
#         try:
#             return _percent_adjust(raw, Decimal(s))
#         except InvalidOperation:
#             # 抽取最后一个数字 token 再试（修复 '80.还有这个问题'）
#             toks = re.findall(_ANYNUM, s)
#             if not toks:
#                 return None
#             t = _clean(toks[-1])

#             mm = re.fullmatch(_MIXED, t)
#             if mm:
#                 whole, a, b = re.match(r'([-+]?\d+)\s+(\d+)\s*/\s*(\d+)', t).groups()
#                 val = Fraction(int(whole), 1) + Fraction(int(a), int(b))
#                 return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))
#             mf = re.fullmatch(r'([-+]?\d+)\s*/\s*(\d+)', t)
#             if mf:
#                 a, b = mf.groups()
#                 val = Fraction(int(a), int(b))
#                 return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))
#             try:
#                 return _percent_adjust(raw, Decimal(t))
#             except InvalidOperation:
#                 return None

#     def numbers_close(a: Optional[Decimal], b: Optional[Decimal],
#                       abs_tol: Decimal = Decimal('1e-8'),
#                       rel_tol: Decimal = Decimal('1e-6')) -> bool:
#         if a is None or b is None:
#             return False
#         diff = abs(a - b)
#         thresh = max(abs_tol, rel_tol * max(Decimal(1), abs(b)))
#         return diff <= thresh

#     def extract_answer(text: Optional[str]) -> Optional[str]:
#         """优先 #### <number>，再 final answer，最后兜底取最后数字样式。"""
#         if not text:
#             return None
#         m = re.findall(r'#{2,}\s*(' + _ANYNUM + r')', text)
#         if m:
#             return m[-1].strip()
#         m = re.findall(r'(?:final\s*answer|最终答案)[^0-9\-]*(' + _ANYNUM + r')',
#                        text, flags=re.IGNORECASE)
#         if m:
#             return m[-1].strip()
#         m = re.findall(_ANYNUM, text)
#         return m[-1].strip() if m else None

#     # ===== 构造消息 =====
#     def build_messages(question: str, prompt_style: str, strategy_template: str, language: str):
#         style_prefix = STYLE_TO_PREFIX.get(prompt_style, "")
#         strategy_text = TEMPLATE_TO_SYSTEM.get(strategy_template, "")
#         lang_suffix = LANG_TO_SUFFIX.get(language, "")
#         system_content = f"{style_prefix}{strategy_text} {lang_suffix}".strip()
#         return [
#             {"role": "system", "content": system_content},
#             {"role": "user",
#              "content": f"{question}\nLet's think step by step. Give your final answer in the format: #### <number>"}
#         ]

#     # ===== 参数 =====
#     strategy_template_idx = int(params.get('strategy_template', 0))
#     style_idx = int(params.get('prompt_style', 0))
#     strategy_template = STRATEGY_TEMPLATE_SET[strategy_template_idx % len(STRATEGY_TEMPLATE_SET)]
#     prompt_style = STYLE_SET[style_idx % len(STYLE_SET)]
#     temperature = float(params.get('temperature', 0.4))
#     top_p = float(params.get('top_p', 0.9))

#     n_eval = int(os.getenv('LLM_EVAL_N', '10') or 10)
#     max_workers = max(1, int(os.getenv('LLM_EVAL_WORKERS', '8') or 8))
#     max_retries = int(os.getenv('LLM_EVAL_RETRIES', '2') or 2)
#     base_backoff = 0.5

#     # ===== 数据 =====
#     try:
#         ds = load_dataset("gsm8k", "main")
#         test_set = ds["test"].select(range(min(n_eval, len(ds["test"]))))
#     except Exception:
#         return 0.0
#     items = list(test_set)
#     if not items:
#         return 0.0

#     # ===== 单样本求解 =====
#     def solve_one(i: int, item) -> Tuple[int, Optional[str], Optional[str], float]:
#         q = item["question"]
#         gt = extract_answer(item["answer"])
#         msgs = build_messages(q, prompt_style, strategy_template, "en")
#         t0 = time.perf_counter()
#         pred = None

#         for attempt in range(max_retries + 1):
#             try:
#                 client = OpenAI(
#                     api_key=os.getenv("DASHSCOPE_API_KEY"),
#                     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
#                 )
#                 out = client.chat.completions.create(
#                     model="qwen2.5-7b-instruct",
#                     messages=msgs,
#                     max_tokens=1024,
#                     temperature=float(temperature),
#                     top_p=float(top_p),
#                 )
#                 pred = extract_answer(out.choices[0].message.content or "")
#                 break
#             except Exception:
#                 if attempt < max_retries:
#                     time.sleep(base_backoff * (2 ** attempt))
#         latency = time.perf_counter() - t0
#         return i, gt, pred, latency

#     # ===== 并发执行 =====
#     from concurrent.futures import ThreadPoolExecutor, as_completed
#     t_wall_start = time.perf_counter()
#     workers = min(max_workers, len(items))

#     futures, results = [], []
#     with ThreadPoolExecutor(max_workers=workers) as ex:
#         for i, it in enumerate(items):
#             futures.append(ex.submit(solve_one, i, it))
#         for f in as_completed(futures):
#             try:
#                 results.append(f.result())
#             except Exception:
#                 pass

#     results.sort(key=lambda x: x[0])

#     # ===== 统计（仅 numeric） =====
#     numeric_correct, latencies = 0, []
#     for _, gt_str, pred_str, t in results:
#         latencies.append(t)
#         gt_num = parse_number(gt_str)
#         pred_num = parse_number(pred_str)
#         if numbers_close(pred_num, gt_num):
#             numeric_correct += 1

#     n_total = len(results)
#     acc = (numeric_correct / n_total) if n_total else 0.0
#     avg_t = (sum(latencies) / len(latencies)) if latencies else 0.0
#     t_wall = time.perf_counter() - t_wall_start

#     # ===== 打印（极简） =====
#     print(f"[time] wall={t_wall:.2f}s | avg req latency={avg_t:.2f}s", flush=True)
#     print(f"[acc] numeric={acc:.3f} ({numeric_correct}/{n_total})", flush=True)
#     print(
#         f"[params] strategy_template={strategy_template}, style={prompt_style}, "
#         f"temperature={float(temperature):.3f}, top_p={float(top_p):.3f}, "
#         f"workers={workers}, retries={max_retries}",
#         flush=True,
#     )

#     return float(acc)

#1003 可以跑的版本 90-94准确率
# def llm_gsm8k(params: dict) -> float:
#     """
#     并发评测（极简）：
#     - 判定：数值等价即正确（整数/小数/科学计数法/分数/混合数/百分号），容差=max(1e-8, 1e-6*max(1,|gt|))
#     - 输出：wall time / avg latency / final numeric accuracy；以及错误样例的 gt 和 pred
#     """
#     import os, re, time
#     from typing import Tuple, Optional

#     try:
#         from datasets import load_dataset
#     except Exception:
#         return 0.0
#     try:
#         from openai import OpenAI
#     except Exception:
#         return 0.0

#     # ===== 配置（保持与原版兼容） =====
#     STRATEGY_TEMPLATE_SET = ["rubric", "critique", "cot"]
#     STYLE_SET = ["step_by_step", "formal", "creative"]

#     TEMPLATE_TO_SYSTEM = {
#         "rubric":   "Knowns → equation → compute → check → final number.",
#         "critique": "Draft briefly, self-check quickly, then corrected final number.",
#         "cot":      "Solve in 2–4 ultra-brief steps."
#     }
#     STYLE_TO_PREFIX = {
#         "step_by_step":  "Show steps briefly. ",
#         "formal":        "Be precise. ",
#         "creative":      "Be engaging. "
#     }
#     LANG_TO_SUFFIX = {
#         "en": "Answer in English.",
#         "zh": "请使用中文回答。",
#         "zh_en_mix": "Answer bilingually: first Chinese, then English."
#     }

#     # ===== 数值提取与比较（稳健） =====
#     from decimal import Decimal, InvalidOperation, getcontext
#     from fractions import Fraction
#     getcontext().prec = 50

#     _NUM_TOKEN = r'[-+]?(?:(?:\d[\d,]*)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
#     _FRAC      = r'\d+\s*/\s*\d+'
#     _MIXED     = r'[-+]?\d+\s+\d+\s*/\s*\d+'
#     _ANYNUM    = r'(?:' + _MIXED + r'|' + _FRAC + r'|' + _NUM_TOKEN + r')'

#     def _clean(s: str) -> str:
#         s = (s or "").strip()
#         s = s.replace("−", "-").replace("\u00a0", " ")
#         s = re.sub(r',', '', s)  # 去千分位
#         s = re.sub(r'^\$', '', s)  # 去前缀货币符
#         return s

#     def _percent_adjust(raw: str, x: Decimal) -> Decimal:
#         # 若原串含 %，将数值缩放为小数（50% -> 0.5）
#         return (x / Decimal(100)) if ('%' in (raw or '')) else x

#     def parse_number(text: Optional[str]) -> Optional[Decimal]:
#         """解析为 Decimal；支持整数/小数/科学计数/分数/混合数；允许尾随文字；处理百分号。"""
#         if not text:
#             return None
#         raw = text
#         s = _clean(text)

#         # 混合数 c a/b
#         m = re.fullmatch(_MIXED, s)
#         if m:
#             whole, a, b = re.match(r'([-+]?\d+)\s+(\d+)\s*/\s*(\d+)', s).groups()
#             val = Fraction(int(whole), 1) + Fraction(int(a), int(b))
#             return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))

#         # 分数 a/b
#         m = re.fullmatch(r'([-+]?\d+)\s*/\s*(\d+)', s)
#         if m:
#             a, b = m.groups()
#             val = Fraction(int(a), int(b))
#             return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))

#         # 纯数字（含科学计数）
#         try:
#             return _percent_adjust(raw, Decimal(s))
#         except InvalidOperation:
#             # 提取最后一个数字 token 再试（例如 '80.还有这个问题'）
#             toks = re.findall(_ANYNUM, s)
#             if not toks:
#                 return None
#             t = _clean(toks[-1])

#             mm = re.fullmatch(_MIXED, t)
#             if mm:
#                 whole, a, b = re.match(r'([-+]?\d+)\s+(\d+)\s*/\s*(\d+)', t).groups()
#                 val = Fraction(int(whole), 1) + Fraction(int(a), int(b))
#                 return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))
#             mf = re.fullmatch(r'([-+]?\d+)\s*/\s*(\d+)', t)
#             if mf:
#                 a, b = mf.groups()
#                 val = Fraction(int(a), int(b))
#                 return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))
#             try:
#                 return _percent_adjust(raw, Decimal(t))
#             except InvalidOperation:
#                 return None

#     def numbers_close(a: Optional[Decimal], b: Optional[Decimal],
#                       abs_tol: Decimal = Decimal('1e-8'),
#                       rel_tol: Decimal = Decimal('1e-6')) -> bool:
#         if a is None or b is None:
#             return False
#         diff = abs(a - b)
#         thresh = max(abs_tol, rel_tol * max(Decimal(1), abs(b)))
#         return diff <= thresh

#     def extract_answer(text: Optional[str]) -> Optional[str]:
#         """优先 #### <number>，再 final answer，最后兜底取最后数字样式。"""
#         if not text:
#             return None
#         m = re.findall(r'#{2,}\s*(' + _ANYNUM + r')', text)
#         if m:
#             return m[-1].strip()
#         m = re.findall(r'(?:final\s*answer|最终答案)[^0-9\-]*(' + _ANYNUM + r')',
#                        text, flags=re.IGNORECASE)
#         if m:
#             return m[-1].strip()
#         m = re.findall(_ANYNUM, text)
#         return m[-1].strip() if m else None

#     # ===== 构造消息 =====
#     def build_messages(question: str, prompt_style: str, strategy_template: str, language: str):
#         style_prefix = STYLE_TO_PREFIX.get(prompt_style, "")
#         strategy_text = TEMPLATE_TO_SYSTEM.get(strategy_template, "")
#         lang_suffix = LANG_TO_SUFFIX.get(language, "")
#         system_content = f"{style_prefix}{strategy_text} {lang_suffix}".strip()
#         return [
#             {"role": "system", "content": system_content},
#             {"role": "user",
#              "content": f"{question}\nLet's think step by step. Give your final answer in the format: #### <number>"}
#         ]

#     # ===== 参数 =====
#     strategy_template_idx = int(params.get('strategy_template', 0))
#     style_idx = int(params.get('prompt_style', 0))
#     strategy_template = STRATEGY_TEMPLATE_SET[strategy_template_idx % len(STRATEGY_TEMPLATE_SET)]
#     prompt_style = STYLE_SET[style_idx % len(STYLE_SET)]
#     temperature = float(params.get('temperature', 0.4))
#     top_p = float(params.get('top_p', 0.9))

#     n_eval = int(os.getenv('LLM_EVAL_N', '10') or 10)
#     max_workers = max(1, int(os.getenv('LLM_EVAL_WORKERS', '8') or 8))
#     max_retries = int(os.getenv('LLM_EVAL_RETRIES', '2') or 2)
#     base_backoff = 0.5

#     # ===== 数据 =====
#     try:
#         ds = load_dataset("gsm8k", "main")
#         test_set = ds["test"].select(range(min(n_eval, len(ds["test"]))))
#     except Exception:
#         return 0.0
#     items = list(test_set)
#     if not items:
#         return 0.0
#     max_tokens = int(params.get('max_tokens', 256))

#     # ===== 单样本求解 =====
#     def solve_one(i: int, item) -> Tuple[int, Optional[str], Optional[str], float]:
#         q = item["question"]
#         gt = extract_answer(item["answer"])
#         msgs = build_messages(q, prompt_style, strategy_template, "en")
#         t0 = time.perf_counter()
#         pred = None

#         for attempt in range(max_retries + 1):
#             try:
#                 client = OpenAI(
#                     api_key=os.getenv("DASHSCOPE_API_KEY"),
#                     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
#                 )
#                 out = client.chat.completions.create(
#                     model="qwen2.5-7b-instruct",
#                     messages=msgs,
#                     max_tokens=max_tokens,
#                     temperature=float(temperature),
#                     top_p=float(top_p),
#                 )
#                 pred = extract_answer(out.choices[0].message.content or "")
#                 break
#             except Exception:
#                 if attempt < max_retries:
#                     time.sleep(base_backoff * (2 ** attempt))
#         latency = time.perf_counter() - t0
#         return i, gt, pred, latency

#     # ===== 并发执行 =====
#     from concurrent.futures import ThreadPoolExecutor, as_completed
#     t_wall_start = time.perf_counter()
#     workers = min(max_workers, len(items))

#     futures, results = [], []
#     with ThreadPoolExecutor(max_workers=workers) as ex:
#         for i, it in enumerate(items):
#             futures.append(ex.submit(solve_one, i, it))
#         for f in as_completed(futures):
#             try:
#                 results.append(f.result())
#             except Exception:
#                 pass

#     results.sort(key=lambda x: x[0])

#     # ===== 统计（numeric）并打印错误的 gt / pred =====
#     numeric_correct, latencies = 0, []
#     errors = []  # 仅保存打印需要：错误的 (gt, pred)

#     for _, gt_str, pred_str, t in results:
#         latencies.append(t)
#         gt_num = parse_number(gt_str)
#         pred_num = parse_number(pred_str)
#         if numbers_close(pred_num, gt_num):
#             numeric_correct += 1
#         else:
#             errors.append((gt_str, pred_str))

#     n_total = len(results)
#     acc = (numeric_correct / n_total) if n_total else 0.0
#     avg_t = (sum(latencies) / len(latencies)) if latencies else 0.0
#     t_wall = time.perf_counter() - t_wall_start

#     # # ===== 打印（极简 + 错误 gt/pred） =====
#     print(f"[time] wall={t_wall:.2f}s | [acc] numeric={acc:.3f} ({numeric_correct}/{n_total})", flush=True)

#     print(
#         f"[params] strategy_template={strategy_template}, style={prompt_style}, "
#         f"temperature={float(temperature):.3f}, top_p={float(top_p):.3f}, "
#         f"workers={workers}, retries={max_retries}, max_tokens={max_tokens}",
#         flush=True,
#     )

#     return float(acc)


#1004有点慢
# def llm_gsm8k(params: dict) -> float:
#     """
#     并发评测（带可选 vote，自一致性）：
#     - 判定：数值等价即正确（整数/小数/科学计数法/分数/混合数/百分号），容差=max(1e-8, 1e-6*max(1,|gt|))
#     - 输出：wall time / avg latency / final numeric accuracy；以及（可选）若干错误样例的 gt/pred
#     - 兼容：vote_k=1 时行为与原版一致；>1 时启用 self-consistency 投票
#     - 投票阶段使用与主推理相同的 temperature / top_p（不再单独可配）
    
#     环境变量（可选；括号内为默认）：
#       - LLM_EVAL_N（10）             : 评测样本数上限
#       - LLM_EVAL_WORKERS（8）        : 并发线程数
#       - LLM_EVAL_RETRIES（2）        : API 调用重试次数
#       - LLM_EVAL_VOTE_K（1）         : 每题采样次数（>1 启用投票）
#       - LLM_EVAL_PRINT_ERRORS（20）  : 打印前多少个错误样例
#       - DASHSCOPE_API_KEY            : DashScope 兼容模式 API Key

#     可通过 params 覆盖：temperature/top_p/max_tokens/strategy_template/prompt_style/vote_k/lang
#     """
#     import os, re, time
#     from typing import Tuple, Optional, List

#     try:
#         from datasets import load_dataset
#     except Exception:
#         return 0.0
#     try:
#         from openai import OpenAI
#     except Exception:
#         return 0.0

#     # ===== 配置（与原版兼容） =====
#     STRATEGY_TEMPLATE_SET = ["rubric", "critique", "cot"]
#     STYLE_SET = ["step_by_step", "formal", "creative"]

#     TEMPLATE_TO_SYSTEM = {
#         "rubric":   "Knowns → equation → compute → check → final number.",
#         "critique": "Draft briefly, self-check quickly, then corrected final number.",
#         "cot":      "Solve in 2–4 ultra-brief steps."
#     }
#     STYLE_TO_PREFIX = {
#         "step_by_step":  "Show steps briefly. ",
#         "formal":        "Be precise. ",
#         "creative":      "Be engaging. "
#     }
#     LANG_TO_SUFFIX = {
#         "en": "Answer in English.",
#         "zh": "请使用中文回答。",
#         "zh_en_mix": "Answer bilingually: first Chinese, then English."
#     }

#     # ===== 数值解析与比较（稳健） =====
#     from decimal import Decimal, InvalidOperation, getcontext
#     from fractions import Fraction
#     getcontext().prec = 50

#     _NUM_TOKEN = r'[-+]?(?:(?:\d[\d,]*)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
#     _MIXED     = r'[-+]?\d+\s+\d+\s*/\s*\d+'
#     _ANYNUM    = r'(?:' + _MIXED + r'|\d+\s*/\s*\d+|' + _NUM_TOKEN + r')'

#     def _clean(s: str) -> str:
#         s = (s or "").strip()
#         s = s.replace("−", "-").replace("\u00a0", " ")
#         s = re.sub(r',', '', s)      # 去千分位
#         s = re.sub(r'^\$', '', s)    # 去前缀货币符
#         return s

#     def _percent_adjust(raw: str, x: Decimal) -> Decimal:
#         # 若原串含 %，将数值缩放为小数（50% -> 0.5）
#         return (x / Decimal(100)) if ('%' in (raw or '')) else x

#     def parse_number(text: Optional[str]) -> Optional[Decimal]:
#         """解析为 Decimal；支持整数/小数/科学计数/分数/混合数；允许尾随文字；处理百分号。"""
#         if not text:
#             return None
#         raw = text
#         s = _clean(text)

#         # 混合数 c a/b
#         m = re.fullmatch(_MIXED, s)
#         if m:
#             whole, a, b = re.match(r'([-+]?\d+)\s+(\d+)\s*/\s*(\d+)', s).groups()
#             val = Fraction(int(whole), 1) + Fraction(int(a), int(b))
#             return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))

#         # 分数 a/b
#         m = re.fullmatch(r'([-+]?\d+)\s*/\s*(\d+)', s)
#         if m:
#             a, b = m.groups()
#             val = Fraction(int(a), int(b))
#             return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))

#         # 纯数字（含科学计数）
#         try:
#             return _percent_adjust(raw, Decimal(s))
#         except InvalidOperation:
#             toks = re.findall(_ANYNUM, s)
#             if not toks:
#                 return None
#             t = _clean(toks[-1])

#             mm = re.fullmatch(_MIXED, t)
#             if mm:
#                 whole, a, b = re.match(r'([-+]?\d+)\s+(\d+)\s*/\s*(\d+)', t).groups()
#                 val = Fraction(int(whole), 1) + Fraction(int(a), int(b))
#                 return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))
#             mf = re.fullmatch(r'([-+]?\d+)\s*/\s*(\d+)', t)
#             if mf:
#                 a, b = mf.groups()
#                 val = Fraction(int(a), int(b))
#                 return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))
#             try:
#                 return _percent_adjust(raw, Decimal(t))
#             except InvalidOperation:
#                 return None

#     def numbers_close(a: Optional[Decimal], b: Optional[Decimal],
#                       abs_tol: Decimal = Decimal('1e-8'),
#                       rel_tol: Decimal = Decimal('1e-6')) -> bool:
#         if a is None or b is None:
#             return False
#         diff = abs(a - b)
#         thresh = max(abs_tol, rel_tol * max(Decimal(1), abs(b)))
#         return diff <= thresh

#     def extract_answer(text: Optional[str]) -> Optional[str]:
#         """优先 #### <number>，再 final answer/最终答案，最后兜底取最后数字样式。"""
#         if not text:
#             return None
#         m = re.findall(r'#{2,}\s*(' + _ANYNUM + r')', text)
#         if m:
#             return m[-1].strip()
#         m = re.findall(r'(?:final\s*answer|最终答案)[^0-9\-]*(' + _ANYNUM + r')',
#                        text, flags=re.IGNORECASE)
#         if m:
#             return m[-1].strip()
#         m = re.findall(_ANYNUM, text)
#         return m[-1].strip() if m else None

#     # ===== 构造消息 =====
#     def build_messages(question: str, prompt_style: str, strategy_template: str, language: str):
#         style_prefix = STYLE_TO_PREFIX.get(prompt_style, "")
#         strategy_text = TEMPLATE_TO_SYSTEM.get(strategy_template, "")
#         lang_suffix = LANG_TO_SUFFIX.get(language, "")
#         system_content = f"{style_prefix}{strategy_text} {lang_suffix}".strip()
#         return [
#             {"role": "system", "content": system_content},
#             {"role": "user",
#              "content": f"{question}\nLet's think step by step. Give your final answer in the format: #### <number>"}
#         ]

#     # ===== 参数 =====
#     strategy_template_idx = int(params.get('strategy_template', 0))
#     style_idx = int(params.get('prompt_style', 0))
#     strategy_template = STRATEGY_TEMPLATE_SET[strategy_template_idx % len(STRATEGY_TEMPLATE_SET)]
#     prompt_style = STYLE_SET[style_idx % len(STYLE_SET)]
#     temperature = float(params.get('temperature', 0.4))
#     top_p = float(params.get('top_p', 0.9))
#     language = str(params.get('lang', 'en'))

#     # vote：默认 1（不开启）；>1 开启自一致性
#     vote_k = int(params.get('vote_k', int(os.getenv('LLM_EVAL_VOTE_K', '1') or 1)))

#     n_eval = int(params.get('n_eval', os.getenv('LLM_EVAL_N', '10')) or 10)
#     max_workers_cfg = int(os.getenv('LLM_EVAL_WORKERS', '8') or 8)
#     max_retries = int(os.getenv('LLM_EVAL_RETRIES', '2') or 2)
#     base_backoff = 0.5
#     max_tokens = int(params.get('max_tokens', 256))
#     print_err_n = int(os.getenv('LLM_EVAL_PRINT_ERRORS', '20') or 20)

#     # 当 vote_k 较大时，适当降并发，避免速率/并发限制
#     if vote_k <= 1:
#         max_workers = max(1, max_workers_cfg)
#     else:
#         max_workers = max(1, max_workers_cfg // max(2, min(vote_k, 8)//2))

#     # ===== 数据 =====
#     try:
#         ds = load_dataset("gsm8k", "main")
#         test_set = ds["test"].select(range(min(n_eval, len(ds["test"]))))
#     except Exception:
#         return 0.0
#     items = list(test_set)
#     if not items:
#         return 0.0

#     # ===== 投票聚类 =====
#     def cluster_vote(nums: List[Decimal],
#                      tol_abs: Decimal = Decimal('1e-8'),
#                      tol_rel: Decimal = Decimal('1e-6')) -> Decimal:
#         """将近似相等的数字聚成簇；返回“最大簇”的代表值（中位数）。"""
#         clusters: List[List[Decimal]] = []
#         for x in nums:
#             placed = False
#             for c in clusters:
#                 if numbers_close(x, c[0], tol_abs, tol_rel):
#                     c.append(x); placed = True; break
#             if not placed:
#                 clusters.append([x])

#         def cluster_rep(c: List[Decimal]) -> Decimal:
#             sc = sorted(c)
#             mid = len(sc)//2
#             return sc[mid] if len(sc)%2==1 else (sc[mid-1] + sc[mid]) / Decimal(2)

#         clusters.sort(key=lambda c: (-len(c), cluster_rep(c)))
#         return cluster_rep(clusters[0])

#     # ===== 单样本求解（支持 vote） =====
#     def solve_one(i: int, item) -> Tuple[int, Optional[str], Optional[str], float]:
#         q = item["question"]
#         gt = extract_answer(item["answer"])
#         msgs = build_messages(q, prompt_style, strategy_template, language)
#         t0 = time.perf_counter()
#         pred_str: Optional[str] = None

#         # 单次模型调用（带重试）
#         def call_once(temp: float, tp: float) -> Optional[str]:
#             for attempt in range(max_retries + 1):
#                 try:
#                     client = OpenAI(
#                         api_key=os.getenv("DASHSCOPE_API_KEY"),
#                         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
#                     )
#                     out = client.chat.completions.create(
#                         model="qwen2.5-7b-instruct",
#                         messages=msgs,
#                         max_tokens=max_tokens,
#                         temperature=float(temp),
#                         top_p=float(tp),
#                     )
#                     return extract_answer(out.choices[0].message.content or "")
#                 except Exception:
#                     if attempt < max_retries:
#                         time.sleep(base_backoff * (2 ** attempt))
#             return None

#         if vote_k <= 1:
#             # 原版：单采样
#             pred_str = call_once(temperature, top_p)
#         else:
#             # vote：多采样 + 聚类投票（使用与主推理相同的 temperature/top_p）
#             preds_num: List[Decimal] = []
#             for _ in range(vote_k):
#                 p = call_once(temperature, top_p)
#                 x = parse_number(p)
#                 if x is not None:
#                     preds_num.append(x)
#             if preds_num:
#                 rep = cluster_vote(preds_num)
#                 pred_str = str(rep)   # 后续 parse_number 会再次解析
#             else:
#                 pred_str = None

#         latency = time.perf_counter() - t0
#         return i, gt, pred_str, latency

#     # ===== 并发执行 =====
#     from concurrent.futures import ThreadPoolExecutor, as_completed
#     t_wall_start = time.perf_counter()
#     workers = min(max_workers, len(items))

#     futures, results = [], []
#     with ThreadPoolExecutor(max_workers=workers) as ex:
#         for i, it in enumerate(items):
#             futures.append(ex.submit(solve_one, i, it))
#         for f in as_completed(futures):
#             try:
#                 results.append(f.result())
#             except Exception:
#                 pass

#     results.sort(key=lambda x: x[0])

#     # ===== 统计（numeric）并打印错误样例 =====
#     numeric_correct, latencies = 0, []
#     errors = []  # 仅用于打印：错误 (gt, pred)

#     for _, gt_str, pred_str, t in results:
#         latencies.append(t)
#         gt_num = parse_number(gt_str)
#         pred_num = parse_number(pred_str)
#         if numbers_close(pred_num, gt_num):
#             numeric_correct += 1
#         else:
#             errors.append((gt_str, pred_str))

#     n_total = len(results)
#     acc = (numeric_correct / n_total) if n_total else 0.0
#     avg_t = (sum(latencies) / len(latencies)) if latencies else 0.0
#     t_wall = time.perf_counter() - t_wall_start

#     # ===== 打印（更全） =====
#     print(f"[time] wall={t_wall:.2f}s | [lat] avg={avg_t:.2f}s | [acc] numeric={acc:.3f} ({numeric_correct}/{n_total})", flush=True)
#     print(
#         f"[params] strategy_template={strategy_template}, style={prompt_style}, "
#         f"language={language}, "
#         f"temperature={float(temperature):.3f}, top_p={float(top_p):.3f}, "
#         f"vote_k={vote_k}, workers={workers}, retries={max_retries}, "
#         f"max_tokens={max_tokens}, n_eval={n_eval}",
#         flush=True,
#     )

#     if errors and print_err_n > 0:
#         print(f"[errors] showing up to {min(print_err_n, len(errors))} cases:", flush=True)
#         for idx, (gt, pd) in enumerate(errors[:print_err_n]):
#             print(f"  #{idx+1:02d} gt={gt!r} | pred={pd!r}", flush=True)

#     return float(acc)

def llm_gsm8k(params: dict) -> float:
    """
    并发评测（带可选 vote，自一致性；按“网格脚本”的更快逻辑改写版）：
    - 关键优化：复用 OpenAI 客户端；优先 n=vote_k 批量采样，不支持再退化为单采补齐；不因 vote_k 下调并发
    - 判定：数值等价即正确（整数/小数/科学计数法/分数/混合数/百分号），容差=max(1e-8, 1e-6*max(1,|gt|))
    - 输出：wall time / avg latency / final numeric accuracy；以及（可选）若干错误样例的 gt/pred
    - 兼容：vote_k=1 时行为与原版一致；>1 时启用 self-consistency 投票（与主推理相同的 temperature/top_p）
    
    环境变量（可选；括号内为默认）：
      - LLM_EVAL_N（10）             : 评测样本数上限
      - LLM_EVAL_WORKERS（8）        : 并发线程数
      - LLM_EVAL_RETRIES（2）        : API 调用重试次数
      - LLM_EVAL_VOTE_K（1）         : 每题采样次数（>1 启用投票）
      - LLM_EVAL_PRINT_ERRORS（20）  : 打印前多少个错误样例
      - DASHSCOPE_API_KEY            : DashScope 兼容模式 API Key

    可通过 params 覆盖：temperature/top_p/max_tokens/strategy_template/prompt_style/vote_k/lang
    """
    import os, re, time
    from typing import Tuple, Optional, List

    try:
        from datasets import load_dataset
    except Exception:
        return 0.0
    try:
        from openai import OpenAI
    except Exception:
        return 0.0

    # ===== 配置（与原版接口兼容） =====
    STRATEGY_TEMPLATE_SET = ["rubric", "critique", "cot"]
    STYLE_SET = ["step_by_step", "formal", "creative"]

    # 提示更“短”：仿照下方脚本，要求 1–3 行推理 + 最后一行严格 #### <number>
    TEMPLATE_TO_SYSTEM = {
        "rubric":   "Think in 1–3 short steps.",
        "critique": "Draft briefly (1–3 lines), self-check quickly.",
        "cot":      "Think in 1–3 short steps."
    }
    STYLE_TO_PREFIX = {
        "step_by_step":  "Be concise. ",
        "formal":        "Be precise. ",
        "creative":      "Be engaging. "
    }
    LANG_TO_SUFFIX = {
        "en": "Answer in English.",
        "zh": "请使用中文回答。",
        "zh_en_mix": "Answer bilingually: first Chinese, then English."
    }

    # ===== 数值解析与比较（保留你原版的鲁棒 Decimal/分数/混合数/百分号） =====
    from decimal import Decimal, InvalidOperation, getcontext
    from fractions import Fraction
    getcontext().prec = 50

    _NUM_TOKEN = r'[-+]?(?:(?:\d[\d,]*)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    _MIXED     = r'[-+]?\d+\s+\d+\s*/\s*\d+'
    _ANYNUM    = r'(?:' + _MIXED + r'|\d+\s*/\s*\d+|' + _NUM_TOKEN + r')'

    def _clean(s: str) -> str:
        s = (s or "").strip()
        s = s.replace("−", "-").replace("\u00a0", " ")
        s = re.sub(r',', '', s)      # 去千分位
        s = re.sub(r'^\$', '', s)    # 去前缀货币符
        return s

    def _percent_adjust(raw: str, x: Decimal) -> Decimal:
        return (x / Decimal(100)) if ('%' in (raw or '')) else x

    def parse_number(text: Optional[str]) -> Optional[Decimal]:
        """解析为 Decimal；支持整数/小数/科学计数/分数/混合数；允许尾随文字；处理百分号。"""
        if not text:
            return None
        raw = text
        s = _clean(text)

        m = re.fullmatch(_MIXED, s)
        if m:
            whole, a, b = re.match(r'([-+]?\d+)\s+(\d+)\s*/\s*(\d+)', s).groups()
            val = Fraction(int(whole), 1) + Fraction(int(a), int(b))
            return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))

        m = re.fullmatch(r'([-+]?\d+)\s*/\s*(\d+)', s)
        if m:
            a, b = m.groups()
            val = Fraction(int(a), int(b))
            return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))

        try:
            return _percent_adjust(raw, Decimal(s))
        except InvalidOperation:
            toks = re.findall(_ANYNUM, s)
            if not toks:
                return None
            t = _clean(toks[-1])

            mm = re.fullmatch(_MIXED, t)
            if mm:
                whole, a, b = re.match(r'([-+]?\d+)\s+(\d+)\s*/\s*(\d+)', t).groups()
                val = Fraction(int(whole), 1) + Fraction(int(a), int(b))
                return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))
            mf = re.fullmatch(r'([-+]?\d+)\s*/\s*(\d+)', t)
            if mf:
                a, b = mf.groups()
                val = Fraction(int(a), int(b))
                return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))
            try:
                return _percent_adjust(raw, Decimal(t))
            except InvalidOperation:
                return None

    def numbers_close(a: Optional[Decimal], b: Optional[Decimal],
                      abs_tol: Decimal = Decimal('1e-8'),
                      rel_tol: Decimal = Decimal('1e-6')) -> bool:
        if a is None or b is None:
            return False
        diff = abs(a - b)
        thresh = max(abs_tol, rel_tol * max(Decimal(1), abs(b)))
        return diff <= thresh

    def extract_answer(text: Optional[str]) -> Optional[str]:
        """优先 #### <number>，再 final answer/最终答案，最后兜底取最后数字样式。"""
        if not text:
            return None
        m = re.findall(r'#{2,}\s*(' + _ANYNUM + r')', text)
        if m:
           return m[-1].strip()
        m = re.findall(r'(?:final\s*answer|最终答案)[^0-9\-]*(' + _ANYNUM + r')',
                       text, flags=re.IGNORECASE)
        if m:
            return m[-1].strip()
        m = re.findall(_ANYNUM, text)
        return m[-1].strip() if m else None

    # ===== 构造消息（更短、更严格的一行答案约束） =====
    def build_messages(question: str, prompt_style: str, strategy_template: str, language: str):
        style_prefix = STYLE_TO_PREFIX.get(prompt_style, "")
        strategy_text = TEMPLATE_TO_SYSTEM.get(strategy_template, "")
        lang_suffix = LANG_TO_SUFFIX.get(language, "")
        # system
        system_content = f"{style_prefix}{strategy_text} {lang_suffix}".strip()
        # user：明确最后只输出一行
        user_content = (
            f"Problem:\n{question}\n\n"
            f"Reason briefly in 1–3 short lines"
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    # ===== 参数 =====
    strategy_template_idx = int(params.get('strategy_template', 2))  # 默认 'cot'
    style_idx = int(params.get('prompt_style', 0))                    # 默认 'step_by_step'
    strategy_template = STRATEGY_TEMPLATE_SET[strategy_template_idx % len(STRATEGY_TEMPLATE_SET)]
    prompt_style = STYLE_SET[style_idx % len(STYLE_SET)]
    temperature = float(params.get('temperature', 0.4))
    top_p = float(params.get('top_p', 0.9))
    language = str(params.get('lang', 'en'))

    # vote：默认 1（不开启）；>1 开启自一致性
    vote_k = int(params.get('vote_k', int(os.getenv('LLM_EVAL_VOTE_K', '1') or 1)))

    n_eval = int(params.get('n_eval', os.getenv('LLM_EVAL_N', '10')) or 10)
    max_workers_cfg = int(os.getenv('LLM_EVAL_WORKERS', '8') or 8)
    max_retries = int(os.getenv('LLM_EVAL_RETRIES', '2') or 2)
    base_backoff = 0.5
    # 默认更短，配合“只输出一行”
    max_tokens = int(params.get('max_tokens', 128))
    print_err_n = int(os.getenv('LLM_EVAL_PRINT_ERRORS', '20') or 20)

    # 新策略：不再因 vote_k 下调并发
    workers = max(1, min(max_workers_cfg, n_eval))

    # ===== 数据 =====
    try:
        ds = load_dataset("gsm8k", "main")
    except Exception:
        try:
            ds = load_dataset("openai/gsm8k", "main")
        except Exception:
            return 0.0
    test_set = ds["test"].select(range(min(n_eval, len(ds["test"]))))
    items = list(test_set)
    if not items:
        return 0.0

    # ===== 投票聚类（保留你原版的“近似相等聚类 → 最大簇中位数”） =====
    def cluster_vote(nums: List[Decimal],
                     tol_abs: Decimal = Decimal('1e-8'),
                     tol_rel: Decimal = Decimal('1e-6')) -> Decimal:
        clusters: List[List[Decimal]] = []
        for x in nums:
            placed = False
            for c in clusters:
                if numbers_close(x, c[0], tol_abs, tol_rel):
                    c.append(x); placed = True; break
            if not placed:
                clusters.append([x])

        def cluster_rep(c: List[Decimal]) -> Decimal:
            sc = sorted(c)
            mid = len(sc)//2
            return sc[mid] if len(sc)%2==1 else (sc[mid-1] + sc[mid]) / Decimal(2)

        clusters.sort(key=lambda c: (-len(c), cluster_rep(c)))
        return cluster_rep(clusters[0])

    # ===== 客户端（复用） =====
    try:
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    except Exception:
        return 0.0

    # ===== 调用封装：优先 n=vote_k；不支持再退化为单采补齐 =====
    def call_n(messages, n_samples: int) -> Optional[List[Optional[str]]]:
        for attempt in range(max_retries + 1):
            try:
                out = client.chat.completions.create(
                    model="qwen2.5-7b-instruct",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    n=n_samples
                )
                res = []
                for ch in getattr(out, "choices", []) or []:
                    msg = getattr(ch, "message", None)
                    txt = getattr(msg, "content", None) if msg is not None else getattr(ch, "content", None)
                    if isinstance(txt, bytes):
                        txt = txt.decode("utf-8", "ignore")
                    res.append(extract_answer(txt))
                return res
            except Exception as e:
                em = str(e)
                # 明确不支持 n 或 n 超范围 → 返回 None 让上层走退化路径
                if "unexpected keyword argument 'n'" in em or "Range of n" in em or "n should be" in em:
                    return None
                if attempt < max_retries:
                    time.sleep(base_backoff * (2 ** attempt))
                else:
                    # 到此视为失败（返回空列表，避免无限重试）
                    return []

    # ===== 单样本求解（支持批量投票） =====
    def solve_one(i: int, item) -> Tuple[int, Optional[str], Optional[str], float]:
        q = item["question"]
        gt = extract_answer(item["answer"])
        msgs = build_messages(q, prompt_style, strategy_template, language)
        t0 = time.perf_counter()
        final_pred_str: Optional[str] = None

        if vote_k <= 1:
            preds = call_n(msgs, 1) or []
        else:
            preds = call_n(msgs, vote_k)
            if preds is None:
                # 不支持 n → 退化为单采补齐
                preds = []
                need = vote_k
                while need > 0:
                    batch = call_n(msgs, 1) or []
                    preds.extend(batch)
                    need -= 1 if batch else 0
                    # 简单保护：避免极端情况下死循环
                    if len(preds) >= vote_k or need <= 0:
                        break

        # 聚合
        nums: List[Decimal] = []
        for p in preds:
            x = parse_number(p)
            if x is not None:
                nums.append(x)
        if nums:
            rep = cluster_vote(nums)
            final_pred_str = str(rep)  # 交给 parse_number 再次解析

        latency = time.perf_counter() - t0
        return i, gt, final_pred_str, latency

    # ===== 并发执行 =====
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t_wall_start = time.perf_counter()
    workers = min(workers, len(items))

    futures, results = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, it in enumerate(items):
            futures.append(ex.submit(solve_one, i, it))
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception:
                pass

    results.sort(key=lambda x: x[0])

    # ===== 统计（numeric）并打印错误样例 =====
    numeric_correct, latencies = 0, []
    errors = []  # 仅用于打印：错误 (gt, pred)

    for _, gt_str, pred_str, t in results:
        latencies.append(t)
        gt_num = parse_number(gt_str)
        pred_num = parse_number(pred_str)
        if numbers_close(pred_num, gt_num):
            numeric_correct += 1
        else:
            errors.append((gt_str, pred_str))

    n_total = len(results)
    acc = (numeric_correct / n_total) if n_total else 0.0
    avg_t = (sum(latencies) / len(latencies)) if latencies else 0.0
    t_wall = time.perf_counter() - t_wall_start

    # ===== 打印（与原版一致的摘要 + 参数） =====
    print(f"[time] wall={t_wall:.2f}s | [lat] avg={avg_t:.2f}s | [acc] numeric={acc:.3f} ({numeric_correct}/{n_total})", flush=True)
    print(
        f"[params] strategy_template={strategy_template}, style={prompt_style}, "
        f"language={language}, "
        f"temperature={float(temperature):.3f}, top_p={float(top_p):.3f}, "
        f"vote_k={vote_k}, workers={workers}, retries={max_retries}, "
        f"max_tokens={max_tokens}, n_eval={n_eval}",
        flush=True,
    )

    if errors and print_err_n > 0:
        print(f"[errors] showing up to {min(print_err_n, len(errors))} cases:", flush=True)
        for idx, (gt, pd) in enumerate(errors[:print_err_n]):
            print(f"  #{idx+1:02d} gt={gt!r} | pred={pd!r}", flush=True)

    return float(acc)

# def llm_agnews(params: dict) -> float:
#     """
#     并发评测（AG News，四分类）：
#     - 标签集合：World, Sports, Business, Sci/Tech
#     - 评估：准确率（pred == gt）
#     - 输出：wall time / avg latency / accuracy；以及若干错误样例（可按需扩展）
#     """
#     import os, re, time
#     from typing import Tuple, Optional

#     try:
#         from datasets import load_dataset
#     except Exception:
#         return 0.0
#     try:
#         from openai import OpenAI
#     except Exception:
#         return 0.0

#     # ===== 与 GSM8K 版保持一致的 prompt 相关可调项 =====
#     STRATEGY_TEMPLATE_SET = ["rubric", "critique", "cot"]
#     STYLE_SET = ["step_by_step", "formal", "creative"]

#     TEMPLATE_TO_SYSTEM = {
#         "rubric":   "Knowns → equation → compute → check → final number.",
#         "critique": "Draft briefly, self-check quickly, then corrected final number.",
#         "cot":      "Solve in 2–4 ultra-brief steps."
#     }
#     STYLE_TO_PREFIX = {
#         "step_by_step":  "Show steps briefly. ",
#         "formal":        "Be precise. ",
#         "creative":      "Be engaging. "
#     }
#     LANG_TO_SUFFIX = {
#         "en": "Answer in English.",
#         "zh": "请使用中文回答。",
#         "zh_en_mix": "Answer bilingually: first Chinese, then English."
#     }

#     # ===== 标签与解析 =====
#     ID2LABEL = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
#     LABEL2ID = {v.lower(): k for k, v in ID2LABEL.items()}

#     # 允许的别名（尽量覆盖常见变体）
#     ALIASES = {
#         "world": {"world", "international", "global", "worldnews"},
#         "sports": {"sports", "sport", "sportsnews"},
#         "business": {"business", "finance", "market", "economy"},
#         "sci/tech": {"sci/tech", "scitech", "science", "tech", "technology", "sci-tech", "sci tech"}
#     }

#     def normalize_label(s: Optional[str]) -> Optional[str]:
#         if not s:
#             return None
#         s = (s or "").strip().lower()
#         s = re.sub(r'[^a-z0-9/ ]+', '', s)  # 清理标点
#         s = re.sub(r'\s+', ' ', s)

#         # 数字类标
#         if s in {"0", "1", "2", "3"}:
#             return ID2LABEL[int(s)]

#         # 直接匹配四类
#         if s in LABEL2ID:
#             return ID2LABEL[LABEL2ID[s]]

#         # 别名匹配
#         for canon, alias_set in ALIASES.items():
#             if s in alias_set:
#                 # 统一到标准写法
#                 if canon == "sci/tech":
#                     return "Sci/Tech"
#                 return canon.capitalize()

#         # 片段匹配（例如 "this is business" -> business）
#         for canon, alias_set in ALIASES.items():
#             for a in alias_set:
#                 if re.search(rf'\b{re.escape(a)}\b', s):
#                     return "Sci/Tech" if canon == "sci/tech" else canon.capitalize()

#         return None

#     def extract_label(text: Optional[str]) -> Optional[str]:
#         """优先 #### <label>，否则抓取最后出现的可解析标签/别名/数字。"""
#         if not text:
#             return None
#         # 1) 优先 #### <...>
#         m = re.findall(r'#{2,}\s*([A-Za-z0-9/][A-Za-z0-9/ \-]*)', text)
#         if m:
#             cand = normalize_label(m[-1])
#             if cand:
#                 return cand

#         # 2) 查找四类关键词出现（从后往前）
#         tokens = re.findall(r'[A-Za-z0-9/]+', text.lower())
#         for tok in reversed(tokens):
#             cand = normalize_label(tok)
#             if cand:
#                 return cand

#         # 3) 兜底：整段归一化再试
#         return normalize_label(text)

#     # ===== 构造消息（与现版保持一致，只是 user 提示语改为分类指令）=====
#     def build_messages(news_text: str, prompt_style: str, strategy_template: str, language: str):
#         style_prefix = STYLE_TO_PREFIX.get(prompt_style, "")
#         strategy_text = TEMPLATE_TO_SYSTEM.get(strategy_template, "")
#         lang_suffix = LANG_TO_SUFFIX.get(language, "")
#         system_content = f"{style_prefix}{strategy_text} {lang_suffix}".strip()

#         instruction = (
#             "Classify the news into one of the categories: World, Sports, Business, Sci/Tech.\n"
#             "Return ONLY the final label in the format: #### <label>\n"
#             "Examples of valid labels: World | Sports | Business | Sci/Tech"
#         )

#         return [
#             {"role": "system", "content": system_content},
#             {"role": "user",
#              "content": f"{instruction}\n\nNews:\n{news_text}\n\nOutput format: #### <label>"}
#         ]

#     # ===== 参数 =====
#     strategy_template_idx = int(params.get('strategy_template', 0))
#     style_idx = int(params.get('prompt_style', 0))
#     strategy_template = STRATEGY_TEMPLATE_SET[strategy_template_idx % len(STRATEGY_TEMPLATE_SET)]
#     prompt_style = STYLE_SET[style_idx % len(STYLE_SET)]
#     temperature = float(params.get('temperature', 0.4))
#     top_p = float(params.get('top_p', 0.9))

#     n_eval = int(os.getenv('LLM_EVAL_N', '200') or 200)  # AG News更快，可适当多评
#     max_workers = max(1, int(os.getenv('LLM_EVAL_WORKERS', '8') or 8))
#     max_retries = int(os.getenv('LLM_EVAL_RETRIES', '2') or 2)
#     base_backoff = 0.5

#     # 输出很短，默认较小 max_tokens
#     max_tokens = int(params.get('max_tokens', 8))

#     # ===== 数据 =====
#     try:
#         ds = load_dataset("ag_news")
#         test_set = ds["test"].select(range(min(n_eval, len(ds["test"]))))
#     except Exception:
#         return 0.0
#     items = list(test_set)
#     if not items:
#         return 0.0

#     # ===== 单样本分类 =====
#     def solve_one(i: int, item) -> Tuple[int, int, Optional[str], float]:
#         text = item["text"]
#         # print("text",text)
#         gt_id = int(item["label"])
#         gt_label = ID2LABEL[gt_id]
#         msgs = build_messages(text, prompt_style, strategy_template, "en")
#         t0 = time.perf_counter()
#         pred_label = None

#         for attempt in range(max_retries + 1):
#             try:
#                 client = OpenAI(
#                     api_key=os.getenv("DASHSCOPE_API_KEY"),
#                     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
#                 )
#                 out = client.chat.completions.create(
#                     model="qwen2.5-7b-instruct",
#                     messages=msgs,
#                     max_tokens=max_tokens,
#                     temperature=float(temperature),
#                     top_p=float(top_p),
#                 )
#                 # print("out",out.choices[0].message.content)
#                 # input()
#                 pred_label = extract_label(out.choices[0].message.content or "")
#                 break
#             except Exception:
#                 if attempt < max_retries:
#                     time.sleep(base_backoff * (2 ** attempt))
#         latency = time.perf_counter() - t0
#         return i, gt_id, pred_label, latency

#     # ===== 并发执行 =====
#     from concurrent.futures import ThreadPoolExecutor, as_completed
#     t_wall_start = time.perf_counter()
#     workers = min(max_workers, len(items))

#     futures, results = [], []
#     with ThreadPoolExecutor(max_workers=workers) as ex:
#         for i, it in enumerate(items):
#             futures.append(ex.submit(solve_one, i, it))
#         for f in as_completed(futures):
#             try:
#                 results.append(f.result())
#             except Exception:
#                 pass

#     results.sort(key=lambda x: x[0])

#     # ===== 统计 =====
#     correct, latencies = 0, []
#     errors = []

#     for _, gt_id, pred_label, t in results:
#         latencies.append(t)
#         gt_label = ID2LABEL[gt_id]
#         if pred_label == gt_label:
#             correct += 1
#         else:
#             errors.append((gt_label, pred_label))

#     n_total = len(results)
#     acc = (correct / n_total) if n_total else 0.0
#     avg_t = (sum(latencies) / len(latencies)) if latencies else 0.0
#     t_wall = time.perf_counter() - t_wall_start

#     # ===== 打印 =====
#     print(f"[time] wall={t_wall:.2f}s | [acc]={acc:.3f} ({correct}/{n_total})", flush=True)
#     print(
#         f"[params] strategy_template={strategy_template}, style={prompt_style}, "
#         f"temperature={float(temperature):.3f}, top_p={float(top_p):.3f}, "
#         f"workers={workers}, retries={max_retries}, max_tokens={max_tokens}",
#         flush=True,
#     )

#     # 如需查看错误样例，可自行打印：
#     # for gt, pd in errors[:10]:
#     #     print(f"[err] gt={gt} | pred={pd}")

#     return float(acc)
def llm_agnews(params: dict) -> float:
    """
    并发评测（AG News，四分类，自一致性投票版）：
    - 标签集合：World, Sports, Business, Sci/Tech
    - 多采样 + 投票：每个样本生成 n_samples 个预测，投票决定最终答案
    - 评估：准确率
    """
    import os, re, time
    from typing import Tuple, Optional
    from collections import Counter

    try:
        from datasets import load_dataset
    except Exception:
        return 0.0
    try:
        from openai import OpenAI
    except Exception:
        return 0.0

    # ===== Prompt 模板配置（保持一致） =====
    STRATEGY_TEMPLATE_SET = ["rubric", "critique", "cot"]
    STYLE_SET = ["step_by_step", "formal", "creative"]

    TEMPLATE_TO_SYSTEM = {
        "rubric":   "Knowns → equation → compute → check → final number.",
        "critique": "Draft briefly, self-check quickly, then corrected final number.",
        "cot":      "Solve in 2–4 ultra-brief steps."
    }
    STYLE_TO_PREFIX = {
        "step_by_step":  "Show steps briefly. ",
        "formal":        "Be precise. ",
        "creative":      "Be engaging. "
    }
    LANG_TO_SUFFIX = {
        "en": "Answer in English.",
        "zh": "请使用中文回答。",
        "zh_en_mix": "Answer bilingually: first Chinese, then English."
    }

    # ===== 标签与解析 =====
    ID2LABEL = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    def normalize_label(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        s = s.strip().lower()
        if s in {"0", "world"}: return "World"
        if s in {"1", "sports", "sport"}: return "Sports"
        if s in {"2", "business", "finance", "economy"}: return "Business"
        if s in {"3", "sci/tech", "scitech", "science", "technology", "tech"}: return "Sci/Tech"
        # 兜底
        if "world" in s: return "World"
        if "sport" in s: return "Sports"
        if "business" in s or "finance" in s or "economy" in s: return "Business"
        if "sci" in s or "tech" in s: return "Sci/Tech"
        return None

    def extract_label(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        m = re.findall(r'#{2,}\s*([A-Za-z0-9/]+)', text)
        if m:
            cand = normalize_label(m[-1])
            if cand: return cand
        return normalize_label(text)

    # ===== 构造消息 =====
    def build_messages(news_text: str, prompt_style: str, strategy_template: str, language: str):
        style_prefix = STYLE_TO_PREFIX.get(prompt_style, "")
        strategy_text = TEMPLATE_TO_SYSTEM.get(strategy_template, "")
        lang_suffix = LANG_TO_SUFFIX.get(language, "")
        system_content = f"{style_prefix}{strategy_text} {lang_suffix}".strip()

        instruction = (
            "Classify the news into one of the categories: World, Sports, Business, Sci/Tech.\n"
            "Return ONLY the final label in the format: #### <label>\n"
            "Examples: World | Sports | Business | Sci/Tech"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{instruction}\n\nNews:\n{news_text}\n\nOutput: #### <label>"}
        ]

    # ===== 参数 =====
    strategy_template_idx = int(params.get('strategy_template', 0))
    style_idx = int(params.get('prompt_style', 0))
    strategy_template = STRATEGY_TEMPLATE_SET[strategy_template_idx % len(STRATEGY_TEMPLATE_SET)]
    prompt_style = STYLE_SET[style_idx % len(STYLE_SET)]
    temperature = float(params.get('temperature', 0.4))
    top_p = float(params.get('top_p', 0.9))

    n_eval = int(os.getenv('LLM_EVAL_N', '200') or 200)
    n_samples = int(os.getenv('LLM_VOTE_N', '5') or 5)   # 每条样本的采样次数
    max_workers = max(1, int(os.getenv('LLM_EVAL_WORKERS', '8') or 8))
    max_retries = int(os.getenv('LLM_EVAL_RETRIES', '2') or 2)
    base_backoff = 0.5
    max_tokens = int(params.get('max_tokens', 8))

    # ===== 数据 =====
    try:
        ds = load_dataset("ag_news")
        test_set = ds["test"].select(range(min(n_eval, len(ds["test"]))))
    except Exception:
        return 0.0
    items = list(test_set)
    if not items:
        return 0.0

    # ===== 单样本求解 + 投票 =====
    def solve_one(i: int, item) -> Tuple[int, int, Optional[str], float]:
        text = item["text"]
        gt_id = int(item["label"])
        gt_label = ID2LABEL[gt_id]
        msgs = build_messages(text, prompt_style, strategy_template, "en")
        t0 = time.perf_counter()
        preds = []

        for _ in range(n_samples):
            for attempt in range(max_retries + 1):
                try:
                    client = OpenAI(
                        api_key=os.getenv("DASHSCOPE_API_KEY"),
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                    )
                    out = client.chat.completions.create(
                        model="qwen2.5-7b-instruct",
                        messages=msgs,
                        max_tokens=max_tokens,
                        temperature=float(temperature),
                        top_p=float(top_p),
                    )
                    pred_label = extract_label(out.choices[0].message.content or "")
                    if pred_label:
                        preds.append(pred_label)
                    break
                except Exception:
                    if attempt < max_retries:
                        time.sleep(base_backoff * (2 ** attempt))
        latency = time.perf_counter() - t0

        # 多数投票
        final_pred = Counter(preds).most_common(1)[0][0] if preds else None
        return i, gt_id, final_pred, latency

    # ===== 并发执行 =====
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t_wall_start = time.perf_counter()
    workers = min(max_workers, len(items))

    futures, results = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, it in enumerate(items):
            futures.append(ex.submit(solve_one, i, it))
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception:
                pass

    results.sort(key=lambda x: x[0])

    # ===== 统计 =====
    correct, latencies = 0, []
    errors = []

    for _, gt_id, pred_label, t in results:
        latencies.append(t)
        gt_label = ID2LABEL[gt_id]
        if pred_label == gt_label:
            correct += 1
        else:
            errors.append((gt_label, pred_label))

    n_total = len(results)
    acc = (correct / n_total) if n_total else 0.0
    avg_t = (sum(latencies) / len(latencies)) if latencies else 0.0
    t_wall = time.perf_counter() - t_wall_start

    # ===== 打印 =====
    print(f"[time] wall={t_wall:.2f}s | [lat] avg={avg_t:.3f}s | [acc]={acc:.3f} ({correct}/{n_total})", flush=True)
    print(
        f"[params] strategy_template={strategy_template}, style={prompt_style}, "
        f"temperature={float(temperature):.3f}, top_p={float(top_p):.3f}, "
        f"n_samples={n_samples}, workers={workers}, retries={max_retries}, max_tokens={max_tokens}",
        flush=True,
    )

    return float(acc)


def llm_triviaqa(params: dict) -> float:
    """
    Closed QA 评测 (TriviaQA 子集)：
    - 指标：Exact Match (EM) + F1
    - categorical 参数：不同的 prompt 模版 (short / complete_sentence / cot / bilingual)
    - continuous 参数：temperature, top_p
    """
    import os, re, time
    from typing import Tuple, Optional

    try:
        from datasets import load_dataset
    except Exception:
        return 0.0
    try:
        from openai import OpenAI
    except Exception:
        return 0.0

    # ===== Prompt 模版 =====
    QA_TEMPLATE_SET = {
        "short": "Answer with only the entity name. No explanation.",
        "complete_sentence": "Answer the question in a complete sentence.",
        "cot": "Think step by step, then give the final short answer.",
        "bilingual": "Answer in English and then in Chinese."
    }

    STYLE_SET = ["short", "complete_sentence", "cot", "bilingual"]

    # ===== 参数解析 =====
    style_idx = int(params.get('prompt_style', 0))
    style = STYLE_SET[style_idx % len(STYLE_SET)]
    system_prompt = QA_TEMPLATE_SET[style]

    temperature = float(params.get('temperature', 0.4))
    top_p = float(params.get('top_p', 0.9))
    max_tokens = int(params.get('max_tokens', 128))

    n_eval = int(os.getenv('LLM_EVAL_N', '50') or 50)  # 默认评测 50 条
    max_workers = max(1, int(os.getenv('LLM_EVAL_WORKERS', '8') or 8))
    max_retries = int(os.getenv('LLM_EVAL_RETRIES', '2') or 2)
    base_backoff = 0.5

    # ===== 数据 =====
    try:
        ds = load_dataset("trivia_qa", "rc")  # TriviaQA Reading Comprehension 版本
        test_set = ds["validation"].select(range(min(n_eval, len(ds["validation"]))))
    except Exception:
        return 0.0
    items = list(test_set)
    if not items:
        return 0.0

    # ===== 工具函数 =====
    def normalize_text(s: str) -> str:
        """简单清洗，用于 EM/F1 匹配"""
        s = s.lower()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = re.sub(r'[^a-z0-9\s]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def f1_score(prediction: str, ground_truth: str) -> float:
        pred_tokens = normalize_text(prediction).split()
        gt_tokens = normalize_text(ground_truth).split()
        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            return 0.0
        prec = len(common) / len(pred_tokens)
        rec = len(common) / len(gt_tokens)
        return 2 * prec * rec / (prec + rec)

    # ===== 构造消息 =====
    def build_messages(question: str, system_prompt: str):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

    # ===== 单样本求解 =====
    def solve_one(i: int, item) -> Tuple[int, str, str, float]:
        q = item["question"]
        gt = item["answer"]["value"] if isinstance(item["answer"], dict) else item["answer"]

        msgs = build_messages(q, system_prompt)
        t0 = time.perf_counter()
        pred = None

        for attempt in range(max_retries + 1):
            try:
                client = OpenAI(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
                out = client.chat.completions.create(
                    model="qwen2.5-7b-instruct",
                    messages=msgs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                pred = out.choices[0].message.content.strip()
                break
            except Exception:
                if attempt < max_retries:
                    time.sleep(base_backoff * (2 ** attempt))

        latency = time.perf_counter() - t0
        return i, gt, pred, latency

    # ===== 并发执行 =====
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t_wall_start = time.perf_counter()
    workers = min(max_workers, len(items))

    futures, results = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, it in enumerate(items):
            futures.append(ex.submit(solve_one, i, it))
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception:
                pass

    results.sort(key=lambda x: x[0])

    # ===== 统计 EM/F1 =====
    em_total, f1_total, latencies = 0, 0, []
    for _, gt, pred, t in results:
        latencies.append(t)
        if not gt or not pred:
            continue
        gt_norm, pred_norm = normalize_text(gt), normalize_text(pred)
        if gt_norm == pred_norm:
            em_total += 1
        f1_total += f1_score(pred, gt)

    n_total = len(results)
    em = em_total / n_total if n_total else 0.0
    f1 = f1_total / n_total if n_total else 0.0
    avg_t = (sum(latencies) / len(latencies)) if latencies else 0.0
    t_wall = time.perf_counter() - t_wall_start

    # ===== 打印 =====
    print(f"[time] wall={t_wall:.2f}s | [acc] EM={em:.3f}, F1={f1:.3f} ({n_total} samples)", flush=True)
    print(f"[params] style={style}, temperature={temperature:.2f}, top_p={top_p:.2f}, max_tokens={max_tokens}", flush=True)

    return float(em)


def llm_translation(params: dict) -> float:
    """
    翻译评测（Closed MT）：
    - 数据：opus100（默认 en→de，可改）
    - 指标：Corpus BLEU（返回值）+ ROUGE-L（日志打印）
    - 连续：temperature, top_p
    - 类别：prompt_style ∈ {faithful, concise, formal, creative}
    环境变量：
      TRANS_SRC=en  TRANS_TGT=de
      LLM_EVAL_N=200  LLM_EVAL_WORKERS=8  LLM_EVAL_RETRIES=2
    """
    import os, re, time
    from typing import List, Tuple, Optional
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        from datasets import load_dataset
    except Exception:
        return 0.0
    try:
        from openai import OpenAI
    except Exception:
        return 0.0

    # ===== 工具：分词 / BLEU / ROUGE-L =====
    _CJK_RANGE = (
        ("\u4E00", "\u9FFF"),  # CJK Unified
        ("\u3400", "\u4DBF"),  # CJK Ext-A
        ("\uF900", "\uFAFF"),  # CJK Compatibility Ideographs
    )
    def _is_cjk(ch: str) -> bool:
        o = ord(ch)
        for lo, hi in _CJK_RANGE:
            if ord(lo) <= o <= ord(hi):
                return True
        return False

    def tokenize(s: str) -> List[str]:
        s = (s or "").strip()
        # CJK 单字切分 + 非CJK按空格和标点切分
        out, buf = [], []
        for ch in s:
            if _is_cjk(ch):
                if buf:
                    out += re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", "".join(buf))
                    buf = []
                out.append(ch)
            else:
                buf.append(ch)
        if buf:
            out += re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", "".join(buf))
        # 去掉纯空白
        out = [t for t in out if not re.fullmatch(r"\s+", t)]
        return out

    def corpus_bleu(list_of_references: List[List[str]], hypotheses: List[List[str]], n_gram: int = 4) -> float:
        # Papineni BLEU（简化版）+ brevity penalty
        from math import log, exp
        # 计数
        import collections
        precisions = []
        for n in range(1, n_gram + 1):
            num, den = 0, 0
            for refs, hyp in zip(list_of_references, hypotheses):
                hyp_ngrams = collections.Counter(tuple(hyp[i:i+n]) for i in range(max(0, len(hyp)-n+1)))
                max_ref_counts = collections.Counter()
                for r in refs:
                    r_ngrams = collections.Counter(tuple(r[i:i+n]) for i in range(max(0, len(r)-n+1)))
                    for k, v in r_ngrams.items():
                        if v > max_ref_counts[k]:
                            max_ref_counts[k] = v
                overlap = {k: min(v, max_ref_counts.get(k, 0)) for k, v in hyp_ngrams.items()}
                num += sum(overlap.values())
                den += max(sum(hyp_ngrams.values()), 1)
            precisions.append(num / den if den > 0 else 0.0)

        # brevity penalty
        hyp_len = sum(len(h) for h in hypotheses)
        # 选与 hyp_len 最接近的参考长度
        ref_len = 0
        for refs, hyp in zip(list_of_references, hypotheses):
            lens = [len(r) for r in refs]
            best = min(lens, key=lambda x: (abs(x - len(hyp)), x))
            ref_len += best
        if hyp_len == 0:
            return 0.0
        if hyp_len > ref_len:
            bp = 1.0
        else:
            from math import exp
            bp = exp(1 - ref_len / hyp_len)
        # 几何平均
        import math
        if any(p == 0 for p in precisions):
            geo = 0.0
        else:
            geo = exp(sum((1/n_gram) * log(p) for p in precisions))
        return bp * geo

    def rouge_l(hyp: List[str], ref: List[str]) -> float:
        # 词级 LCS-based ROUGE-L F-score
        m, n = len(ref), len(hyp)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                if ref[i] == hyp[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
        lcs = dp[m][n]
        prec = lcs / max(n, 1)
        rec  = lcs / max(m, 1)
        return 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)

    # ===== 模板（categorical）=====
    STYLES = ["faithful", "concise", "formal", "creative"]
    STYLE_TO_SYSTEM = {
        "faithful": "Translate faithfully. Preserve meaning and terminology. Output only the translation.",
        "concise":  "Translate concisely and naturally. Output only the translation.",
        "formal":   "Translate in a formal tone and clear grammar. Output only the translation.",
        "creative": "Translate freely with natural phrasing while staying faithful to meaning. Output only the translation."
    }

    style_idx = int(params.get("prompt_style", 0))
    style = STYLES[style_idx % len(STYLES)]
    temperature = float(params.get("temperature", 0.5))
    top_p = float(params.get("top_p", 0.9))
    max_tokens = int(params.get("max_tokens", 128))

    src = os.getenv("TRANS_SRC", "en")
    tgt = os.getenv("TRANS_TGT", "de")
    n_eval = int(os.getenv("LLM_EVAL_N", "200") or 200)
    max_workers = max(1, int(os.getenv("LLM_EVAL_WORKERS", "8") or 8))
    max_retries = int(os.getenv("LLM_EVAL_RETRIES", "2") or 2)
    base_backoff = 0.5

    # ===== 数据：opus100（多语翻译对）=====
    try:
        ds = load_dataset("opus100", f"{src}-{tgt}")
        split = "test" if "test" in ds else "validation"
        data = ds[split].select(range(min(n_eval, len(ds[split]))))
    except Exception:
        return 0.0

    # ===== OpenAI 兼容客户端 =====
    try:
        client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    except Exception:
        return 0.0

    def build_messages(text: str) -> list:
        sys = STYLE_TO_SYSTEM[style]
        return [
            {"role": "system", "content": sys + f" The target language is {tgt}."},
            {"role": "user", "content": f"Source ({src}): {text}\nTranslate to {tgt}."}
        ]

    # ===== 单样本翻译 =====
    def solve_one(i, item) -> Tuple[int, str, str, float]:
        # opus100 每条在 'translation' 字段里：{src_lang: "...", tgt_lang: "..."}
        src_text = item["translation"].get(src, "")
        ref_text = item["translation"].get(tgt, "")
        msgs = build_messages(src_text)
        t0 = time.perf_counter()
        pred = ""
        for attempt in range(max_retries + 1):
            try:
                out = client.chat.completions.create(
                    model="qwen2.5-7b-instruct",
                    messages=msgs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                pred = (out.choices[0].message.content or "").strip()
                break
            except Exception:
                if attempt < max_retries:
                    time.sleep(base_backoff * (2 ** attempt))
        latency = time.perf_counter() - t0
        return i, ref_text, pred, latency

    # ===== 并发跑 =====
    futs, results = [], []
    t_wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, it in enumerate(data):
            futs.append(ex.submit(solve_one, i, it))
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception:
                pass
    results.sort(key=lambda x: x[0])

    # ===== 计算 BLEU / ROUGE-L =====
    refs_tok, hyps_tok, lats = [], [], []
    for _, ref, hyp, lat in results:
        lats.append(lat)
        refs_tok.append([tokenize(ref)])
        hyps_tok.append(tokenize(hyp))
    bleu = corpus_bleu(refs_tok, hyps_tok, n_gram=4)
    rouge_l_avg = sum(rouge_l(h, r[0]) for r, h in zip(refs_tok, hyps_tok)) / max(len(hyps_tok), 1)
    t_wall = time.perf_counter() - t_wall0
    avg_lat = sum(lats)/len(lats) if lats else 0.0

    print(f"[time] wall={t_wall:.2f}s | [lat] avg={avg_lat:.2f}s | [BLEU]={bleu:.3f} | [ROUGE-L]={rouge_l_avg:.3f}")
    print(f"[params] style={style}, temp={temperature:.2f}, top_p={top_p:.2f}, max_tokens={max_tokens}, pair={src}->{tgt}")
    return float(bleu)


def llm_paraphrase(params: dict) -> float:
    """
    改写（Paraphrase Generation）评测：
    - 数据：GLUE/MRPC（仅取 label==1 的同义句对），用 s1 作为输入，s2 作为参考
    - 指标：Corpus BLEU（返回值）+ ROUGE-L（打印）
    - 连续：temperature, top_p
    - 类别：prompt_style ∈ {faithful, concise, formal, creative}
    环境变量：
      LLM_EVAL_N=200  LLM_EVAL_WORKERS=8  LLM_EVAL_RETRIES=2
    """
    import os, re, time
    from typing import List, Tuple
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        from datasets import load_dataset
    except Exception:
        return 0.0
    try:
        from openai import OpenAI
    except Exception:
        return 0.0

    # 复用上面的分词/指标实现
    _CJK_RANGE = (("\u4E00", "\u9FFF"), ("\u3400", "\u4DBF"), ("\uF900", "\uFAFF"))
    def _is_cjk(ch: str) -> bool:
        o = ord(ch)
        for lo, hi in _CJK_RANGE:
            if ord(lo) <= o <= ord(hi):
                return True
        return False
    def tokenize(s: str) -> List[str]:
        import re
        s = (s or "").strip()
        out, buf = [], []
        for ch in s:
            if _is_cjk(ch):
                if buf:
                    out += re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", "".join(buf))
                    buf = []
                out.append(ch)
            else:
                buf.append(ch)
        if buf:
            out += re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", "".join(buf))
        out = [t for t in out if not re.fullmatch(r"\s+", t)]
        return out
    def corpus_bleu(list_of_references, hypotheses, n_gram=4) -> float:
        from math import log, exp
        import collections
        precisions = []
        for n in range(1, n_gram + 1):
            num, den = 0, 0
            for refs, hyp in zip(list_of_references, hypotheses):
                hyp_ngrams = collections.Counter(tuple(hyp[i:i+n]) for i in range(max(0, len(hyp)-n+1)))
                max_ref_counts = collections.Counter()
                for r in refs:
                    r_ngrams = collections.Counter(tuple(r[i:i+n]) for i in range(max(0, len(r)-n+1)))
                    for k, v in r_ngrams.items():
                        if v > max_ref_counts[k]:
                            max_ref_counts[k] = v
                overlap = {k: min(v, max_ref_counts.get(k, 0)) for k, v in hyp_ngrams.items()}
                num += sum(overlap.values()); den += max(sum(hyp_ngrams.values()), 1)
            precisions.append(num/den if den>0 else 0.0)
        hyp_len = sum(len(h) for h in hypotheses)
        ref_len = 0
        for refs, hyp in zip(list_of_references, hypotheses):
            lens = [len(r) for r in refs]
            best = min(lens, key=lambda x: (abs(x - len(hyp)), x))
            ref_len += best
        if hyp_len == 0:
            return 0.0
        bp = 1.0 if hyp_len > ref_len else exp(1 - ref_len / hyp_len)
        if any(p == 0 for p in precisions): return 0.0
        return bp * exp(sum((1/n_gram)*log(p) for p in precisions))
    def rouge_l(hyp, ref) -> float:
        m, n = len(ref), len(hyp)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                if ref[i]==hyp[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
        lcs = dp[m][n]
        prec = lcs / max(n, 1); rec = lcs / max(m, 1)
        return 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)

    STYLES = ["faithful", "concise", "formal", "creative"]
    STYLE_TO_SYSTEM = {
        "faithful": "Paraphrase the sentence faithfully, preserving meaning. Output only the rewritten sentence.",
        "concise":  "Paraphrase concisely and naturally, avoiding redundancy. Output only the rewritten sentence.",
        "formal":   "Paraphrase in a formal tone with clear grammar. Output only the rewritten sentence.",
        "creative": "Paraphrase with varied wording while keeping meaning. Output only the rewritten sentence."
    }

    style_idx = int(params.get("prompt_style", 0))
    style = STYLES[style_idx % len(STYLES)]
    temperature = float(params.get("temperature", 0.6))
    top_p = float(params.get("top_p", 0.9))
    max_tokens = int(params.get("max_tokens", 64))

    n_eval = int(os.getenv("LLM_EVAL_N", "200") or 200)
    max_workers = max(1, int(os.getenv("LLM_EVAL_WORKERS", "8") or 8))
    max_retries = int(os.getenv("LLM_EVAL_RETRIES", "2") or 2)
    base_backoff = 0.5

    # ===== 数据：GLUE/MRPC（选 label==1 的同义句对）=====
    try:
        ds = load_dataset("glue", "mrpc")
        split = "validation" if "validation" in ds else "train"
        cand = [ex for ex in ds[split] if int(ex["label"]) == 1]
        data = cand[: min(n_eval, len(cand))]
    except Exception:
        return 0.0

    # ===== OpenAI 兼容客户端 =====
    try:
        client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    except Exception:
        return 0.0

    def build_messages(src: str) -> list:
        sys = STYLE_TO_SYSTEM[style]
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": f"Original: {src}\nParaphrase:"}
        ]

    def solve_one(i, item) -> Tuple[int, str, str, float]:
        s1, s2 = item["sentence1"], item["sentence2"]  # s1→生成；s2→参考
        msgs = build_messages(s1)
        t0 = time.perf_counter()
        pred = ""
        for attempt in range(max_retries + 1):
            try:
                out = client.chat.completions.create(
                    model="qwen2.5-7b-instruct",
                    messages=msgs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                pred = (out.choices[0].message.content or "").strip()
                break
            except Exception:
                if attempt < max_retries:
                    time.sleep(base_backoff * (2 ** attempt))
        lat = time.perf_counter() - t0
        return i, s2, pred, lat

    futs, results = [], []
    t_wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, it in enumerate(data):
            futs.append(ex.submit(solve_one, i, it))
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception:
                pass
    results.sort(key=lambda x: x[0])

    refs_tok, hyps_tok, lats = [], [], []
    for _, ref, hyp, lat in results:
        lats.append(lat)
        refs_tok.append([tokenize(ref)])
        hyps_tok.append(tokenize(hyp))
    bleu = corpus_bleu(refs_tok, hyps_tok, n_gram=4)
    rouge_l_avg = sum(rouge_l(h, r[0]) for r, h in zip(refs_tok, hyps_tok)) / max(len(hyps_tok), 1)
    t_wall = time.perf_counter() - t_wall0
    avg_lat = sum(lats)/len(lats) if lats else 0.0

    print(f"[time] wall={t_wall:.2f}s | [lat] avg={avg_lat:.2f}s | [BLEU]={bleu:.3f} | [ROUGE-L]={rouge_l_avg:.3f}")
    print(f"[params] style={style}, temp={temperature:.2f}, top_p={top_p:.2f}, max_tokens={max_tokens}")
    return float(bleu)





def llm_gsm8k_penalty(params: dict) -> float:
    """
    并发评测（带可选 vote，自一致性；按“网格脚本”的更快逻辑改写版）：
    - 关键优化：复用 OpenAI 客户端；优先 n=vote_k 批量采样，不支持再退化为单采补齐；不因 vote_k 下调并发
    - 判定：数值等价即正确（整数/小数/科学计数法/分数/混合数/百分号），容差=max(1e-8, 1e-6*max(1,|gt|))
    - 输出：wall time / avg latency / final numeric accuracy；以及（可选）若干错误样例的 gt/pred
    - 兼容：vote_k=1 时行为与原版一致；>1 时启用 self-consistency 投票（与主推理相同的 temperature/top_p）

    环境变量（可选；括号内为默认）：
      - LLM_EVAL_N（10）             : 评测样本数上限
      - LLM_EVAL_WORKERS（8）        : 并发线程数
      - LLM_EVAL_RETRIES（2）        : API 调用重试次数
      - LLM_EVAL_VOTE_K（1）         : 每题采样次数（>1 启用投票）
      - LLM_EVAL_PRINT_ERRORS（20）  : 打印前多少个错误样例
      - DASHSCOPE_API_KEY            : DashScope 兼容模式 API Key

    可通过 params 覆盖：temperature/top_p/max_tokens/strategy_template/prompt_style/vote_k/lang/presence_penalty
    """
    import os, re, time
    from typing import Tuple, Optional, List

    try:
        from datasets import load_dataset
    except Exception:
        return 0.0
    try:
        from openai import OpenAI
    except Exception:
        return 0.0

    # ===== 配置（与原版接口兼容） =====
    STRATEGY_TEMPLATE_SET = ["rubric", "critique", "cot"]
    STYLE_SET = ["step_by_step", "formal", "creative"]

    # 提示更“短”：仿照下方脚本，要求 1–3 行推理 + 最后一行严格 #### <number>
    TEMPLATE_TO_SYSTEM = {
        "rubric":   "Think in 1–3 short steps.",
        "critique": "Draft briefly (1–5 lines), self-check quickly.",
        "cot":      "Think in 1–2 short steps."
    }
    STYLE_TO_PREFIX = {
        "step_by_step":  "Be concise. ",
#     "step_by_step": "Explain the steps clearly before the final answer.",
        "formal":        "Be precise. ",
#     "formal": "Use a formal and precise tone.",
        "creative":      "Be engaging. "

 #     "creative": "Use an engaging tone while staying accurate."
    }
    LANG_TO_SUFFIX = {
        "en": "Answer in English.",
        "zh": "请使用中文回答。",
        "zh_en_mix": "Answer bilingually: first Chinese, then English."
    }

    # ===== 数值解析与比较（保留你原版的鲁棒 Decimal/分数/混合数/百分号） =====
    from decimal import Decimal, InvalidOperation, getcontext
    from fractions import Fraction
    getcontext().prec = 50

    _NUM_TOKEN = r'[-+]?(?:(?:\d[\d,]*)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    _MIXED     = r'[-+]?\d+\s+\d+\s*/\s*\d+'
    _ANYNUM    = r'(?:' + _MIXED + r'|\d+\s*/\s*\d+|' + _NUM_TOKEN + r')'

    def _clean(s: str) -> str:
        s = (s or "").strip()
        s = s.replace("−", "-").replace("\u00a0", " ")
        s = re.sub(r',', '', s)      # 去千分位
        s = re.sub(r'^\$', '', s)    # 去前缀货币符
        return s

    def _percent_adjust(raw: str, x: Decimal) -> Decimal:
        return (x / Decimal(100)) if ('%' in (raw or '')) else x

    def parse_number(text: Optional[str]) -> Optional[Decimal]:
        """解析为 Decimal；支持整数/小数/科学计数/分数/混合数；允许尾随文字；处理百分号。"""
        if not text:
            return None
        raw = text
        s = _clean(text)

        m = re.fullmatch(_MIXED, s)
        if m:
            whole, a, b = re.match(r'([-+]?\d+)\s+(\d+)\s*/\s*(\d+)', s).groups()
            val = Fraction(int(whole), 1) + Fraction(int(a), int(b))
            return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))

        m = re.fullmatch(r'([-+]?\d+)\s*/\s*(\d+)', s)
        if m:
            a, b = m.groups()
            val = Fraction(int(a), int(b))
            return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))

        try:
            return _percent_adjust(raw, Decimal(s))
        except InvalidOperation:
            toks = re.findall(_ANYNUM, s)
            if not toks:
                return None
            t = _clean(toks[-1])

            mm = re.fullmatch(_MIXED, t)
            if mm:
                whole, a, b = re.match(r'([-+]?\d+)\s+(\d+)\s*/\s*(\d+)', t).groups()
                val = Fraction(int(whole), 1) + Fraction(int(a), int(b))
                return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))
            mf = re.fullmatch(r'([-+]?\d+)\s*/\s*(\d+)', t)
            if mf:
                a, b = mf.groups()
                val = Fraction(int(a), int(b))
                return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))
            try:
                return _percent_adjust(raw, Decimal(t))
            except InvalidOperation:
                return None

    def numbers_close(a: Optional[Decimal], b: Optional[Decimal],
                      abs_tol: Decimal = Decimal('1e-8'),
                      rel_tol: Decimal = Decimal('1e-6')) -> bool:
        if a is None or b is None:
            return False
        diff = abs(a - b)
        thresh = max(abs_tol, rel_tol * max(Decimal(1), abs(b)))
        return diff <= thresh

    def extract_answer(text: Optional[str]) -> Optional[str]:
        """优先 #### <number>，再 final answer/最终答案，最后兜底取最后数字样式。"""
        if not text:
            return None
        m = re.findall(r'#{2,}\s*(' + _ANYNUM + r')', text)
        if m:
           return m[-1].strip()
        m = re.findall(r'(?:final\s*answer|最终答案)[^0-9\-]*(' + _ANYNUM + r')',
                       text, flags=re.IGNORECASE)
        if m:
            return m[-1].strip()
        m = re.findall(_ANYNUM, text)
        return m[-1].strip() if m else None

    # ===== 构造消息（更短、更严格的一行答案约束） =====
    def build_messages(question: str, prompt_style: str, strategy_template: str, language: str):
        style_prefix = STYLE_TO_PREFIX.get(prompt_style, "")
        strategy_text = TEMPLATE_TO_SYSTEM.get(strategy_template, "")
        lang_suffix = LANG_TO_SUFFIX.get(language, "")
        # system
        system_content = f"{style_prefix}{strategy_text} {lang_suffix}".strip()
        # user：明确最后只输出一行
        user_content = (
            f"Problem:\n{question}\n\n"
            f"Reason briefly in 1–3 short lines"
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    # ===== 参数 =====
    strategy_template_idx = int(params.get('strategy_template', 2))  # 默认 'cot'
    style_idx = int(params.get('prompt_style', 0))                    # 默认 'step_by_step'
    strategy_template = STRATEGY_TEMPLATE_SET[strategy_template_idx % len(STRATEGY_TEMPLATE_SET)]
    prompt_style = STYLE_SET[style_idx % len(STYLE_SET)]
    temperature = float(params.get('temperature', 0.4))
    top_p = float(params.get('top_p', 0.9))
    presence_penalty = float(params.get('presence_penalty', 0.0))      # ← 新增：presence_penalty
    language = str(params.get('lang', 'en'))

    # vote：默认 1（不开启）；>1 开启自一致性
    import os as _os
    vote_k = int(params.get('vote_k', int(_os.getenv('LLM_EVAL_VOTE_K', '1') or 1)))

    n_eval = int(params.get('n_eval', _os.getenv('LLM_EVAL_N', '10')) or 10)
    max_workers_cfg = int(_os.getenv('LLM_EVAL_WORKERS', '8') or 8)
    max_retries = int(_os.getenv('LLM_EVAL_RETRIES', '2') or 2)
    base_backoff = 0.5
    # 默认更短，配合“只输出一行”
    max_tokens = int(params.get('max_tokens', 128))
    print_err_n = int(_os.getenv('LLM_EVAL_PRINT_ERRORS', '20') or 20)

    # 新策略：不再因 vote_k 下调并发
    workers = max(1, min(max_workers_cfg, n_eval))

    # ===== 数据 =====
    try:
        ds = load_dataset("gsm8k", "main")
    except Exception:
        try:
            ds = load_dataset("openai/gsm8k", "main")
        except Exception:
            return 0.0
    test_set = ds["test"].select(range(min(n_eval, len(ds["test"]))))
    items = list(test_set)
    if not items:
        return 0.0

    # ===== 投票聚类（保留你原版的“近似相等聚类 → 最大簇中位数”） =====
    def cluster_vote(nums: List[Decimal],
                     tol_abs: Decimal = Decimal('1e-8'),
                     tol_rel: Decimal = Decimal('1e-6')) -> Decimal:
        clusters: List[List[Decimal]] = []
        for x in nums:
            placed = False
            for c in clusters:
                if numbers_close(x, c[0], tol_abs, tol_rel):
                    c.append(x); placed = True; break
            if not placed:
                clusters.append([x])

        def cluster_rep(c: List[Decimal]) -> Decimal:
            sc = sorted(c)
            mid = len(sc)//2
            return sc[mid] if len(sc)%2==1 else (sc[mid-1] + sc[mid]) / Decimal(2)

        clusters.sort(key=lambda c: (-len(c), cluster_rep(c)))
        return cluster_rep(clusters[0])

    # ===== 客户端（复用） =====
    try:
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    except Exception:
        return 0.0

    # 全局开关：后端是否支持 presence_penalty（遇到报错会自动关掉并重试）
    presence_supported = True

    # ===== 调用封装：优先 n=vote_k；不支持再退化为单采补齐 =====
    def call_n(messages, n_samples: int) -> Optional[List[Optional[str]]]:
        nonlocal presence_supported  # 需要在异常时下调它
        for attempt in range(max_retries + 1):
            try:
                kwargs = dict(
                    model="qwen2.5-7b-instruct",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    n=n_samples
                )
                # 仅当设置了非零 presence_penalty 且当前判定支持时才传参
                if presence_supported and float(presence_penalty) != 0.0:
                    kwargs["presence_penalty"] = float(presence_penalty)

                out = client.chat.completions.create(**kwargs)

                res = []
                for ch in getattr(out, "choices", []) or []:
                    msg = getattr(ch, "message", None)
                    txt = getattr(msg, "content", None) if msg is not None else getattr(ch, "content", None)
                    if isinstance(txt, bytes):
                        txt = txt.decode("utf-8", "ignore")
                    res.append(extract_answer(txt))
                return res

            except Exception as e:
                em = str(e)

                # ① 明确不支持 n 或 n 超范围 → 返回 None 让上层走退化路径
                if ("unexpected keyword argument 'n'" in em) or ("Range of n" in em) or ("n should be" in em):
                    return None

                # ② 明确不支持 presence_penalty → 自动禁用并立即重试本次
                if ("presence_penalty" in em) or ("presence penalty" in em):
                    if presence_supported:
                        presence_supported = False
                        # 不计入重试配额，立即再来一次
                        continue

                # ③ 其他错误：退避重试
                if attempt < max_retries:
                    time.sleep(base_backoff * (2 ** attempt))
                else:
                    # 到此视为失败（返回空列表，避免无限重试）
                    return []

    # ===== 单样本求解（支持批量投票） =====
    def solve_one(i: int, item) -> Tuple[int, Optional[str], Optional[str], float]:
        q = item["question"]
        gt = extract_answer(item["answer"])
        msgs = build_messages(q, prompt_style, strategy_template, language)
        t0 = time.perf_counter()
        final_pred_str: Optional[str] = None

        if vote_k <= 1:
            preds = call_n(msgs, 1) or []
        else:
            preds = call_n(msgs, vote_k)
            if preds is None:
                # 不支持 n → 退化为单采补齐
                preds = []
                need = vote_k
                while need > 0:
                    batch = call_n(msgs, 1) or []
                    preds.extend(batch)
                    need -= 1 if batch else 0
                    # 简单保护：避免极端情况下死循环
                    if len(preds) >= vote_k or need <= 0:
                        break

        # 聚合
        nums: List[Decimal] = []
        for p in preds:
            x = parse_number(p)
            if x is not None:
                nums.append(x)
        if nums:
            rep = cluster_vote(nums)
            final_pred_str = str(rep)  # 交给 parse_number 再次解析

        latency = time.perf_counter() - t0
        return i, gt, final_pred_str, latency

    # ===== 并发执行 =====
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t_wall_start = time.perf_counter()
    workers = min(workers, len(items))

    futures, results = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, it in enumerate(items):
            futures.append(ex.submit(solve_one, i, it))
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception:
                pass

    results.sort(key=lambda x: x[0])

    # ===== 统计（numeric）并打印错误样例 =====
    numeric_correct, latencies = 0, []
    errors = []  # 仅用于打印：错误 (gt, pred)

    for _, gt_str, pred_str, t in results:
        latencies.append(t)
        gt_num = parse_number(gt_str)
        pred_num = parse_number(pred_str)
        if numbers_close(pred_num, gt_num):
            numeric_correct += 1
        else:
            errors.append((gt_str, pred_str))

    n_total = len(results)
    acc = (numeric_correct / n_total) if n_total else 0.0
    avg_t = (sum(latencies) / len(latencies)) if latencies else 0.0
    t_wall = time.perf_counter() - t_wall_start

    # ===== 打印（与原版一致的摘要 + 参数） =====
    print(f"[time] wall={t_wall:.2f}s | [lat] avg={avg_t:.2f}s | [acc] numeric={acc:.3f} ({numeric_correct}/{n_total})", flush=True)
    print(
        f"[params] strategy_template={strategy_template}, style={prompt_style}, "
        f"language={language}, "
        f"temperature={float(temperature):.3f}, top_p={float(top_p):.3f}, "
        f"presence_penalty={float(presence_penalty):.3f}, "
        f"vote_k={vote_k}, workers={workers}, retries={max_retries}, "
        f"max_tokens={max_tokens}, n_eval={n_eval}",
        flush=True,
    )

    if errors and print_err_n > 0:
        print(f"[errors] showing up to {min(print_err_n, len(errors))} cases:", flush=True)
        for idx, (gt, pd) in enumerate(errors[:print_err_n]):
            print(f"  #{idx+1:02d} gt={gt!r} | pred={pd!r}", flush=True)

    return float(acc)



def llm_math(params: dict) -> float:
    """
    并发评测（带可选 vote，自一致性；按“网格脚本”的更快逻辑改写版）：
    - 关键优化：复用 OpenAI 客户端；优先 n=vote_k 批量采样，不支持再退化为单采补齐；不因 vote_k 下调并发
    - 判定：数值等价即正确（整数/小数/科学计数法/分数/混合数/百分号），容差=max(1e-8, 1e-6*max(1,|gt|))
    - 输出：wall time / avg latency / final numeric accuracy；以及（可选）若干错误样例的 gt/pred
    - 兼容：vote_k=1 时行为与原版一致；>1 时启用 self-consistency 投票（与主推理相同的 temperature/top_p）

    环境变量（可选；括号内为默认）：
      - LLM_EVAL_N（10）             : 评测样本数上限
      - LLM_EVAL_WORKERS（8）        : 并发线程数
      - LLM_EVAL_RETRIES（2）        : API 调用重试次数
      - LLM_EVAL_VOTE_K（1）         : 每题采样次数（>1 启用投票）
      - LLM_EVAL_PRINT_ERRORS（20）  : 打印前多少个错误样例
      - DASHSCOPE_API_KEY            : DashScope 兼容模式 API Key

    可通过 params 覆盖：temperature/top_p/max_tokens/strategy_template/prompt_style/vote_k/lang/presence_penalty
    """
    import os, re, time
    from typing import Tuple, Optional, List

    try:
        from datasets import load_dataset
    except Exception:
        return 0.0
    try:
        from openai import OpenAI
    except Exception:
        return 0.0

    # ===== 配置（与原版接口兼容） =====
    STRATEGY_TEMPLATE_SET = ["rubric", "critique", "cot"]
    STYLE_SET = ["step_by_step", "formal", "creative"]

    # 提示更“短”：仿照下方脚本，要求 1–3 行推理 + 最后一行严格 #### <number>
    TEMPLATE_TO_SYSTEM = {
        "rubric":   "Think in 1–3 short steps.",
        "critique": "Draft briefly (1–5 lines), self-check quickly.",
        "cot":      "Think in 1–2 short steps."
    }
    STYLE_TO_PREFIX = {
        "step_by_step":  "Be concise. ",
#     "step_by_step": "Explain the steps clearly before the final answer.",
        "formal":        "Be precise. ",
#     "formal": "Use a formal and precise tone.",
        "creative":      "Be engaging. "

 #     "creative": "Use an engaging tone while staying accurate."
    }
    LANG_TO_SUFFIX = {
        "en": "Answer in English.",
        "zh": "请使用中文回答。",
        "zh_en_mix": "Answer bilingually: first Chinese, then English."
    }

    # ===== 数值解析与比较（保留你原版的鲁棒 Decimal/分数/混合数/百分号） =====
    from decimal import Decimal, InvalidOperation, getcontext
    from fractions import Fraction
    getcontext().prec = 50

    _NUM_TOKEN = r'[-+]?(?:(?:\d[\d,]*)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    _MIXED     = r'[-+]?\d+\s+\d+\s*/\s*\d+'
    _ANYNUM    = r'(?:' + _MIXED + r'|\d+\s*/\s*\d+|' + _NUM_TOKEN + r')'

    def _clean(s: str) -> str:
        s = (s or "").strip()
        s = s.replace("−", "-").replace("\u00a0", " ")
        s = re.sub(r',', '', s)      # 去千分位
        s = re.sub(r'^\$', '', s)    # 去前缀货币符
        return s

    def _percent_adjust(raw: str, x: Decimal) -> Decimal:
        return (x / Decimal(100)) if ('%' in (raw or '')) else x

    def parse_number(text: Optional[str]) -> Optional[Decimal]:
        """解析为 Decimal；支持整数/小数/科学计数/分数/混合数；允许尾随文字；处理百分号。"""
        if not text:
            return None
        raw = text
        s = _clean(text)

        m = re.fullmatch(_MIXED, s)
        if m:
            whole, a, b = re.match(r'([-+]?\d+)\s+(\d+)\s*/\s*(\d+)', s).groups()
            val = Fraction(int(whole), 1) + Fraction(int(a), int(b))
            return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))

        m = re.fullmatch(r'([-+]?\d+)\s*/\s*(\d+)', s)
        if m:
            a, b = m.groups()
            val = Fraction(int(a), int(b))
            return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))

        try:
            return _percent_adjust(raw, Decimal(s))
        except InvalidOperation:
            toks = re.findall(_ANYNUM, s)
            if not toks:
                return None
            t = _clean(toks[-1])

            mm = re.fullmatch(_MIXED, t)
            if mm:
                whole, a, b = re.match(r'([-+]?\d+)\s+(\d+)\s*/\s*(\d+)', t).groups()
                val = Fraction(int(whole), 1) + Fraction(int(a), int(b))
                return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))
            mf = re.fullmatch(r'([-+]?\d+)\s*/\s*(\d+)', t)
            if mf:
                a, b = mf.groups()
                val = Fraction(int(a), int(b))
                return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))
            try:
                return _percent_adjust(raw, Decimal(t))
            except InvalidOperation:
                return None

    def numbers_close(a: Optional[Decimal], b: Optional[Decimal],
                      abs_tol: Decimal = Decimal('1e-8'),
                      rel_tol: Decimal = Decimal('1e-6')) -> bool:
        if a is None or b is None:
            return False
        diff = abs(a - b)
        thresh = max(abs_tol, rel_tol * max(Decimal(1), abs(b)))
        return diff <= thresh

    def extract_answer(text: Optional[str]) -> Optional[str]:
        """优先 #### <number>，再 final answer/最终答案，最后兜底取最后数字样式。"""
        if not text:
            return None
        m = re.findall(r'#{2,}\s*(' + _ANYNUM + r')', text)
        if m:
           return m[-1].strip()
        m = re.findall(r'(?:final\s*answer|最终答案)[^0-9\-]*(' + _ANYNUM + r')',
                       text, flags=re.IGNORECASE)
        if m:
            return m[-1].strip()
        m = re.findall(_ANYNUM, text)
        return m[-1].strip() if m else None

    # ===== 构造消息（更短、更严格的一行答案约束） =====
    def build_messages(question: str, prompt_style: str, strategy_template: str, language: str):
        style_prefix = STYLE_TO_PREFIX.get(prompt_style, "")
        strategy_text = TEMPLATE_TO_SYSTEM.get(strategy_template, "")
        lang_suffix = LANG_TO_SUFFIX.get(language, "")
        # system
        system_content = f"{style_prefix}{strategy_text} {lang_suffix}".strip()
        # user：明确最后只输出一行
        user_content = (
            f"Problem:\n{question}\n\n"
            f"Reason briefly in 1–3 short lines"
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    # ===== 参数 =====
    strategy_template_idx = int(params.get('strategy_template', 2))  # 默认 'cot'
    style_idx = int(params.get('prompt_style', 0))                    # 默认 'step_by_step'
    strategy_template = STRATEGY_TEMPLATE_SET[strategy_template_idx % len(STRATEGY_TEMPLATE_SET)]
    prompt_style = STYLE_SET[style_idx % len(STYLE_SET)]
    temperature = float(params.get('temperature', 0.4))
    top_p = float(params.get('top_p', 0.9))
    presence_penalty = float(params.get('presence_penalty', 0.0))      # ← 新增：presence_penalty
    language = str(params.get('lang', 'en'))

    # vote：默认 1（不开启）；>1 开启自一致性
    import os as _os
    vote_k = int(params.get('vote_k', int(_os.getenv('LLM_EVAL_VOTE_K', '1') or 1)))

    n_eval = int(params.get('n_eval', _os.getenv('LLM_EVAL_N', '10')) or 10)
    max_workers_cfg = int(_os.getenv('LLM_EVAL_WORKERS', '8') or 8)
    max_retries = int(_os.getenv('LLM_EVAL_RETRIES', '2') or 2)
    base_backoff = 0.5
    # 默认更短，配合“只输出一行”
    max_tokens = int(params.get('max_tokens', 128))
    print_err_n = int(_os.getenv('LLM_EVAL_PRINT_ERRORS', '20') or 20)

    # 新策略：不再因 vote_k 下调并发
    workers = max(1, min(max_workers_cfg, n_eval))

    # ===== 数据 =====
    try:
        ds = load_dataset("math")
        print("ok")

    except Exception:
        try:
            ds = load_dataset("openai/gsm8k", "main")
        except Exception:
            return 0.0
    test_set = ds["test"].select(range(min(n_eval, len(ds["test"]))))
    items = list(test_set)
    if not items:
        return 0.0

    # ===== 投票聚类（保留你原版的“近似相等聚类 → 最大簇中位数”） =====
    def cluster_vote(nums: List[Decimal],
                     tol_abs: Decimal = Decimal('1e-8'),
                     tol_rel: Decimal = Decimal('1e-6')) -> Decimal:
        clusters: List[List[Decimal]] = []
        for x in nums:
            placed = False
            for c in clusters:
                if numbers_close(x, c[0], tol_abs, tol_rel):
                    c.append(x); placed = True; break
            if not placed:
                clusters.append([x])

        def cluster_rep(c: List[Decimal]) -> Decimal:
            sc = sorted(c)
            mid = len(sc)//2
            return sc[mid] if len(sc)%2==1 else (sc[mid-1] + sc[mid]) / Decimal(2)

        clusters.sort(key=lambda c: (-len(c), cluster_rep(c)))
        return cluster_rep(clusters[0])

    # ===== 客户端（复用） =====
    try:
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    except Exception:
        return 0.0

    # 全局开关：后端是否支持 presence_penalty（遇到报错会自动关掉并重试）
    presence_supported = True

    # ===== 调用封装：优先 n=vote_k；不支持再退化为单采补齐 =====
    def call_n(messages, n_samples: int) -> Optional[List[Optional[str]]]:
        nonlocal presence_supported  # 需要在异常时下调它
        for attempt in range(max_retries + 1):
            try:
                kwargs = dict(
                    model="qwen2.5-7b-instruct",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    n=n_samples
                )
                # 仅当设置了非零 presence_penalty 且当前判定支持时才传参
                if presence_supported and float(presence_penalty) != 0.0:
                    kwargs["presence_penalty"] = float(presence_penalty)

                out = client.chat.completions.create(**kwargs)

                res = []
                for ch in getattr(out, "choices", []) or []:
                    msg = getattr(ch, "message", None)
                    txt = getattr(msg, "content", None) if msg is not None else getattr(ch, "content", None)
                    if isinstance(txt, bytes):
                        txt = txt.decode("utf-8", "ignore")
                    res.append(extract_answer(txt))
                return res

            except Exception as e:
                em = str(e)

                # ① 明确不支持 n 或 n 超范围 → 返回 None 让上层走退化路径
                if ("unexpected keyword argument 'n'" in em) or ("Range of n" in em) or ("n should be" in em):
                    return None

                # ② 明确不支持 presence_penalty → 自动禁用并立即重试本次
                if ("presence_penalty" in em) or ("presence penalty" in em):
                    if presence_supported:
                        presence_supported = False
                        # 不计入重试配额，立即再来一次
                        continue

                # ③ 其他错误：退避重试
                if attempt < max_retries:
                    time.sleep(base_backoff * (2 ** attempt))
                else:
                    # 到此视为失败（返回空列表，避免无限重试）
                    return []

    # ===== 单样本求解（支持批量投票） =====
    def solve_one(i: int, item) -> Tuple[int, Optional[str], Optional[str], float]:
        q = item["question"]
        gt = extract_answer(item["answer"])
        msgs = build_messages(q, prompt_style, strategy_template, language)
        t0 = time.perf_counter()
        final_pred_str: Optional[str] = None

        if vote_k <= 1:
            preds = call_n(msgs, 1) or []
        else:
            preds = call_n(msgs, vote_k)
            if preds is None:
                # 不支持 n → 退化为单采补齐
                preds = []
                need = vote_k
                while need > 0:
                    batch = call_n(msgs, 1) or []
                    preds.extend(batch)
                    need -= 1 if batch else 0
                    # 简单保护：避免极端情况下死循环
                    if len(preds) >= vote_k or need <= 0:
                        break

        # 聚合
        nums: List[Decimal] = []
        for p in preds:
            x = parse_number(p)
            if x is not None:
                nums.append(x)
        if nums:
            rep = cluster_vote(nums)
            final_pred_str = str(rep)  # 交给 parse_number 再次解析

        latency = time.perf_counter() - t0
        return i, gt, final_pred_str, latency

    # ===== 并发执行 =====
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t_wall_start = time.perf_counter()
    workers = min(workers, len(items))

    futures, results = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, it in enumerate(items):
            futures.append(ex.submit(solve_one, i, it))
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception:
                pass

    results.sort(key=lambda x: x[0])

    # ===== 统计（numeric）并打印错误样例 =====
    numeric_correct, latencies = 0, []
    errors = []  # 仅用于打印：错误 (gt, pred)

    for _, gt_str, pred_str, t in results:
        latencies.append(t)
        gt_num = parse_number(gt_str)
        pred_num = parse_number(pred_str)
        if numbers_close(pred_num, gt_num):
            numeric_correct += 1
        else:
            errors.append((gt_str, pred_str))

    n_total = len(results)
    acc = (numeric_correct / n_total) if n_total else 0.0
    avg_t = (sum(latencies) / len(latencies)) if latencies else 0.0
    t_wall = time.perf_counter() - t_wall_start

    # ===== 打印（与原版一致的摘要 + 参数） =====
    print(f"[time] wall={t_wall:.2f}s | [lat] avg={avg_t:.2f}s | [acc] numeric={acc:.3f} ({numeric_correct}/{n_total})", flush=True)
    print(
        f"[params] strategy_template={strategy_template}, style={prompt_style}, "
        f"language={language}, "
        f"temperature={float(temperature):.3f}, top_p={float(top_p):.3f}, "
        f"presence_penalty={float(presence_penalty):.3f}, "
        f"vote_k={vote_k}, workers={workers}, retries={max_retries}, "
        f"max_tokens={max_tokens}, n_eval={n_eval}",
        flush=True,
    )

    if errors and print_err_n > 0:
        print(f"[errors] showing up to {min(print_err_n, len(errors))} cases:", flush=True)
        for idx, (gt, pd) in enumerate(errors[:print_err_n]):
            print(f"  #{idx+1:02d} gt={gt!r} | pred={pd!r}", flush=True)

    return float(acc)



def llm_gsm8k_1008(params: dict) -> float:
    """
    并发评测（带可选 vote，自一致性；按“网格脚本”的更快逻辑改写版）：
    - 关键优化：复用 OpenAI 客户端；优先 n=vote_k 批量采样，不支持再退化为单采补齐；不因 vote_k 下调并发
    - 判定：数值等价即正确（整数/小数/科学计数法/分数/混合数/百分号），容差=max(1e-8, 1e-6*max(1,|gt|))
    - 输出：wall time / avg latency / final numeric accuracy；以及（可选）若干错误样例的 gt/pred
    - 兼容：vote_k=1 时行为与原版一致；>1 时启用 self-consistency 投票（与主推理相同的 temperature/top_p）
    
    环境变量（可选；括号内为默认）：
      - LLM_EVAL_N（10）             : 评测样本数上限
      - LLM_EVAL_WORKERS（8）        : 并发线程数
      - LLM_EVAL_RETRIES（2）        : API 调用重试次数
      - LLM_EVAL_VOTE_K（1）         : 每题采样次数（>1 启用投票）
      - LLM_EVAL_PRINT_ERRORS（20）  : 打印前多少个错误样例
      - DASHSCOPE_API_KEY            : DashScope 兼容模式 API Key

    可通过 params 覆盖：temperature/top_p/max_tokens/strategy_template/prompt_style/vote_k/lang
    """
    import os, re, time
    from typing import Tuple, Optional, List

    try:
        from datasets import load_dataset
    except Exception:
        return 0.0
    try:
        from openai import OpenAI
    except Exception:
        return 0.0

    # ===== 配置（与原版接口兼容） =====
    STRATEGY_TEMPLATE_SET = ["rubric", "critique", "cot"]
    STYLE_SET = ["step_by_step", "formal", "creative"]

    # 提示更“短”：仿照下方脚本，要求 1–3 行推理 + 最后一行严格 #### <number>
    TEMPLATE_TO_SYSTEM = {
        "rubric":   "Think in 1–3 short steps.",
        "critique": "Draft briefly (1–3 lines), self-check quickly.",
        "cot":      "Think in 1–3 short steps."
    }
    STYLE_TO_PREFIX = {
        "step_by_step":  "Be concise. ",
        "formal":        "Be precise. ",
        "creative":      "Be engaging. "
    }
    LANG_TO_SUFFIX = {
        "en": "Answer in English.",
        "zh": "请使用中文回答。",
        "zh_en_mix": "Answer bilingually: first Chinese, then English."
    }

    # ===== 数值解析与比较（保留你原版的鲁棒 Decimal/分数/混合数/百分号） =====
    from decimal import Decimal, InvalidOperation, getcontext
    from fractions import Fraction
    getcontext().prec = 50

    _NUM_TOKEN = r'[-+]?(?:(?:\d[\d,]*)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    _MIXED     = r'[-+]?\d+\s+\d+\s*/\s*\d+'
    _ANYNUM    = r'(?:' + _MIXED + r'|\d+\s*/\s*\d+|' + _NUM_TOKEN + r')'

    def _clean(s: str) -> str:
        s = (s or "").strip()
        s = s.replace("−", "-").replace("\u00a0", " ")
        s = re.sub(r',', '', s)      # 去千分位
        s = re.sub(r'^\$', '', s)    # 去前缀货币符
        return s

    def _percent_adjust(raw: str, x: Decimal) -> Decimal:
        return (x / Decimal(100)) if ('%' in (raw or '')) else x

    def parse_number(text: Optional[str]) -> Optional[Decimal]:
        """解析为 Decimal；支持整数/小数/科学计数/分数/混合数；允许尾随文字；处理百分号。"""
        if not text:
            return None
        raw = text
        s = _clean(text)

        m = re.fullmatch(_MIXED, s)
        if m:
            whole, a, b = re.match(r'([-+]?\d+)\s+(\d+)\s*/\s*(\d+)', s).groups()
            val = Fraction(int(whole), 1) + Fraction(int(a), int(b))
            return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))

        m = re.fullmatch(r'([-+]?\d+)\s*/\s*(\d+)', s)
        if m:
            a, b = m.groups()
            val = Fraction(int(a), int(b))
            return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))

        try:
            return _percent_adjust(raw, Decimal(s))
        except InvalidOperation:
            toks = re.findall(_ANYNUM, s)
            if not toks:
                return None
            t = _clean(toks[-1])

            mm = re.fullmatch(_MIXED, t)
            if mm:
                whole, a, b = re.match(r'([-+]?\d+)\s+(\d+)\s*/\s*(\d+)', t).groups()
                val = Fraction(int(whole), 1) + Fraction(int(a), int(b))
                return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))
            mf = re.fullmatch(r'([-+]?\d+)\s*/\s*(\d+)', t)
            if mf:
                a, b = mf.groups()
                val = Fraction(int(a), int(b))
                return _percent_adjust(raw, Decimal(val.numerator) / Decimal(val.denominator))
            try:
                return _percent_adjust(raw, Decimal(t))
            except InvalidOperation:
                return None

    def numbers_close(a: Optional[Decimal], b: Optional[Decimal],
                      abs_tol: Decimal = Decimal('1e-8'),
                      rel_tol: Decimal = Decimal('1e-6')) -> bool:
        if a is None or b is None:
            return False
        diff = abs(a - b)
        thresh = max(abs_tol, rel_tol * max(Decimal(1), abs(b)))
        return diff <= thresh

    def extract_answer(text: Optional[str]) -> Optional[str]:
        """优先 #### <number>，再 final answer/answer is，最后兜底取最后数字样式。"""
        if not text:
            return None
        m = re.findall(r'#{2,}\s*(' + _ANYNUM + r')', text)
        if m:
           return m[-1].strip()
        m = re.findall(r'(?:final\s*answer|answer\s*is)\D*(' + _ANYNUM + r')',
                       text, flags=re.IGNORECASE)
        if m:
            return m[-1].strip()
        m = re.findall(_ANYNUM, text)
        return m[-1].strip() if m else None

    # ===== 构造消息（更短、更严格的一行答案约束） =====
    def build_messages(question: str, prompt_style: str, strategy_template: str):
        style_prefix = STYLE_TO_PREFIX.get(prompt_style, "")
        strategy_text = TEMPLATE_TO_SYSTEM.get(strategy_template, "")
        lang_suffix = "en"
        # system
        system_content = f"{style_prefix}{strategy_text} {lang_suffix}".strip()
        # user：明确最后只输出一行并要求严格输出格式
        user_content = (
            f"Problem:\n{question}\n\n"
            f"Reason briefly in 1–3 short lines. Then output the final numeric answer in the format: #### <number>"
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    # ===== 参数 =====
    strategy_template_idx = int(params.get('strategy_template', 2))  # 默认 'cot'
    style_idx = int(params.get('prompt_style', 0))                    # 默认 'step_by_step'
    strategy_template = STRATEGY_TEMPLATE_SET[strategy_template_idx % len(STRATEGY_TEMPLATE_SET)]
    prompt_style = STYLE_SET[style_idx % len(STYLE_SET)]
    temperature = float(params.get('temperature', 0.4))
    top_p = float(params.get('top_p', 0.9))
    # 评测样本数默认与 synthetic 版本一致（2），可被 env/params 覆盖
    n_eval = int(params.get('n_eval', os.getenv('LLM_EVAL_N', '2')) or 2)
    max_workers_cfg = int(os.getenv('LLM_EVAL_WORKERS', '8') or 8)
    max_retries = int(os.getenv('LLM_EVAL_RETRIES', '2') or 2)
    base_backoff = 0.5
    # 生成长度优先 params，其次 env；默认 128
    try:
        max_tokens = int(params.get('max_tokens', os.getenv('LLM_EVAL_MAX_TOKENS', '128')) or 128)
    except Exception:
        max_tokens = 128

    workers = max(1, min(max_workers_cfg, n_eval))

    # ===== 数据 =====
    try:
        ds = load_dataset("gsm8k", "main")
    except Exception:
        try:
            ds = load_dataset("openai/gsm8k", "main")
        except Exception:
            return 0.0
    test_set = ds["test"].select(range(min(n_eval, len(ds["test"]))))
    items = list(test_set)
    if not items:
        return 0.0

    # ===== 客户端（复用） =====
    try:
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    except Exception:
        return 0.0

    # ===== 单样本求解（单采样 + 重试） =====
    def solve_one(i: int, item) -> Tuple[int, Optional[str]]:
        q = item["question"]
        msgs = build_messages(q, prompt_style, strategy_template)
        for attempt in range(max_retries + 1):
            try:
                out = client.chat.completions.create(
                    model="qwen2.5-7b-instruct",
                    messages=msgs,
                    max_tokens=max_tokens,
                    temperature=float(temperature),
                    top_p=float(top_p)
                )
                pred_text = getattr(out.choices[0].message, 'content', None) or ""
                pred = extract_answer(pred_text)
                return i, pred
            except Exception:
                if attempt < max_retries:
                    time.sleep(base_backoff * (2 ** attempt))
        return i, None

    # ===== 并发执行 =====
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t_wall_start = time.perf_counter()
    workers = min(workers, len(items))

    futures, results = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, it in enumerate(items):
            futures.append(ex.submit(solve_one, i, it))
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception:
                pass
    # ===== 统计并输出（与 synthetic 版本一致，仅打印 acc） =====
    results.sort(key=lambda x: x[0])
    preds: List[Optional[str]] = [None] * len(results)
    for idx, pr in results:
        preds[idx] = pr

    correct = 0
    for i, it in enumerate(items):
        gt_text = it["answer"]
        gt = extract_answer(gt_text)
        gt_num = parse_number(gt)
        pred_num = parse_number(preds[i])
        if numbers_close(pred_num, gt_num):
            correct += 1

    acc = correct / max(1, len(items))

    print(f"acc: {acc}", flush=True)

    return float(acc)


def llm_gsm8k_lg(params: dict) -> float:
    """
    LLM on GSM8K（含语言作为第三个类别）：
      - 连续：top_p ∈ [0.7,0.9]，temperature ∈ [0.2,0.9]
      - 类别：strategy_template(4)、prompt_style(4)、language(3={en,zh,mixed})

    返回：numeric accuracy ∈ [0,1]
    可用环境变量：LLM_EVAL_N, LLM_EVAL_WORKERS, LLM_EVAL_RETRIES, LLM_EVAL_MAX_TOKENS, DASHSCOPE_API_KEY
    """
    import os, re, time, unicodedata
    from typing import Optional

    try:
        from datasets import load_dataset
    except Exception:
        return 0.0
    try:
        from openai import OpenAI
    except Exception:
        return 0.0

    STRATEGY_TEMPLATE_SET = ["qa", "rubric", "critique", "cot"]
    STYLE_SET = ["concise", "step_by_step", "formal", "creative"]
    LANGUAGE_SET = ["en", "zh", "mixed"]

    TEMPLATE_TO_SYSTEM_EN = {
        "qa":       "Answer directly; keep reasoning minimal.",
        "rubric":   "Think in 1–3 short steps.",
        "critique": "Draft briefly (1–3 lines), self-check quickly.",
        "cot":      "Think in 1–3 short steps."
    }
    STYLE_TO_PREFIX_EN = {
        "concise":       "Be concise and direct. ",
        "step_by_step":  "Be concise. ",
        "formal":        "Be precise. ",
        "creative":      "Be engaging. "
    }
    TEMPLATE_TO_SYSTEM_ZH = {
        "qa":       "尽量直接回答，推理要简洁。",
        "rubric":   "简要思考 1–3 步。",
        "critique": "先简要起草 1–3 行，再快速自检。",
        "cot":      "简要思考 1–3 步。"
    }
    STYLE_TO_PREFIX_ZH = {
        "concise":       "请简洁。",
        "step_by_step":  "请简洁。",
        "formal":        "请精确。",
        "creative":      "请有创意。"
    }

    # 解析参数
    try:
        strategy_template = STRATEGY_TEMPLATE_SET[int(params.get('strategy_template', 0)) % len(STRATEGY_TEMPLATE_SET)]
    except Exception:
        strategy_template = STRATEGY_TEMPLATE_SET[0]
    try:
        prompt_style = STYLE_SET[int(params.get('prompt_style', 0)) % len(STYLE_SET)]
    except Exception:
        prompt_style = STYLE_SET[0]
    try:
        language = LANGUAGE_SET[int(params.get('language', 0)) % len(LANGUAGE_SET)]
    except Exception:
        language = "en"
    try:
        top_p = float(params.get('top_p', 0.85))
    except Exception:
        top_p = 0.85
    try:
        temperature = float(params.get('temperature', 0.4))
    except Exception:
        temperature = 0.4

    # 数据集
    try:
        n_eval = int(os.getenv('LLM_EVAL_N', '2') or 2)
    except Exception:
        n_eval = 2
    try:
        ds = load_dataset("gsm8k", "main")
    except Exception:
        try:
            ds = load_dataset("openai/gsm8k", "main")
        except Exception:
            return 0.0
    test_set = ds["test"].select(range(min(n_eval, len(ds["test"])))); items = list(test_set)
    if not items:
        return 0.0

    # 数值解析（含中文/全角）
    _NUM_TOKEN = r'[-+]?(?:(?:\d[\d,]*)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    _MIXED     = r'[-+]?\d+\s+\d+\s*/\s*\d+'
    _ANYNUM    = r'(?:' + _MIXED + r'|\d+\s*/\s*\d+|' + _NUM_TOKEN + r')'

    def _clean(s: str) -> str:
        s = (s or "").strip()
        try:
            s = unicodedata.normalize("NFKC", s)
        except Exception:
            pass
        s = s.replace("−", "-").replace("－", "-").replace("\u00a0", " ")
        s = s.replace("，", ",").replace("．", ".")
        s = re.sub(r',', '', s)
        s = re.sub(r'^\$', '', s)
        s = s.replace("负", "-")
        return s

    from decimal import Decimal, InvalidOperation, getcontext
    from fractions import Fraction
    getcontext().prec = 50

    def parse_number(text: Optional[str]):
        if not text:
            return None
        raw = text
        s = _clean(text)
        m = re.fullmatch(_MIXED, s)
        if m:
            whole, a, b = re.match(r'([-+]?\d+)\s+(\d+)\s*/\s*(\d+)', s).groups()
            val = Fraction(int(whole), 1) + Fraction(int(a), int(b))
            return (Decimal(val.numerator) / Decimal(val.denominator)) / (Decimal(100)) if ('%' in (raw or '')) else Decimal(val.numerator) / Decimal(val.denominator)
        m = re.fullmatch(r'([-+]?\d+)\s*/\s*(\d+)', s)
        if m:
            a, b = m.groups(); val = Fraction(int(a), int(b))
            return (Decimal(val.numerator) / Decimal(val.denominator)) / (Decimal(100)) if ('%' in (raw or '')) else Decimal(val.numerator) / Decimal(val.denominator)
        try:
            x = Decimal(s)
            return (x / Decimal(100)) if ('%' in (raw or '')) else x
        except InvalidOperation:
            toks = re.findall(_ANYNUM, s)
            if not toks:
                return None
            t = _clean(toks[-1])
            try:
                x = Decimal(t)
                return (x / Decimal(100)) if ('%' in (raw or '')) else x
            except InvalidOperation:
                return None

    def numbers_close(a: Optional[Decimal], b: Optional[Decimal],
                      abs_tol: Decimal = Decimal('1e-8'),
                      rel_tol: Decimal = Decimal('1e-6')) -> bool:
        if a is None or b is None:
            return False
        diff = abs(a - b)
        thresh = max(abs_tol, rel_tol * max(Decimal(1), abs(b)))
        return diff <= thresh

    def build_messages(question: str, prompt_style: str, strategy_template: str, lang: str):
        if lang == 'zh':
            style_prefix = STYLE_TO_PREFIX_ZH.get(prompt_style, "")
            strategy_text = TEMPLATE_TO_SYSTEM_ZH.get(strategy_template, "")
            system_content = f"{style_prefix}{strategy_text} 中文作答。".strip()
            user_content = (
                f"问题:\n{question}\n\n"
                f"请先用中文在 1–3 行内简要推理。然后用格式：#### <number> 输出最终数值答案"
            )
        elif lang == 'mixed':
            style_prefix = STYLE_TO_PREFIX_ZH.get(prompt_style, "")
            strategy_text = TEMPLATE_TO_SYSTEM_ZH.get(strategy_template, "")
            system_content = f"{style_prefix}{strategy_text} 先中文思考，再英文给出最终答案。".strip()
            user_content = (
                f"问题:\n{question}\n\n"
                f"先用中文 1–3 行简要推理。然后用英文按如下格式输出最终数值答案: #### <number>"
            )
        else:
            style_prefix = STYLE_TO_PREFIX_EN.get(prompt_style, "")
            strategy_text = TEMPLATE_TO_SYSTEM_EN.get(strategy_template, "")
            system_content = f"{style_prefix}{strategy_text} en".strip()
            user_content = (
                f"Problem:\n{question}\n\n"
                f"Reason briefly in 1–3 short lines. Then output the final numeric answer in the format: #### <number>"
            )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def extract_answer(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        m = re.findall(r'#{2,}\s*(' + _ANYNUM + r')', text)
        if m:
            return m[-1].strip()
        m = re.findall(r'(?:final\s*answer|answer\s*is)\D*(' + _ANYNUM + r')', text, flags=re.IGNORECASE)
        if m:
            return m[-1].strip()
        m = re.findall(r'(?:最终\s*答案|答案\s*(?:是|为)|结果\s*(?:是|为))\D*(' + _ANYNUM + r')', text)
        if m:
            return m[-1].strip()
        m = re.findall(_ANYNUM, text)
        return m[-1].strip() if m else None

    # 客户端
    try:
        client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    except Exception:
        return 0.0

    # 并发求解
    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        max_workers = int(os.getenv('LLM_EVAL_WORKERS', '8') or 8)
    except Exception:
        max_workers = 8
    try:
        max_tokens = int(os.getenv('LLM_EVAL_MAX_TOKENS', '128') or 128)
    except Exception:
        max_tokens = 128
    try:
        retries = int(os.getenv('LLM_EVAL_RETRIES', '2') or 2)
    except Exception:
        retries = 2

    def solve_one(i, item):
        q = item["question"]
        msgs = build_messages(q, prompt_style, strategy_template, language)
        for attempt in range(retries + 1):
            try:
                out = client.chat.completions.create(
                    model="qwen2.5-7b-instruct",
                    messages=msgs,
                    max_tokens=max_tokens,
                    temperature=float(temperature),
                    top_p=float(top_p)
                )
                pred_text = getattr(out.choices[0].message, 'content', None) or ""
                pred = extract_answer(pred_text)
                return i, pred
            except Exception:
                if attempt < retries:
                    time.sleep(0.5 * (2 ** attempt))
        return i, None

    workers = max(1, min(max_workers, len(items)))
    preds = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(solve_one, i, it) for i, it in enumerate(items)]
        for f in as_completed(futures):
            try:
                idx, pr = f.result()
                preds[idx] = pr
            except Exception:
                pass

    # 评估
    correct = 0
    for i, it in enumerate(items):
        gt_text = it["answer"]
        gt = extract_answer(gt_text)
        from decimal import Decimal
        gt_num = parse_number(gt)
        pred_num = parse_number(preds[i])
        if numbers_close(pred_num, gt_num):
            correct += 1
    acc = correct / max(1, len(items))
    print("acc",acc)
    return float(acc)


# -*- coding: utf-8 -*-
"""
目标函数：风格仿照你的 if/elif 分支写法
- 接收 params（含 a ∈ {0,1,2} 与 o1..o6 ∈ {0..4}、eta、lam）
- 把类别映射到 JAHS 要求的 Activation/OpID
- 调用 JAHS surrogate，返回要最大化的指标（默认 valid-acc）
依赖：pip install jahs-bench
"""

from typing import Dict

# --- sklearn OneHotEncoder compatibility shim ---
# Some pre-trained pipelines (e.g., JAHS surrogate) expect private attributes
# that differ across scikit-learn versions (e.g., `_infrequent_enabled`).
# If the current sklearn is newer and the loaded estimator is older (or vice versa),
# accessing these missing attributes can raise errors during transform().
# We defensively patch OneHotEncoder.transform to set reasonable defaults
# when these private attributes are missing, avoiding full env rollback.
try:
    from sklearn.preprocessing import OneHotEncoder as _Sk_OH
    _orig_oh_transform = _Sk_OH.transform

    def _oh_transform_safe(self, X, *args, **kwargs):
        if not hasattr(self, "_infrequent_enabled"):
            # Older estimators don't have this; default to False
            self._infrequent_enabled = False
        # Some versions also expect `_legacy_mode` when unpickled across versions
        if not hasattr(self, "_legacy_mode"):
            self._legacy_mode = False
        return _orig_oh_transform(self, X, *args, **kwargs)

    _Sk_OH.transform = _oh_transform_safe
except Exception:
    pass

import jahs_bench

# 可按需修改
_JAHS_TASK = "cifar10"
_JAHS_EPOCHS = 200
_TARGET_METRIC = "valid-acc"

# --- 映射：整数类别 -> JAHS 名称/ID ---
def _map_act(a_code: int) -> str:
    if a_code == 0:
        return "ReLU"
    elif a_code == 1:
        return "Mish"
    elif a_code == 2:
        return "Hardswish"
    else:
        raise ValueError(f"激活编码不合法: {a_code}，应为 0/1/2")

def _map_op(op_code: int) -> int:
    # 0..4: zero, skip-connect, 1x1-conv, 3x3-conv, 3x3-avg-pool
    if op_code not in (0, 1, 2, 3, 4):
        raise ValueError(f"算子编码不合法: {op_code}，应为 0..4")
    return int(op_code)

def _make_cfg(p: Dict) -> Dict:
    # 分支写法（仿照你的函数风格）：对每个类别做显式判断
    # 激活
    if p['a'] == 0:
        act = "ReLU"
    elif p['a'] == 1:
        act = "Mish"
    elif p['a'] == 2:
        act = "Hardswish"
    else:
        raise ValueError(f"未知 a: {p['a']}")

    # 6 条边
    def op_field(name: str):
        v = int(p[name])
        if v == 0:
            return 0   # zero
        elif v == 1:
            return 1   # skip-connect
        elif v == 2:
            return 2   # 1x1-conv
        elif v == 3:
            return 3   # 3x3-conv
        elif v == 4:
            return 4   # 3x3-avg-pool
        else:
            raise ValueError(f"未知 {name}: {v}（应为 0..4）")

    cfg = {
        "Optimizer": "SGD",
        "LearningRate": float(p['eta']),
        "WeightDecay": float(p['lam']),
        "Activation": act,

        # 固定默认
        "TrivialAugment": False,
        "N": 5,
        "W": 16,
        "Resolution": 1.0,

        "Op1": op_field('o1'),
        "Op2": op_field('o2'),
        "Op3": op_field('o3'),
        "Op4": op_field('o4'),
        "Op5": op_field('o5'),
        "Op6": op_field('o6'),
    }
    return cfg

# 可选：缓存代理，避免反复初始化/下载
_JAHS_BENCH = None
def _bench():
    global _JAHS_BENCH
    if _JAHS_BENCH is None:
        _JAHS_BENCH = jahs_bench.Benchmark(task=_JAHS_TASK, kind="surrogate", download=True)
    return _JAHS_BENCH

def jash(params: Dict) -> float:
    """
    返回要最大化的值（默认 valid-acc）。
    若你要最小化某损失，可改成：return -float(last['valid-loss']) 等。
    """
    cfg = _make_cfg(params)
    bench = _bench()
    results = bench(cfg, nepochs=_JAHS_EPOCHS, full_trajectory=False)

    last = results[_JAHS_EPOCHS]
    metric = _TARGET_METRIC
    if metric not in last:
        # 兜底尝试常见字段名
        for cand in ("valid-acc", "val-acc", "train-acc", "test-acc"):
            if cand in last:
                metric = cand
                break
        else:
            raise KeyError(f"未找到指标 '{_TARGET_METRIC}'；可用：{sorted(last.keys())}")

    y = float(last[metric])  # 越大越好
    return y
