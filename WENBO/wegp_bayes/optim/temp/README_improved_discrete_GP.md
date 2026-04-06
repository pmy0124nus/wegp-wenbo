# 改进版离散GP优化策略

## 概述

这个改进版本解决了原始离散GP优化中L-BFGS无法找到最优解的问题，通过结合局部搜索和全局搜索来找到更好的起始点。

## 主要改进

### 1. 局部搜索改进起始点
- 对每个随机起始点进行局部搜索
- 使用高斯采样在起始点周围探索
- 设置距离阈值和最大迭代次数
- 提前停止机制避免过度计算

### 2. 全局搜索策略
- **拉丁超立方采样** (40%): 确保空间覆盖
- **随机采样** (30%): 随机探索
- **边界采样** (20%): 探索边界区域
- **中心采样** (10%): 探索中心区域

### 3. 混合优化策略
- 结合随机采样、局部搜索和全局搜索
- 选择最好的几个点作为L-BFGS的起始点
- 可以灵活控制是否启用局部搜索和全局搜索

## 文件结构

```
wegp_bayes/optim/
├── optimization_acq_discreteGP_0724.py          # 原始版本
├── optimization_acq_discreteGP_improved.py      # 改进版本
├── example_usage_improved.py                    # 使用示例
└── README_improved_discrete_GP.md              # 本文件
```

## 使用方法

### 基本使用

```python
from wegp_bayes.optim.optimization_acq_discreteGP_improved import MixedBayesOptGPDiscreteImproved

# 创建优化器
optimizer = MixedBayesOptGPDiscreteImproved(
    acq_obj=your_acq_obj,
    rng=np.random.RandomState(42),
    config_fun=your_config_fun,
    combined_cat_index=your_cat_index,
    model=your_model,
    kernel=your_kernel
)

# 使用混合策略（推荐）
result = optimizer.run(
    init_size=5,      # 初始采样5个离散点
    N_cand=10,        # 每轮选择10个候选
    T=20,             # 总共20次迭代
    strategy="hybrid"  # 使用混合策略
)
```

### 可用的优化策略

1. **"original"**: 原始策略，仅使用随机采样
2. **"hybrid"**: 混合策略（推荐），结合局部搜索和全局搜索
3. **"local_only"**: 仅使用局部搜索
4. **"global_only"**: 仅使用全局搜索

### 高级参数配置

```python
# 直接调用混合优化方法，可以更精细地控制参数
result = optimizer.optimize_x_given_h_hybrid(
    h_idx=0,                    # 离散选项索引
    n_starts=100,               # 初始采样数
    top_k=10,                   # L-BFGS起始点数
    use_local_search=True,       # 是否启用局部搜索
    use_global_search=True       # 是否启用全局搜索
)
```

## 参数说明

### 局部搜索参数
- `max_iter`: 最大迭代次数（默认20）
- `sigma`: 高斯采样标准差（默认0.1）
- `delta`: 距离阈值（默认0.3）

### 全局搜索参数
- `n_samples`: 采样数量（默认20）
- 采样比例：拉丁超立方40%，随机30%，边界20%，中心10%

### 优化策略参数
- `n_starts`: 初始随机采样数（默认100）
- `top_k`: 选择多少个点进行L-BFGS优化（默认10）

## 性能比较

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| original | 计算量小 | 可能陷入局部最优 | 计算资源有限 |
| hybrid | 效果好，平衡探索与开发 | 计算量较大 | 推荐使用 |
| local_only | 局部搜索能力强 | 可能错过全局最优 | 已知大致最优区域 |
| global_only | 全局探索能力强 | 计算量最大 | 需要全局探索 |

## 使用建议

### 1. 推荐配置
```python
# 对于大多数问题，推荐使用混合策略
result = optimizer.run(
    init_size=5,
    N_cand=10,
    T=20,
    strategy="hybrid"
)
```

### 2. 计算资源有限时
```python
# 减少采样数和迭代次数
result = optimizer.run(
    init_size=3,
    N_cand=5,
    T=10,
    strategy="original"  # 或 "local_only"
)
```

### 3. 需要高质量结果时
```python
# 增加采样数和迭代次数
result = optimizer.run(
    init_size=10,
    N_cand=20,
    T=50,
    strategy="hybrid"
)
```

## 测试和验证

运行测试示例：
```bash
cd WEBO
python wegp_bayes/optim/optimization_acq_discreteGP_improved.py
```

运行使用示例：
```bash
cd WEBO
python wegp_bayes/optim/example_usage_improved.py
```

## 与原始版本的兼容性

改进版本保持了与原始版本的API兼容性，主要区别是：
1. 类名改为 `MixedBayesOptGPDiscreteImproved`
2. 新增 `strategy` 参数来选择优化策略
3. 新增 `optimize_x_given_h_hybrid` 方法

## 故障排除

### 常见问题

1. **评估次数过多**
   - 减少 `n_starts` 和 `top_k`
   - 使用 `"original"` 或 `"local_only"` 策略

2. **结果不理想**
   - 增加 `n_starts` 和 `top_k`
   - 使用 `"hybrid"` 策略
   - 调整局部搜索的 `sigma` 和 `delta`

3. **计算时间过长**
   - 减少 `max_iter` 和 `n_samples`
   - 使用 `"original"` 策略

## 贡献

如果您发现bug或有改进建议，请：
1. 检查现有代码
2. 运行测试确保没有破坏现有功能
3. 提交改进建议

## 许可证

与原始代码保持相同的许可证。 