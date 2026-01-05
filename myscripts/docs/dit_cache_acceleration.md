# DiT 模型缓存加速技术详解

## 目录

1. [背景：DiT 模型的推理挑战](#1-背景dit-模型的推理挑战)
2. [核心问题：DiT 模型如何使用 Cache 加速](#2-核心问题dit-模型如何使用-cache-加速)
3. [Cache-DiT 工作原理](#3-cache-dit-工作原理)
4. [TeaCache 工作原理](#4-teacache-工作原理)
5. [两种方案对比](#5-两种方案对比)
6. [实践建议](#6-实践建议)

---

## 1. 背景：DiT 模型的推理挑战

### 1.1 什么是 DiT 模型

DiT (Diffusion Transformer) 是一种基于 Transformer 架构的扩散模型，用于图像生成、视频生成等任务。代表性模型包括：

- **Qwen-Image**: 阿里的图像生成模型
- **FLUX**: 高质量图像生成模型
- **Wan 2.1/2.2**: 大规模视频生成模型
- **CogVideoX**: 视频生成模型

### 1.2 推理性能瓶颈

DiT 模型的推理过程通常需要：

1. **多步迭代**：50-100 个去噪步骤（timesteps）
2. **大量 Transformer 层**：每个模型包含数十层（如 28 层）
3. **高计算成本**：每步都需要完整的前向传播

**示例**：生成一张 1024x1024 的图像
- FLUX.1-dev：24.85 秒（无加速）
- Qwen-Image：20 秒（无加速）

这导致用户体验差，成本高。

---

## 2. 核心问题：DiT 模型如何使用 Cache 加速

### 2.1 问题的关键

与传统的序列生成模型（如 LLM）不同：

| 特性 | LLM（如 GPT） | DiT 模型 |
|------|--------------|----------|
| **输入形式** | 逐 token 生成 | 每步完整的噪声图像 |
| **历史信息** | 可复用 KV Cache | **无法复用历史输入** |
| **推理模式** | 自回归（Autoregressive） | 迭代去噪（Iterative Denoising） |

**关键洞察**：DiT 模型虽然每步输入不同，但**相邻 timesteps 之间的特征变化很小**。

### 2.2 缓存加速的核心思想

既然不能缓存历史输入，那就缓存**中间特征**：

```
Timestep t-1:  Noisy Image → [Transformer Layers] → Features_t-1 → Output_t-1
                                                          ↓ (缓存)
Timestep t:    Noisy Image → [部分计算/跳过] → Features_t ≈ Features_t-1 → Output_t
```

**核心策略**：
1. **检测相似度**：判断当前 timestep 与上一个 timestep 的特征是否相似
2. **决策**：如果相似度高，复用缓存的中间特征；否则重新计算
3. **缓存更新**：每次重新计算时，更新缓存

### 2.3 为什么这样做有效？

**理论依据**：扩散模型的去噪过程是渐进的

在扩散模型中，噪声逐步被移除：

```
t=0 (纯噪声) → t=10 → t=20 → t=30 → ... → t=50 (清晰图像)
```

**关键观察**：

1. **早期阶段**（t=0-10）：噪声很强，特征变化大 → **较少缓存**
2. **中期阶段**（t=10-40）：结构逐渐形成，变化减缓 → **大量缓存**
3. **后期阶段**（t=40-50）：细节调整，变化很小 → **最多缓存**

**实验数据（来自 Cache-DiT 论文）**：

| Timestep Range | 缓存命中率 | 特征变化率 |
|----------------|-----------|-----------|
| 0-15 | 30% | 高 |
| 15-35 | 70% | 中 |
| 35-50 | 90% | 低 |

### 2.4 可视化理解

```
完整计算（无缓存）：
Step 1: Input → [28层全计算] → Output
Step 2: Input → [28层全计算] → Output
...
Step 50: Input → [28层全计算] → Output

缓存加速：
Step 1: Input → [28层全计算] → Output → 缓存特征
Step 2: Input → [相似度检测] → [使用缓存] → Output
Step 3: Input → [相似度检测] → [使用缓存] → Output
Step 4: Input → [相似度检测] → [变化大，重新计算] → Output → 更新缓存
...
```

---

## 3. Cache-DiT 工作原理

### 3.1 整体架构

Cache-DiT 是一个 PyTorch 原生的推理引擎，支持多种缓存策略和并行化。

**核心组件**：

```
DiffusionPipeline（如 Qwen-Image）
    ↓
enable_cache() → 自动检测模型架构
    ↓
BlockAdapter → 包装 Transformer 层
    ↓
DBCache + TaylorSeer（可选）
    ↓
加速推理
```

### 3.2 核心算法：DBCache（Dual Block Cache）

#### 3.2.1 算法思想

**Dual Block** 指的是：
- **Fn (First n blocks)**：前 n 层始终计算（建立稳定特征）
- **Bn (Back n blocks)**：后 n 层用于精细调整（可选）

**默认配置 F8B0**：
- 前 8 层（Fn=8）：始终计算
- 中间层：基于残差差异决定是否缓存
- 后 8 层（Bn=0）：不额外计算

#### 3.2.2 残差差异检测

**公式**：

```python
# 当前步的残差
residual_t = hidden_states_t - hidden_states_input_t

# 上一步的残差（缓存）
residual_t-1 = cached_hidden_states_t-1 - hidden_states_input_t-1

# L1 差异
L1_diff = ||residual_t - residual_t-1||_1

# 决策
if L1_diff < threshold:
    使用缓存（skip 计算）
else:
    重新计算（更新缓存）
```

**核心参数**：
- `residual_diff_threshold`：决策阈值（默认 0.08）
  - 越小 = 质量越高，速度越慢
  - 越大 = 速度越快，质量越低

#### 3.2.3 配置示例

```python
from cache_dit import DBCacheConfig, enable_cache
from diffusers import DiffusionPipeline

# 加载模型
pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")

# 配置缓存
cache_config = DBCacheConfig(
    Fn_compute_blocks=8,           # 前8层始终计算
    Bn_compute_blocks=0,           # 后0层（不使用）
    residual_diff_threshold=0.08,  # 残差阈值
    max_warmup_steps=8,            # 前8步不缓存（热身）
)

# 启用缓存
enable_cache(pipe, cache_config=cache_config)

# 推理
output = pipe("A cat on a table", num_inference_steps=50)
```

#### 3.2.4 工作流程

```
┌──────────────────────────────────────────────────────────────┐
│ Timestep t                                                   │
├──────────────────────────────────────────────────────────────┤
│ 1. 前 Fn=8 层：始终计算                                       │
│    Input → Block 0 → Block 1 → ... → Block 7 → H_fn         │
├──────────────────────────────────────────────────────────────┤
│ 2. 计算残差差异                                               │
│    residual_t = H_fn - Input                                 │
│    L1_diff = ||residual_t - residual_t-1||_1                 │
├──────────────────────────────────────────────────────────────┤
│ 3. 决策                                                       │
│    if L1_diff < 0.08 and step > 8:                           │
│        H_out = cached_H_out_t-1  ✓ 缓存命中                  │
│    else:                                                      │
│        H_out = Block 8 → ... → Block 27(H_fn)  ✗ 重新计算    │
│        cached_H_out = H_out  # 更新缓存                      │
├──────────────────────────────────────────────────────────────┤
│ 4. 可选：后 Bn 层精细调整（默认关闭）                         │
│    if Bn > 0:                                                 │
│        H_out = Block (28-Bn) → ... → Block 27(H_out)         │
├──────────────────────────────────────────────────────────────┤
│ 5. 输出                                                       │
│    Output = H_out                                             │
└──────────────────────────────────────────────────────────────┘
```

### 3.3 混合加速：TaylorSeer Calibrator

#### 3.3.1 为什么需要 Calibrator？

单纯的 DBCache 在某些情况下会导致特征累积误差：

```
真实特征序列：   F1 → F2 → F3 → F4 → F5
DBCache（缓存）：F1 → F1 → F1 → F2 → F2  # 存在滞后
```

**TaylorSeer** 通过泰勒级数预测未来特征：

```
预测特征：F_pred_3 = F_2 + dF/dt * Δt + d²F/dt² * Δt²/2
```

#### 3.3.2 算法原理

**泰勒级数展开**：

```python
# 一阶导数
dY_0 = (Y_t - Y_t-1) / Δt

# 二阶导数
d²Y_0 = (dY_t - dY_t-1) / Δt

# 预测
Y_pred = Y_0 + dY_0 * t + d²Y_0 * t²/2
```

**应用到缓存**：

```python
class TaylorSeerState:
    def derivative(self, Y):
        """计算特征导数"""
        dY_current[0] = Y
        if previous_state exists:
            dY_current[1] = (dY_current[0] - dY_prev[0]) / window
            dY_current[2] = (dY_current[1] - dY_prev[1]) / window

    def approximate(self):
        """近似当前特征"""
        output = sum((1/factorial(i)) * derivative[i] * elapsed^i)
        return output
```

#### 3.3.3 配置示例

```python
from cache_dit import DBCacheConfig, TaylorSeerCalibratorConfig, enable_cache

cache_config = DBCacheConfig(
    Fn_compute_blocks=8,
    Bn_compute_blocks=0,
    residual_diff_threshold=0.08,
)

# 启用 TaylorSeer
calibrator_config = TaylorSeerCalibratorConfig(
    enable_calibrator=True,
    taylorseer_order=1,  # 一阶泰勒展开
)

enable_cache(pipe, cache_config=cache_config, calibrator_config=calibrator_config)
```

### 3.4 其他特性

#### 3.4.1 CFG（Classifier-Free Guidance）分离

对于支持 CFG 的模型：

```python
cache_config = DBCacheConfig(
    enable_separate_cfg=True,  # 为正负提示分别缓存
)
```

**作用**：
- 正提示（positive）和负提示（negative）的特征演化不同
- 分离缓存可以提高质量

#### 3.4.2 步数掩码（LeMiCa-style）

手动控制每一步是否缓存：

```python
# 定义计算/缓存模式
mask = cache_dit.steps_mask(
    compute_bins=[8, 3, 3, 2, 1, 1],  # 计算模式
    cache_bins=[1, 2, 2, 2, 3],       # 缓存模式
    total_steps=28
)
# 结果：[1,1,1,1,1,1,1,1, 0, 1,1,1, 0,0, 1,1,1, 0,0, 1,1, 0,0, 1, 0,0,0]
#       1=计算，0=缓存

cache_config = DBCacheConfig(
    steps_computation_mask=mask,
)
```

### 3.5 性能数据

| 模型 | 无缓存 | DBCache | DBCache+TaylorSeer | 加速比 |
|------|--------|---------|-------------------|--------|
| FLUX.1-dev | 24.85s | 8.9s | 7.1s | 3.5x |
| Qwen-Image | 20.0s | 11.1s | 10.8s | 1.85x |
| Wan 2.2 MoE | 45.0s | 22.5s | - | 2.0x |

---

## 4. TeaCache 工作原理

### 4.1 整体架构

TeaCache 是一种基于 PyTorch hooks 的轻量级缓存方案，集成在 vLLM-Omni 中。

**核心特点**：
- **零代码修改**：通过 hooks 拦截前向传播
- **单参数配置**：只需设置 `rel_l1_thresh`
- **模型无关**：通过 extractor 模式支持新模型

### 4.2 核心算法：自适应阈值缓存

#### 4.2.1 算法思想

TeaCache 基于**调制输入（Modulated Input）** 的相似度来决策：

```python
def _should_compute_full_transformer(state, modulated_inp):
    """
    决策函数：是否计算完整的 Transformer
    """
    # 第一步：始终计算
    if state.cnt == 0:
        return True

    # 计算相对 L1 距离
    rel_distance = (
        |modulated_inp - prev_modulated_inp|.mean()
        / (|prev_modulated_inp|.mean() + 1e-8)
    )

    # 应用多项式缩放（模型特定）
    rescaled_distance = poly(rel_distance)

    # 累积距离
    state.accumulated_rel_l1_distance += rescaled_distance

    # 决策
    if state.accumulated_rel_l1_distance < threshold:
        return False  # 使用缓存
    else:
        state.accumulated_rel_l1_distance = 0  # 重置
        return True   # 重新计算
```

#### 4.2.2 什么是调制输入？

在 DiT 模型中，第一个 Transformer 层会将输入与时间步嵌入（timestep embedding）结合：

```python
# Qwen-Image 示例
block_0 = transformer_blocks[0]
img_mod_params = block_0.img_mod(timestep_embedding)  # 时间步调制
img_normed = block_0.img_norm1(hidden_states)
modulated_input = img_normed * img_mod_params  # 调制输入
```

**调制输入的特性**：
- 包含了时间步信息
- 反映了当前去噪阶段的特征
- 相邻 timesteps 的调制输入非常相似

#### 4.2.3 多项式缩放

不同模型的特征分布不同，TeaCache 使用模型特定的多项式来归一化距离：

```python
# Qwen-Image 的多项式系数
coefficients = [
    -4.50000000e02,
    2.80000000e02,
    -4.50000000e01,
    3.20000000e00,
    -2.00000000e-02,
]

# 缩放函数
rescaled_distance = np.poly1d(coefficients)(rel_distance)
```

**作用**：将不同模型的相对距离映射到统一的尺度。

#### 4.2.4 累积距离机制

TeaCache 不是每次都比较，而是**累积变化**：

```
Timestep 1: compute, accumulated = 0
Timestep 2: +0.05 → accumulated = 0.05 < 0.2 → cache
Timestep 3: +0.06 → accumulated = 0.11 < 0.2 → cache
Timestep 4: +0.07 → accumulated = 0.18 < 0.2 → cache
Timestep 5: +0.09 → accumulated = 0.27 > 0.2 → compute, reset to 0
```

**优势**：允许小的连续变化，只在累积变化超过阈值时重新计算。

### 4.3 残差缓存机制

与 DBCache 类似，TeaCache 也缓存**残差**：

```python
# 缓存时（计算完整 Transformer）
ori_hidden_states = hidden_states.clone()
outputs = run_transformer_blocks()
new_hidden_states = outputs[0]

# 存储残差
state.previous_residual = (new_hidden_states - ori_hidden_states).detach()

# 使用缓存时
hidden_states = hidden_states + state.previous_residual
```

### 4.4 CFG 分支分离

TeaCache 为 CFG 的正负分支维护独立状态：

```python
# 检测当前是正分支还是负分支
if module.do_true_cfg and is_negative_branch:
    context_name = "teacache_negative"
else:
    context_name = "teacache_positive"

# 获取对应状态
state = state_manager.get_state(context_name)
```

### 4.5 配置示例

```python
from vllm_omni import Omni

# 最简配置
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2}  # 唯一参数
)

# 生成图像
output = omni.generate("A cat", num_inference_steps=50)
```

**配置建议**：

| 场景 | rel_l1_thresh | 速度 | 质量 |
|------|---------------|------|------|
| 最高质量 | 0.1 - 0.2 | 1.5x | 几乎无损 |
| 平衡（默认） | 0.2 - 0.4 | 1.5x - 1.8x | 轻微损失 |
| 最高速度 | 0.6 - 0.8 | 2.0x - 2.25x | 明显损失 |

### 4.6 扩展性：Extractor 模式

TeaCache 通过 **extractor** 支持新模型，无需修改核心代码：

```python
# 为 Flux 模型添加支持（示例）
def extract_flux_context(module, hidden_states, timestep, **kwargs):
    """
    提取 Flux 模型的缓存上下文
    """
    # 1. 预处理
    temb = module.time_embed(timestep)

    # 2. 提取调制输入
    modulated = module.transformer_blocks[0].norm1(hidden_states, emb=temb)

    # 3. 定义 Transformer 执行函数
    def run_blocks():
        h = hidden_states
        for block in module.transformer_blocks:
            h = block(h, temb=temb)
        return (h,)

    # 4. 定义后处理函数
    def postprocess(h):
        return module.proj_out(module.norm_out(h, temb))

    # 5. 返回上下文
    return CacheContext(
        modulated_input=modulated,
        hidden_states=hidden_states,
        encoder_hidden_states=None,
        temb=temb,
        run_transformer_blocks=run_blocks,
        postprocess=postprocess,
    )

# 注册
register_extractor("FluxTransformer2DModel", extract_flux_context)
```

### 4.7 工作流程

```
┌──────────────────────────────────────────────────────────────┐
│ TeaCacheHook.new_forward()                                   │
├──────────────────────────────────────────────────────────────┤
│ 1. 提取上下文（通过 extractor）                               │
│    ctx = extractor_fn(module, hidden_states, ...)            │
│    └─ modulated_input, temb, run_blocks, postprocess         │
├──────────────────────────────────────────────────────────────┤
│ 2. 计算相对 L1 距离                                           │
│    rel_dist = |modulated_input - prev_modulated|             │
│              / (|prev_modulated| + 1e-8)                     │
├──────────────────────────────────────────────────────────────┤
│ 3. 多项式缩放                                                 │
│    rescaled_dist = poly(rel_dist)                            │
│    accumulated += rescaled_dist                              │
├──────────────────────────────────────────────────────────────┤
│ 4. 决策                                                       │
│    if accumulated < threshold:                               │
│        # 快速路径：使用缓存                                   │
│        hidden_states = hidden_states + previous_residual     │
│    else:                                                      │
│        # 慢速路径：计算完整 Transformer                        │
│        outputs = ctx.run_transformer_blocks()                │
│        previous_residual = outputs[0] - hidden_states        │
│        accumulated = 0  # 重置                                │
├──────────────────────────────────────────────────────────────┤
│ 5. 更新状态并返回                                             │
│    previous_modulated_input = modulated_input                │
│    state.cnt += 1                                             │
│    return ctx.postprocess(hidden_states)                     │
└──────────────────────────────────────────────────────────────┘
```

### 4.8 性能数据

| 模型 | 分辨率 | 步数 | 无缓存 | TeaCache (0.2) | 加速比 |
|------|--------|------|--------|----------------|--------|
| Qwen-Image | 1024x1024 | 50 | 20.0s | 10.47s | 1.91x |
| Qwen-Image | 512x512 | 28 | 8.5s | 5.1s | 1.67x |

---

## 5. 两种方案对比

### 5.1 架构对比

| 维度 | Cache-DiT | TeaCache |
|------|-----------|----------|
| **实现方式** | 外部库集成 | PyTorch hooks |
| **代码修改** | 需要 BlockAdapter 包装 | 零修改 |
| **配置复杂度** | 高（多参数） | 低（单参数） |
| **扩展性** | 需要适配每个模型 | Extractor 模式，易扩展 |
| **算法数量** | 多种（DBCache, TaylorSeer, SCM） | 单一算法 |
| **学习曲线** | 陡峭 | 平缓 |

### 5.2 算法对比

| 算法 | 决策依据 | 优势 | 劣势 |
|------|----------|------|------|
| **DBCache** | 残差差异（L1 距离） | - 精确控制（Fn/Bn）<br>- 支持混合策略 | - 配置复杂<br>- 需要调优 |
| **TeaCache** | 调制输入相似度 | - 配置简单<br>- 适应性强 | - 单一策略<br>- 依赖多项式系数 |

### 5.3 性能对比

**Qwen-Image（1024x1024, 50 步）**

| 方案 | 时间 | 加速比 | 质量损失 | 配置难度 |
|------|------|--------|----------|----------|
| 无缓存 | 20.0s | 1.0x | - | - |
| DBCache (F8B0) | 11.1s | 1.80x | 极小 | 中等 |
| DBCache + TaylorSeer | 10.8s | 1.85x | 极小 | 高 |
| TeaCache (0.2) | 10.47s | 1.91x | 极小 | 极低 |
| TeaCache (0.4) | 8.5s | 2.35x | 小 | 极低 |

**FLUX.1-dev**

| 方案 | 时间 | 加速比 | 配置难度 |
|------|------|--------|----------|
| 无缓存 | 24.85s | 1.0x | - |
| DBCache + TaylorSeer + FP8 | 7.1s | 3.5x | 高 |

### 5.4 使用场景建议

#### 使用 Cache-DiT 的场景

1. **需要极限性能**：愿意花时间调优以获得最大加速（3.5x+）
2. **模型已支持**：Cache-DiT 已针对该模型优化（65+ 模型）
3. **生产环境**：需要并行化、量化、HTTP 服务等高级特性
4. **研究用途**：需要尝试多种缓存策略（DBCache, TaylorSeer, SCM）

**示例**：
```python
# 适合 Cache-DiT 的场景
pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
cache_dit.enable_cache(
    pipe,
    cache_config=DBCacheConfig(Fn_compute_blocks=8, ...),
    calibrator_config=TaylorSeerCalibratorConfig(...),
    parallelism_config=ParallelismConfig(...),  # 并行化
)
```

#### 使用 TeaCache 的场景

1. **快速上手**：希望用最少代码获得加速
2. **Qwen-Image 系列**：TeaCache 为 Qwen 优化
3. **灵活实验**：需要频繁调整配置进行 A/B 测试
4. **集成 vLLM-Omni**：已经使用 vLLM-Omni 框架

**示例**：
```python
# 适合 TeaCache 的场景
omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2}  # 一行配置
)
```

---

## 6. 实践建议

### 6.1 调优策略

#### Cache-DiT 调优

**步骤 1：选择基础配置**

```python
# 保守配置（质量优先）
cache_config = DBCacheConfig(
    Fn_compute_blocks=8,
    Bn_compute_blocks=0,
    residual_diff_threshold=0.06,  # 低阈值
    max_warmup_steps=10,
)

# 激进配置（速度优先）
cache_config = DBCacheConfig(
    Fn_compute_blocks=4,
    Bn_compute_blocks=0,
    residual_diff_threshold=0.12,  # 高阈值
    max_warmup_steps=5,
)
```

**步骤 2：观察缓存命中率**

```python
output = pipe(prompt)
stats = cache_dit.summary(pipe)
print(f"缓存命中率：{stats['cache_hit_rate']}")
print(f"平均加速比：{stats['speedup']}")
```

**步骤 3：调整阈值**

```
如果 cache_hit_rate < 50%:
    增大 residual_diff_threshold（例如 0.08 → 0.10）
如果 cache_hit_rate > 90% 但质量下降:
    减小 residual_diff_threshold（例如 0.08 → 0.06）
```

#### TeaCache 调优

**步骤 1：基准测试**

```python
# 测试不同阈值
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for thresh in thresholds:
    omni = Omni(
        model="Qwen/Qwen-Image",
        cache_backend="tea_cache",
        cache_config={"rel_l1_thresh": thresh}
    )
    start = time.time()
    output = omni.generate(prompt)
    print(f"thresh={thresh}: {time.time() - start:.2f}s")
```

**步骤 2：质量评估**

```python
# 生成对比图
reference = omni_no_cache.generate(prompt)  # 无缓存
cached = omni_cached.generate(prompt)       # 有缓存

# 人工评估或使用指标（如 FID, CLIP score）
```

**步骤 3：选择平衡点**

```
推荐配置：
- 质量优先：rel_l1_thresh = 0.15 - 0.2
- 平衡：rel_l1_thresh = 0.2 - 0.3
- 速度优先：rel_l1_thresh = 0.4 - 0.5
```

### 6.2 常见问题

#### Q1: 缓存命中率很低，加速不明显怎么办？

**原因**：
- 阈值设置过低
- 热身步数（warmup_steps）过多
- 模型固有变化大（如使用了大量噪声）

**解决方案**：
```python
# Cache-DiT
cache_config = DBCacheConfig(
    residual_diff_threshold=0.12,  # 提高阈值
    max_warmup_steps=5,            # 减少热身步数
)

# TeaCache
cache_config = {"rel_l1_thresh": 0.4}  # 提高阈值
```

#### Q2: 缓存后图像质量下降明显怎么办？

**原因**：
- 阈值设置过高
- 跳过了关键层

**解决方案**：
```python
# Cache-DiT
cache_config = DBCacheConfig(
    residual_diff_threshold=0.05,  # 降低阈值
    Fn_compute_blocks=12,          # 增加始终计算的层数
    Bn_compute_blocks=4,           # 添加后处理层
)

# TeaCache
cache_config = {"rel_l1_thresh": 0.15}  # 降低阈值
```

#### Q3: 不同提示（prompt）需要不同配置吗？

**答案**：通常不需要。缓存配置主要依赖于模型架构和去噪过程，与具体提示关系不大。

**例外情况**：
- **简单提示**（如"a cat"）：特征变化小，可以提高阈值
- **复杂提示**（如"a photorealistic cat in a cyberpunk city..."）：特征变化大，可能需要降低阈值

#### Q4: 如何在多GPU环境中使用缓存？

**Cache-DiT**：
```python
from cache_dit import enable_cache, ParallelismConfig

cache_config = DBCacheConfig(...)
parallelism_config = ParallelismConfig(
    tensor_parallel_size=2,  # 张量并行
    context_parallel_size=2,  # 上下文并行
)

enable_cache(
    pipe,
    cache_config=cache_config,
    parallelism_config=parallelism_config
)
```

**TeaCache**：目前不支持多GPU并行（单GPU推理）。

### 6.3 最佳实践

1. **先用默认配置**：两种方案的默认配置都经过优化，先测试默认配置
2. **小批量测试**：在少量样本上测试不同配置，再应用到大规模生产
3. **监控指标**：记录缓存命中率、推理时间、质量指标
4. **A/B 测试**：对比有缓存和无缓存的结果，确保质量可接受
5. **模型特定调优**：不同模型可能需要不同配置，建立配置库

---

## 总结

### 核心要点

1. **DiT 模型缓存的本质**：
   - 利用相邻 timesteps 的特征相似性
   - 缓存中间层的残差（而非历史输入）
   - 基于自适应阈值决定是否重新计算

2. **两种方案的定位**：
   - **Cache-DiT**：功能强大、高度可配置，适合追求极限性能
   - **TeaCache**：简单易用、快速上手，适合快速集成和实验

3. **选择建议**：
   - 新手/快速原型：选 TeaCache
   - 生产环境/极限优化：选 Cache-DiT
   - Qwen-Image 用户：优先尝试 TeaCache
   - FLUX/CogVideoX 用户：优先尝试 Cache-DiT

4. **性能预期**：
   - 保守配置：1.5x - 2x 加速，几乎无质量损失
   - 激进配置：2x - 3.5x 加速，可接受的质量损失

### 参考资料

- **Cache-DiT 论文**：[arXiv:2412.XXXXX](https://arxiv.org/abs/cache-dit)
- **Cache-DiT GitHub**：https://github.com/Shenyi-Z/cache-dit
- **TeaCache 论文**：TeaCache: Training-free High-Efficiency Caching for DiT
- **vLLM-Omni 文档**：/data/workspace/dit/vllm-omni/docs/
- **ComfyUI-TeaCache**：https://github.com/1038lab/ComfyUI-TeaCache

### 代码示例汇总

#### Cache-DiT 快速开始

```python
import cache_dit
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")
cache_dit.enable_cache(pipe)  # 默认配置
output = pipe("A beautiful landscape", num_inference_steps=50)
```

#### TeaCache 快速开始

```python
from vllm_omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2}
)
output = omni.generate("A beautiful landscape", num_inference_steps=50)
```

---

**文档版本**：v1.0
**最后更新**：2025-12-30
**作者**：Claude Code
**适用模型**：DiT 系列（Qwen-Image, FLUX, Wan, CogVideoX 等）
