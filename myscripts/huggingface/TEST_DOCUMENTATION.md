# CosyVoice3 DiT Cache 加速测试 - 技术文档

## 1. 测试过程

### 1.1 分离的 DiT 模型

#### 1.1.1 模块组成

从 CosyVoice3 中提取的 DiT (Diffusion Transformer) 模型包含以下核心模块：

**模型架构** (`cosyvoice3_dit_model.py`):

```
CosyVoice3DiT
├── Time Embedding (时间步嵌入)
│   ├── SinusoidalPosEmb (正弦位置编码, dim=1024)
│   └── TimeEmbedding (时间投影, 1024 → 1024)
│
├── Input Projection (输入投影)
│   ├── x_embedder: Linear(240 → 1024)
│   │   - 输入: x (mel) + mu (mean) + spk (speaker)
│   │   - 240 = 80 (mel) + 80 (mu) + 80 (spk)
│   └── RMSNorm(1024)
│
├── Transformer Blocks (22层 DiT blocks)
│   └── DiTBlock × 22
│       ├── Attention (多头自注意力)
│       │   ├── norm1: RMSNorm(1024)
│       │   ├── to_qkv: Linear(1024 → 3072)
│       │   │   - 16 heads × 64 dim_head = 1024
│       │   ├── MultiheadAttention
│       │   └── to_out: Linear(1024 → 1024)
│       │
│       ├── FeedForward (前馈网络)
│       │   ├── norm2: RMSNorm(1024)
│       │   ├── ff.0: Linear(1024 → 2048)  # ff_mult=2
│       │   ├── GELU activation
│       │   └── ff.2: Linear(2048 → 1024)
│       │
│       └── Timestep Conditioning (时间步条件)
│           ├── adaLN_modulation (自适应层归一化调制)
│           └── shift, scale, gate parameters
│
└── Output Projection (输出投影)
    ├── final_norm: RMSNorm(1024)
    └── final_layer: Linear(1024 → 80)
```

**参数统计**:
- 总参数量: **193.41M** (193,408,680)
- 主要参数分布:
  - Transformer blocks: ~190M (98%)
  - Time embedding: ~1M
  - Input/Output projection: ~2M

**模型配置** (CosyVoice3-0.5B):
```python
dim = 1024              # 隐藏层维度
depth = 22              # Transformer 层数
heads = 16              # 注意力头数
dim_head = 64           # 每个头的维度
ff_mult = 2             # FeedForward 扩展倍数
mel_dim = 80            # Mel 特征维度
mu_dim = 80             # Mean 特征维度
spk_dim = 80            # Speaker 特征维度
out_channels = 80       # 输出通道数
```

#### 1.1.2 Forward Pipeline

**完整的前向传播流程**:

```python
def forward(x, timesteps, mu, spk):
    """
    Args:
        x: [batch, seq_len, 80] - Mel spectrogram
        timesteps: [batch] - Diffusion timestep (0-1000)
        mu: [batch, seq_len, 80] - Mean features
        spk: [batch, seq_len, 80] - Speaker embeddings

    Returns:
        out: [batch, seq_len, 80] - Predicted noise/velocity
    """

    # 1. Time Embedding (时间步编码)
    t_emb = self.time_embed(timesteps)  # [batch, 1024]

    # 2. Input Projection (输入投影)
    # 拼接三个输入特征
    x_concat = torch.cat([x, mu, spk], dim=-1)  # [batch, seq_len, 240]
    hidden = self.x_embedder(x_concat)          # [batch, seq_len, 1024]
    hidden = self.norm_x(hidden)                # RMSNorm

    # 3. Transformer Blocks (逐层处理)
    for block in self.blocks:
        # 每个 block 的内部流程:
        # a) Self-Attention with AdaLN (自适应层归一化)
        shift, scale, gate = block.adaLN(t_emb)

        # 层归一化 + 时间步调制
        normed = block.norm1(hidden)
        normed = normed * (1 + scale) + shift

        # 多头注意力
        attn_out = block.attention(normed)

        # 残差连接 + gate 调制
        hidden = hidden + gate * attn_out

        # b) FeedForward with AdaLN
        shift_ff, scale_ff, gate_ff = block.adaLN_ff(t_emb)

        # 层归一化 + 时间步调制
        normed_ff = block.norm2(hidden)
        normed_ff = normed_ff * (1 + scale_ff) + shift_ff

        # 前馈网络
        ff_out = block.feedforward(normed_ff)

        # 残差连接 + gate 调制
        hidden = hidden + gate_ff * ff_out

    # 4. Output Projection (输出投影)
    hidden = self.final_norm(hidden)      # [batch, seq_len, 1024]
    out = self.final_layer(hidden)        # [batch, seq_len, 80]

    return out
```

**关键特性**:

1. **AdaLN (Adaptive Layer Normalization)**:
   - 使用时间步信息调制每一层的归一化
   - 提供 shift, scale, gate 三个参数
   - 使特征能够根据扩散时间步动态调整

2. **残差连接**:
   - 每个子模块（Attention/FeedForward）都有残差连接
   - 通过 gate 参数控制残差的权重

3. **ForwardPattern.Pattern_3**:
   - 输入: `(hidden_states,)` - 只有一个张量
   - 输出: `(hidden_states,)` - 返回一个张量
   - 适配 cache-dit 的 Pattern_3 模式

### 1.2 测试参数

#### 1.2.1 输入输出

**输入参数**:

| 参数 | 维度 | 类型 | 说明 | 生成方式 |
|------|------|------|------|----------|
| `x` | `[B, T, 80]` | `torch.Tensor` | Mel 频谱特征 | `torch.randn(B, T, 80)` |
| `mu` | `[B, T, 80]` | `torch.Tensor` | 均值特征（条件） | `torch.randn(B, T, 80)` |
| `spk` | `[B, T, 80]` | `torch.Tensor` | 说话人嵌入 | `torch.randn(B, T, 80)` |
| `timesteps` | `[B]` | `torch.Tensor` | 扩散时间步 | `torch.zeros(B)` 或 `t * 1000` |

其中:
- `B` = batch_size (默认 1)
- `T` = seq_len (默认 200，表示 200 帧 mel 特征)
- 时间步范围: 0-1000 (归一化到 [0, 1])

**输出**:

| 参数 | 维度 | 说明 |
|------|------|------|
| `out` | `[B, T, 80]` | 预测的噪声/速度场 |

**测试配置**:

```python
# 默认测试参数
batch_size = 1          # 批大小
seq_len = 200           # 序列长度（mel 帧数）
num_steps = 28          # 扩散采样步数
device = 'cpu' or 'cuda'

# 模拟扩散过程
for step in range(num_steps):
    t = step / (num_steps - 1)  # 归一化到 [0, 1]
    timesteps = torch.full((B,), t * 1000)
    output = model(x, timesteps, mu, spk)
```

#### 1.2.2 Warmup

**Warmup 的必要性**:

在深度学习推理基准测试中，warmup 至关重要，原因包括：

1. **GPU 预热**: GPU 初次运行时需要时间初始化内核、分配内存
2. **JIT 编译**: PyTorch 的 JIT (Just-In-Time) 编译器需要预热
3. **缓存预热**: CPU/GPU 缓存需要预热以达到稳定状态
4. **内存分配**: 避免首次内存分配的额外开销
5. **CUDA 流**: 确保 CUDA 流和异步操作稳定

**Warmup 实现**:

```python
def warmup_model(model, inputs, num_warmup: int = 3):
    """
    执行 warmup 迭代以稳定性能

    Args:
        model: DiT 模型
        inputs: 输入字典 {'x', 'timesteps', 'mu', 'spk'}
        num_warmup: warmup 迭代次数
    """
    device = next(model.parameters()).device

    for i in range(num_warmup):
        # 执行前向传播
        with torch.no_grad():
            _ = model(**inputs)

        # GPU 同步（确保操作完成）
        if device.type == 'cuda':
            torch.cuda.synchronize()
```

**Warmup 配置**:

| 配置 | Warmup 次数 | 适用场景 |
|------|-------------|----------|
| 快速测试 | 3 | 快速验证，结果可能有波动 |
| **标准测试** | **5** | **推荐，平衡速度和稳定性** |
| 精确测试 | 10 | 需要非常稳定的结果 |
| GPU 测试 | 10-20 | GPU 上需要更多 warmup |

#### 1.2.3 指标计算方法

**1. TTFT (Time To First Token)**

首次推理的时间，反映冷启动性能：

```python
# 在 warmup 后立即测量
if device == 'cuda':
    torch.cuda.synchronize()

start_time = time.perf_counter()
with torch.no_grad():
    _ = model(x, timesteps, mu, spk)

if device == 'cuda':
    torch.cuda.synchronize()

ttft = time.perf_counter() - start_time  # 秒
```

**2. Time per Step (每步推理时间)**

每个扩散步的推理时间：

```python
times = []
for step in range(num_steps):
    t = step / (num_steps - 1)
    timesteps = torch.full((B,), t * 1000)

    if device == 'cuda':
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        _ = model(x, timesteps, mu, spk)

    if device == 'cuda':
        torch.cuda.synchronize()

    step_time = time.perf_counter() - start_time
    times.append(step_time)

# 统计指标
avg_time = sum(times) / len(times)      # 平均时间
min_time = min(times)                   # 最小时间
max_time = max(times)                   # 最大时间
```

**3. Total Time (总推理时间)**

完成所有扩散步的总时间：

```python
total_time = sum(times)  # 秒
```

**4. Throughput (吞吐量)**

每秒可完成的推理步数：

```python
throughput = num_steps / total_time  # steps/s
```

**5. Speedup (加速比)**

相对于 baseline 的加速倍数：

```python
speedup = baseline_avg_time / current_avg_time  # 倍数
```

**6. Cache Hit Rate (缓存命中率)**

```python
cached_steps = len(cache_stats.cached_steps)
cache_hit_rate = cached_steps / num_steps * 100  # 百分比
```

**7. GPU Memory Usage (GPU 显存)**

```python
# 在推理前重置统计
torch.cuda.reset_peak_memory_stats()

# 执行推理
...

# 获取峰值显存
memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
```

**指标汇总表**:

| 指标 | 单位 | 公式 | 说明 |
|------|------|------|------|
| TTFT | ms | 首次推理时间 | 冷启动性能 |
| Avg Time/Step | ms | Σ(times) / n | 平均推理速度 |
| Min Time/Step | ms | min(times) | 最快推理时间 |
| Max Time/Step | ms | max(times) | 最慢推理时间 |
| Total Time | s | Σ(times) | 总耗时 |
| Throughput | steps/s | n / total_time | 吞吐量 |
| Speedup | x | baseline / current | 加速比 |
| Cache Hit Rate | % | cached / total × 100 | 缓存效率 |
| GPU Memory | MB | peak_memory / 1024² | 显存占用 |

**测量精度保证**:

1. **CUDA 同步**: 每次测量前后都执行 `torch.cuda.synchronize()`
2. **高精度计时**: 使用 `time.perf_counter()`（纳秒级精度）
3. **禁用梯度**: 使用 `torch.no_grad()` 避免梯度计算开销
4. **多次测量**: 取多步平均值，减少单次测量误差
5. **Warmup 隔离**: Warmup 结果不计入最终统计

## 2. Cache-DiT 配置

### 2.1 基础配置

**DBCacheConfig 参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `Fn_compute_blocks` | 8 | 前 N 个 block 用于计算差异 |
| `Bn_compute_blocks` | 0 | 后 N 个 block 用于融合 |
| `residual_diff_threshold` | 0.08 | 残差差异阈值，触发缓存 |
| `max_warmup_steps` | 8 | 前 N 步不使用缓存 |
| `max_cached_steps` | -1 | 最大连续缓存步数（-1=无限） |
| `num_inference_steps` | 28 | 总推理步数（必需） |

**测试策略**:

| 策略 | Fn | Bn | Threshold | Warmup | 特点 |
|------|----|----|-----------|--------|------|
| Baseline | 22 | 0 | 0.0 | 0 | 无缓存，作为基准 |
| Aggressive | 1 | 0 | 0.15 | 4 | 最快速度，高缓存率 |
| Balanced | 8 | 0 | 0.08 | 8 | 速度与质量平衡 |
| Conservative | 12 | 4 | 0.05 | 10 | 高质量，低阈值 |

### 2.2 TaylorSeer Calibrator

**配置**:

```python
TaylorSeerCalibratorConfig(
    enable_calibrator=True,
    taylorseer_order=1,  # 1 或 2（Taylor 级数阶数）
)
```

**原理**:

使用 Taylor 级数展开预测缓存步的特征：

$$
\mathcal{F}_{\text{pred},m}(x_{t-k}^l) = \mathcal{F}(x_t^l) + \sum_{i=1}^m \frac{\Delta^i \mathcal{F}(x_t^l)}{i! \cdot N^i}(-k)^i
$$

- 阶数 1: 一阶导数近似
- 阶数 2: 二阶导数近似，更精确但计算量稍大

## 3. 测试脚本使用

### 3.1 生成结果表

```bash
# 完整基准测试（12 次测试：4 策略 × 3 配置）
python benchmark_table.py --output RESULTS_TABLE.md

# 自定义配置
python benchmark_table.py \
    --seq-len 300 \
    --num-steps 28 \
    --num-warmup 5 \
    --device cuda \
    --output MY_RESULTS.md
```

### 3.2 详细对比测试

```bash
# 基础策略对比
python test_cosyvoice3_dit_cache_improved.py --test-mode compare

# TaylorSeer 对比
python test_cosyvoice3_dit_cache_improved.py --test-mode compare-taylorseer

# 单独测试
python test_cosyvoice3_dit_cache_improved.py --test-mode aggressive --enable-taylorseer
```

## 4. 预期结果

基于 CPU 测试（batch=1, seq=200, steps=28）：

| 策略 | Time/Step | Speedup | Cache Rate |
|------|-----------|---------|------------|
| Baseline | ~130ms | 1.00x | 0% |
| Conservative | ~112ms | 1.16x | 64% |
| Balanced | ~72ms | 1.81x | 71% |
| **Aggressive** | **~23ms** | **5.68x** | **86%** |

TaylorSeer 影响: ±1-2%（可忽略）

## 5. 参考资料

- **模型**: CosyVoice3-0.5B (193M parameters)
- **论文**: CosyVoice 3: Scalable Multilingual Speech Synthesis via LLM-like AutoRegressive Modeling
- **加速库**: cache-dit v1.1.8
- **文档**: `/data/workspace/dit/cache-dit/docs/User_Guide.md`

---

**文档版本**: v1.0
**测试日期**: 2026-01-04
**测试环境**: CPU, PyTorch, Linux 5.4
