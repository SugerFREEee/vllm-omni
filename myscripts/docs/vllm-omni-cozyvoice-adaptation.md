# vllm-omni 适配 CosyVoice3 技术文档

## 1. 测试过程

### 1.1 vllm-omni 适配

#### 1.1.1 vllm-omni 注册模型

vllm-omni 使用双层注册机制来集成 CosyVoice3 模型：

**模型层注册 (Model Executor)**

位置: `vllm-omni/vllm_omni/model_executor/models/registry.py:52-56`

```python
_OMNI_MODELS = {
    "CosyVoice3DiTModel": (
        "cosyvoice3",              # 模块文件夹
        "cosyvoice3_dit_vllm",     # 模块相对名称
        "CosyVoice3DiTVllm",       # 类名
    )
}
```

注册机制特点:
- 使用懒加载 `_LazyRegisteredModel`，避免启动时加载所有模型
- 完整模块路径: `vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_dit_vllm`
- 通过 `OmniModelRegistry` 统一管理

**Diffusion Pipeline 注册**

位置: `vllm-omni/vllm_omni/diffusion/registry.py:72-76`

```python
_DIFFUSION_MODELS = {
    "CosyVoice3Pipeline": (
        "cosyvoice3",              # 模块文件夹
        "pipeline_cosyvoice3",     # 模块相对名称
        "CosyVoice3Pipeline",      # 类名
    )
}
```

注册作用:
- 将 DiT 模型暴露给 Omni Diffusion 引擎
- 提供统一的推理接口和性能测量
- 支持与缓存后端 (cache-dit/tea_cache) 集成

#### 1.1.2 cosyvoice3 在 vllm-omni 上的模块和 forward pipeline

**模块架构概览**

```
CosyVoice3 适配层
├── cosyvoice3_config.py        # 配置文件
├── cosyvoice3_dit_model.py     # 核心 DiT 模型实现
├── cosyvoice3_dit_vllm.py      # vLLM 包装器
└── pipeline_cosyvoice3.py      # Diffusion Pipeline
```

##### 1.1.2.1 配置层 (CosyVoice3DiTConfig)

位置: `cosyvoice3_config.py`

**核心配置参数:**

```python
# 模型架构
hidden_size = 1024              # Transformer 隐藏层维度
num_hidden_layers = 22          # Transformer 层数
num_attention_heads = 16        # 多头注意力头数
dim_head = 64                   # 每个注意力头的维度

# 输入输出维度
mel_dim = 80                    # Mel频谱维度
mu_dim = 80                     # 条件向量维度
spk_dim = 80                    # 说话人嵌入维度
out_channels = 80               # 输出通道数

# cache-dit 加速配置
enable_cache_dit = False        # 是否启用缓存
cache_Fn = 8                    # 前向计算块数
cache_Bn = 0                    # 后向融合块数
cache_threshold = 0.08          # 残差差异阈值
cache_warmup_steps = 8          # 缓存预热步数
num_inference_steps = 28        # 推理步数

# TaylorSeer 配置
enable_taylorseer = False       # 是否启用 Taylor 级数加速
taylorseer_order = 1            # Taylor 级数阶数
```

##### 1.1.2.2 DiT 模型核心 (CosyVoice3DiT)

位置: `cosyvoice3_dit_model.py:121-218`

**模型结构:**

```
CosyVoice3DiT
├── 时间编码模块
│   ├── SinusoidalPosEmb       # 正弦位置编码 (diffusion timestep)
│   └── TimeEmbedding          # MLP 时间嵌入 (映射到 hidden_size)
│
├── 输入投影层
│   └── input_proj: Linear(mel_dim+mu_dim+spk_dim → hidden_size)
│
├── Transformer 块 (×22 层)
│   ├── Attention              # 多头自注意力
│   │   ├── norm: RMSNorm     # 前规范化
│   │   ├── to_q, to_k, to_v  # Q/K/V 投影
│   │   └── to_out: Linear    # 输出投影
│   │
│   └── FeedForward            # 前馈网络
│       ├── norm: RMSNorm     # 前规范化
│       ├── proj_in: Linear   # 扩展到 4×hidden_size
│       ├── gelu: GELU        # 激活函数
│       └── proj_out: Linear  # 投影回 hidden_size
│
└── 输出层
    ├── norm_out: RMSNorm      # 最终规范化
    └── output_proj: Linear(hidden_size → out_channels)
```

**Forward Pipeline:**

```python
def forward(
    x: torch.Tensor,           # [B, T, mel_dim] - 带噪 Mel 频谱
    timesteps: torch.Tensor,   # [B] - Diffusion 时间步
    mu: torch.Tensor,          # [B, T, mu_dim] - 条件向量
    spk: torch.Tensor,         # [B, T, spk_dim] - 说话人嵌入
):
    # 第1步: 时间步编码
    t_emb = self.time_embed(timesteps)  # [B, hidden_size]

    # 第2步: 输入拼接和投影
    x = torch.cat([x, mu, spk], dim=-1)  # [B, T, mel+mu+spk]
    x = self.input_proj(x)                # [B, T, hidden_size]
    x = x + t_emb.unsqueeze(1)           # 加入时间信息

    # 第3步: 逐层 Transformer 处理 (22层)
    for block in self.blocks:
        # 注意力 + 残差连接
        x = x + block.attn(x)
        # 前馈网络 + 残差连接
        x = x + block.ff(x)

    # 第4步: 输出投影
    x = self.norm_out(x)                 # [B, T, hidden_size]
    x = self.output_proj(x)              # [B, T, out_channels]

    return x  # 预测的噪声或去噪后的频谱
```

**关键设计:**
- 使用 RMSNorm 而非 LayerNorm (更高效)
- 预规范化架构 (norm → attention/ffn → residual)
- 时间步通过加法注入，而非拼接 (节省计算)
- 单输入单输出模式 (Pattern_3) 适配 cache-dit

##### 1.1.2.3 vLLM 包装器 (CosyVoice3DiTVllm)

位置: `cosyvoice3_dit_vllm.py:24-249`

**功能职责:**
1. 包装原始 DiT 模型为 vLLM 兼容格式
2. 管理 cache-dit 加速的生命周期
3. 提供统一的缓存控制接口

**核心方法:**

```python
class CosyVoice3DiTVllm(nn.Module):
    def __init__(self, config):
        self.model = CosyVoice3DiT(config)
        self.config = config
        self._refresh_cache_context_fn = None

        # 如果配置启用缓存，则初始化
        if config.enable_cache_dit:
            self._setup_cache_dit()

    def _setup_cache_dit(self):
        """初始化 cache-dit 加速"""
        from vllm_omni.diffusion.cache.cache_dit_backend import (
            enable_cache_for_cosyvoice3
        )

        # 构建缓存配置
        cache_config = DiffusionCacheConfig(
            Fn_compute_blocks=self.config.cache_Fn,
            Bn_compute_blocks=self.config.cache_Bn,
            residual_diff_threshold=self.config.cache_threshold,
            max_warmup_steps=self.config.cache_warmup_steps,
            num_inference_steps=self.config.num_inference_steps,
            enable_taylorseer=self.config.enable_taylorseer,
            taylorseer_order=self.config.taylorseer_order,
        )

        # 启用缓存并获取刷新函数
        self._refresh_cache_context_fn = enable_cache_for_cosyvoice3(
            pipeline=self,
            cache_config=cache_config,
        )

    def forward(self, x, timesteps, mu, spk):
        """vLLM 兼容的前向传播"""
        return self.model(x, timesteps, mu, spk)

    def refresh_cache_context(self, num_inference_steps, verbose=False):
        """刷新缓存上下文 (改变推理步数时调用)"""
        if self._refresh_cache_context_fn:
            self._refresh_cache_context_fn(
                self, num_inference_steps, verbose
            )

    def get_cache_stats(self):
        """获取缓存统计信息"""
        import cache_dit
        return cache_dit.summary(self.model, details=True)

    def disable_cache(self):
        """禁用缓存"""
        import cache_dit
        cache_dit.disable_cache(self.model)
```

##### 1.1.2.4 Diffusion Pipeline (CosyVoice3Pipeline)

位置: `pipeline_cosyvoice3.py:33-167`

**Pipeline 职责:**
- 在 Omni Diffusion 引擎中暴露 CosyVoice3 DiT
- 生成随机输入进行基准测试
- 测量性能指标 (TTFT, 平均时间, 总时间, 缓存率)

**Forward 流程:**

```python
def forward(self, request: OmniDiffusionRequest):
    # 第1步: 解析请求参数
    batch_size = request.batch_size
    seq_len = request.seq_len
    num_steps = request.num_steps
    num_warmup = request.num_warmup

    # 第2步: 生成随机输入 (模拟真实推理)
    device = next(self.model.parameters()).device
    hidden_states = torch.randn(batch_size, seq_len, 80, device=device)
    condition_vector = torch.randn(batch_size, seq_len, 80, device=device)
    speaker_embedding = torch.randn(batch_size, seq_len, 80, device=device)
    timesteps = torch.zeros(batch_size, device=device)

    # 第3步: 预热 (cache-dit 需要观察前几步以建立缓存策略)
    for _ in range(num_warmup):
        with torch.no_grad():
            self.model(hidden_states, timesteps, condition_vector, speaker_embedding)

    # 第4步: 测量 TTFT (Time To First Token)
    torch.cuda.synchronize()
    ttft_start = time.perf_counter()

    with torch.no_grad():
        self.model(hidden_states, timesteps, condition_vector, speaker_embedding)

    torch.cuda.synchronize()
    ttft = time.perf_counter() - ttft_start

    # 第5步: 逐步推理并测量每步时间
    step_times = []
    for step in range(num_steps):
        # 计算当前时间步 (线性插值 0 → 1000)
        t = step / (num_steps - 1) * 1000
        timesteps = torch.full((batch_size,), t, device=device)

        # 测量单步时间
        torch.cuda.synchronize()
        step_start = time.perf_counter()

        with torch.no_grad():
            self.model(hidden_states, timesteps, condition_vector, speaker_embedding)

        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - step_start)

    # 第6步: 收集缓存统计
    cached_steps = self._collect_cached_steps()

    # 第7步: 计算指标
    avg_time = sum(step_times) / len(step_times)
    total_time = sum(step_times)

    return {
        "ttft": ttft,                  # 首Token时间 (ms)
        "avg_time": avg_time,          # 平均每步时间 (ms)
        "total_time": total_time,      # 总推理时间 (ms)
        "cached_steps": cached_steps,  # 被缓存的步数
    }

def _collect_cached_steps(self):
    """从 cache-dit 收集缓存步数统计"""
    try:
        import cache_dit
        stats = cache_dit.summary(self.model.transformer, details=False)
        if stats and hasattr(stats[0], "cached_steps"):
            return len(stats[0].cached_steps)
    except Exception:
        pass
    return 0
```

#### 1.1.3 如何利用 vllm-omni 封装的 cache-dit

##### 1.1.3.1 cache-dit 核心概念

cache-dit 是一种针对 Diffusion Transformer 的加速技术，通过以下策略减少计算:

**核心思想:**
- Diffusion 模型在相邻时间步之间，某些 Transformer 块的输出变化很小
- 对于变化小的块，可以复用上一步的输出 (缓存)，跳过当前步的计算
- 通过残差差异 (residual difference) 判断是否可以缓存

**关键配置参数:**

| 参数 | 说明 | CosyVoice3 推荐值 |
|------|------|------------------|
| `Fn_compute_blocks` | 用于计算残差差异的前向块数 | 8 (平衡) / 1 (激进) |
| `Bn_compute_blocks` | 融合的后向块数 | 0 |
| `residual_diff_threshold` | L1 残差差异阈值 (超过则重新计算) | 0.08 (平衡) / 0.15 (激进) |
| `max_warmup_steps` | 缓存前的预热步数 | 8 |
| `max_continuous_cached_steps` | 连续缓存的最大步数 | 3 |

##### 1.1.3.2 启用 cache-dit 的流程

**后端封装层:**

位置: `vllm-omni/vllm_omni/diffusion/cache/cache_dit_backend.py:344-415`

```python
def enable_cache_for_cosyvoice3(pipeline, cache_config):
    """为 CosyVoice3 启用 cache-dit 加速"""
    import cache_dit
    from cache_dit import BlockAdapter, ForwardPattern

    # 第1步: 构建 cache-dit 配置
    db_cache_config = DBCacheConfig(
        Fn_compute_blocks=cache_config.Fn_compute_blocks,
        Bn_compute_blocks=cache_config.Bn_compute_blocks,
        residual_diff_threshold=cache_config.residual_diff_threshold,
        max_warmup_steps=cache_config.max_warmup_steps,
        max_cached_steps=cache_config.max_cached_steps,
        max_continuous_cached_steps=cache_config.max_continuous_cached_steps,
    )

    # 第2步: 配置 TaylorSeer (可选)
    calibrator = None
    if cache_config.enable_taylorseer:
        calibrator = TaylorSeerCalibratorConfig(
            order=cache_config.taylorseer_order,
        )

    # 第3步: 创建参数修改器
    modifier = ParamsModifier(
        cache_config=db_cache_config,
        calibrator_config=calibrator,
    )

    # 第4步: 创建 BlockAdapter (适配 CosyVoice3 的块结构)
    block_adapter = BlockAdapter(
        transformer=pipeline.transformer,       # DiT 模型
        blocks=pipeline.transformer.blocks,     # Transformer 块列表
        forward_pattern=ForwardPattern.Pattern_3,  # 单输入单输出模式
        params_modifiers=[modifier],
    )

    # 第5步: 启用缓存
    cache_dit.enable_cache(
        block_adapter,
        cache_config=db_cache_config,
    )

    # 第6步: 返回刷新函数 (改变推理步数时需要调用)
    def refresh_cache_context(pipeline, num_inference_steps, verbose=False):
        cache_dit.refresh_context(
            pipeline.transformer,
            num_inference_steps=num_inference_steps,
            verbose=verbose
        )

    return refresh_cache_context
```

##### 1.1.3.3 BlockAdapter 详解

**作用:** 告诉 cache-dit 如何适配特定的 Transformer 架构

```python
@dataclass
class BlockAdapter:
    # 必需字段
    transformer: nn.Module              # DiT 模型对象
    blocks: nn.ModuleList               # Transformer 块列表

    # 可选字段
    blocks_name: str = "blocks"         # 块的属性名
    forward_pattern: ForwardPattern = ForwardPattern.Pattern_3
    params_modifiers: List[ParamsModifier] = None
    check_forward_pattern: bool = False
    has_separate_cfg: bool = False
    auto: bool = False
```

**Forward Pattern 说明:**

cache-dit 支持 6 种前向模式 (Pattern_0 到 Pattern_5)，根据块的输入输出签名区分:

| Pattern | 输入 | 输出 | 适用场景 |
|---------|------|------|----------|
| Pattern_0 | (h, e) | (h, e) | 标准双输入双输出 (如 Stable Diffusion UNet) |
| Pattern_1 | (h, e) | (e, h) | 交换输出顺序 |
| Pattern_2 | (h, e) | (h,) | 仅返回 hidden_states |
| **Pattern_3** | **(h,)** | **(h,)** | **单输入单输出 (CosyVoice3 使用)** |
| Pattern_4 | (h,) | (h, e) | 单输入双输出 |
| Pattern_5 | (h,) | (e, h) | 单输入交换输出 |

**CosyVoice3 使用 Pattern_3 的原因:**
- CosyVoice3 的 DiTBlock 只接受 hidden_states 作为输入
- 不使用 cross-attention (无 encoder_hidden_states)
- 时间步和条件信息通过加法注入，而非拼接

##### 1.1.3.4 缓存策略示例

**平衡策略 (推荐):**

```python
cache_config = DiffusionCacheConfig(
    Fn_compute_blocks=8,                # 计算前8个块来估计残差差异
    Bn_compute_blocks=0,                # 不融合后向块
    residual_diff_threshold=0.08,       # L1 差异 < 0.08 时缓存
    max_warmup_steps=8,                 # 前8步用于观察
    max_continuous_cached_steps=3,      # 最多连续缓存3步
)
```

**工作流程:**
1. 前8步: 预热阶段，所有块正常计算
2. 第9步开始: 计算前8个块的残差差异
3. 如果差异 < 0.08: 复用上一步的输出 (缓存命中)
4. 如果差异 ≥ 0.08: 重新计算 (缓存未命中)
5. 如果连续缓存3步: 强制重新计算 (避免误差累积)

**激进策略 (追求极致速度):**

```python
cache_config = DiffusionCacheConfig(
    Fn_compute_blocks=1,                # 仅计算第1个块
    residual_diff_threshold=0.15,       # 更宽松的阈值
    max_warmup_steps=4,                 # 更少的预热步数
)
```

**保守策略 (追求质量):**

```python
cache_config = DiffusionCacheConfig(
    Fn_compute_blocks=12,               # 计算更多块
    residual_diff_threshold=0.05,       # 更严格的阈值
    max_warmup_steps=8,
)
```

##### 1.1.3.5 运行时控制缓存

```python
# 示例: 在 vLLM 包装器中使用缓存
model = CosyVoice3DiTVllm(config)

# 刷新缓存上下文 (改变推理步数时必须调用)
model.refresh_cache_context(num_inference_steps=50, verbose=True)

# 获取缓存统计
stats = model.get_cache_stats()
print(f"缓存步数: {stats[0].cached_steps}")
print(f"残差差异: {stats[0].residual_diffs}")

# 禁用缓存 (调试时)
model.disable_cache()
```

### 1.2 测试过程

#### 1.2.1 输入输出

##### 1.2.1.1 基准测试输入

位置: `myscripts/benchmark_vllm.py`

**标准输入配置:**

```python
# 请求参数
request = OmniDiffusionRequest(
    batch_size=1,              # 批量大小
    seq_len=256,               # 序列长度 (Mel频谱帧数)
    num_steps=28,              # Diffusion 推理步数
    num_warmup=8,              # 预热步数
)

# 模型自动生成的输入张量 (在 pipeline 内部)
hidden_states = torch.randn(1, 256, 80)      # [B, T, mel_dim]
condition_vector = torch.randn(1, 256, 80)   # [B, T, mu_dim]
speaker_embedding = torch.randn(1, 256, 80)  # [B, T, spk_dim]
timesteps = torch.tensor([500.0])            # [B] - 时间步
```

**输入张量说明:**

| 张量 | 形状 | 说明 |
|------|------|------|
| `hidden_states` | `[B, T, 80]` | 带噪 Mel 频谱 (Diffusion 输入) |
| `condition_vector` | `[B, T, 80]` | 条件向量 (如文本编码器输出) |
| `speaker_embedding` | `[B, T, 80]` | 说话人嵌入 (控制音色) |
| `timesteps` | `[B]` | Diffusion 时间步 (0-1000) |

##### 1.2.1.2 输出格式

**模型输出 (单步):**

```python
# DiT 模型的单步输出
output = model(hidden_states, timesteps, condition_vector, speaker_embedding)
# output.shape: [B, T, 80] - 预测的噪声或去噪后的频谱
```

**Pipeline 输出 (完整推理):**

```python
result = pipeline(request)

# 返回字典结构
{
    "ttft": 0.0234,          # Time To First Token (秒)
    "avg_time": 0.0187,      # 平均每步时间 (秒)
    "total_time": 0.5236,    # 总推理时间 (秒)
    "cached_steps": 15,      # 被缓存的步数 (共28步中有15步命中缓存)
}
```

##### 1.2.1.3 真实使用场景的输入

在实际 TTS 应用中 (非基准测试):

```python
# 第1步: 文本编码
text_tokens = tokenizer.encode("你好世界")
text_embedding = text_encoder(text_tokens)  # [1, T_text, hidden_size]

# 第2步: 获取说话人嵌入
speaker_embedding = speaker_encoder(speaker_id)  # [1, T_mel, spk_dim]

# 第3步: Diffusion 采样循环
hidden_states = torch.randn(1, T_mel, 80)  # 初始噪声
for t in reversed(range(num_steps)):
    timesteps = torch.tensor([t])

    # 预测噪声
    noise_pred = model(hidden_states, timesteps, text_embedding, speaker_embedding)

    # 去噪步骤 (DDPM/DDIM)
    hidden_states = scheduler.step(noise_pred, t, hidden_states)

# 第4步: 解码为音频
mel_spectrogram = hidden_states
audio = vocoder(mel_spectrogram)
```

#### 1.2.2 如何获取指标

##### 1.2.2.1 性能指标

**指标定义:**

| 指标 | 说明 | 计算方式 | 单位 |
|------|------|----------|------|
| **TTFT** | Time To First Token (首Token延迟) | 第一次前向传播的时间 | 秒 (s) 或 毫秒 (ms) |
| **avg_time** | 平均每步时间 | `sum(step_times) / num_steps` | 秒 (s) |
| **total_time** | 总推理时间 | `sum(step_times)` | 秒 (s) |
| **cached_steps** | 缓存步数 | 命中缓存的步数 / 总步数 | 整数 |
| **cache_rate** | 缓存命中率 | `cached_steps / num_steps` | 百分比 (%) |
| **speedup** | 加速比 | `baseline_time / cache_time` | 倍数 (×) |

**测量代码:**

位置: `pipeline_cosyvoice3.py:91-166`

```python
# 第1步: 测量 TTFT
torch.cuda.synchronize()  # 等待 GPU 完成
ttft_start = time.perf_counter()

model(hidden_states, timesteps, condition_vector, speaker_embedding)

torch.cuda.synchronize()
ttft = time.perf_counter() - ttft_start

# 第2步: 测量每步时间
step_times = []
for step in range(num_steps):
    t = step / (num_steps - 1) * 1000
    timesteps = torch.full((batch_size,), t)

    torch.cuda.synchronize()
    step_start = time.perf_counter()

    model(hidden_states, timesteps, condition_vector, speaker_embedding)

    torch.cuda.synchronize()
    step_times.append(time.perf_counter() - step_start)

# 第3步: 计算统计指标
avg_time = sum(step_times) / len(step_times)
total_time = sum(step_times)

# 第4步: 获取缓存统计
import cache_dit
stats = cache_dit.summary(model.transformer, details=False)
cached_steps = len(stats[0].cached_steps)
```

##### 1.2.2.2 缓存统计指标

**CacheStats 数据结构:**

位置: `cache-dit/src/cache_dit/summary.py:24-41`

```python
@dataclass
class CacheStats:
    # 缓存配置
    cache_options: dict                 # 使用的缓存参数

    # Dual Block Cache 统计
    cached_steps: list[int]             # 被缓存的步索引 [8, 10, 12, ...]
    residual_diffs: dict[str, float]    # 每步的残差差异 {"step_8": 0.032, ...}

    # CFG 统计 (如果启用)
    cfg_cached_steps: list[int]
    cfg_residual_diffs: dict[str, float]

    # Dynamic Block Prune 统计 (如果启用)
    pruned_steps: list[int]             # 被剪枝的步索引
    pruned_blocks: list[int]            # 每步剪枝的块数
    actual_blocks: list[int]            # 每步实际计算的块数
    pruned_ratio: float                 # 平均剪枝比例
```

**获取详细统计:**

```python
import cache_dit

# 获取详细统计
stats = cache_dit.summary(model.transformer, details=True)

# 打印缓存信息
print(f"缓存配置: {stats[0].cache_options}")
print(f"缓存步数: {len(stats[0].cached_steps)}/{num_inference_steps}")
print(f"缓存命中率: {len(stats[0].cached_steps) / num_inference_steps * 100:.1f}%")

# 打印残差差异 (判断缓存决策是否合理)
for step, diff in stats[0].residual_diffs.items():
    print(f"{step}: L1_diff = {diff:.4f}")

# 示例输出:
# step_8: L1_diff = 0.0321  ✓ 缓存 (< 0.08)
# step_9: L1_diff = 0.0567  ✓ 缓存
# step_10: L1_diff = 0.1123  ✗ 重新计算 (≥ 0.08)
```

##### 1.2.2.3 基准测试脚本

**完整测试流程:**

位置: `myscripts/benchmark_vllm.py`

```python
import torch
from vllm_omni.diffusion import OmniDiffusionRequest
from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_config import CosyVoice3DiTConfig

# 定义测试策略
strategies = {
    'baseline': {
        'enable_cache': False,
        'Fn': 22, 'Bn': 0, 'threshold': 0.0,
    },
    'aggressive': {
        'enable_cache': True,
        'Fn': 1, 'Bn': 0, 'threshold': 0.15,
    },
    'balanced': {
        'enable_cache': True,
        'Fn': 8, 'Bn': 0, 'threshold': 0.08,
    },
    'conservative': {
        'enable_cache': True,
        'Fn': 12, 'Bn': 4, 'threshold': 0.05,
    },
}

# 测试参数
test_configs = [
    {"batch_size": 1, "seq_len": 256, "num_steps": 28},
    {"batch_size": 1, "seq_len": 512, "num_steps": 28},
    {"batch_size": 2, "seq_len": 256, "num_steps": 28},
]

# 运行基准测试
results = {}
for strategy_name, strategy_config in strategies.items():
    for test_config in test_configs:
        # 创建模型配置
        model_config = CosyVoice3DiTConfig(
            enable_cache_dit=strategy_config['enable_cache'],
            cache_Fn=strategy_config['Fn'],
            cache_Bn=strategy_config['Bn'],
            cache_threshold=strategy_config['threshold'],
            num_inference_steps=test_config['num_steps'],
        )

        # 加载模型
        pipeline = DiffusionModelRegistry.load_pipeline(
            "CosyVoice3Pipeline",
            config=model_config,
        )
        pipeline = pipeline.cuda()

        # 创建请求
        request = OmniDiffusionRequest(
            batch_size=test_config['batch_size'],
            seq_len=test_config['seq_len'],
            num_steps=test_config['num_steps'],
            num_warmup=8,
        )

        # 运行推理
        stats = pipeline(request)

        # 保存结果
        key = f"{strategy_name}_b{test_config['batch_size']}_s{test_config['seq_len']}"
        results[key] = stats

        print(f"\n{key}:")
        print(f"  TTFT: {stats['ttft']*1000:.2f} ms")
        print(f"  Avg Time: {stats['avg_time']*1000:.2f} ms")
        print(f"  Total Time: {stats['total_time']:.3f} s")
        print(f"  Cached Steps: {stats['cached_steps']}/{test_config['num_steps']}")

        # 计算加速比
        if strategy_name != 'baseline':
            baseline_key = f"baseline_b{test_config['batch_size']}_s{test_config['seq_len']}"
            baseline_time = results[baseline_key]['total_time']
            speedup = baseline_time / stats['total_time']
            print(f"  Speedup: {speedup:.2f}x")
```

##### 1.2.2.4 输出示例

```
baseline_b1_s256:
  TTFT: 23.45 ms
  Avg Time: 18.76 ms
  Total Time: 0.525 s
  Cached Steps: 0/28

balanced_b1_s256:
  TTFT: 23.12 ms
  Avg Time: 12.34 ms
  Total Time: 0.346 s
  Cached Steps: 15/28
  Speedup: 1.52x

aggressive_b1_s256:
  TTFT: 22.89 ms
  Avg Time: 9.87 ms
  Total Time: 0.277 s
  Cached Steps: 20/28
  Speedup: 1.90x

conservative_b1_s256:
  TTFT: 23.34 ms
  Avg Time: 14.56 ms
  Total Time: 0.408 s
  Cached Steps: 10/28
  Speedup: 1.29x
```

##### 1.2.2.5 质量评估指标

除了性能指标，还需要评估缓存对质量的影响:

**客观指标:**

```python
# 第1步: 生成两个输出 (baseline vs cache)
output_baseline = model_baseline(...)  # [B, T, 80]
output_cache = model_cache(...)        # [B, T, 80]

# 第2步: 计算 MSE (均方误差)
mse = torch.mean((output_baseline - output_cache) ** 2).item()

# 第3步: 计算 MAE (平均绝对误差)
mae = torch.mean(torch.abs(output_baseline - output_cache)).item()

# 第4步: 计算 PSNR (峰值信噪比)
import numpy as np
mse_np = mse
psnr = 20 * np.log10(1.0 / np.sqrt(mse_np))

print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"PSNR: {psnr:.2f} dB")
```

**主观指标 (需要人工评估):**

- MOS (Mean Opinion Score): 音质主观评分 (1-5分)
- SMOS (Speaker Mean Opinion Score): 说话人相似度评分
- ABX 测试: 盲测对比 baseline vs cache

##### 1.2.2.6 关键文件路径总结

| 功能 | 文件路径 |
|------|---------|
| 基准测试脚本 | `myscripts/benchmark_vllm.py` |
| Pipeline 推理 | `vllm-omni/vllm_omni/diffusion/models/cosyvoice3/pipeline_cosyvoice3.py` |
| 缓存统计 | `cache-dit/src/cache_dit/summary.py` |
| 缓存后端 | `vllm-omni/vllm_omni/diffusion/cache/cache_dit_backend.py` |
| 模型配置 | `vllm-omni/vllm_omni/model_executor/models/cosyvoice3/cosyvoice3_config.py` |

---

## 附录: 常见问题

### Q1: 如何调整缓存策略以平衡速度和质量?

**A1:** 调整以下参数:
- 提高速度: 减小 `Fn_compute_blocks` (1-4)，提高 `residual_diff_threshold` (0.12-0.20)
- 提高质量: 增大 `Fn_compute_blocks` (12-16)，降低 `residual_diff_threshold` (0.03-0.06)
- 平衡设置: `Fn=8`, `threshold=0.08`

### Q2: 为什么需要 `refresh_cache_context`?

**A2:** 当改变推理步数时，cache-dit 需要重新计算缓存决策策略。例如从 28 步切换到 50 步，缓存的时间步分布会改变。

### Q3: 如何禁用缓存进行调试?

**A3:**
```python
# 方法1: 配置时禁用
config = CosyVoice3DiTConfig(enable_cache_dit=False)

# 方法2: 运行时禁用
model.disable_cache()
```

### Q4: 缓存统计中的 `residual_diffs` 如何解读?

**A4:**
- 值越小: 表示当前步与上一步的输出越接近，缓存越安全
- 值越大: 表示输出变化大，需要重新计算
- 一般规律: Diffusion 早期步 (高噪声) 变化大，后期步 (低噪声) 变化小

### Q5: 如何为新模型适配 cache-dit?

**A5:**
1. 确定 Forward Pattern (单输入单输出 → Pattern_3)
2. 在模型配置中添加 cache 参数
3. 创建 `enable_cache_for_xxx()` 函数
4. 在包装器中调用 `_setup_cache_dit()`
5. 测试和调优缓存参数
