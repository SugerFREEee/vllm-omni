# vllm-omni 适配 cozyvoice3

## 1 整体流程

### 1.1 初始化
#### 1.1.1 服务器入口 (`vllm_server.py`)

启动命令示例
```bash
python myscripts/vllm_server.py \
  --model-dir /home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --num-steps 10 \
  --host 0.0.0.0 \
  --port 8001
```
服务器启动后，会执行以下关键步骤：
1. **参数解析**：解析命令行参数，包括模型路径、步数、主机和端口等
2. **路径设置**：添加项目根路径到 Python 环境变量
3. **运行时初始化**：创建 `VLLMRuntime` 实例，这是服务器的核心运行时组件
4. **FastAPI 启动**：启动 FastAPI 服务器，监听指定的主机和端口

#### 1.1.2 从OmniDiffusion 初始化到 Pipeline被创建

1. `VLLMRuntime` 在初始化时会创建 `OmniDiffusion` 实例，这是与 vLLM-Omni 扩散模型交互的主要接口：
    - **配置创建**：创建 `OmniDiffusionConfig` 配置对象，设置模型路径、缓存后端、数据类型等
    - **模型类名设置**：显式设置模型类名为 `CosyVoice3Pipeline`
    - **Transformer 配置**：创建 `TransformerConfig`，设置模型的隐藏层大小、注意力头数、梅尔维度等
    - **引擎创建**：调用 `OmniDiffusion` 构造函数创建引擎实例

2. `OmniDiffusion` 在初始化时会创建 `DiffusionEngine` 实例，这是实际处理扩散过程的引擎：
    - **配置验证**：验证配置的有效性
    - **调度器初始化**：初始化调度器，用于管理请求队列
    - **Worker 启动**：启动多个 worker 进程，每个 worker 对应一个 GPU
    - **资源管理**：创建 `BackgroundResources` 实例，用于管理后台资源的清理

3. `DiffusionEngine` 会为每个 GPU 启动一个 worker 进程：
    - **多进程设置**：设置多进程启动方式为 `spawn`
    - **Worker 创建**：为每个 GPU 创建一个 `WorkerProc` 实例
    - **初始化完成通知**：当 worker 初始化完成后，向主进程发送通知
    - **结果队列设置**：设置结果队列，用于 worker 向主进程返回结果

4. 每个 worker 进程中，`WorkerProc` 会创建 `GPUWorker` 实例，负责初始化设备和加载模型：
    - **分布式环境设置**：设置分布式环境变量，包括 MASTER_ADDR、MASTER_PORT、LOCAL_RANK、RANK 和 WORLD_SIZE
    - **设备设置**：设置 CUDA 设备，确保每个 worker 使用正确的 GPU
    - **模型并行初始化**：初始化模型并行环境
    - **模型加载**：使用 `DiffusersPipelineLoader` 加载模型

5. `DiffusersPipelineLoader` 负责从磁盘加载模型权重：
    - **权重准备**：检查模型路径，如果不是本地路径则下载
    - **文件过滤**：过滤重复的 safetensors 文件和推理不需要的文件
    - **模型初始化**：调用 `initialize_model` 函数初始化模型
    - **权重加载**：将权重加载到模型中

6. `initialize_model` 函数根据模型类名从注册表中加载相应的模型类：
    - **模型类查找**：在 `DiffusionModelRegistry` 中查找模型类
    - **模型实例创建**：创建模型实例，传入配置
    - **模型优化设置**：配置 VAE 的内存优化设置
    - **Pipeline创建**：在 `initialize_model`函数中
        - 通过 `DiffusionModelRegistry._try_load_model_cls(od_config, model_class_name)` 加载 `CosyVoice3Pipeline` 类
        - 最终创建实例： `model = CosyVoice3Pipeline(od_config=od_config)`



### 1.2 请求推理

#### 1.2.1 请求发出

```python
# /home/wjs/workspace/CosyVoice/cosyvoice/flow/flow.py
class CausalMaskedDiffWithDiT(torch.nn.Module):
    # ...
    feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=streaming
        )

```

服务器接收推理请求后，会执行以下步骤：

1. **请求验证**：验证请求的有效性
2. **运行时获取**：获取 `VLLMRuntime` 实例
3. **推理调用**：调用 `infer` 方法执行推理

```python
# Call vllm-omni engine
output = self.omni.generate(
    prompt="CosyVoice3-RealInference",
    num_inference_steps=num_steps,
    extra={
        "condition_vector": payload.condition_vector,
        "speaker_embedding": payload.speaker_embedding,
        "cond": payload.cond,
        "seq_len": payload.seq_len,
        "batch_size": payload.batch_size,
        "benchmark_mode": False,  # 真实推理模式
    },
)
```

1. **OmniDiffusion.generate** (`/home/wjs/workspace/CosyVoice/third_party/vllm-omni/vllm_omni/entrypoints/omni_diffusion.py`)
   - 将字符串提示转换为请求列表
   - 为每个请求生成请求ID
   - 调用 `prepare_requests()` 创建 `OmniDiffusionRequest` 对象
   - 最终调用 `self._run_engine(requests)`

2. **OmniDiffusion._run_engine**
   - 简单转发请求：`return self.engine.step(requests)`

3. **DiffusionEngine.step** (`/home/wjs/workspace/CosyVoice/third_party/vllm-omni/vllm_omni/diffusion/diffusion_engine.py`)
   - 应用预处理函数（如果有）
   - 调用 `self.add_req_and_wait_for_response(requests)`
   - 应用后处理函数（如果有）
   - 将结果转换为 `OmniRequestOutput` 格式返回

4. **DiffusionEngine.add_req_and_wait_for_response**
   - 转发请求到调度器：`return scheduler.add_req(requests)`

5. **Scheduler.add_req** (`/home/wjs/workspace/CosyVoice/third_party/vllm-omni/vllm_omni/diffusion/scheduler.py`)
   - 创建RPC请求消息
   - 通过共享内存队列 `self.mq.enqueue(rpc_request)` 将请求广播给所有worker
   - 等待并从结果队列获取响应

6. **WorkerProc.worker_busy_loop** (`/home/wjs/workspace/CosyVoice/third_party/vllm-omni/vllm_omni/diffusion/worker/gpu_worker.py`)
   - 从共享内存队列接收RPC请求
   - 调用 `self.execute_rpc(msg)` 处理请求

7. **WorkerProc.execute_rpc**
   - 获取worker对象的 `generate` 方法
   - 调用 `func(*args, **kwargs)` 执行方法

8. **GPUWorker.generate**
   - 转发请求到 `self.execute_model(requests, self.od_config)`

9. **GPUWorker.execute_model**
   - 获取第一个请求对象 `req = reqs[0]`
   - 刷新缓存上下文（如果需要）
   - 设置前向传播上下文
   - 最终调用 `self.pipeline.forward(req)`

10. **CosyVoice3Pipeline.forward** (`/home/wjs/workspace/CosyVoice/third_party/vllm-omni/vllm_omni/diffusion/models/cosyvoice3/pipeline_cosyvoice3.py`)
    - 检查请求类型（基准测试模式或实际推理模式）
    - 根据请求类型调用相应的处理方法
    - 返回推理结果


#### 1.2.2 pipeline



`CosyVoice3Pipeline` 的 `forward` 方法基本上沿用了CosyVoice原始的推理逻辑，但经过了一些适配以确保与原始实现的一致性：

```python
def _forward_inference(self, request: OmniDiffusionRequest) -> DiffusionOutput:

        extra = request.extra or {}

        mu = extra.get("condition_vector")  # [batch, mel_dim, seq]
        spks = extra.get("speaker_embedding")  # [batch, spk_dim]
        cond = extra.get("cond")  # [batch, mel_dim, seq]
        seq_len = extra.get("seq_len")
        batch_size = extra.get("batch_size", 1)


        num_steps = int(request.num_inference_steps or self.model_config.num_inference_steps)

        # 时间步调度器设置
        t_span = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device, dtype=torch.float32)
        # 如果配置了余弦调度器 ，将线性时间步转换为余弦调度的时间步
        if self._t_scheduler == "cosine":
            t_span = 1.0 - torch.cos(t_span * 0.5 * torch.pi)

        # 确定性初始噪声（与CosyVoice的 CausalConditionalCFM.rand_noise 用法保持一致）
        # 添加温度参数支持
        temperature = 1.0  # Default temperature
        z = self._rand_noise[:, :, :seq_len].to(self.device) * temperature

        # 使用make_pad_mask生成正确的掩码，匹配原始CosyVoice实现
        # Create token_len_total (assuming no padding in this case)
        token_len_total = torch.tensor([seq_len], dtype=torch.int32, device=self.device).repeat(batch_size)
        mask = (~make_pad_mask(token_len_total)).unsqueeze(1).to(self.device)

        # 将初始噪声扩展到批次大小
        x = z.expand(batch_size, -1, -1).contiguous()

        # 在扩散步骤循环之前预先分配内存，避免在每个步骤中重复创建张量
        # 使用输入数据的dtype而不是硬编码float32
        dtype = mu.dtype if mu is not None else torch.float32
        x_in = torch.zeros((2 * batch_size, self.model_config.mel_dim, seq_len), device=self.device, dtype=dtype)
        mask_in = torch.zeros((2 * batch_size, 1, seq_len), device=self.device, dtype=dtype)
        mu_in = torch.zeros((2 * batch_size, self.model_config.mel_dim, seq_len), device=self.device, dtype=dtype)
        t_in = torch.zeros((2 * batch_size,), device=self.device, dtype=dtype)
        spks_in = torch.zeros((2 * batch_size, self.model_config.mel_dim), device=self.device, dtype=dtype)
        cond_in = torch.zeros((2 * batch_size, self.model_config.mel_dim, seq_len), device=self.device, dtype=dtype)

        # 初始化时间步（与原始CosyVoice保持一致）
        t = t_span[0]
        dt = (t_span[1] - t_span[0]).item()

        with torch.no_grad():
            for step in range(1, len(t_span)):
                # 填充输入数据
                x_in[:] = x.repeat(2, 1, 1)        # 扩散状态复制到两个批次
                mask_in[:] = mask.repeat(2, 1, 1)  # 掩码复制到两个批次
                mu_in[:batch_size] = mu            # 只填充前半部分的条件向量
                spks_in[:batch_size] = spks        # 只填充前半部分的说话人嵌入
                cond_in[:batch_size] = cond        # 只填充前半部分的条件输入
                t_in[:] = t                        # 时间步填充所有批次

                # Call the real DiT estimator (signature matches CosyVoice).
                dphi_dt = self.model.dit(
                    x=x_in,
                    mask=mask_in,
                    mu=mu_in,
                    t=t_in,
                    spks=spks_in,
                    cond=cond_in,
                    streaming=False,
                )

                guided, cfg = torch.split(dphi_dt, [batch_size, batch_size], dim=0)
                # 引导输出 = 有条件输出 + CFG强度 × (有条件输出 - 无条件输出)
                guided = (1.0 + self._inference_cfg_rate) * guided - self._inference_cfg_rate * cfg
                # 使用欧拉方法更新扩散状态
                x = x + dt * guided

                # 更新当前时间步（与原始CosyVoice保持一致）
                t = t + dt
                # 计算下一时间步长
                if step < len(t_span) - 1:
                    dt = t_span[step + 1] - t

        return DiffusionOutput(output=x.float())
```

### 关键适配点

1. **温度参数支持**：
   - 在初始噪声生成时添加了温度参数：`z = self._rand_noise[:, :, :seq_len].to(self.device) * temperature`
   - 目前默认值为1.0，可根据需要调整

2. **掩码生成优化**：
   - 使用 `make_pad_mask` 函数生成正确形状的掩码：`mask = (~make_pad_mask(token_len_total)).unsqueeze(1)`
   - 掩码形状为 `[batch_size, 1, seq_len]`，与原始CosyVoice保持一致

3. **数据类型处理**：
   - 使用输入数据的dtype：`dtype = mu.dtype if mu is not None else torch.float32`
   - 避免硬编码为float32，提高灵活性

4. **时间步计算**：
   - 使用累积方式更新时间步：`t = t + dt`
   - 与原始CosyVoice的时间步计算逻辑保持一致
   - 确保每个时间步的更新与原始实现完全相同

5. **完整的Dit调用**：
   - 展示了完整的 `self.model.dit()` 调用参数，便于理解实际调用过程

#### 1.2.3 CosyVoice3DiTVllm


```python
# CosyVoice/third_party/vllm-omni/vllm_omni/diffusion/models/cosyvoice3/cosyvoice3_dit_vllm.py
class CosyVoice3DiTVllm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        condition_vector: torch.Tensor,
        speaker_embedding: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:

        # Transpose inputs from [batch, seq, dim] to [batch, dim, seq] for real DiT
        x = hidden_states.transpose(1, 2)  # [batch, mel_dim, seq]
        mu = condition_vector.transpose(1, 2)  # [batch, mel_dim, seq]

        # Speaker embedding: take first token (they're all the same after expansion)
        spks = speaker_embedding[:, 0, :]  # [batch, mel_dim]

        if cond is not None:
            cond = cond.transpose(1, 2)  # [batch, mel_dim, seq]
        else:
            # If no cond provided, use zeros
            cond = torch.zeros_like(x)

        # Create mask: ones for all positions (no masking)
        batch, _, seq_len = x.shape
        mask = torch.ones(batch, seq_len, device=x.device, dtype=torch.bool)

        # Call real DiT
        # from .DiT.dit import DiT
        # self.dit = DiT(...)
        output = self.dit(
            x=x,
            mask=mask,
            mu=mu,
            t=timesteps,
            spks=spks,
            cond=cond,
            streaming=False
        )

        # Transpose output back to [batch, seq, dim]
        output = output.transpose(1, 2)
        return output
```

