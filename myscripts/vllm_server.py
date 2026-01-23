"""
python myscripts/vllm_server.py \
  --model-dir /home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --num-steps 10 \
  --disable-cache-dit

extreme
python myscripts/vllm_server.py \
  --model-dir /home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --num-steps 10 \
  --cache-dit-fn 1 \
  --cache-dit-bn 0 \
  --max-warmup-steps 0 \
  --cache-dit-residual-threshold 0.4 \
  --cache-logging off


python myscripts/vllm_server.py \
    --model-dir /home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --num-steps 10 \
    --max-workers 10 \
    --disable-cache-dit
--cache-logging命令行参数有4种 json,on,return,off。off=禁用缓存统计, return=仅返回缓存信息, on=打印信息并返回, json=打印并写json日志
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time
import json
import msgpack
from typing import Any, Dict, List, Optional

import torch

# 添加ZeroMQ支持
try:
    import zmq
    import zmq.asyncio as zmq_async
except ImportError:
    print("[ERROR] ZeroMQ not installed. Please run 'pip install pyzmq' to enable ZeroMQ support.")
    sys.exit(1)

# 确保zmq可用
if zmq is None:
    print("[ERROR] ZeroMQ not available.")
    sys.exit(1)

# Add paths BEFORE importing vllm_omni
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
project_root = os.path.dirname(current_dir)

sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "third_party", "vllm-omni"))


from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion





class VLLMRuntime:
    def __init__(
        self,
        model_dir: str,
        num_steps: int,
        dtype: str,
        enable_cache_dit: bool = True,
        cache_dit_fn: int = 8,
        cache_dit_bn: int = 0,
        cache_dit_enable_taylorseer: bool = False,
        cache_dit_taylorseer_order: int = 1,
        cache_dit_residual_threshold: float = 0.08,
        max_warmup_steps: int = 0,
        enable_cache_logging: Optional[bool] = None,
        cache_logging_mode: Optional[str] = None,
    ):
        self.model_dir = model_dir
        self.num_steps = num_steps
        self.dtype_str = dtype
        self.cache_dit_enabled = enable_cache_dit
        self.cache_dit_fn = cache_dit_fn
        self.cache_dit_bn = cache_dit_bn
        self.cache_dit_enable_taylorseer = cache_dit_enable_taylorseer
        self.cache_dit_taylorseer_order = cache_dit_taylorseer_order
        self.cache_dit_residual_threshold = cache_dit_residual_threshold
        self.max_warmup_steps = max_warmup_steps
        self.enable_cache_logging = enable_cache_logging
        self.cache_logging_mode = cache_logging_mode
        self.average_infer_time = 0.0
        self.cnt_infer = 0

        # Initialize vllm-omni engine
        print(f"[vllm_server] Initializing vLLM-Omni engine...")
        self.omni = self._create_omni_engine(model_dir, num_steps, dtype)
        print(f"[vllm_server] Runtime initialized!")

    def _create_omni_engine(self, model_path: str, num_steps: int, dtype: str) -> OmniDiffusion:
        """Create OmniDiffusion engine with CosyVoice3Pipeline"""
        # Force float32 for all operations; fp16 is unstable for this model/implementation.
        torch_dtype = torch.float32
        if dtype.lower() != "float32":
            print(f"[vllm_server] 忽略请求的 dtype={dtype}，强制使用 float32")
        print(f"[vllm_server] 正在使用 dtype: float32")
        
        # cache-dit 配置 - 确保所有命令行参数都能被正确传递
        cache_backend = "cache_dit" if self.cache_dit_enabled else "none"
        cache_config = None
        if self.cache_dit_enabled:
            cache_config = {
                "Fn_compute_blocks": self.cache_dit_fn,
                "Bn_compute_blocks": self.cache_dit_bn,
                "residual_diff_threshold": self.cache_dit_residual_threshold,
                "max_warmup_steps": self.max_warmup_steps,
                "max_cached_steps": -1,
                "max_continuous_cached_steps": num_steps,  # 设置为num_steps允许连续缓存所有步骤
                "num_inference_steps": num_steps,
                "enable_taylorseer": self.cache_dit_enable_taylorseer,
                "taylorseer_order": self.cache_dit_taylorseer_order,
            }
        else:
            print("[vllm_server] cache-dit disabled for this runtime.")
        
        od_config = OmniDiffusionConfig.from_kwargs(
            model=model_path,
            cache_backend=cache_backend,
            cache_config=cache_config,
            dtype=torch_dtype,
            enable_cache_logging=self.enable_cache_logging,
            cache_logging_mode=self.cache_logging_mode,
        )
        od_config.model_class_name = "CosyVoice3Pipeline"

        tf_cfg = {
            "hidden_size": 1024,
            "num_hidden_layers": 22,
            "num_attention_heads": 16,
            "mel_dim": 80,
            "num_inference_steps": num_steps,
            "seq_len": 200,  # 默认值，实际会被请求覆盖
            "batch_size": 1,
            "num_warmup": 0,
        }
        od_config.tf_model_config = TransformerConfig.from_dict(tf_cfg)

        return OmniDiffusion(od_config=od_config)
    
    def infer(self, payload: dict) -> torch.Tensor:
        """DiT 推理：接收预处理好的 condition_vector 和 speaker_embedding"""
        # 只处理字典格式的请求（ZeroMQ）
        num_steps = payload.get('num_inference_steps', self.num_steps)
        condition_vector = payload.get('condition_vector')
        speaker_embedding = payload.get('speaker_embedding')
        cond = payload.get('cond')
        seq_len = payload.get('seq_len')
        mel_len1 = payload.get('mel_len1')
        batch_size = payload.get('batch_size', 1)

        # print(f"[DEBUG] DiT inference")
        # print(f"[DEBUG]   condition_vector shape: {len(condition_vector)}x{len(condition_vector[0])}x{len(condition_vector[0][0])}")
        # print(f"[DEBUG]   speaker_embedding shape: {len(speaker_embedding)}x{len(speaker_embedding[0])}")
        # print(f"[DEBUG]   cond shape: {len(cond)}x{len(cond[0])}x{len(cond[0][0])}")
        # print(f"[DEBUG]   seq_len: {seq_len}, mel_len1: {mel_len1}")

        perf_start = time.perf_counter()
        # Call vllm-omni engine
        output = self.omni.generate(
            prompt="CosyVoice3-RealInference",
            num_inference_steps=num_steps,
            extra={
                "condition_vector": condition_vector,
                "speaker_embedding": speaker_embedding,
                "cond": cond,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "benchmark_mode": False,  # 真实推理模式
            },
        )
        latency = time.perf_counter() - perf_start
        per_step = latency / num_steps if num_steps else latency
        
        # 更新平均推理时间
        self.average_infer_time = (self.average_infer_time * self.cnt_infer + latency) / (self.cnt_infer + 1)
        self.cnt_infer += 1
        
        print(
            f"[vllm_server] Inference finished in {latency:.3f}s "
            f"(~{per_step * 1000:.2f} ms/step for {num_steps} steps)"
            f" | Average: {self.average_infer_time:.3f}s"
        )

        # Extract mel from output
        print(f"[DEBUG] Output type: {type(output)}")

        # Handle OmniRequestOutput
        if hasattr(output, 'images') and output.images:
            # 虽然我们是生成音频但是 vllm-omni的 diffusion引擎的返回值默认把生产的放入默认的images字段中
            payload_data = output.images[0]

            if isinstance(payload_data, dict):
                if "mel" in payload_data:
                    mel_tensor = payload_data["mel"]
                    
                    # Extract only the generated part (exclude prompt)
                    # mel shape: [batch, mel_dim, total_len]
                    mel = mel_tensor[:, :, mel_len1:]
                    
                    # Print cache hit ratio if cached_steps is available
                    if "cached_steps" in payload_data:
                        cached_steps = payload_data["cached_steps"]
                        if cached_steps is not None:
                            cache_hit_ratio = cached_steps / num_steps if num_steps > 0 else 0.0
                            print(f"[DEBUG] Cache Hit Ratio: {cached_steps}/{num_steps} ({cache_hit_ratio:.2%})")
                    
                    return mel.to(dtype=torch.float32)
            elif isinstance(payload_data, torch.Tensor):
                # 历史遗留下来的，因为原本payload_data是torch.Tensor，现在改成了 dict
                # (wjs)TODO:删除这个历史遗留
                mel_tensor = payload_data
                mel = mel_tensor[:, :, mel_len1:]
                return mel.to(dtype=torch.float32)
            else:
                print(f"[DEBUG] payload_data is PIL Image or other type")

        raise RuntimeError(f"Failed to extract mel from vllm-omni output. Output type: {type(output)}")


runtime: Optional[VLLMRuntime] = None
_executor: Optional[ThreadPoolExecutor] = None


def _require_runtime() -> VLLMRuntime:
    if runtime is None:
        raise RuntimeError("Runtime is not initialized")
    return runtime


def _setup_runtime(
    model_dir: str, 
    num_steps: int, 
    dtype: str,
    enable_cache_dit: bool = True,
    cache_dit_fn: int = 8,
    cache_dit_bn: int = 0,
    cache_dit_enable_taylorseer: bool = False,
    cache_dit_taylorseer_order: int = 1,
    cache_dit_residual_threshold: float = 0.08,
    max_warmup_steps: int = 0,
    enable_cache_logging: Optional[bool] = None,
    cache_logging_mode: Optional[str] = None,
):
    global runtime  # noqa: PLW0603
    runtime = VLLMRuntime(
        model_dir=model_dir,
        num_steps=num_steps,
        dtype=dtype,
        enable_cache_dit=enable_cache_dit,
        cache_dit_fn=cache_dit_fn,
        cache_dit_bn=cache_dit_bn,
        cache_dit_enable_taylorseer=cache_dit_enable_taylorseer,
        cache_dit_taylorseer_order=cache_dit_taylorseer_order,
        cache_dit_residual_threshold=cache_dit_residual_threshold,
        max_warmup_steps=max_warmup_steps,
        enable_cache_logging=enable_cache_logging,
        cache_logging_mode=cache_logging_mode,
    )
    print(f"[vllm_server] Ready! (steps={num_steps}, dtype={dtype})")
    if enable_cache_dit:
        print(f"[vllm_server] cache-dit 配置: Fn={cache_dit_fn}, Bn={cache_dit_bn}, residual_threshold={cache_dit_residual_threshold}, max_warmup_steps={max_warmup_steps}")
        print(f"[vllm_server] TaylorSeer: enabled={cache_dit_enable_taylorseer}, order={cache_dit_taylorseer_order}")
    else:
        print("[vllm_server] cache-dit 已禁用，将执行全量计算。")
    if cache_logging_mode:
        print(f"[vllm_server] cache logging mode: {cache_logging_mode}")
    elif enable_cache_logging is None:
        print("[vllm_server] cache logging: auto (follows backend settings)")
    else:
        log_state = "enabled" if enable_cache_logging else "disabled"
        print(f"[vllm_server] cache logging forced to {log_state}.")


def _serialize_mel(mel: torch.Tensor) -> bytes:
    import numpy as np

    mel_cpu = mel.detach().to("cpu", non_blocking=True)
    mel_np = mel_cpu.numpy()
    response = {
        "shape": mel_np.shape,
        "dtype": str(mel_np.dtype),
        "data": mel_np.tobytes()
    }
    return msgpack.packb(response, use_bin_type=True)


async def _handle_request(socket, client_id, message, sem: asyncio.Semaphore):
    async with sem:
        try:
            dit_input = msgpack.unpackb(message, raw=False)
            rt = _require_runtime()
            loop = asyncio.get_running_loop()
            mel = await loop.run_in_executor(_executor, rt.infer, dit_input)
            serialized_response = _serialize_mel(mel)
            await socket.send_multipart([client_id, serialized_response])
        except Exception as e:
            print(f"[ERROR] Error handling ZeroMQ request: {e}")
            import traceback
            traceback.print_exc()
            try:
                error_response = {"error": str(e)}
                await socket.send_multipart([client_id, json.dumps(error_response).encode()])
            except Exception:
                pass


async def run_zmq_server(socket_address, max_workers: int):
    """运行ZeroMQ服务端（异步 + 线程池推理）"""
    if zmq is None:
        print("[ERROR] ZeroMQ not available. Please install pyzmq first.")
        return

    context = zmq_async.Context()
    socket = context.socket(zmq.ROUTER)  # ROUTER模式支持多客户端

    try:
        socket.bind(socket_address)
        print(f"[vllm_server] ZeroMQ server started, listening on {socket_address}, max_workers={max_workers}")
    except Exception as e:
        print(f"[ERROR] Failed to bind ZeroMQ socket: {e}")
        return

    sem = asyncio.Semaphore(max_workers)

    while True:
        client_id, message = await socket.recv_multipart()
        asyncio.create_task(_handle_request(socket, client_id, message, sem))


def main():
    parser = argparse.ArgumentParser(description="vLLM Omni DiT server")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to CosyVoice checkpoint")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of diffusion steps")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"], help="Computation dtype")
    parser.add_argument("--disable-cache-dit", action="store_true", help="禁用 cache-dit 加速")
    parser.add_argument(
        "--cache-logging",
        type=str,
        choices=["json", "on", "return", "off"],
        default="return",
        help="cache 日志行为: off=禁用缓存统计, return=仅返回缓存信息, on=打印信息并返回, json=打印并写json日志",
    )
    # ZeroMQ相关参数
    parser.add_argument("--zmq-address", type=str, default="ipc:///tmp/vllm.sock", help="ZeroMQ服务端地址")
    
    # cache-dit 配置参数
    parser.add_argument("--cache-dit-fn", type=int, default=8, help="cache-dit: 计算块数量")
    parser.add_argument("--cache-dit-bn", type=int, default=0, help="cache-dit: 批处理块数量")
    parser.add_argument("--cache-dit-enable-taylorseer", action="store_true", default=False, help="cache-dit: 是否启用taylorseer")
    parser.add_argument("--cache-dit-taylorseer-order", type=int, default=1, help="cache-dit: taylorseer阶数")
    parser.add_argument("--cache-dit-residual-threshold", type=float, default=0.08, help="cache-dit: 残差差异阈值")
    parser.add_argument("--max-warmup-steps", type=int, default=0, help="cache-dit: 最大预热步骤数")
    parser.add_argument("--max-workers", type=int, default=4, help="ZeroMQ并发处理线程数")
    
    args = parser.parse_args()
    cache_logging_mode = args.cache_logging
    enable_cache_logging = None
    if args.cache_logging == "json":
        enable_cache_logging = True
    elif args.cache_logging == "off":
        enable_cache_logging = False

    _setup_runtime(
        args.model_dir, 
        args.num_steps, 
        args.dtype,
        enable_cache_dit=not args.disable_cache_dit,
        cache_dit_fn=args.cache_dit_fn,
        cache_dit_bn=args.cache_dit_bn,
        cache_dit_enable_taylorseer=args.cache_dit_enable_taylorseer,
        cache_dit_taylorseer_order=args.cache_dit_taylorseer_order,
        cache_dit_residual_threshold=args.cache_dit_residual_threshold,
        max_warmup_steps=args.max_warmup_steps,
        enable_cache_logging=enable_cache_logging,
        cache_logging_mode=cache_logging_mode,
    )

    # 启动ZeroMQ服务端
    global _executor  # noqa: PLW0603
    _executor = ThreadPoolExecutor(max_workers=args.max_workers)
    asyncio.run(run_zmq_server(args.zmq_address, args.max_workers))


if __name__ == "__main__":
    main()
