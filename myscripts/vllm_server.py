"""
python myscripts/vllm_server.py \
  --model-dir /home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --num-steps 10 \
  --host 0.0.0.0 \
  --port 8001 \
  --disable-cache-dit

python myscripts/vllm_server.py \
  --model-dir /home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --num-steps 10 \
  --host 0.0.0.0 \
  --port 8001 \
  --cache-dit-fn 1 \
  --cache-dit-bn 0 \
  --cache-dit-residual-threshold 0.4 \
  --cache-logging off

--cache-logging命令行参数有4种 json,on,return,off。off=禁用缓存统计, return=仅返回缓存信息, on=打印信息并返回, json=打印并写json日志
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add paths BEFORE importing vllm_omni
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
project_root = os.path.dirname(current_dir)

sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "third_party", "vllm-omni"))


from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion


class DiTInferRequest(BaseModel):
    """DiT 推理请求，包含预处理好的输入"""
    condition_vector: List[List[List[float]]]  # [batch, mel_dim, seq_len]
    speaker_embedding: List[List[float]]  # [batch, spk_dim]
    cond: List[List[List[float]]]  # [batch, mel_dim, seq_len] - prompt mel features + zeros
    seq_len: int
    mel_len1: int  # prompt mel 长度，用于后处理切分
    batch_size: Optional[int] = 1
    num_inference_steps: Optional[int] = None


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
        self.enable_cache_logging = enable_cache_logging
        self.cache_logging_mode = cache_logging_mode

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
                "max_warmup_steps": 0,
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
    
    def infer(self, payload: DiTInferRequest) -> torch.Tensor:
        """DiT 推理：接收预处理好的 condition_vector 和 speaker_embedding"""
        num_steps = payload.num_inference_steps or self.num_steps

        # print(f"[DEBUG] DiT inference")
        # print(f"[DEBUG]   condition_vector shape: {len(payload.condition_vector)}x{len(payload.condition_vector[0])}x{len(payload.condition_vector[0][0])}")
        # print(f"[DEBUG]   speaker_embedding shape: {len(payload.speaker_embedding)}x{len(payload.speaker_embedding[0])}")
        # print(f"[DEBUG]   cond shape: {len(payload.cond)}x{len(payload.cond[0])}x{len(payload.cond[0][0])}")
        # print(f"[DEBUG]   seq_len: {payload.seq_len}, mel_len1: {payload.mel_len1}")

        perf_start = time.perf_counter()
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
        latency = time.perf_counter() - perf_start
        per_step = latency / num_steps if num_steps else latency
        print(
            f"[vllm_server] Inference finished in {latency:.3f}s "
            f"(~{per_step * 1000:.2f} ms/step for {num_steps} steps)"
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
                    mel = mel_tensor[:, :, payload.mel_len1:]
                    
                    # Print cache hit ratio if cached_steps is available
                    if "cached_steps" in payload_data:
                        cached_steps = payload_data["cached_steps"]
                        if cached_steps is not None:
                            num_steps = payload.num_inference_steps or self.num_steps
                            cache_hit_ratio = cached_steps / num_steps if num_steps > 0 else 0.0
                            print(f"[DEBUG] Cache Hit Ratio: {cached_steps}/{num_steps} ({cache_hit_ratio:.2%})")
                    
                    return mel.to(dtype=torch.float32)
            elif isinstance(payload_data, torch.Tensor):
                # 历史遗留下来的，因为原本payload_data是torch.Tensor，现在改成了 dict
                # (wjs)TODO:删除这个历史遗留
                mel_tensor = payload_data
                mel = mel_tensor[:, :, payload.mel_len1:]
                return mel.to(dtype=torch.float32)
            else:
                print(f"[DEBUG] payload_data is PIL Image or other type")

        raise RuntimeError(f"Failed to extract mel from vllm-omni output. Output type: {type(output)}")


app = FastAPI(title="vLLM Omni DiT Server", version="1.0.0")
runtime: Optional[VLLMRuntime] = None


def _require_runtime() -> VLLMRuntime:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime is not initialized")
    return runtime


@app.get("/health")
def health() -> Dict[str, Any]:
    status = "ready" if runtime else "initializing"
    defaults = {}
    if runtime:
        defaults = {
            "num_inference_steps": runtime.num_steps,
            "dtype": runtime.dtype_str,
        }
    return {"status": status, "defaults": defaults}


@app.post("/infer")
def infer(req: DiTInferRequest) -> Dict[str, Any]:
    rt = _require_runtime()
    try:
        mel = rt.infer(req)
        return {"tts_mel": mel.cpu().tolist(), "mel_dtype": "float32"}
    except Exception as exc:
        import traceback
        print(f"[ERROR] Inference failed:")
        print(f"[ERROR]   {type(exc).__name__}: {str(exc)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {str(exc)}") from exc


@app.post("/echo")
def echo(req: DiTInferRequest) -> Dict[str, Any]:
    """Echo endpoint for testing data transmission. Returns the received data unchanged."""
    print("[DEBUG] Received echo request:")
    print(f"[DEBUG]   condition_vector shape: {len(req.condition_vector)}x{len(req.condition_vector[0])}x{len(req.condition_vector[0][0])}")
    print(f"[DEBUG]   speaker_embedding shape: {len(req.speaker_embedding)}x{len(req.speaker_embedding[0])}")
    print(f"[DEBUG]   cond shape: {len(req.cond)}x{len(req.cond[0])}x{len(req.cond[0][0])}")
    print(f"[DEBUG]   seq_len: {req.seq_len}, mel_len1: {req.mel_len1}")
    
    # Return the received data unchanged
    return {
        "condition_vector": req.condition_vector,
        "speaker_embedding": req.speaker_embedding,
        "cond": req.cond,
        "seq_len": req.seq_len,
        "mel_len1": req.mel_len1,
        "batch_size": req.batch_size,
        "num_inference_steps": req.num_inference_steps
    }


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
        enable_cache_logging=enable_cache_logging,
        cache_logging_mode=cache_logging_mode,
    )
    print(f"[vllm_server] Ready! (steps={num_steps}, dtype={dtype})")
    if enable_cache_dit:
        print(f"[vllm_server] cache-dit 配置: Fn={cache_dit_fn}, Bn={cache_dit_bn}, residual_threshold={cache_dit_residual_threshold}")
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


def main():
    parser = argparse.ArgumentParser(description="vLLM Omni DiT server")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to CosyVoice checkpoint")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of diffusion steps")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"], help="Computation dtype")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--disable-cache-dit", action="store_true", help="禁用 cache-dit 加速")
    parser.add_argument(
        "--cache-logging",
        type=str,
        choices=["json", "on", "return", "off"],
        default="return",
        help="cache 日志行为: off=禁用缓存统计, return=仅返回缓存信息, on=打印信息并返回, json=打印并写json日志",
    )
    
    # cache-dit 配置参数
    parser.add_argument("--cache-dit-fn", type=int, default=8, help="cache-dit: 计算块数量")
    parser.add_argument("--cache-dit-bn", type=int, default=0, help="cache-dit: 批处理块数量")
    parser.add_argument("--cache-dit-enable-taylorseer", action="store_true", default=False, help="cache-dit: 是否启用taylorseer")
    parser.add_argument("--cache-dit-taylorseer-order", type=int, default=1, help="cache-dit: taylorseer阶数")
    parser.add_argument("--cache-dit-residual-threshold", type=float, default=0.08, help="cache-dit: 残差差异阈值")
    
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
        enable_cache_logging=enable_cache_logging,
        cache_logging_mode=cache_logging_mode,
    )

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
