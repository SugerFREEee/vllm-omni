"""
python myscripts/vllm_server.py \
  --model-dir /home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --num-steps 10 \
  --host 0.0.0.0 \
  --port 8001
"""

from __future__ import annotations

import argparse
import os
import sys
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
    ):
        self.model_dir = model_dir
        self.num_steps = num_steps
        self.dtype_str = dtype

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
        
        od_config = OmniDiffusionConfig.from_kwargs(
            model=model_path,
            cache_backend="cache_dit",
            cache_config={},
            dtype=torch_dtype,
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

        print(f"[DEBUG] DiT inference")
        print(f"[DEBUG]   condition_vector shape: {len(payload.condition_vector)}x{len(payload.condition_vector[0])}x{len(payload.condition_vector[0][0])}")
        print(f"[DEBUG]   speaker_embedding shape: {len(payload.speaker_embedding)}x{len(payload.speaker_embedding[0])}")
        print(f"[DEBUG]   cond shape: {len(payload.cond)}x{len(payload.cond[0])}x{len(payload.cond[0][0])}")
        print(f"[DEBUG]   seq_len: {payload.seq_len}, mel_len1: {payload.mel_len1}")

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

        # Extract mel from output
        print(f"[DEBUG] Output type: {type(output)}")

        # Handle OmniRequestOutput
        if hasattr(output, 'images') and output.images:
            print(f"[DEBUG] output.images length: {len(output.images)}")
            payload_data = output.images[0]
            print(f"[DEBUG] payload_data type: {type(payload_data)}")

            # Check if it's directly a tensor (our case after modification)
            if isinstance(payload_data, torch.Tensor):
                mel_tensor = payload_data
                print(f"[DEBUG] Got tensor directly! Shape: {mel_tensor.shape}")
                # Extract only the generated part (exclude prompt)
                # mel shape: [batch, mel_dim, total_len]
                mel = mel_tensor[:, :, payload.mel_len1:]
                print(f"[DEBUG] Generated mel shape: {mel.shape}")
                return mel.to(dtype=torch.float32)
            # Or check if it's a dict
            elif isinstance(payload_data, dict):
                print(f"[DEBUG] payload_data keys: {list(payload_data.keys())}")
                if "mel" in payload_data:
                    mel_tensor = payload_data["mel"]
                    if isinstance(mel_tensor, torch.Tensor):
                        print(f"[DEBUG] Full mel shape: {mel_tensor.shape}")
                        mel = mel_tensor[:, :, payload.mel_len1:]
                        print(f"[DEBUG] Generated mel shape: {mel.shape}")
                        return mel.to(dtype=torch.float32)
            else:
                print(f"[DEBUG] payload_data is PIL Image or other type")
                # Check if we can access the raw tensor
                if hasattr(output, '__dict__'):
                    print(f"[DEBUG] output fields: {list(output.__dict__.keys())}")

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


def _setup_runtime(model_dir: str, num_steps: int, dtype: str):
    global runtime  # noqa: PLW0603
    runtime = VLLMRuntime(
        model_dir=model_dir,
        num_steps=num_steps,
        dtype=dtype,
    )
    print(f"[vllm_server] Ready! (steps={num_steps}, dtype={dtype})")


def main():
    parser = argparse.ArgumentParser(description="vLLM Omni DiT server")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to CosyVoice checkpoint")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of diffusion steps")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"], help="Computation dtype")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    args = parser.parse_args()

    _setup_runtime(args.model_dir, args.num_steps, args.dtype)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
