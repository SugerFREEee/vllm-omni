# CosyVoice3 DiT vLLM-Omni Benchmark

## Quick Start

```bash
cd /data/workspace/dit/vllm-omni

# Run benchmark and generate table
python benchmark_vllm.py --output VLLM_RESULTS.md

# Custom configuration
python benchmark_vllm.py --batch-size 2 --seq-len 300 --num-steps 50 --device cuda
```

## Output Format

The benchmark generates TSV tables with the following metrics:

- **TTFT (ms)**: Time to first token
- **Time/step (ms)**: Average time per diffusion step
- **Total Time (s)**: Total inference time
- **Cached Steps**: Number of steps using cache
- **Speedup**: Speedup vs baseline (no cache)

## Test Strategies

1. **baseline**: No cache (Fn=22, Bn=0)
2. **conservative**: Conservative cache (Fn=12, Bn=4, threshold=0.05)
3. **balanced**: Balanced cache (Fn=8, Bn=0, threshold=0.08)
4. **aggressive**: Aggressive cache (Fn=1, Bn=0, threshold=0.15)

Each strategy is tested with 3 TaylorSeer configurations:
- No TaylorSeer
- TaylorSeer order 1
- TaylorSeer order 2

## Example Output

```
TTFT (ms)	No TaylorSeer	TaylorSeer 1	TaylorSeer 2
baseline(Fn=22,Bn=0)	145.23	-	-
aggressive(Fn=1,Bn=0)	126.45	128.31	130.12

Time/step (ms)	No TaylorSeer	TaylorSeer 1	TaylorSeer 2
baseline(Fn=22,Bn=0)	146.53	-	-
aggressive(Fn=1,Bn=0)	21.36	22.15	23.08

Speedup (vs baseline)	No TaylorSeer	TaylorSeer 1	TaylorSeer 2
baseline(Fn=22,Bn=0)	1.00x	-	-
aggressive(Fn=1,Bn=0)	6.86x	6.61x	6.35x
```

## Model Integration

Model location: `vllm_omni/model_executor/models/cosyvoice3/`

Files:
- `cosyvoice3_config.py` - Configuration class
- `cosyvoice3_dit_model.py` - DiT model (193M params)
- `cosyvoice3_dit_vllm.py` - vLLM wrapper with cache-dit
- `__init__.py` - Module exports

Registered in: `vllm_omni/model_executor/models/registry.py`
