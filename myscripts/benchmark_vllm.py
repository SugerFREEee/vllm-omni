"""
CosyVoice3 DiT vLLM-Omni Benchmark Table Generator

Usage:
    python benchmark_vllm.py --output VLLM_RESULTS.md
"""

import sys
import os
import torch
import time
import argparse
import importlib.util

# Load modules directly
model_dir = '/data/workspace/dit/vllm-omni/vllm_omni/model_executor/models/cosyvoice3'
sys.path.insert(0, model_dir)

# Load config
spec_config = importlib.util.spec_from_file_location(
    "cosyvoice3_config",
    os.path.join(model_dir, "cosyvoice3_config.py")
)
config_module = importlib.util.module_from_spec(spec_config)
spec_config.loader.exec_module(config_module)
CosyVoice3DiTConfig = config_module.CosyVoice3DiTConfig

# Load vLLM wrapper
spec_vllm = importlib.util.spec_from_file_location(
    "cosyvoice3_dit_vllm",
    os.path.join(model_dir, "cosyvoice3_dit_vllm.py")
)
vllm_module = importlib.util.module_from_spec(spec_vllm)
spec_vllm.loader.exec_module(vllm_module)
CosyVoice3DiTVllm = vllm_module.CosyVoice3DiTVllm


def warmup_model(model, inputs, num_warmup=3):
    """Warmup model"""
    device = next(model.parameters()).device
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(**inputs)
        if device.type == 'cuda':
            torch.cuda.synchronize()


def benchmark_config(
    strategy_name,
    enable_cache,
    Fn,
    Bn,
    threshold,
    max_warmup_steps,
    taylorseer_order=0,
    batch_size=1,
    seq_len=200,
    num_steps=28,
    num_warmup=3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    """Run benchmark for single configuration"""

    # Create config
    config = CosyVoice3DiTConfig(
        hidden_size=1024,
        num_hidden_layers=22,
        num_attention_heads=16,
        mel_dim=80,
        enable_cache_dit=enable_cache,
        cache_Fn=Fn,
        cache_Bn=Bn,
        cache_threshold=threshold,
        cache_warmup_steps=max_warmup_steps,
        num_inference_steps=num_steps,
        enable_taylorseer=(taylorseer_order > 0),
        taylorseer_order=taylorseer_order,
    )

    # Create model
    model = CosyVoice3DiTVllm(config)
    model = model.to(device)
    model.eval()

    # Prepare inputs
    inputs = {
        'hidden_states': torch.randn(batch_size, seq_len, 80, device=device),
        'condition_vector': torch.randn(batch_size, seq_len, 80, device=device),
        'speaker_embedding': torch.randn(batch_size, seq_len, 80, device=device),
        'timesteps': torch.zeros(batch_size, device=device),
    }

    # Warmup
    warmup_model(model, inputs, num_warmup=num_warmup)

    # Refresh cache context
    if enable_cache:
        try:
            model.refresh_cache_context(num_steps)
        except:
            pass

    # Measure TTFT
    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        _ = model(**inputs)
    if device == 'cuda':
        torch.cuda.synchronize()
    ttft = time.perf_counter() - start_time

    # Measure per-step time
    times = []
    for step in range(num_steps):
        t = step / max(num_steps - 1, 1)
        inputs['timesteps'] = torch.full((batch_size,), t * 1000, device=device)

        if device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)
        if device == 'cuda':
            torch.cuda.synchronize()

        step_time = time.perf_counter() - start_time
        times.append(step_time)

    avg_time = sum(times) / len(times)
    total_time = sum(times)

    # Get cache stats
    cached_steps = 0
    if enable_cache:
        try:
            stats = model.get_cache_stats()
            if stats and isinstance(stats, list) and len(stats) > 0:
                cached_steps = len(stats[0].cached_steps) if hasattr(stats[0], 'cached_steps') else 0
        except:
            pass

    # Cleanup
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return {
        'ttft': ttft,
        'avg_time': avg_time,
        'total_time': total_time,
        'cached_steps': cached_steps,
    }


def run_full_benchmark(
    batch_size=1,
    seq_len=200,
    num_steps=28,
    num_warmup=3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    output_file=None,
):
    """Run full benchmark and generate table"""

    # Define strategies
    strategies = {
        'baseline': {
            'enable_cache': False,
            'Fn': 22,
            'Bn': 0,
            'threshold': 0.0,
            'max_warmup_steps': 0,
        },
        'aggressive': {
            'enable_cache': True,
            'Fn': 1,
            'Bn': 0,
            'threshold': 0.15,
            'max_warmup_steps': 4,
        },
        'balanced': {
            'enable_cache': True,
            'Fn': 8,
            'Bn': 0,
            'threshold': 0.08,
            'max_warmup_steps': 8,
        },
        'conservative': {
            'enable_cache': True,
            'Fn': 12,
            'Bn': 4,
            'threshold': 0.05,
            'max_warmup_steps': 10,
        },
    }

    results = {}

    print("\n" + "="*80)
    print("🎯 CosyVoice3 DiT vLLM-Omni Benchmark - Generating Result Table")
    print("="*80)
    print(f"Config: batch={batch_size}, seq={seq_len}, steps={num_steps}, device={device}")
    print("="*80 + "\n")

    strategy_order = ['baseline', 'conservative', 'balanced', 'aggressive']

    for strategy_name in strategy_order:
        config = strategies[strategy_name]
        results[strategy_name] = {}

        print(f"Testing {strategy_name}(Fn={config['Fn']},Bn={config['Bn']})...")

        # Test 3 TaylorSeer configurations
        for ts_order in [0, 1, 2]:
            ts_label = f"TaylorSeer_{ts_order}" if ts_order > 0 else "No_TaylorSeer"
            print(f"  - {ts_label}...", end=" ", flush=True)

            try:
                result = benchmark_config(
                    strategy_name,
                    enable_cache=config['enable_cache'],
                    Fn=config['Fn'],
                    Bn=config['Bn'],
                    threshold=config['threshold'],
                    max_warmup_steps=config['max_warmup_steps'],
                    taylorseer_order=ts_order,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_steps=num_steps,
                    num_warmup=num_warmup,
                    device=device,
                )
                results[strategy_name][ts_label] = result
                print(f"✓ {result['avg_time']*1000:.2f}ms/step")

            except Exception as e:
                print(f"✗ Failed: {e}")
                results[strategy_name][ts_label] = None

    # Generate tables
    print("\n" + "="*80)
    print("📊 BENCHMARK RESULTS TABLE")
    print("="*80 + "\n")

    output_lines = []

    # TTFT table
    output_lines.append("TTFT (ms)\tNo TaylorSeer\tTaylorSeer 1\tTaylorSeer 2")
    for strategy_name in strategy_order:
        row = f"{strategy_name}(Fn={strategies[strategy_name]['Fn']},Bn={strategies[strategy_name]['Bn']})"
        for ts_label in ["No_TaylorSeer", "TaylorSeer_1", "TaylorSeer_2"]:
            result = results[strategy_name].get(ts_label)
            if result:
                row += f"\t{result['ttft']*1000:.2f}"
            else:
                row += "\t-"
        output_lines.append(row)

    output_lines.append("")

    # Time/step table
    output_lines.append("Time/step (ms)\tNo TaylorSeer\tTaylorSeer 1\tTaylorSeer 2")
    for strategy_name in strategy_order:
        row = f"{strategy_name}(Fn={strategies[strategy_name]['Fn']},Bn={strategies[strategy_name]['Bn']})"
        for ts_label in ["No_TaylorSeer", "TaylorSeer_1", "TaylorSeer_2"]:
            result = results[strategy_name].get(ts_label)
            if result:
                row += f"\t{result['avg_time']*1000:.2f}"
            else:
                row += "\t-"
        output_lines.append(row)

    output_lines.append("")

    # Total Time table
    output_lines.append("Total Time (s)\tNo TaylorSeer\tTaylorSeer 1\tTaylorSeer 2")
    for strategy_name in strategy_order:
        row = f"{strategy_name}(Fn={strategies[strategy_name]['Fn']},Bn={strategies[strategy_name]['Bn']})"
        for ts_label in ["No_TaylorSeer", "TaylorSeer_1", "TaylorSeer_2"]:
            result = results[strategy_name].get(ts_label)
            if result:
                row += f"\t{result['total_time']:.3f}"
            else:
                row += "\t-"
        output_lines.append(row)

    output_lines.append("")

    # Cached Steps table
    output_lines.append(f"Cached Steps (of {num_steps})\tNo TaylorSeer\tTaylorSeer 1\tTaylorSeer 2")
    for strategy_name in strategy_order:
        row = f"{strategy_name}(Fn={strategies[strategy_name]['Fn']},Bn={strategies[strategy_name]['Bn']})"
        for ts_label in ["No_TaylorSeer", "TaylorSeer_1", "TaylorSeer_2"]:
            result = results[strategy_name].get(ts_label)
            if result:
                row += f"\t{result['cached_steps']}"
            else:
                row += "\t-"
        output_lines.append(row)

    output_lines.append("")

    # Speedup table (vs baseline)
    baseline_no_ts = results['baseline'].get('No_TaylorSeer')
    if baseline_no_ts:
        output_lines.append("Speedup (vs baseline)\tNo TaylorSeer\tTaylorSeer 1\tTaylorSeer 2")
        for strategy_name in strategy_order:
            row = f"{strategy_name}(Fn={strategies[strategy_name]['Fn']},Bn={strategies[strategy_name]['Bn']})"
            for ts_label in ["No_TaylorSeer", "TaylorSeer_1", "TaylorSeer_2"]:
                result = results[strategy_name].get(ts_label)
                if result:
                    speedup = baseline_no_ts['avg_time'] / result['avg_time']
                    row += f"\t{speedup:.2f}x"
                else:
                    row += "\t-"
            output_lines.append(row)

    # Print to console
    for line in output_lines:
        print(line)

    print("\n" + "="*80)

    # Save to file
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"✓ Results saved to: {output_file}")
        print("="*80 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='CosyVoice3 DiT vLLM-Omni Benchmark Table Generator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device (cuda/cpu)'
    )
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seq-len', type=int, default=200)
    parser.add_argument('--num-steps', type=int, default=28)
    parser.add_argument('--num-warmup', type=int, default=3)
    parser.add_argument(
        '--output',
        type=str,
        default='VLLM_RESULTS_TABLE.md',
        help='Output file path'
    )

    args = parser.parse_args()

    run_full_benchmark(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_steps=args.num_steps,
        num_warmup=args.num_warmup,
        device=args.device,
        output_file=args.output,
    )


if __name__ == '__main__':
    main()
