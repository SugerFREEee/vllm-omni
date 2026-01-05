"""
CosyVoice3 DiT 性能基准测试表生成器

python benchmark_table.py --output MY_RESULTS.md
"""

import torch
import time
import argparse
from pathlib import Path
import cache_dit
from cache_dit import BlockAdapter, ForwardPattern, DBCacheConfig, TaylorSeerCalibratorConfig
from cosyvoice3_dit_model import CosyVoice3DiT


def warmup_model(model, inputs, num_warmup: int = 3, verbose: bool = False):
    """Warmup 模型"""
    device = next(model.parameters()).device
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(**inputs)
        if device.type == 'cuda':
            torch.cuda.synchronize()


def benchmark_config(
    model_path: str,
    strategy_name: str,
    Fn: int,
    Bn: int,
    threshold: float,
    max_warmup_steps: int,
    taylorseer_order: int = 0,  # 0 = 不使用 TaylorSeer
    batch_size: int = 1,
    seq_len: int = 200,
    num_steps: int = 28,
    num_warmup: int = 3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """运行单个配置的基准测试"""

    # 加载模型
    model = CosyVoice3DiT.from_pretrained(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    # 配置 cache-dit
    adapter = BlockAdapter(
        pipe=None,
        transformer=model,
        blocks=model.blocks,
        forward_pattern=ForwardPattern.Pattern_3,
    )

    cache_config = DBCacheConfig(
        Fn_compute_blocks=Fn,
        Bn_compute_blocks=Bn,
        residual_diff_threshold=threshold,
        max_warmup_steps=max_warmup_steps,
        max_cached_steps=-1,
        max_continuous_cached_steps=-1,
        num_inference_steps=num_steps,
    )

    calibrator_config = None
    if taylorseer_order > 0:
        calibrator_config = TaylorSeerCalibratorConfig(
            enable_calibrator=True,
            taylorseer_order=taylorseer_order,
        )

    cache_dit.enable_cache(adapter, cache_config=cache_config, calibrator_config=calibrator_config)

    # 准备输入
    x = torch.randn(batch_size, seq_len, 80, device=device)
    mu = torch.randn(batch_size, seq_len, 80, device=device)
    spk = torch.randn(batch_size, seq_len, 80, device=device)
    timesteps = torch.zeros(batch_size, device=device)

    inputs = {
        'x': x,
        'timesteps': timesteps,
        'mu': mu,
        'spk': spk,
    }

    # Warmup
    warmup_model(model, inputs, num_warmup=num_warmup)

    # 刷新 cache context
    try:
        cache_dit.refresh_context(model, num_inference_steps=num_steps, verbose=False)
    except:
        pass

    # 测量 TTFT
    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        _ = model(**inputs)
    if device == 'cuda':
        torch.cuda.synchronize()
    ttft = time.perf_counter() - start_time

    # 测量每步推理时间
    times = []
    for step in range(num_steps):
        t = step / max(num_steps - 1, 1)
        timesteps_step = torch.full((inputs['x'].shape[0],), t * 1000, device=device)
        inputs_step = {**inputs, 'timesteps': timesteps_step}

        if device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs_step)
        if device == 'cuda':
            torch.cuda.synchronize()

        step_time = time.perf_counter() - start_time
        times.append(step_time)

    avg_time = sum(times) / len(times)
    total_time = sum(times)

    # 获取缓存统计
    try:
        stats = cache_dit.summary(adapter, details=False)
        if stats and isinstance(stats, list) and len(stats) > 0:
            cached_steps = len(stats[0].cached_steps) if hasattr(stats[0], 'cached_steps') else 0
        else:
            cached_steps = 0
    except:
        cached_steps = 0

    # 清理
    try:
        cache_dit.disable_cache(adapter)
    except:
        pass

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
    model_path: str,
    batch_size: int = 1,
    seq_len: int = 200,
    num_steps: int = 28,
    num_warmup: int = 3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    output_file: str = None,
):
    """运行完整基准测试并生成表格"""

    # 定义测试策略
    strategies = {
        'baseline': {
            'Fn': 22,  # 所有 blocks
            'Bn': 0,
            'threshold': 0.0,  # 不触发缓存
            'max_warmup_steps': 0,
        },
        'aggressive': {
            'Fn': 1,
            'Bn': 0,
            'threshold': 0.15,
            'max_warmup_steps': 4,
        },
        'balanced': {
            'Fn': 8,
            'Bn': 0,
            'threshold': 0.08,
            'max_warmup_steps': 8,
        },
        'conservative': {
            'Fn': 12,
            'Bn': 4,
            'threshold': 0.05,
            'max_warmup_steps': 10,
        },
    }

    # 存储结果
    results = {}

    print("\n" + "="*80)
    print("🎯 CosyVoice3 DiT Benchmark - Generating Result Table")
    print("="*80)
    print(f"Config: batch={batch_size}, seq={seq_len}, steps={num_steps}, device={device}")
    print("="*80 + "\n")

    strategy_order = ['baseline', 'conservative', 'balanced', 'aggressive']

    for strategy_name in strategy_order:
        config = strategies[strategy_name]
        results[strategy_name] = {}

        print(f"Testing {strategy_name}(Fn={config['Fn']},Bn={config['Bn']})...")

        # 测试 3 种配置：无 TaylorSeer, order=1, order=2
        for ts_order in [0, 1, 2]:
            ts_label = f"TaylorSeer_{ts_order}" if ts_order > 0 else "No_TaylorSeer"
            print(f"  - {ts_label}...", end=" ", flush=True)

            try:
                result = benchmark_config(
                    model_path,
                    strategy_name,
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

    # 生成表格
    print("\n" + "="*80)
    print("📊 BENCHMARK RESULTS TABLE")
    print("="*80 + "\n")

    output_lines = []

    # TTFT 表
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

    # Time/step 表
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

    # Total Time 表
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

    # Cached Steps 表
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

    # Speedup 表（相对于 baseline）
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

    # 打印到控制台
    for line in output_lines:
        print(line)

    print("\n" + "="*80)

    # 保存到文件
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"✓ Results saved to: {output_file}")
        print("="*80 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='CosyVoice3 DiT Benchmark Table Generator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='/data/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/flow.pt',
        help='Path to flow.pt'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device (cuda/cpu)'
    )
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seq-len', type=int, default=200)
    parser.add_argument('--num-steps', type=int, default=10)
    parser.add_argument('--num-warmup', type=int, default=3)
    parser.add_argument(
        '--output',
        type=str,
        default='RESULTS_TABLE.md',
        help='Output file path'
    )

    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"❌ Model not found: {args.model_path}")
        return

    run_full_benchmark(
        args.model_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_steps=args.num_steps,
        num_warmup=args.num_warmup,
        device=args.device,
        output_file=args.output,
    )


if __name__ == '__main__':
    main()
