"""
CosyVoice3 DiT vLLM-Omni Benchmark Table Generator.

This script drives CosyVoice3 DiT through the official Omni diffusion engine
using the new CosyVoice3Pipeline, so cache backends (cache-dit / tea_cache)
work exactly like the built-in image/video pipelines.

Example:
    python myscripts/benchmark_vllm.py --output VLLM_RESULTS.md
"""

import argparse
from typing import Dict, Optional, Tuple

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
from vllm_omni.outputs import OmniRequestOutput


def build_od_config(
    model_path: str,
    cache_backend: str,
    cache_config: Dict,
    dtype: str,
    num_steps: int,
    seq_len: int,
    batch_size: int,
    num_warmup: int,
) -> OmniDiffusionConfig:
    """Construct an OmniDiffusionConfig for CosyVoice benchmarking."""

    od_config = OmniDiffusionConfig.from_kwargs(
        model=model_path,
        cache_backend=cache_backend,
        cache_config=cache_config,
        dtype=dtype,
    )
    od_config.model_class_name = "CosyVoice3Pipeline"
    tf_cfg = {
        "hidden_size": 1024,
        "num_hidden_layers": 22,
        "num_attention_heads": 16,
        "mel_dim": 80,
        "num_inference_steps": num_steps,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_warmup": num_warmup,
    }
    od_config.tf_model_config = TransformerConfig.from_dict(tf_cfg)
    return od_config


def run_strategy(
    model_path: str,
    batch_size: int,
    seq_len: int,
    num_steps: int,
    num_warmup: int,
    cache_backend: str,
    cache_overrides: Dict,
    strategy_cfg: Dict,
    dtype: str,
) -> Optional[Dict]:
    """Execute one benchmark strategy via OmniDiffusion."""

    enable_cache = strategy_cfg["enable_cache"] and cache_backend != "none"
    backend = cache_backend if enable_cache else "none"
    cache_config = {}

    if backend == "cache_dit":
        cache_config.update(
            {
                "Fn_compute_blocks": strategy_cfg["Fn"],
                "Bn_compute_blocks": strategy_cfg["Bn"],
                "max_warmup_steps": strategy_cfg["max_warmup_steps"],
                "residual_diff_threshold": strategy_cfg["threshold"],
                "enable_taylorseer": False,
                "taylorseer_order": 1,
                "num_inference_steps": num_steps,
            }
        )
        cache_config.update(cache_overrides)
    elif backend == "tea_cache":
        cache_config.update(cache_overrides)

    od_config = build_od_config(
        model_path=model_path,
        cache_backend=backend,
        cache_config=cache_config,
        dtype=dtype,
        num_steps=num_steps,
        seq_len=seq_len,
        batch_size=batch_size,
        num_warmup=num_warmup,
    )

    omni = OmniDiffusion(od_config=od_config)
    try:
        output = omni.generate(
            prompt="CosyVoice3-Benchmark",
            num_inference_steps=num_steps,
            extra={"seq_len": seq_len, "batch_size": batch_size, "num_warmup": num_warmup},
        )
    finally:
        omni.close()

    req_output = output
    if isinstance(output, list):
        req_output = output[0]
    if not isinstance(req_output, OmniRequestOutput):
        return None

    # Try to get stats from images field (CosyVoice3Pipeline returns data here)
    payload = None
    if req_output.images:
        payload = req_output.images
    elif req_output.request_output and req_output.request_output[0].images:
        payload = req_output.request_output[0].images

    if not payload:
        return None

    stats = payload[0]
    if not isinstance(stats, dict):
        return None

    return {
        "ttft": stats.get("ttft", 0.0),
        "avg_time": stats.get("avg_time", 0.0),
        "total_time": stats.get("total_time", 0.0),
        "cached_steps": stats.get("cached_steps", 0),
    }


def run_full_benchmark(
    model_path: str,
    batch_size: int,
    seq_len: int,
    num_steps: int,
    num_warmup: int,
    cache_backend: str,
    cache_overrides: Dict,
    dtype: str,
    output_file: Optional[str] = None,
) -> Dict:
    """Run all benchmark configurations and print tables."""

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

    results: Dict[str, Dict[str, Optional[Dict]]] = {}

    print("\n" + "=" * 80)
    print("🎯 CosyVoice3 DiT vLLM-Omni Benchmark - Generating Result Table")
    print("=" * 80)
    print(f"Config: batch={batch_size}, seq={seq_len}, steps={num_steps}, cache_backend={cache_backend}")
    print("=" * 80 + "\n")

    strategy_order = ['baseline', 'conservative', 'balanced', 'aggressive']

    for strategy_name in strategy_order:
        cfg = strategies[strategy_name]
        results[strategy_name] = {}
        print(f"Testing {strategy_name}(Fn={cfg['Fn']},Bn={cfg['Bn']})...")

        for ts_order in [0, 1, 2]:
            label = f"TaylorSeer_{ts_order}" if ts_order > 0 else "No_TaylorSeer"
            print(f"  - {label}...", end=" ", flush=True)
            cfg_copy = cfg.copy()
            cfg_copy['enable_cache'] = cfg['enable_cache'] and (ts_order == 0 or cache_backend != "tea_cache")
            stats = run_strategy(
                model_path=model_path,
                batch_size=batch_size,
                seq_len=seq_len,
                num_steps=num_steps,
                num_warmup=num_warmup,
                cache_backend=cache_backend,
                cache_overrides=cache_overrides,
                strategy_cfg=cfg_copy,
                dtype=dtype,
            )
            results[strategy_name][label] = stats
            if stats:
                print(f"✓ {stats['avg_time'] * 1000:.2f}ms/step")
            else:
                print("✗ Failed")

    # Build tables
    def _collect_row(metric: str, format_spec: str = "{:.2f}") -> Tuple[str, list[str]]:
        row_label = f"{metric}\tNo TaylorSeer\tTaylorSeer 1\tTaylorSeer 2"
        rows = []
        for strategy_name in strategy_order:
            row = f"{strategy_name}(Fn={strategies[strategy_name]['Fn']},Bn={strategies[strategy_name]['Bn']})"
            for label in ["No_TaylorSeer", "TaylorSeer_1", "TaylorSeer_2"]:
                stats = results[strategy_name].get(label)
                if stats and metric in stats:
                    value = stats[metric]
                    if isinstance(value, (float, int)):
                        row += f"\t{format_spec.format(value * (1000 if metric in ['ttft', 'avg_time'] else 1))}"
                    else:
                        row += "\t-"
                else:
                    row += "\t-"
            rows.append(row)
        return row_label, rows

    output_lines = []
    label, rows = _collect_row("ttft")
    output_lines.append("TTFT (ms)\tNo TaylorSeer\tTaylorSeer 1\tTaylorSeer 2")
    output_lines.extend(rows)
    output_lines.append("")

    label, rows = _collect_row("avg_time")
    output_lines.append("Time/step (ms)\tNo TaylorSeer\tTaylorSeer 1\tTaylorSeer 2")
    output_lines.extend(rows)
    output_lines.append("")

    label, rows = _collect_row("total_time", "{:.3f}")
    output_lines.append("Total Time (s)\tNo TaylorSeer\tTaylorSeer 1\tTaylorSeer 2")
    output_lines.extend(rows)
    output_lines.append("")

    output_lines.append(f"Cached Steps (of {num_steps})\tNo TaylorSeer\tTaylorSeer 1\tTaylorSeer 2")
    for strategy_name in strategy_order:
        row = f"{strategy_name}(Fn={strategies[strategy_name]['Fn']},Bn={strategies[strategy_name]['Bn']})"
        for label in ["No_TaylorSeer", "TaylorSeer_1", "TaylorSeer_2"]:
            stats = results[strategy_name].get(label)
            if stats:
                row += f"\t{stats.get('cached_steps', 0)}"
            else:
                row += "\t-"
        output_lines.append(row)

    output_lines.append("")

    baseline_stats = results['baseline'].get('No_TaylorSeer')
    if baseline_stats and baseline_stats.get("avg_time"):
        output_lines.append("Speedup (vs baseline)\tNo TaylorSeer\tTaylorSeer 1\tTaylorSeer 2")
        base_avg = baseline_stats["avg_time"]
        for strategy_name in strategy_order:
            row = f"{strategy_name}(Fn={strategies[strategy_name]['Fn']},Bn={strategies[strategy_name]['Bn']})"
            for label in ["No_TaylorSeer", "TaylorSeer_1", "TaylorSeer_2"]:
                stats = results[strategy_name].get(label)
                if stats and stats.get("avg_time"):
                    row += f"\t{base_avg / stats['avg_time']:.2f}x"
                else:
                    row += "\t-"
            output_lines.append(row)

    for line in output_lines:
        print(line)

    print("\n" + "=" * 80)

    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"✓ Results saved to: {output_file}")
        print("=" * 80 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='CosyVoice3 DiT vLLM-Omni Benchmark Table Generator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='/home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
        help='Path to CosyVoice3 directory or flow.pt checkpoint'
    )
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seq-len', type=int, default=200)
    parser.add_argument('--num-steps', type=int, default=10)
    parser.add_argument('--num-warmup', type=int, default=3)
    parser.add_argument(
        '--cache-backend',
        type=str,
        default='cache_dit',
        choices=['cache_dit', 'tea_cache', 'none'],
        help='Use vLLM-Omni cache backend (cache_dit/tea_cache) to accelerate CosyVoice DiT',
    )
    parser.add_argument(
        '--cache-dit-max-continuous-cached-steps',
        type=int,
        default=3,
        help='cache-dit: maximum continuous cached steps',
    )
    parser.add_argument(
        '--cache-dit-scm-mask-policy',
        type=str,
        default=None,
        choices=[None, 'slow', 'medium', 'fast', 'ultra'],
        help='cache-dit: SCM mask policy (None disables step masking)',
    )
    parser.add_argument(
        '--cache-dit-scm-steps-policy',
        type=str,
        default='dynamic',
        choices=['dynamic', 'static'],
        help='cache-dit: SCM steps policy (only used when mask policy is set)',
    )
    parser.add_argument(
        '--tea-cache-rel-l1-thresh',
        type=float,
        default=0.2,
        help='tea_cache: accumulated relative L1 threshold',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='bfloat16',
        help='Torch dtype for the OmniDiffusionConfig (e.g., bfloat16/float16)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='VLLM_RESULTS_TABLE.md',
        help='Output file path'
    )

    args = parser.parse_args()

    cache_overrides: Dict = {}
    if args.cache_backend == 'cache_dit':
        cache_overrides['max_continuous_cached_steps'] = args.cache_dit_max_continuous_cached_steps
        if args.cache_dit_scm_mask_policy:
            cache_overrides['scm_steps_mask_policy'] = args.cache_dit_scm_mask_policy
            cache_overrides['scm_steps_policy'] = args.cache_dit_scm_steps_policy
    elif args.cache_backend == 'tea_cache':
        cache_overrides['rel_l1_thresh'] = args.tea_cache_rel_l1_thresh

    run_full_benchmark(
        model_path=args.model_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_steps=args.num_steps,
        num_warmup=args.num_warmup,
        cache_backend=args.cache_backend,
        cache_overrides=cache_overrides,
        dtype=args.dtype,
        output_file=args.output,
    )


if __name__ == '__main__':
    main()
