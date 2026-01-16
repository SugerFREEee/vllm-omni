#!/usr/bin/env python3
"""
分析cache-dit日志文件，展示per-step-per-block的残差diff

使用方法:
    python myscripts/analyze_cache_log.py <log_file_path>
    python myscripts/analyze_cache_log.py <log_file_path> --output heatmap.png
    python myscripts/analyze_cache_log.py <log_file_path> --output-dir ./outputs
    python myscripts/analyze_cache_log.py <log_file_path> --output heatmap.png --output-dir ./outputs
"""

import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def parse_cache_log(log_file):
    """解析cache-dit日志文件，提取per-step-per-block的残差diff数据"""
    with open(log_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取配置信息
    config = data.get('cache_config', {})
    
    # 提取per-step-per-block的残差diff数据
    per_step_data = data.get('per_layer_residual_diffs', [])
    
    # 转换为表格格式
    rows = []
    for step_entry in per_step_data:
        step = step_entry.get('step')
        layers = step_entry.get('layers', [])
        
        for layer_entry in layers:
            layer_name = layer_entry.get('layer')
            diff_value = layer_entry.get('value')
            executed = layer_entry.get('executed')
            cached = layer_entry.get('cached')
            
            # 提取层号（如从transformer_block_0提取0）
            try:
                layer_num = int(layer_name.split('_')[-1])
            except (ValueError, IndexError):
                layer_num = -1
            
            rows.append({
                'step': step,
                'layer': layer_name,
                'layer_num': layer_num,
                'diff_value': diff_value,
                'executed': executed,
                'cached': cached
            })
    
    df = pd.DataFrame(rows)
    
    # 按step和layer_num排序
    df = df.sort_values(['step', 'layer_num'])
    
    return df, config


def display_as_table(df, config):
    """以表格形式展示数据"""
    print("=" * 80)
    print("CACHE-DIT 日志分析结果")
    print("=" * 80)
    
    # 显示配置信息
    print("\n配置信息:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 显示统计信息
    print("\n统计信息:")
    print(f"  总步骤数: {df['step'].nunique()}")
    print(f"  总层数: {df['layer'].nunique()}")
    print(f"  平均残差diff: {df['diff_value'].mean():.4f}")
    print(f"  最小残差diff: {df['diff_value'].min():.4f}")
    print(f"  最大残差diff: {df['diff_value'].max():.4f}")
    print(f"  缓存命中率: {df['cached'].sum() / len(df) * 100:.2f}%")
    
    # 显示表格（只显示有diff_value的行）
    print("\n\n每步每层残差diff表格:")
    
    # 创建pivot表格并按layer_num排序行索引
    pivot_df = df.pivot(index='layer', columns='step', values='diff_value')
    
    # 获取layer到layer_num的映射
    layer_to_num = df.set_index('layer')['layer_num'].drop_duplicates().to_dict()
    
    # 按layer_num对行索引进行排序
    sorted_layers = sorted(pivot_df.index, key=lambda x: layer_to_num[x])
    pivot_df = pivot_df.loc[sorted_layers]
    
    print(pivot_df.round(4))
    
    return pivot_df


def plot_heatmap(pivot_df, output_file=None):
    """生成热力图"""
    plt.figure(figsize=(12, 10))
    
    # 只显示有数据的部分
    mask = pivot_df.isnull()
    
    sns.heatmap(
        pivot_df,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        cbar_kws={"label": "residual diff"},
        linewidths=0.5,
        vmin=0.1,
        vmax=0.5  # 根据数据范围调整
    )
    
    plt.title("Cache-Dit per-step-per-block residual diff heatmap")
    
    plt.xlabel(" Step")
    plt.ylabel("Transformer block (Layer)")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n热力图已保存到: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='分析cache-dit日志文件')
    parser.add_argument('log_file', type=str, help='日志文件路径')
    parser.add_argument('--output', type=str, help='热力图输出文件路径（如heatmap.png）')
    parser.add_argument('--output-dir', type=str, help='输出文件目录')
    args = parser.parse_args()
    
    log_file_path = Path(args.log_file)
    if not log_file_path.exists():
        print(f"错误: 文件不存在: {log_file_path}")
        return 1
    
    # 解析日志
    print(f"正在解析日志文件: {log_file_path}")
    df, config = parse_cache_log(log_file_path)
    
    # 显示表格
    pivot_df = display_as_table(df, config)
    
    # 生成热力图
    # 处理输出路径逻辑
    output_file = None
    if args.output:
        output_file = Path(args.output)
    else:
        # 默认使用日志文件名加_heatmap.png
        output_file = Path(f"{log_file_path.stem}_heatmap.png")
    
    # 如果指定了输出目录，将输出文件路径调整到该目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        # 将输出文件放在指定目录
        output_file = output_dir / output_file.name
    
    plot_heatmap(pivot_df, str(output_file))
    
    return 0


if __name__ == '__main__':
    exit(main())
