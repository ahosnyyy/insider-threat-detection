import argparse
import json
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_reports(input_path: Path):
    """Load one or more reports from directory or file."""
    results = {}
    
    if input_path.is_file():
        # Load single file
        with open(input_path, 'r') as f:
            data = json.load(f)
            results.update(data)
    elif input_path.is_dir():
        # Look for evaluation_report.json in subdirectories
        for report_file in input_path.glob("*/evaluation_report.json"):
            try:
                with open(report_file, 'r') as f:
                    data = json.load(f)
                    results.update(data)
            except Exception as e:
                logger.warning(f"Could not load {report_file}: {e}")
                
        # Also check root if exists
        root_report = input_path / "evaluation_report.json"
        if root_report.exists():
            with open(root_report, 'r') as f:
                data = json.load(f)
                results.update(data)
                
    return results

def compare_models(input_path: Path, output_dir: Path):
    """Visualize model comparison."""
    results = load_reports(input_path)
        
    if not results:
        logger.error(f"No results found at {input_path}")
        return 1
        
    logger.info(f"Comparing models: {list(results.keys())}")
    
    # metrics to plot
    metrics = ['auc_roc', 'f1_score', 'precision', 'recall']
    
    # Prepare data for plotting
    data = []
    for model_name, res in results.items():
        for level in ['session_level', 'user_level']:
            for metric in metrics:
                val = res[level].get(metric, 0)
                data.append({
                    'Model': model_name,
                    'Level': level.replace('_', ' ').title(),
                    'Metric': metric.upper(),
                    'Score': val
                })
                
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # FacetGrid for Metric? Or just hue?
    # Simple bar chart: x=Metric, y=Score, hue=Model, col=Level
    
    g = sns.catplot(
        data=df, 
        x='Metric', 
        y='Score', 
        hue='Model', 
        col='Level', 
        kind='bar',
        height=5, 
        aspect=1.2,
        palette='viridis'
    )
    
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Model Comparison: Phase 1 (Autoencoders)')
    
    save_path = output_dir / "model_comparison.png"
    plt.savefig(save_path)
    logger.info(f"Comparison plot saved to {save_path}")
    
    # TTD Comparison if available
    ttd_data = []
    for model_name, res in results.items():
        if 'ttd' in res:
            ttd = res['ttd']
            ttd_data.append({
                'Model': model_name,
                'TTD (Hours)': ttd.get('ttd_hours_mean', 0),
                'TTD (Sessions)': ttd.get('ttd_sessions_mean', 0)
            })
            
    # Performance Comparison
    perf_data = []
    for model_name, res in results.items():
        if 'performance' in res:
            perf = res['performance']
            perf_data.append({
                'Model': model_name,
                'Throughput (sessions/sec)': perf.get('throughput_sessions_per_sec', 0),
                'Latency (ms)': perf.get('latency_ms_per_session', 0),
                'Memory (MB)': perf.get('peak_cpu_memory_mb', 0)
            })

    if ttd_data:
        df_ttd = pd.DataFrame(ttd_data)
        
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_ttd, x='Model', y='TTD (Hours)', hue='Model', palette='rocket', legend=False)
        plt.title('Time-to-Detect (Mean Hours)')
        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison_ttd.png")
        logger.info(f"TTD Comparison saved to {output_dir / 'model_comparison_ttd.png'}")
        
    if perf_data:
        df_perf = pd.DataFrame(perf_data)
        
        # Plot Throughput
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_perf, x='Model', y='Throughput (sessions/sec)', hue='Model', palette='Greens', legend=False)
        plt.title('Inference Throughput (Higher is Better)')
        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison_throughput.png")
        logger.info(f"Throughput plot saved to {output_dir / 'model_comparison_throughput.png'}")
        
        # Plot Memory
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_perf, x='Model', y='Memory (MB)', hue='Model', palette='Blues', legend=False)
        plt.title('Peak Memory Usage (Lower is Better)')
        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison_memory.png")
        logger.info(f"Memory plot saved to {output_dir / 'model_comparison_memory.png'}")
    
    return 0

def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Compare models from evaluation report")
    
    # Use config output_path but point to results directory usually
    # Assuming report is in results/evaluation_report.json relative to project root
    # But cfg['data']['output_path'] is data/outputs
    
    parser.add_argument("--input", type=Path, default=Path("results"),
                        help="Path to results directory or specific report file")
    parser.add_argument("--output-dir", type=Path, default=Path("results/comparison"),
                        help="Directory to save plots")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = compare_models(args.input, args.output_dir)
    return result if result is not None else 0

if __name__ == "__main__":
    sys.exit(main())
