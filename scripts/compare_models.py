import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_models(report_path: Path, output_dir: Path):
    """Visualize model comparison."""
    if not report_path.exists():
        logger.error(f"Report not found: {report_path}")
        return 1

    with open(report_path, 'r') as f:
        results = json.load(f)
        
    if not results:
        logger.error("Empty report")
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
            
    if ttd_data:
        df_ttd = pd.DataFrame(ttd_data)
        
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_ttd, x='Model', y='TTD (Hours)', palette='rocket')
        plt.title('Time-to-Detect (Mean Hours)')
        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison_ttd.png")
        logger.info(f"TTD Comparison saved to {output_dir / 'model_comparison_ttd.png'}")
    
    return 0

def main():
    parser = argparse.ArgumentParser(description="Compare models from evaluation report")
    parser.add_argument("--input", type=Path, default=Path("results/evaluation_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = compare_models(args.input, args.output_dir)
    return result if result is not None else 0

if __name__ == "__main__":
    sys.exit(main())
