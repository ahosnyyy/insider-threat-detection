import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_embeddings(model_type: str, method: str = 'tsne', output_dir: Path = Path("results")):
    """Visualize 2D embeddings."""
    models_dir = Path("models")
    emb_path = models_dir / f"embeddings_{model_type}.npy"
    label_path = models_dir / f"labels_{model_type}.npy"
    
    if not emb_path.exists() or not label_path.exists():
        logger.error(f"Embeddings or labels not found for {model_type}. Run evaluate_model.py first.")
        return

    logger.info(f"Loading embeddings from {emb_path}")
    embeddings = np.load(emb_path)
    labels = np.load(label_path)
    
    # Subsample if too large (t-SNE is slow)
    max_samples = 5000
    if len(embeddings) > max_samples:
        logger.info(f"Subsampling {len(embeddings)} -> {max_samples}")
        idx = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        
    logger.info(f"Computing {method.upper()}...")
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    else:
        reducer = PCA(n_components=2)
        
    emb_2d = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot Normal
    normal_mask = (labels == 0)
    plt.scatter(emb_2d[normal_mask, 0], emb_2d[normal_mask, 1], 
                c='blue', alpha=0.3, label='Normal', s=10)
    
    # Plot Insider
    insider_mask = (labels == 1)
    plt.scatter(emb_2d[insider_mask, 0], emb_2d[insider_mask, 1], 
                c='red', alpha=0.8, label='Insider', s=20)
    
    plt.title(f"{model_type.upper()} Embeddings ({method.upper()})")
    plt.legend()
    plt.tight_layout()
    
    save_path = output_dir / f"embeddings_{model_type}_{method}.png"
    plt.savefig(save_path)
    logger.info(f"Plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize model embeddings")
    parser.add_argument("--model", choices=["lstm", "transformer"], required=True)
    parser.add_argument("--method", choices=["tsne", "pca"], default="tsne")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visualize_embeddings(args.model, args.method, args.output_dir)

if __name__ == "__main__":
    main()
