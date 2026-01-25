#!/usr/bin/env python
"""
Embedding Visualization Script
Visualize session embeddings using t-SNE/UMAP with insider highlighting.

Usage:
    python scripts/visualize_embeddings.py --model lstm
    python scripts/visualize_embeddings.py --model lstm --method umap --output plots/
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.data import get_session_dataframe, prepare_training_data
from src.models import LSTMAutoencoder, TransformerAutoencoder
from src.utils import setup_logging, load_ground_truth


def load_model(model_path: Path, model_type: str, feature_dim: int):
    """Load saved model checkpoint."""
    if model_type == "lstm":
        model = LSTMAutoencoder(input_dim=feature_dim)
    else:
        model = TransformerAutoencoder(input_dim=feature_dim)
    
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def extract_embeddings(model, sequences, masks, device="cuda"):
    """Extract embeddings from the model's encoder."""
    model = model.to(device)
    model.eval()
    
    embeddings = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seq = torch.FloatTensor(sequences[i:i+batch_size]).to(device)
            batch_mask = torch.FloatTensor(masks[i:i+batch_size]).to(device)
            
            # Get embedding from encoder
            embedding = model.encode(batch_seq, batch_mask)
            # Take final hidden state or mean pool
            if len(embedding.shape) == 3:
                embedding = embedding.mean(dim=1)
            embeddings.append(embedding.cpu().numpy())
    
    return np.concatenate(embeddings, axis=0)


def plot_embeddings_2d(embeddings_2d, labels_session, labels_user, output_path, title=""):
    """Create dual visualization for session and user level labels."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Session-level plot
    ax = axes[0]
    normal_mask = labels_session == 0
    insider_mask = labels_session == 1
    
    ax.scatter(embeddings_2d[normal_mask, 0], embeddings_2d[normal_mask, 1],
               c='#3498db', s=2, alpha=0.3, label=f'Normal ({normal_mask.sum():,})')
    ax.scatter(embeddings_2d[insider_mask, 0], embeddings_2d[insider_mask, 1],
               c='#e74c3c', s=20, alpha=0.8, label=f'Insider ({insider_mask.sum():,})')
    ax.set_title(f"Session-Level Labels\n(Specific Malicious Sessions)", fontsize=12)
    ax.legend(loc='upper right')
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    
    # User-level plot
    ax = axes[1]
    normal_mask = labels_user == 0
    insider_mask = labels_user == 1
    
    ax.scatter(embeddings_2d[normal_mask, 0], embeddings_2d[normal_mask, 1],
               c='#3498db', s=2, alpha=0.3, label=f'Normal Users ({normal_mask.sum():,})')
    ax.scatter(embeddings_2d[insider_mask, 0], embeddings_2d[insider_mask, 1],
               c='#f39c12', s=5, alpha=0.5, label=f'Insider Users ({insider_mask.sum():,})')
    ax.set_title(f"User-Level Labels\n(All Sessions from Insider Users)", fontsize=12)
    ax.legend(loc='upper right')
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize session embeddings")
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--method", choices=["tsne", "umap"], default="tsne")
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/cert.duckdb"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--answers-dir", type=Path, default=Path("data/raw/answers"))
    parser.add_argument("--output", type=Path, default=Path("plots"))
    parser.add_argument("--max-samples", type=int, default=10000, help="Max samples to visualize")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    setup_logging()
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Embedding Visualization ({args.method.upper()})")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df = get_session_dataframe(args.db_path)
    
    ground_truth = load_ground_truth(args.answers_dir, dataset='4.2')
    insider_users = ground_truth.get('insider_users', [])
    insider_incidents = ground_truth.get('insider_incidents', [])
    
    data = prepare_training_data(
        df,
        sequence_length=100,
        insider_users=insider_users,
        insider_incidents=insider_incidents,
    )
    
    test_seq = data['test_sequences']
    test_masks = data['test_masks']
    labels_session = data['test_labels']
    labels_user = data['test_labels_user']
    feature_dim = data['feature_dim']
    
    print(f"Test set: {len(test_seq):,} sequences")
    
    # Subsample if needed
    if len(test_seq) > args.max_samples:
        print(f"Subsampling to {args.max_samples} samples...")
        # Keep all insiders, random sample normals
        insider_idx = np.where(labels_session == 1)[0]
        normal_idx = np.where(labels_session == 0)[0]
        
        n_normal_sample = min(len(normal_idx), args.max_samples - len(insider_idx))
        normal_sample = np.random.choice(normal_idx, n_normal_sample, replace=False)
        sample_idx = np.concatenate([insider_idx, normal_sample])
        np.random.shuffle(sample_idx)
        
        test_seq = test_seq[sample_idx]
        test_masks = test_masks[sample_idx]
        labels_session = labels_session[sample_idx]
        labels_user = labels_user[sample_idx]
    
    # Load model
    model_path = args.models_dir / f"{args.model}_autoencoder_best.pt"
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return 1
    
    print(f"\nLoading {args.model.upper()} model...")
    model = load_model(model_path, args.model, feature_dim)
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, test_seq, test_masks, args.device)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Reduce dimensionality
    print(f"\nApplying {args.method.upper()}...")
    if args.method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        except ImportError:
            print("UMAP not installed, falling back to t-SNE")
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create visualization
    output_file = args.output / f"{args.model}_embeddings_{args.method}.png"
    plot_embeddings_2d(
        embeddings_2d, 
        labels_session, 
        labels_user, 
        output_file,
        title=f"{args.model.upper()} Autoencoder Embeddings ({args.method.upper()})"
    )
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
