"""
Evaluation Module
Compute metrics and compare models.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation metrics."""
    model_name: str
    val_mse: float
    auc_roc: float
    average_precision: float
    precision_at_100: float
    recall_at_fpr_5: float
    optimal_threshold: float
    f1_at_optimal: float
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ModelEvaluator:
    """Evaluate autoencoder models on insider threat detection."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def compute_reconstruction_errors(
        self,
        sequences: np.ndarray,
        masks: np.ndarray,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Compute reconstruction error for all sequences.
        
        Returns:
            errors: (N,) array of reconstruction errors
        """
        dataset = TensorDataset(
            torch.FloatTensor(sequences),
            torch.FloatTensor(masks),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_errors = []
        
        with torch.no_grad():
            for seq_batch, mask_batch in tqdm(loader, desc="Computing errors"):
                seq_batch = seq_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                
                errors = self.model.get_reconstruction_error(seq_batch, mask_batch)
                all_errors.append(errors.cpu().numpy())
        
        return np.concatenate(all_errors)
    
    def compute_embeddings(
        self,
        sequences: np.ndarray,
        masks: np.ndarray,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Compute embeddings for all sequences.
        
        Returns:
            embeddings: (N, embedding_dim) array
        """
        dataset = TensorDataset(
            torch.FloatTensor(sequences),
            torch.FloatTensor(masks),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_embeddings = []
        
        with torch.no_grad():
            for seq_batch, mask_batch in loader:
                seq_batch = seq_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                
                embeddings = self.model.get_embedding(seq_batch, mask_batch)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings)
    
    def evaluate(
        self,
        sequences: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
        model_name: str = "model",
    ) -> EvaluationResults:
        """
        Full evaluation with ground truth labels.
        
        Args:
            sequences: (N, seq_len, feature_dim)
            masks: (N, seq_len)
            labels: (N,) binary labels (1 = insider/anomaly)
            model_name: Name for reporting
            
        Returns:
            EvaluationResults with all metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Compute reconstruction errors
        errors = self.compute_reconstruction_errors(sequences, masks)
        
        # Validation MSE (on normal samples only)
        normal_mask = labels == 0
        val_mse = errors[normal_mask].mean() if normal_mask.sum() > 0 else errors.mean()
        
        # AUC-ROC
        auc_roc = roc_auc_score(labels, errors)
        
        # Average Precision
        avg_precision = average_precision_score(labels, errors)
        
        # Precision@100 (top 100 highest errors)
        sorted_indices = np.argsort(errors)[::-1]
        top_100_labels = labels[sorted_indices[:100]]
        precision_at_100 = top_100_labels.sum() / 100
        
        # Recall@FPR=5%
        fpr, tpr, thresholds = roc_curve(labels, errors)
        idx_5_fpr = np.argmin(np.abs(fpr - 0.05))
        recall_at_fpr_5 = tpr[idx_5_fpr]
        
        # Optimal threshold (max F1)
        precision, recall, pr_thresholds = precision_recall_curve(labels, errors)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores[:-1])  # Last element is undefined
        optimal_threshold = pr_thresholds[optimal_idx]
        f1_at_optimal = f1_scores[optimal_idx]
        
        results = EvaluationResults(
            model_name=model_name,
            val_mse=float(val_mse),
            auc_roc=float(auc_roc),
            average_precision=float(avg_precision),
            precision_at_100=float(precision_at_100),
            recall_at_fpr_5=float(recall_at_fpr_5),
            optimal_threshold=float(optimal_threshold),
            f1_at_optimal=float(f1_at_optimal),
        )
        
        logger.info(f"Results: AUC-ROC={auc_roc:.4f}, P@100={precision_at_100:.4f}, R@FPR5%={recall_at_fpr_5:.4f}")
        
        return results


def compare_models(
    models: Dict[str, nn.Module],
    sequences: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    output_path: Optional[Path] = None,
) -> Dict[str, EvaluationResults]:
    """
    Compare multiple models.
    
    Args:
        models: Dict mapping model name to model
        sequences, masks, labels: Test data
        output_path: Optional path to save comparison JSON
        
    Returns:
        Dict mapping model name to results
    """
    results = {}
    
    for name, model in models.items():
        evaluator = ModelEvaluator(model)
        results[name] = evaluator.evaluate(sequences, masks, labels, name)
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model':<20} {'AUC-ROC':>10} {'P@100':>10} {'R@FPR5%':>10} {'F1':>10}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<20} {r.auc_roc:>10.4f} {r.precision_at_100:>10.4f} {r.recall_at_fpr_5:>10.4f} {r.f1_at_optimal:>10.4f}")
    print("=" * 80)
    
    # Save if path provided
    if output_path:
        comparison = {name: r.to_dict() for name, r in results.items()}
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Saved comparison to {output_path}")
    
    return results


def compute_percentiles(
    errors: np.ndarray,
    percentiles: List[float] = [90, 95, 99],
) -> Dict[int, float]:
    """Compute percentile thresholds for reconstruction errors."""
    return {p: float(np.percentile(errors, p)) for p in percentiles}


if __name__ == "__main__":
    from src.models import LSTMAutoencoder
    
    # Quick test
    model = LSTMAutoencoder(input_dim=24)
    
    n_samples = 100
    sequences = np.random.randn(n_samples, 50, 24).astype(np.float32)
    masks = np.ones((n_samples, 50), dtype=np.float32)
    labels = np.random.binomial(1, 0.1, n_samples)  # 10% anomalies
    
    evaluator = ModelEvaluator(model)
    results = evaluator.evaluate(sequences, masks, labels, "test_lstm")
    print(f"\nResults: {results}")
