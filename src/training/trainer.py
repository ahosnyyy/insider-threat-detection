"""
Training Module
Training loop with early stopping and checkpoint saving.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from .metrics import compute_ttd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0  # L2 regularization (e.g. 1e-5) for better validation loss
    warmup_epochs: int = 0  # Linear LR warm-up for first N epochs, then ReduceLROnPlateau
    patience: int = 10
    min_delta: float = 0.0001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path = Path("models")
    log_dir: Path = Path("runs")  # TensorBoard log directory
    tensorboard: bool = True  # Enable TensorBoard logging
    eval_every: int = 5  # Evaluate on test set every N epochs
    # Thresholding strategy for converting reconstruction error to anomaly label.
    # - "percentile": use fixed percentile of error distribution
    # - "f1_optimal": choose threshold that maximizes F1 on the evaluated set
    threshold_method: str = "percentile"
    threshold_percentile: float = 95.0  # Percentile for anomaly threshold (fallback/default)
    accumulation_steps: int = 1  # Gradient accumulation steps
    

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """Trainer for autoencoder models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        model_name: str = "autoencoder",
    ):
        self.model = model.to(config.device)
        self.config = config
        self.model_name = model_name
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        self._base_lr = config.learning_rate
        self._warmup_epochs = config.warmup_epochs
        self.criterion = nn.MSELoss(reduction='none')
        self.early_stopping = EarlyStopping(config.patience, config.min_delta)
        
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard setup
        self.writer = None
        if config.tensorboard:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = config.log_dir / f"{model_name}_{timestamp}"
            log_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_path))
            logger.info(f"TensorBoard logs: {log_path}")
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_time': [],
            # Session-level classification metrics
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'fpr': [],
            'auc_roc': [],
            # Confusion matrix values (session-level)
            'tp': [],
            'tn': [],
            'fp': [],
            'fn': [],
            # User-level metrics
            'user_accuracy': [],
            'user_precision': [],
            'user_recall': [],
            'user_f1_score': [],
            'user_fpr': [],
            'user_auc_roc': [],
            # Confusion matrix values (user-level)
            'user_tp': [],
            'user_tn': [],
            'user_fp': [],
            'user_fn': [],
            # Time-to-Detect metrics
            'ttd_hours_mean': [],
            'ttd_hours_median': [],
            'ttd_sessions_mean': [],
            'ttd_lag_mean': [],
            'pct_first_session': [],
            'n_incidents_detected': [],
            # Total training time
            'total_training_time': 0.0,
        }
    
    def train(
        self,
        train_sequences: np.ndarray,
        train_masks: np.ndarray,
        val_sequences: np.ndarray,
        val_masks: np.ndarray,
        test_sequences: Optional[np.ndarray] = None,
        test_masks: Optional[np.ndarray] = None,
        test_labels: Optional[np.ndarray] = None,
        test_labels_user: Optional[np.ndarray] = None,
        test_session_ids: Optional[List[str]] = None,
        test_timestamps: Optional[List] = None,
        test_user_ids: Optional[List[str]] = None,
        insider_incidents: Optional[List[Dict]] = None,
    ) -> Dict[str, list]:
        """
        Train the model with periodic evaluation on test set.
        
        Args:
            train_sequences: (N, seq_len, feature_dim)
            train_masks: (N, seq_len)
            val_sequences: Validation sequences (normal only)
            val_masks: Validation masks
            test_sequences: Test sequences (normal + insider)
            test_masks: Test masks
            test_labels: Test labels (session-level)
            test_labels_user: Test labels (user-level)
            test_session_ids: Session IDs for test sequences
            test_timestamps: Timestamps for test sessions (for TTD)
            insider_incidents: List of incident dicts with user, start, end
            
        Returns:
            Training history with classification metrics and TTD
        """
        # Store for TTD calculation
        self.test_session_ids = test_session_ids
        self.test_timestamps = test_timestamps
        self.insider_incidents = insider_incidents
        
        # Create data loaders
        train_loader = self._create_loader(train_sequences, train_masks, shuffle=True)
        val_loader = self._create_loader(val_sequences, val_masks, shuffle=False)
        
        # Test loader if provided
        has_test_data = test_sequences is not None and test_labels is not None
        if has_test_data:
            test_loader = self._create_loader(test_sequences, test_masks, shuffle=False)
            logger.info(f"Test set: {len(test_sequences)} samples, {test_labels.sum()} insiders")
        
        logger.info(f"Starting training on {self.config.device}")
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        best_val_loss = float('inf')
        train_start_time = time.time()
        
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            # Train epoch
            train_loss = self._train_epoch(train_loader)
            
            # Validation
            val_loss = self._validate(val_loader)
            
            epoch_time = time.time() - start_time
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epoch_time'].append(epoch_time)
            
            # Learning rate: warm-up then ReduceLROnPlateau
            if self._warmup_epochs > 0 and epoch < self._warmup_epochs:
                warmup_lr = self._base_lr * (epoch + 1) / self._warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            else:
                self.scheduler.step(val_loss)
            
            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"Time: {epoch_time:.1f}s, LR: {current_lr:.6f}"
            )
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalars('Loss', {
                    'train': train_loss,
                    'validation': val_loss,
                }, epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                self.writer.add_scalar('Epoch_Time', epoch_time, epoch)
            
            # Periodic evaluation on test set
            if has_test_data and (epoch + 1) % self.config.eval_every == 0:
                # 1. Evaluate Session-Level Metrics (Default)
                metrics_session = self._evaluate_test_set(test_loader, test_labels)
                
                # 2. Evaluate User-Level Metrics (Secondary)
                metrics_user = None
                if test_labels_user is not None:
                    # Reuse predictions if possible or re-evaluate
                    # Ideally we'd reuse raw errors, but _evaluate_test_set encapsulates that.
                    # For simplicity, we re-run eval with user labels (low cost since inference is cached/fast enough)
                    metrics_user = self._evaluate_test_set(test_loader, test_labels_user)

                # Store session metrics in history (primary)
                self.history['accuracy'].append(metrics_session['accuracy'])
                self.history['precision'].append(metrics_session['precision'])
                self.history['recall'].append(metrics_session['recall'])
                self.history['f1_score'].append(metrics_session['f1_score'])
                self.history['fpr'].append(metrics_session['fpr'])
                self.history['auc_roc'].append(metrics_session['auc_roc'])
                self.history['tp'].append(metrics_session['tp'])
                self.history['tn'].append(metrics_session['tn'])
                self.history['fp'].append(metrics_session['fp'])
                self.history['fn'].append(metrics_session['fn'])
                
                # Store user-level metrics in history
                if metrics_user:
                    self.history['user_accuracy'].append(metrics_user['accuracy'])
                    self.history['user_precision'].append(metrics_user['precision'])
                    self.history['user_recall'].append(metrics_user['recall'])
                    self.history['user_f1_score'].append(metrics_user['f1_score'])
                    self.history['user_fpr'].append(metrics_user['fpr'])
                    self.history['user_auc_roc'].append(metrics_user['auc_roc'])
                    self.history['user_tp'].append(metrics_user['tp'])
                    self.history['user_tn'].append(metrics_user['tn'])
                    self.history['user_fp'].append(metrics_user['fp'])
                    self.history['user_fn'].append(metrics_user['fn'])
                
                # Compute Time-to-Detect metrics (always uses session predictions)
                ttd_metrics = self._compute_ttd(
                    metrics_session['predictions'], 
                    test_labels,
                    user_ids=test_user_ids
                )
                self.history['ttd_hours_mean'].append(ttd_metrics['ttd_hours_mean'])
                self.history['ttd_hours_median'].append(ttd_metrics['ttd_hours_median'])
                self.history['ttd_sessions_mean'].append(ttd_metrics['ttd_sessions_mean'])
                self.history['ttd_lag_mean'].append(ttd_metrics['ttd_lag_mean'])
                self.history['pct_first_session'].append(ttd_metrics.get('pct_first_session', 0.0))
                self.history['n_incidents_detected'].append(ttd_metrics.get('n_incidents_detected', 0))
                
                # Log Session Metrics
                logger.info(f"  Session Metrics: AUC={metrics_session['auc_roc']:.4f}, Acc={metrics_session['accuracy']:.4f}, P={metrics_session['precision']:.4f}, R={metrics_session['recall']:.4f}, F1={metrics_session['f1_score']:.4f}, FPR={metrics_session['fpr']:.4f}")
                logger.debug(f"    Confusion: TP={metrics_session['tp']}, FP={metrics_session['fp']}, FN={metrics_session['fn']}, TN={metrics_session['tn']}")
                
                # Log User Metrics
                if metrics_user:
                    logger.info(f"  User Metrics:    AUC={metrics_user['auc_roc']:.4f}, Acc={metrics_user['accuracy']:.4f}, P={metrics_user['precision']:.4f}, R={metrics_user['recall']:.4f}, F1={metrics_user['f1_score']:.4f}, FPR={metrics_user['fpr']:.4f}")
                    logger.info(f"    User Confusion: TP={metrics_user['tp']}, TN={metrics_user['tn']}, FP={metrics_user['fp']}, FN={metrics_user['fn']}")

                # Log TTD
                if ttd_metrics.get('n_incidents_detected', 0) > 0:
                    logger.info(
                        f"  Time-to-Detect: {ttd_metrics['ttd_hours_mean']:.1f}h mean, "
                        f"{ttd_metrics['pct_first_session']:.1f}% first session"
                    )
                
                # TensorBoard logging
                if self.writer:
                    # Session Level
                    self.writer.add_scalars('Classification/Session', {
                        'accuracy': metrics_session['accuracy'],
                        'precision': metrics_session['precision'],
                        'recall': metrics_session['recall'],
                        'f1_score': metrics_session['f1_score'],
                        'auc_roc': metrics_session['auc_roc'],
                        'fpr': metrics_session['fpr'],
                    }, epoch)
                    
                    # User Level
                    if metrics_user:
                        self.writer.add_scalars('Classification/User', {
                            'accuracy': metrics_user['accuracy'],
                            'precision': metrics_user['precision'],
                            'recall': metrics_user['recall'],
                            'f1_score': metrics_user['f1_score'],
                            'auc_roc': metrics_user['auc_roc'],
                            'fpr': metrics_user['fpr'],
                        }, epoch)
                    
                    # TTD
                    self.writer.add_scalars('Time_to_Detect', {
                        'hours_mean': ttd_metrics['ttd_hours_mean'],
                        'hours_median': ttd_metrics['ttd_hours_median'],
                        'sessions_mean': ttd_metrics['ttd_sessions_mean'],
                        'lag_percent': ttd_metrics['ttd_lag_mean'],
                        'pct_first_session': ttd_metrics['pct_first_session'],
                    }, epoch)
                    
                    # Confusion Matrix (Session)
                    self.writer.add_scalars('Confusion_Matrix/Session', {
                        'TP': metrics_session['tp'], 'TN': metrics_session['tn'],
                        'FP': metrics_session['fp'], 'FN': metrics_session['fn'],
                    }, epoch)
                    
                    cm_fig = self._create_confusion_matrix_figure(
                        metrics_session['tp'], metrics_session['tn'], metrics_session['fp'], metrics_session['fn']
                    )
                    self.writer.add_figure('Confusion_Matrix_Plot/Session', cm_fig, epoch)
                    
                    # Confusion Matrix (User)
                    if metrics_user:
                        self.writer.add_scalars('Confusion_Matrix/User', {
                            'TP': metrics_user['tp'], 'TN': metrics_user['tn'],
                            'FP': metrics_user['fp'], 'FN': metrics_user['fn'],
                        }, epoch)
                        
                        cm_fig_user = self._create_confusion_matrix_figure(
                            metrics_user['tp'], metrics_user['tn'], metrics_user['fp'], metrics_user['fn']
                        )
                        self.writer.add_figure('Confusion_Matrix_Plot/User', cm_fig_user, epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=True)
            
            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
            logger.info("TensorBoard logging complete")
        
        # Load best model
        self._load_best_checkpoint()
        
        # Save total training time
        self.history['total_training_time'] = time.time() - train_start_time
        
        return self.history
    
    def _train_epoch(self, loader: DataLoader) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for i, (sequences, masks) in enumerate(tqdm(loader, desc="Training", leave=False)):
            sequences = sequences.to(self.config.device)
            masks = masks.to(self.config.device)
            
            # Forward
            # Model returns (reconstructed, embedding)
            output = self.model(sequences, masks)
            if isinstance(output, tuple):
                reconstructed, _ = output
            else:
                reconstructed = output
            
            # Masked MSE loss
            loss = self._masked_mse_loss(sequences, reconstructed, masks)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.accumulation_steps
            
            # Backward
            loss.backward()
            
            # Accumulate and step
            if (i + 1) % self.config.accumulation_steps == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.accumulation_steps
        
        return total_loss / len(loader)
    
    def _validate(self, loader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for sequences, masks in loader:
                sequences = sequences.to(self.config.device)
                masks = masks.to(self.config.device)
                
                reconstructed, _ = self.model(sequences, masks)
                loss = self._masked_mse_loss(sequences, reconstructed, masks)
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _masked_mse_loss(
        self,
        target: torch.Tensor,
        output: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss with masking for padded positions."""
        loss = self.criterion(output, target)
        
        # Apply mask (expand to feature dim)
        mask = mask.unsqueeze(-1)
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def _evaluate_test_set(
        self,
        test_loader: DataLoader,
        test_labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate model on test set and compute classification metrics.
        
        Uses reconstruction error as anomaly score.
        Higher error = more likely anomaly.
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix,
            precision_recall_curve,
        )
        
        self.model.eval()
        all_errors = []
        
        with torch.no_grad():
            for sequences, masks in test_loader:
                sequences = sequences.to(self.config.device)
                masks = masks.to(self.config.device)
                
                reconstructed, _ = self.model(sequences, masks)
                
                # Compute per-sample reconstruction error
                mse = (sequences - reconstructed) ** 2
                mask_expanded = masks.unsqueeze(-1)
                sample_errors = (mse * mask_expanded).sum(dim=(1, 2)) / mask_expanded.sum(dim=(1, 2))
                
                all_errors.extend(sample_errors.cpu().numpy())
        
        all_errors = np.array(all_errors)
        
        # Determine threshold for converting scores to binary predictions
        if self.config.threshold_method == "f1_optimal":
            # Use F1-optimal threshold based on precision-recall curve.
            # Note: This optimizes on the current evaluation set.
            try:
                precision_arr, recall_arr, pr_thresholds = precision_recall_curve(
                    test_labels, all_errors
                )
                # Last element of precision/recall corresponds to a threshold
                # that is effectively +inf; ignore it when computing F1.
                if pr_thresholds.size > 0:
                    f1_scores = 2 * (precision_arr[:-1] * recall_arr[:-1]) / (
                        precision_arr[:-1] + recall_arr[:-1] + 1e-8
                    )
                    best_idx = int(np.argmax(f1_scores))
                    threshold = pr_thresholds[best_idx]
                else:
                    # Fallback to percentile if thresholds are not available
                    threshold = np.percentile(all_errors, self.config.threshold_percentile)
            except Exception:
                # Any issue with PR curve computation: fall back to percentile
                threshold = np.percentile(all_errors, self.config.threshold_percentile)
        else:
            # Default: fixed percentile of reconstruction error distribution
            threshold = np.percentile(all_errors, self.config.threshold_percentile)
        
        # Predict: 1 if error > threshold (anomaly), 0 otherwise
        predictions = (all_errors > threshold).astype(int)
        
        # Compute metrics
        accuracy = accuracy_score(test_labels, predictions)
        
        # Handle edge cases (no positives predicted)
        if predictions.sum() == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            precision = precision_score(test_labels, predictions, zero_division=0)
            recall = recall_score(test_labels, predictions, zero_division=0)
            f1 = f1_score(test_labels, predictions, zero_division=0)
        
        # False Positive Rate
        tn, fp, fn, tp = confusion_matrix(test_labels, predictions, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # AUC-ROC (uses raw scores, not binary predictions)
        try:
            auc_roc = roc_auc_score(test_labels, all_errors)
        except ValueError:
            auc_roc = 0.5  # Default if only one class
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'fpr': fpr,
            'auc_roc': auc_roc,
            'threshold': threshold,
            # Confusion matrix values
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            # For TTD calculation
            'predictions': predictions,
        }
    
    def _create_confusion_matrix_figure(
        self, tp: int, tn: int, fp: int, fn: int
    ):
        """Create a confusion matrix figure for TensorBoard."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues')
        
        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Anomaly'])
        ax.set_yticklabels(['Normal', 'Anomaly'])
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14)
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                value = cm[i, j]
                color = 'white' if value > cm.max() / 2 else 'black'
                ax.text(j, i, f'{value:,}', ha='center', va='center', 
                        color=color, fontsize=14, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        return fig
    
    def _compute_ttd(
        self,
        predictions: np.ndarray,
        test_labels: np.ndarray,
        user_ids: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute Time-to-Detect metrics for insider incidents.
        
        Args:
            predictions: Binary predictions (0=normal, 1=anomaly)
            test_labels: Ground truth labels
            user_ids: Optional[List[str]] of user IDs
            
        Returns:
            Dict with TTD metrics:
            - ttd_hours_mean: Mean time from incident start to first detection (hours)
            - ttd_hours_median: Median TTD in hours
            - ttd_sessions_mean: Mean number of sessions before first detection
            - ttd_lag_mean: Mean detection lag as % of incident duration
        """
        return compute_ttd(
            predictions=predictions,
            test_labels=test_labels,
            test_session_ids=self.test_session_ids,
            test_timestamps=self.test_timestamps,
            insider_incidents=self.insider_incidents,
            test_user_ids=user_ids,
        )
    
    def _create_loader(
        self,
        sequences: np.ndarray,
        masks: np.ndarray,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create PyTorch DataLoader."""
        dataset = TensorDataset(
            torch.FloatTensor(sequences),
            torch.FloatTensor(masks),
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
        )
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
        }
        
        path = self.config.checkpoint_dir / f"{self.model_name}.pt"
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.config.checkpoint_dir / f"{self.model_name}_best.pt"
            torch.save(checkpoint, best_path)
    
    def _load_best_checkpoint(self) -> None:
        """Load best checkpoint."""
        best_path = self.config.checkpoint_dir / f"{self.model_name}_best.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.config.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1}")


def train_autoencoder(
    model: nn.Module,
    train_data: Dict[str, np.ndarray],
    config: Optional[TrainConfig] = None,
    model_name: str = "autoencoder",
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    High-level training function.
    
    Args:
        model: Autoencoder model
        train_data: Dict with train/val sequences and masks
        config: Training configuration
        model_name: Name for checkpoint saving
        
    Returns:
        Trained model and history
    """
    if config is None:
        config = TrainConfig()
    
    trainer = Trainer(model, config, model_name)
    
    history = trainer.train(
        train_data['train_sequences'],
        train_data['train_masks'],
        train_data['val_sequences'],
        train_data['val_masks'],
    )
    
    return trainer.model, history


if __name__ == "__main__":
    from src.models import LSTMAutoencoder
    
    # Quick test with dummy data
    batch_size = 100
    seq_len = 50
    feature_dim = 24
    
    model = LSTMAutoencoder(input_dim=feature_dim)
    
    train_data = {
        'train_sequences': np.random.randn(batch_size, seq_len, feature_dim).astype(np.float32),
        'train_masks': np.ones((batch_size, seq_len), dtype=np.float32),
        'val_sequences': np.random.randn(20, seq_len, feature_dim).astype(np.float32),
        'val_masks': np.ones((20, seq_len), dtype=np.float32),
    }
    
    config = TrainConfig(epochs=2, batch_size=16)
    
    model, history = train_autoencoder(model, train_data, config, "test_model")
    print(f"\nTraining complete. Final val loss: {history['val_loss'][-1]:.6f}")
