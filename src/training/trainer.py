"""
Training Module
Training loop with early stopping and checkpoint saving.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    patience: int = 10
    min_delta: float = 0.0001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path = Path("models")
    

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
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        self.criterion = nn.MSELoss(reduction='none')
        self.early_stopping = EarlyStopping(config.patience, config.min_delta)
        
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_time': [],
        }
    
    def train(
        self,
        train_sequences: np.ndarray,
        train_masks: np.ndarray,
        val_sequences: np.ndarray,
        val_masks: np.ndarray,
    ) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            train_sequences: (N, seq_len, feature_dim)
            train_masks: (N, seq_len)
            val_sequences: Validation sequences
            val_masks: Validation masks
            
        Returns:
            Training history
        """
        # Create data loaders
        train_loader = self._create_loader(train_sequences, train_masks, shuffle=True)
        val_loader = self._create_loader(val_sequences, val_masks, shuffle=False)
        
        logger.info(f"Starting training on {self.config.device}")
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        best_val_loss = float('inf')
        
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
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"Time: {epoch_time:.1f}s, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=True)
            
            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self._load_best_checkpoint()
        
        return self.history
    
    def _train_epoch(self, loader: DataLoader) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for sequences, masks in tqdm(loader, desc="Training", leave=False):
            sequences = sequences.to(self.config.device)
            masks = masks.to(self.config.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            reconstructed, _ = self.model(sequences, masks)
            
            # Masked MSE loss
            loss = self._masked_mse_loss(sequences, reconstructed, masks)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
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
            checkpoint = torch.load(best_path, map_location=self.config.device)
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
