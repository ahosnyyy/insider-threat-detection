"""
Transformer Autoencoder Model
Encoder-Decoder architecture using self-attention for session reconstruction.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer Encoder that compresses sequence to bottleneck."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        max_len: int = 500,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Bottleneck projection (mean pooling + linear)
        self.fc = nn.Linear(d_model, embedding_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len) - 1 for real, 0 for padding
            
        Returns:
            embedding: (batch, embedding_dim)
            encoder_output: (batch, seq_len, d_model) - for decoder cross-attention
        """
        # Project input
        x = self.input_proj(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create attention mask (True = ignore)
        if mask is not None:
            src_key_padding_mask = (mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer forward
        encoder_output = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Mean pooling (over non-padded positions)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            pooled = (encoder_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = encoder_output.mean(dim=1)
        
        # Project to embedding
        embedding = self.fc(pooled)
        
        return embedding, encoder_output


class TransformerDecoder(nn.Module):
    """Transformer Decoder that reconstructs sequence from bottleneck."""
    
    def __init__(
        self,
        output_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        max_len: int = 500,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Project embedding to sequence
        self.embedding_proj = nn.Linear(embedding_dim, d_model)
        
        # Learnable query embeddings
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer decoder layers (with cross-attention to encoder output)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.fc_out = nn.Linear(d_model, output_dim)
    
    def forward(
        self,
        embedding: torch.Tensor,
        encoder_output: torch.Tensor,
        seq_len: int,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embedding: (batch, embedding_dim)
            encoder_output: (batch, seq_len, d_model)
            seq_len: Target sequence length
            memory_mask: (batch, seq_len) - mask for encoder output
            
        Returns:
            output: (batch, seq_len, output_dim)
        """
        batch_size = embedding.size(0)
        
        # Create query sequence from embedding (broadcast + positional)
        query = self.embedding_proj(embedding).unsqueeze(1).expand(-1, seq_len, -1)
        query = self.pos_encoding(query)
        
        # Memory key padding mask
        if memory_mask is not None:
            memory_key_padding_mask = (memory_mask == 0)
        else:
            memory_key_padding_mask = None
        
        # Transformer decoder (cross-attention to encoder output)
        output = self.transformer(
            query,
            encoder_output,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        
        # Project to output dim
        output = self.fc_out(output)
        
        return output


class TransformerAutoencoder(nn.Module):
    """
    Full Transformer Autoencoder for session anomaly detection.
    
    Architecture:
        Input -> Transformer Encoder -> Bottleneck (128d) -> Transformer Decoder -> Output
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        max_len: int = 500,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            embedding_dim=embedding_dim,
            dropout=dropout,
            max_len=max_len,
        )
        
        self.decoder = TransformerDecoder(
            output_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            embedding_dim=embedding_dim,
            dropout=dropout,
            max_len=max_len,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len)
            
        Returns:
            reconstructed: (batch, seq_len, input_dim)
            embedding: (batch, embedding_dim)
        """
        seq_len = x.size(1)
        
        # Encode
        embedding, encoder_output = self.encoder(x, mask)
        
        # Decode
        reconstructed = self.decoder(embedding, encoder_output, seq_len, mask)
        
        return reconstructed, embedding
    
    def get_embedding(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get only the embedding (for inference)."""
        embedding, _ = self.encoder(x, mask)
        return embedding
    
    def get_reconstruction_error(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute per-sample reconstruction error.
        
        Returns:
            error: (batch,) - MSE per sample
        """
        reconstructed, _ = self.forward(x, mask)
        
        # Compute MSE per sample
        error = (x - reconstructed) ** 2
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            error = error * mask
            # Mean over non-padded elements
            error = error.sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-8)
        else:
            error = error.mean(dim=(1, 2))
        
        return error


def create_transformer_autoencoder(config: dict) -> TransformerAutoencoder:
    """Factory function to create model from config."""
    return TransformerAutoencoder(
        input_dim=config.get("feature_dim", 32),
        d_model=config.get("hidden_dim", 256),
        nhead=config.get("num_heads", 8),
        num_layers=config.get("num_layers", 6),
        dim_feedforward=config.get("ff_dim", 1024),
        embedding_dim=config.get("embedding_dim", 128),
        dropout=config.get("dropout", 0.1),
    )


if __name__ == "__main__":
    # Quick test
    batch_size = 4
    seq_len = 50
    input_dim = 24
    
    model = TransformerAutoencoder(input_dim=input_dim)
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones(batch_size, seq_len)
    
    reconstructed, embedding = model(x, mask)
    error = model.get_reconstruction_error(x, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Error shape: {error.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
