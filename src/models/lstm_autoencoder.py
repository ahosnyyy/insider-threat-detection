"""
LSTM Autoencoder Model
Encoder-Decoder architecture for session sequence reconstruction.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LSTMEncoder(nn.Module):
    """LSTM Encoder that compresses sequence to bottleneck."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Project to embedding dimension
        self.fc = nn.Linear(hidden_dim * self.num_directions, embedding_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len) - 1 for real, 0 for padding
            
        Returns:
            embedding: (batch, embedding_dim)
            hidden: Tuple of hidden states for decoder
        """
        # LSTM forward
        # output: (batch, seq_len, hidden_dim * num_directions)
        # For bidirectional: already concatenated [forward; backward]
        output, (h_n, c_n) = self.lstm(x)
        
        # Mask-aware pooling: use mean of non-padded positions
        if mask is not None:
            # output: (batch, seq_len, hidden_dim * num_directions)
            # mask: (batch, seq_len)
            mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            
            # Zero out padded positions
            masked_output = output * mask_expanded
            
            # Compute sequence lengths (number of non-padded positions)
            seq_lengths = mask.sum(dim=1, keepdim=True)  # (batch, 1)
            seq_lengths = torch.clamp(seq_lengths, min=1.0)  # Avoid division by zero
            
            # Mean pooling over sequence length
            pooled = masked_output.sum(dim=1) / seq_lengths  # (batch, hidden_dim * num_directions)
        else:
            # Fallback: use last hidden state (original behavior)
            if self.bidirectional:
                # Concat last layer forward and backward
                pooled = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                pooled = h_n[-1]
        
        # Project to embedding
        embedding = self.fc(pooled)
        
        return embedding, (h_n, c_n)


class LSTMDecoder(nn.Module):
    """LSTM Decoder that reconstructs sequence from bottleneck."""
    
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Project embedding to hidden for decoder initialization
        self.fc_hidden = nn.Linear(embedding_dim, hidden_dim * num_layers)
        self.fc_cell = nn.Linear(embedding_dim, hidden_dim * num_layers)
        
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self,
        embedding: torch.Tensor,
        seq_len: int,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embedding: (batch, embedding_dim)
            seq_len: Length of sequence to generate
            target: Optional target for teacher forcing during training
            
        Returns:
            output: (batch, seq_len, output_dim)
        """
        batch_size = embedding.size(0)
        
        # Initialize hidden state from embedding
        h_0 = self.fc_hidden(embedding).view(self.num_layers, batch_size, self.hidden_dim)
        c_0 = self.fc_cell(embedding).view(self.num_layers, batch_size, self.hidden_dim)
        
        # Initialize decoder input (zeros)
        decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=embedding.device)
        
        outputs = []
        hidden = (h_0, c_0)
        
        for t in range(seq_len):
            # Decode one step
            output, hidden = self.lstm(decoder_input, hidden)
            output = self.fc_out(output)
            outputs.append(output)
            
            # Teacher forcing or use own output
            if target is not None and self.training:
                decoder_input = target[:, t:t+1, :]
            else:
                decoder_input = output
        
        return torch.cat(outputs, dim=1)


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder for session anomaly detection.
    
    Architecture:
        Input -> BiLSTM Encoder -> Bottleneck (128d) -> LSTM Decoder -> Output
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        )
        
        self.decoder = LSTMDecoder(
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
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
        embedding, _ = self.encoder(x, mask)
        
        # Decode
        reconstructed = self.decoder(embedding, seq_len, target=x)
        
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


def create_lstm_autoencoder(config: dict) -> LSTMAutoencoder:
    """Factory function to create model from config."""
    return LSTMAutoencoder(
        input_dim=config.get("feature_dim", 32),
        hidden_dim=config.get("hidden_dim", 256),
        embedding_dim=config.get("embedding_dim", 128),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.2),
    )


if __name__ == "__main__":
    # Quick test
    batch_size = 4
    seq_len = 50
    input_dim = 24
    
    model = LSTMAutoencoder(input_dim=input_dim)
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones(batch_size, seq_len)
    
    reconstructed, embedding = model(x, mask)
    error = model.get_reconstruction_error(x, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Error shape: {error.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
