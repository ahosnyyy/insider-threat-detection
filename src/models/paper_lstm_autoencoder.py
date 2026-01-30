"""
Paper LSTM Autoencoder
Matches the reference paper: unidirectional encoder (last-step bottleneck),
RepeatVector, unidirectional decoder in one forward pass.
Input (B,T,F) -> Encoder -> (B, bottleneck) -> Repeat -> (B,T,bottleneck) -> Decoder -> (B,T,F).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class PaperLSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder matching the paper architecture:
    - Encoder: 2 unidirectional LSTMs (input_dim -> enc_h1 -> bottleneck_dim), last valid step as bottleneck.
    - RepeatVector: bottleneck repeated seq_len times.
    - Decoder: 2 unidirectional LSTMs (bottleneck -> dec_h1 -> input_dim), one forward pass.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 16,
        enc_hidden: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_dim: Feature dimension (e.g. 32).
            bottleneck_dim: Latent size (paper uses 16).
            enc_hidden: First encoder LSTM hidden size (paper uses 32). Defaults to input_dim.
            dropout: Dropout between LSTM layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        enc_h1 = enc_hidden if enc_hidden is not None else input_dim

        # Encoder: layer1 (input_dim -> enc_h1), layer2 (enc_h1 -> bottleneck_dim)
        self.enc_lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=enc_h1,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        self.enc_lstm2 = nn.LSTM(
            input_size=enc_h1,
            hidden_size=bottleneck_dim,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        self.enc_dropout = nn.Dropout(dropout)

        # Decoder: layer1 (bottleneck_dim -> bottleneck_dim), layer2 (bottleneck_dim -> input_dim)
        self.dec_lstm1 = nn.LSTM(
            input_size=bottleneck_dim,
            hidden_size=bottleneck_dim,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        self.dec_lstm2 = nn.LSTM(
            input_size=bottleneck_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        self.dec_dropout = nn.Dropout(dropout)

    def _last_valid_step(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Take output at last valid (mask=1) position per sample. x: (B, T, D), mask: (B, T). Returns (B, D)."""
        B, T, D = x.shape
        # last_idx[b] = index of last valid step (0 if none)
        lengths = mask.sum(dim=1).long()  # (B,)
        last_idx = (lengths - 1).clamp(min=0)  # (B,)
        # Gather: we want x[b, last_idx[b], :]
        batch_idx = torch.arange(B, device=x.device)
        return x[batch_idx, last_idx, :]  # (B, D)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len) - 1 for real, 0 for padding. Used for encoder last-step only.

        Returns:
            reconstructed: (batch, seq_len, input_dim)
            embedding: (batch, bottleneck_dim)
        """
        B, T, _ = x.shape
        if mask is None:
            mask = torch.ones(B, T, device=x.device, dtype=x.dtype)

        # Encoder: two layers, take last valid step as bottleneck
        out1, _ = self.enc_lstm1(x)  # (B, T, enc_h1)
        out1 = self.enc_dropout(out1)
        out2, _ = self.enc_lstm2(out1)  # (B, T, bottleneck_dim)
        embedding = self._last_valid_step(out2, mask)  # (B, bottleneck_dim)

        # RepeatVector: (B, bottleneck_dim) -> (B, T, bottleneck_dim)
        repeated = embedding.unsqueeze(1).expand(-1, T, -1)

        # Decoder: one forward pass
        dec1, _ = self.dec_lstm1(repeated)  # (B, T, bottleneck_dim)
        dec1 = self.dec_dropout(dec1)
        reconstructed, _ = self.dec_lstm2(dec1)  # (B, T, input_dim)

        return reconstructed, embedding

    def get_embedding(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get only the bottleneck embedding (for inference)."""
        _, embedding = self.forward(x, mask)
        return embedding

    def get_reconstruction_error(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-sample reconstruction error (MSE over non-padded elements). Returns (batch,)."""
        reconstructed, _ = self.forward(x, mask)
        error = (x - reconstructed) ** 2
        if mask is not None:
            mask_exp = mask.unsqueeze(-1)
            error = error * mask_exp
            error = error.sum(dim=(1, 2)) / (mask_exp.sum(dim=(1, 2)) + 1e-8)
        else:
            error = error.mean(dim=(1, 2))
        return error


def create_paper_lstm_autoencoder(config: dict) -> PaperLSTMAutoencoder:
    """Factory: create paper LSTM from config."""
    return PaperLSTMAutoencoder(
        input_dim=config.get("feature_dim", 32),
        bottleneck_dim=config.get("paper_lstm_bottleneck", 16),
        enc_hidden=config.get("paper_lstm_enc_hidden"),  # None -> use input_dim
        dropout=config.get("dropout", 0.0),
    )


if __name__ == "__main__":
    batch_size, seq_len, input_dim = 4, 20, 32
    model = PaperLSTMAutoencoder(input_dim=input_dim, bottleneck_dim=16, enc_hidden=32)
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones(batch_size, seq_len)
    mask[0, 15:] = 0  # one short sequence
    rec, emb = model(x, mask)
    err = model.get_reconstruction_error(x, mask)
    print(f"Input: {x.shape}, Reconstructed: {rec.shape}, Embedding: {emb.shape}, Error: {err.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
