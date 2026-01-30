#!/usr/bin/env python
"""Quick test that PaperLSTMAutoencoder runs and matches expected shapes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models import PaperLSTMAutoencoder

m = PaperLSTMAutoencoder(input_dim=32, bottleneck_dim=16, enc_hidden=32)
x = torch.randn(2, 20, 32)
mask = torch.ones(2, 20)
rec, emb = m(x, mask)
print("OK", rec.shape, emb.shape)
print("Params:", sum(p.numel() for p in m.parameters()))
