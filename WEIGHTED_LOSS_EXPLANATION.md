# Weighted Loss Explained

## Implemented: Hard Mining (CLI)

**Branch:** `feature/weighted-loss`

- **CLI:** `--weighted-loss` enables hard mining (focus loss on top-k% hardest samples per batch). **Training only**; validation loss is unchanged (mean over all val samples).
- **Config:** `training.weighted_loss: none | hard_mining`, `training.hard_mining_ratio: 0.1`.
- **Usage:** `python scripts/train.py --model lstm --weighted-loss` or `--weighted-loss --hard-mining-ratio 0.2`.

---

## What is Weighted Loss?

**Weighted loss** is a technique to handle **class imbalance** by giving more importance to underrepresented classes during training. Instead of treating all samples equally, you multiply each sample's loss by a weight that reflects its importance.

---

## Current Implementation (Unweighted)

In your current code (`src/training/trainer.py`), the loss is computed as:

```python
def _masked_mse_loss(self, target, output, mask):
    """Compute MSE loss with masking for padded positions."""
    loss = self.criterion(output, target)  # MSE per element
    mask = mask.unsqueeze(-1)
    loss = (loss * mask).sum() / mask.sum()  # Average over non-padded positions
    return loss
```

**Current behavior:**
- All samples contribute equally to the loss
- Normal sessions (99.6%) dominate the gradient updates
- Insider sessions (0.4%) have minimal impact
- Result: Model learns to reconstruct normal sessions well, but ignores anomalies

---

## The Problem: Class Imbalance

Your dataset has:
- **Normal sessions:** 113,243 (99.6%)
- **Insider sessions:** 1,831 (0.4%)

**What happens without weighting:**
```
Total loss = (99.6% × normal_loss) + (0.4% × insider_loss)
           ≈ normal_loss (insider_loss is negligible)
```

The model optimizes for normal sessions and ignores insider patterns.

---

## How Weighted Loss Works

### Basic Concept

Multiply each sample's loss by a weight based on its class:

```python
weighted_loss = weight_normal × loss_normal + weight_insider × loss_insider
```

### Weight Calculation Strategies

#### 1. **Inverse Frequency Weighting** (Most Common)

```python
# Calculate weights inversely proportional to class frequency
n_normal = 113_243
n_insider = 1_831
total = n_normal + n_insider

weight_normal = total / (2 * n_normal)  # ≈ 0.5
weight_insider = total / (2 * n_insider)  # ≈ 31.4
```

**Result:** Insider samples get ~63× more weight than normal samples.

#### 2. **Balanced Weighting**

```python
weight_normal = 1.0
weight_insider = n_normal / n_insider  # ≈ 61.8
```

**Result:** Each class contributes equally to the loss.

#### 3. **Manual Weighting** (Tunable)

```python
weight_normal = 1.0
weight_insider = 20.0  # Or 50.0, or 100.0 - tune based on results
```

**Result:** You control the trade-off between precision and recall.

---

## Implementation for Your Project

### Option 1: Sample-Level Weighting (If Labels Available During Training)

**When to use:** If you have insider labels in your training set (semi-supervised)

```python
def _masked_mse_loss_weighted(
    self,
    target: torch.Tensor,
    output: torch.Tensor,
    mask: torch.Tensor,
    sample_weights: torch.Tensor,  # (batch_size,) - weight per sample
) -> torch.Tensor:
    """Compute weighted MSE loss with masking."""
    # Compute per-sample loss
    loss = self.criterion(output, target)  # (batch, seq_len, features)
    
    # Apply mask and sum over sequence/features
    mask_expanded = mask.unsqueeze(-1)
    sample_losses = (loss * mask_expanded).sum(dim=(1, 2)) / mask_expanded.sum(dim=(1, 2))
    # sample_losses shape: (batch_size,)
    
    # Apply sample weights
    weighted_loss = (sample_losses * sample_weights).mean()
    
    return weighted_loss
```

**Usage:**
```python
# In training loop
labels = batch_labels  # 0 = normal, 1 = insider
weights = torch.where(labels == 0, 1.0, 50.0)  # Normal=1.0, Insider=50.0
loss = self._masked_mse_loss_weighted(sequences, reconstructed, masks, weights)
```

---

### Option 2: Feature-Level Weighting (Unsupervised)

**When to use:** If training on normal-only data (current setup)

Since you're training on **normal-only** data, you can't weight by class. Instead, you can:

#### A. Weight by Reconstruction Error Magnitude

Give more weight to samples with higher reconstruction error (potential anomalies):

```python
def _adaptive_weighted_loss(
    self,
    target: torch.Tensor,
    output: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Weight samples by their reconstruction error."""
    loss = self.criterion(output, target)
    mask_expanded = mask.unsqueeze(-1)
    
    # Compute per-sample error
    sample_errors = (loss * mask_expanded).sum(dim=(1, 2)) / mask_expanded.sum(dim=(1, 2))
    
    # Weight by error magnitude (samples with higher error get more weight)
    # Use softmax to normalize weights
    weights = torch.softmax(sample_errors / sample_errors.std(), dim=0)
    
    # Apply weights
    weighted_loss = (sample_errors * weights).mean()
    
    return weighted_loss
```

#### B. Focus on Hard Examples

Weight samples that are harder to reconstruct:

```python
def _hard_example_mining_loss(
    self,
    target: torch.Tensor,
    output: torch.Tensor,
    mask: torch.Tensor,
    top_k_ratio: float = 0.1,  # Focus on top 10% hardest examples
) -> torch.Tensor:
    """Focus loss on hardest-to-reconstruct samples."""
    loss = self.criterion(output, target)
    mask_expanded = mask.unsqueeze(-1)
    
    # Per-sample loss
    sample_losses = (loss * mask_expanded).sum(dim=(1, 2)) / mask_expanded.sum(dim=(1, 2))
    
    # Select top-k hardest examples
    k = max(1, int(len(sample_losses) * top_k_ratio))
    top_k_losses, _ = torch.topk(sample_losses, k)
    
    # Average only the hardest examples
    return top_k_losses.mean()
```

---

## Recommended Approach for Your Project

### Current Situation:
- **Training set:** Normal-only (unsupervised)
- **Test set:** Normal + Insider (for evaluation)
- **Goal:** Detect insider sessions as anomalies

### Best Strategy: **Hard Example Mining**

Since you don't have insider labels during training, focus on samples that are hardest to reconstruct:

```python
class Trainer:
    def __init__(self, ...):
        # ... existing code ...
        self.hard_example_ratio = 0.1  # Focus on top 10% hardest
    
    def _masked_mse_loss(
        self,
        target: torch.Tensor,
        output: torch.Tensor,
        mask: torch.Tensor,
        use_hard_mining: bool = True,
    ) -> torch.Tensor:
        """Compute MSE loss with optional hard example mining."""
        loss = self.criterion(output, target)
        mask_expanded = mask.unsqueeze(-1)
        
        # Per-sample loss
        sample_losses = (loss * mask_expanded).sum(dim=(1, 2)) / mask_expanded.sum(dim=(1, 2))
        
        if use_hard_mining:
            # Focus on hardest examples (potential anomalies)
            k = max(1, int(len(sample_losses) * self.hard_example_ratio))
            top_k_losses, _ = torch.topk(sample_losses, k)
            return top_k_losses.mean()
        else:
            # Standard average
            return sample_losses.mean()
```

**Why this works:**
- Hard-to-reconstruct samples are likely anomalies
- Focusing on them improves anomaly detection
- No need for labels during training

---

## Alternative: If You Add Insider Data to Training

If you decide to include insider sessions in training (semi-supervised), use **sample-level weighting**:

```python
# In prepare_training_data or train()
# Add insider sessions to training set with labels

# Calculate weights
n_normal = len(normal_sessions)
n_insider = len(insider_sessions)
weight_normal = 1.0
weight_insider = n_normal / n_insider  # ~61.8

# Create weight tensor for each sample
sample_weights = torch.where(
    labels == 0,
    weight_normal,
    weight_insider
)

# Use in loss calculation
loss = self._masked_mse_loss_weighted(
    sequences, reconstructed, masks, sample_weights
)
```

---

## Expected Impact

### Without Weighting (Current):
- **Recall:** 0.87% (very low)
- **F1:** 0.0042 (very low)
- Model ignores anomalies

### With Hard Example Mining:
- **Recall:** Expected 5-15% improvement
- **F1:** Expected 2-5× improvement
- Model focuses on hard-to-reconstruct patterns

### With Sample-Level Weighting (if labels available):
- **Recall:** Expected 10-30% improvement
- **F1:** Expected 5-10× improvement
- Better balance between classes

---

## Trade-offs

### Hard Example Mining:
✅ **Pros:**
- Works with normal-only training
- No labels needed
- Focuses on potential anomalies

❌ **Cons:**
- May be noisy (some hard examples are just outliers, not anomalies)
- Less stable than class-based weighting

### Sample-Level Weighting:
✅ **Pros:**
- More stable and predictable
- Directly addresses class imbalance
- Well-established technique

❌ **Cons:**
- Requires labels during training
- May need tuning of weight values

---

## Implementation Checklist

If implementing weighted loss:

1. **Decide on approach:**
   - [ ] Hard example mining (normal-only training)
   - [ ] Sample-level weighting (if labels available)

2. **Modify `_masked_mse_loss`:**
   - [ ] Add weight parameter or hard mining logic
   - [ ] Update loss calculation

3. **Update training loop:**
   - [ ] Pass weights or enable hard mining
   - [ ] Log weighted vs unweighted loss

4. **Tune hyperparameters:**
   - [ ] Weight values (if sample-level)
   - [ ] Hard example ratio (if hard mining)

5. **Evaluate:**
   - [ ] Compare recall/F1 before/after
   - [ ] Check if loss values are reasonable

---

## Example Code Snippet

```python
# In trainer.py

def _masked_mse_loss(
    self,
    target: torch.Tensor,
    output: torch.Tensor,
    mask: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
    hard_mining_ratio: Optional[float] = None,
) -> torch.Tensor:
    """Compute MSE loss with optional weighting or hard mining."""
    loss = self.criterion(output, target)
    mask_expanded = mask.unsqueeze(-1)
    
    # Per-sample loss
    sample_losses = (loss * mask_expanded).sum(dim=(1, 2)) / mask_expanded.sum(dim=(1, 2))
    
    # Apply weighting or hard mining
    if sample_weights is not None:
        # Sample-level weighting
        weighted_loss = (sample_losses * sample_weights).mean()
        return weighted_loss
    elif hard_mining_ratio is not None:
        # Hard example mining
        k = max(1, int(len(sample_losses) * hard_mining_ratio))
        top_k_losses, _ = torch.topk(sample_losses, k)
        return top_k_losses.mean()
    else:
        # Standard average
        return sample_losses.mean()
```

---

## Summary

**Weighted loss** helps handle class imbalance by:
1. **Giving more importance to underrepresented classes** (if labels available)
2. **Focusing on hard-to-reconstruct examples** (if normal-only training)

For your current **normal-only training setup**, **hard example mining** is the best approach. It focuses the model on samples that are hardest to reconstruct, which are likely anomalies.

**Expected improvement:** 5-15% recall increase, 2-5× F1 improvement.
