# LSTM Autoencoder Architecture Analysis

## Current Architecture

```
Input (batch, 100, 74)
  ↓
BiLSTM Encoder (2 layers, hidden=256, bidirectional=True)
  ↓
Last Hidden State → Linear → Embedding (128d bottleneck)
  ↓
LSTM Decoder (2 layers, hidden=256)
  ↓
Output (batch, 100, 74)
```

**Parameters:** 3,339,978

---

## 🔴 Critical Issues Found

### 1. **Mask Not Used in Encoder Forward Pass** (CRITICAL)

**Location:** `LSTMEncoder.forward()` line 42-65

**Problem:**
```python
def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
    # LSTM forward
    output, (h_n, c_n) = self.lstm(x)  # ❌ Processes ALL positions including padding
    
    # Use last hidden state
    if self.bidirectional:
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # ❌ Uses last position regardless of padding
    else:
        h_last = h_n[-1]
    
    embedding = self.fc(h_last)
    return embedding, (h_n, c_n)
```

**Impact:**
- Padding tokens (zeros) are processed by LSTM
- Last hidden state may come from a padded position
- Encoder doesn't know which positions are real vs padding
- This hurts reconstruction quality, especially for shorter sequences

**Fix Required:** ✅ **YES - CRITICAL**

---

### 2. **No Mask-Aware Pooling**

**Problem:**
- Currently uses only the last hidden state
- Should use mean pooling over non-padded positions
- Better representation for variable-length sequences

**Impact:**
- Short sequences (e.g., 20 events) get same treatment as full sequences (100 events)
- Information from all valid positions not utilized

**Fix Required:** ✅ **YES - HIGH PRIORITY**

---

### 3. **Decoder Doesn't Use Mask**

**Problem:**
- Decoder doesn't receive or use mask information
- Less critical than encoder, but still suboptimal

**Impact:**
- Decoder tries to reconstruct padded positions
- Loss function handles this, but decoder could be more efficient

**Fix Required:** ⚠️ **OPTIONAL - LOW PRIORITY**

---

## 🟡 Potential Improvements

### 4. **Bottleneck Size (embedding_dim=128)**

**Current:** 128 dimensions for 74 input features

**Analysis:**
- Input: 74 features × 100 sequence length = 7,400 values
- Bottleneck: 128 values
- Compression ratio: ~58:1

**Question:** Is 128 too small?
- With 74 features, might need larger bottleneck
- But larger bottleneck = easier reconstruction = worse anomaly detection
- **Need to test:** Try 256 embedding_dim if underfitting

**Fix Required:** ⚠️ **TEST FIRST - MEDIUM PRIORITY**

---

### 5. **No Attention Mechanism**

**Current:** Simple LSTM encoder-decoder

**Potential Benefit:**
- Attention could focus on anomalous parts of sequences
- Better feature importance understanding
- More interpretable

**Impact:** Medium-High (but requires significant changes)

**Fix Required:** 🔵 **FUTURE WORK - LOW PRIORITY**

---

### 6. **No Per-Feature Error Analysis**

**Current:** Only total MSE error

**Potential Benefit:**
- Understand which features are most discriminative
- Could weight features differently
- Better anomaly scoring

**Fix Required:** 🟢 **NICE TO HAVE - LOW PRIORITY**

---

## ✅ What's Working Well

1. **Bidirectional Encoder** - Good for capturing context
2. **Teacher Forcing in Decoder** - Helps training stability
3. **Proper Loss Masking** - Loss function correctly ignores padding
4. **Architecture Structure** - Standard autoencoder design is sound

---

## Recommended Modifications

### Priority 1: Fix Mask Usage (CRITICAL)

**Modify `LSTMEncoder.forward()`:**

```python
def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: (batch, seq_len, input_dim)
        mask: (batch, seq_len) - 1 for real, 0 for padding
    """
    # LSTM forward
    output, (h_n, c_n) = self.lstm(x)
    
    # Mask-aware pooling: use mean of non-padded positions
    if mask is not None:
        # output: (batch, seq_len, hidden_dim * num_directions)
        # mask: (batch, seq_len)
        mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Weighted mean over sequence length
        masked_output = output * mask_expanded  # Zero out padded positions
        seq_lengths = mask.sum(dim=1, keepdim=True)  # (batch, 1)
        seq_lengths = torch.clamp(seq_lengths, min=1.0)  # Avoid division by zero
        
        # Mean pooling over sequence
        pooled = masked_output.sum(dim=1) / seq_lengths  # (batch, hidden_dim * num_directions)
    else:
        # Fallback: use last position (current behavior)
        if self.bidirectional:
            pooled = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            pooled = h_n[-1]
    
    # Project to embedding
    embedding = self.fc(pooled)
    
    return embedding, (h_n, c_n)
```

**Expected Impact:**
- Better encoding for variable-length sequences
- Improved reconstruction quality
- Better anomaly detection (5-10% improvement possible)

---

### Priority 2: Add Per-Feature Error Method (MEDIUM)

**Add to `LSTMAutoencoder` class:**

```python
def get_per_feature_error(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute per-feature reconstruction error.
    
    Returns:
        error: (batch, feature_dim) - MSE per feature
    """
    reconstructed, _ = self.forward(x, mask)
    
    # Compute MSE per feature
    error = (x - reconstructed) ** 2  # (batch, seq_len, feature_dim)
    
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
        error = error * mask_expanded
        # Mean over sequence length
        error = error.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
    else:
        error = error.mean(dim=1)
    
    return error
```

**Use Case:**
- Feature importance analysis
- Better anomaly scoring (weight important features)
- Interpretability

---

### Priority 3: Increase Bottleneck Size (IF UNDERFITTING)

**Test with larger embedding_dim:**

```python
# In config.yaml or training script
embedding_dim: 256  # Instead of 128
```

**Test First:**
- Check if model is underfitting (train_loss ≈ val_loss but both high)
- If yes, try 256 embedding_dim
- Monitor if recall improves

---

## Implementation Plan

### Phase 1: Critical Fixes (Do First)
1. ✅ Fix mask usage in encoder (Priority 1)
2. ✅ Test with same config
3. ✅ Compare results with baseline

### Phase 2: Enhancements (After Phase 1)
1. ✅ Add per-feature error method (Priority 2)
2. ✅ Test bottleneck size increase (Priority 3)
3. ✅ Analyze feature importance

### Phase 3: Future Work
1. ⬜ Add attention mechanism
2. ⬜ Add decoder mask support
3. ⬜ Advanced architectures (VAE, etc.)

---

## Expected Improvements

### After Fixing Mask Usage:
- **Reconstruction Quality:** +5-15% improvement
- **Recall:** +2-5% improvement (0.87% → 2-5%)
- **F1 Score:** +0.01-0.02 improvement (0.0042 → 0.01-0.02)

### Combined with Other Improvements (batch size, threshold, etc.):
- **Recall:** 0.87% → 10-30% (target)
- **F1:** 0.0042 → 0.05-0.15 (target)

---

## Code Changes Summary

### Files to Modify:
1. `src/models/lstm_autoencoder.py`
   - `LSTMEncoder.forward()` - Add mask-aware pooling
   - `LSTMAutoencoder` - Add `get_per_feature_error()` method

### Testing:
- Run training with fixed encoder
- Compare metrics with baseline
- Verify mask-aware pooling works correctly
- Check reconstruction quality improvement

---

## Conclusion

**Must Fix:**
- ✅ Mask usage in encoder (CRITICAL)

**Should Add:**
- ✅ Per-feature error analysis (useful for debugging/analysis)

**Consider:**
- ⚠️ Larger bottleneck if underfitting

**Future:**
- 🔵 Attention mechanism
- 🔵 Other architecture improvements

**Recommendation:** Fix mask usage first, then proceed with other improvements from the main plan.
