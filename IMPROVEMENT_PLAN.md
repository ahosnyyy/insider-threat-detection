# Insider Threat Detection - Model Improvement Plan

## Executive Summary

**Current Performance:**
- Session-Level Recall: 0.87% (16/1,831 insider sessions detected)
- User-Level Recall: 4.86% 
- Incidents Detected: 2/70 (2.9%)
- F1 Score: 0.0042 (session-level), 0.0797 (user-level)

**Root Causes Identified:**
1. Batch size too large (4096) → insufficient gradient updates
2. Severe class imbalance (98.4% normal) → model biased toward normal
3. Fixed threshold (95th percentile) → suboptimal for imbalanced data
4. Early stopping too aggressive (stopped at epoch 12)
5. No explicit handling of insider sessions during training

**Target Improvements:**
- Session-Level Recall: 10-30% (realistic)
- User-Level Recall: 20-40%
- Incidents Detected: 10-20/70 (14-29%)
- F1 Score: 0.05-0.15 (session-level)

---

## Phase 1: Quick Wins (High Impact, Low Effort)

### 1.1 Reduce Batch Size
**Priority:** 🔴 Critical  
**Effort:** Low  
**Expected Impact:** High

**Action:**
- Change batch size from 4096 → 512
- Test with 1024 if stable
- Update config or command-line argument

**Implementation:**
```bash
python scripts/train.py --model lstm --epochs 200 --batch-size 512
```

**Expected Results:**
- More gradient updates per epoch (~300 vs 76)
- Smoother learning curves
- Better convergence
- Training time: ~5-10 min/epoch (vs 44s, but better quality)

**Success Criteria:**
- Training loss decreases more smoothly
- Validation loss stabilizes better
- More epochs before early stopping

---

### 1.2 Optimize Threshold on Validation Set
**Priority:** 🔴 Critical  
**Effort:** Medium  
**Expected Impact:** Very High

**Current Issue:**
- Fixed 95th percentile threshold
- Config mentions `f1_optimal` but code uses percentile

**Action:**
- Implement threshold optimization on validation set
- Find threshold that maximizes F1 score
- Use this threshold for test evaluation

**Implementation Steps:**
1. Modify `_evaluate_test_set()` in `src/training/trainer.py`
2. Add threshold optimization function
3. Use validation set to find optimal threshold
4. Apply optimal threshold to test set

**Code Changes:**
- Add `find_optimal_threshold()` function
- Modify evaluation to use optimized threshold
- Add threshold to history/metrics

**Expected Results:**
- Recall: 0.87% → 10-25%
- F1: 0.0042 → 0.05-0.12
- Better balance between precision and recall

**Success Criteria:**
- F1 score improves significantly
- Recall increases without excessive FPR increase
- Threshold value logged and tracked

---

### 1.3 Increase Early Stopping Patience
**Priority:** 🟡 High  
**Effort:** Low  
**Expected Impact:** Medium

**Current Issue:**
- Patience: 10 epochs
- Stopped at epoch 12 (best at epoch 2)
- Insufficient training time

**Action:**
- Increase patience to 20 epochs
- Add metric-based early stopping (monitor recall/F1)
- Disable early stopping for first 20 epochs (warm-up)

**Implementation:**
- Update `config.yaml`: `patience: 20`
- Optionally: Add custom early stopping based on recall/F1

**Expected Results:**
- More training epochs
- Better model convergence
- Potentially better final metrics

**Success Criteria:**
- Training continues for more epochs
- Best model found later in training
- Final metrics improve

---

## Phase 2: Class Imbalance Handling (High Impact, Medium Effort)

### 2.1 Oversample Insider Sessions in Training Batches
**Priority:** 🔴 Critical  
**Effort:** Medium  
**Expected Impact:** Very High

**Current Issue:**
- Training on normal-only data (or heavily imbalanced)
- Model never sees insider patterns during training
- Cannot learn to distinguish insider behavior

**Action:**
- Modify data loading to oversample insider sessions
- Ensure each batch contains some insider sessions
- Balance batches: e.g., 80% normal, 20% insider (or adjust ratio)

**Implementation Steps:**
1. Modify `prepare_training_data()` in `src/data/dataset.py`
2. Create balanced sampler or modify DataLoader
3. Ensure insider sessions are included in training set
4. Repeat insider sessions to balance batches

**Code Changes:**
- Add `oversample_insider` parameter to data preparation
- Create custom sampler or modify batch creation
- Update `_create_loader()` to use balanced batches

**Expected Results:**
- Model learns insider patterns
- Recall: 0.87% → 15-30%
- F1: 0.0042 → 0.08-0.15
- Incidents detected: 2/70 → 10-20/70

**Success Criteria:**
- Training batches contain insider sessions
- Model shows improved recall on validation set
- Insider sessions have higher reconstruction errors

**Alternative Approach (if oversampling difficult):**
- Use weighted reconstruction loss
- Weight insider sessions higher in loss calculation
- Requires labels during training

---

### 2.2 Weighted Loss Function
**Priority:** 🟡 High  
**Effort:** Medium  
**Expected Impact:** High

**Current Issue:**
- Standard MSE loss treats all samples equally
- Insider sessions (rare) get same weight as normal (common)

**Action:**
- Implement weighted MSE loss
- Weight insider sessions 20-50× higher than normal
- Apply weights during reconstruction loss calculation

**Implementation Steps:**
1. Modify `_masked_mse_loss()` in `src/training/trainer.py`
2. Add sample weights based on labels
3. Weight insider sessions higher
4. Requires labels during training (may need to modify data loading)

**Code Changes:**
```python
# In _masked_mse_loss or _train_epoch
if labels is not None:
    weights = torch.where(labels == 1, 
                         torch.tensor(insider_weight, device=device),
                         torch.tensor(1.0, device=device))
    loss = loss * weights.unsqueeze(-1)
```

**Expected Results:**
- Model focuses more on reconstructing insider sessions correctly
- Better separation between normal and insider
- Improved recall

**Success Criteria:**
- Loss values for insider sessions are weighted higher
- Model shows improved discrimination
- Validation metrics improve

**Note:** This requires insider labels during training. If using normal-only training, skip this and use oversampling instead.

---

## Phase 3: Training Strategy Improvements (Medium Impact, Medium Effort)

### 3.1 Learning Rate Scheduling
**Priority:** 🟢 Medium  
**Effort:** Low  
**Expected Impact:** Medium

**Current:**
- ReduceLROnPlateau already implemented
- Factor: 0.5, Patience: 5

**Action:**
- Verify scheduler is working correctly
- Consider CosineAnnealingLR for longer training
- Add LR logging to TensorBoard

**Implementation:**
- Already implemented, just verify it's working
- Add LR to history and logging

**Expected Results:**
- Better convergence
- Avoids getting stuck in local minima

---

### 3.2 Multi-Metric Early Stopping
**Priority:** 🟢 Medium  
**Effort:** Medium  
**Expected Impact:** Medium

**Current Issue:**
- Early stopping only monitors validation loss
- With class imbalance, loss may not reflect detection quality

**Action:**
- Implement early stopping based on recall or F1
- Monitor multiple metrics
- Stop when recall/F1 plateaus

**Implementation Steps:**
1. Modify `EarlyStopping` class in `src/training/trainer.py`
2. Add metric-based stopping option
3. Track best recall/F1 alongside loss

**Code Changes:**
- Extend `EarlyStopping` to support metric-based stopping
- Add `best_recall`, `best_f1` tracking
- Stop when recall/F1 doesn't improve

**Expected Results:**
- Model stops at better point for detection task
- Better final recall/F1 scores

**Success Criteria:**
- Early stopping considers detection metrics
- Final model has better recall/F1

---

### 3.3 Extended Training with Warm-up
**Priority:** 🟢 Medium  
**Effort:** Low  
**Expected Impact:** Low-Medium

**Action:**
- Disable early stopping for first 20-30 epochs
- Allow model to learn basic patterns before stopping
- Then enable early stopping

**Implementation:**
- Add `warmup_epochs` parameter
- Skip early stopping check during warm-up

**Expected Results:**
- Model gets minimum training time
- Better baseline before stopping

---

## Phase 4: Evaluation Improvements (Medium Impact, Low-Medium Effort)

### 4.1 Temporal Split for Final Evaluation
**Priority:** 🟡 High  
**Effort:** Low  
**Expected Impact:** Medium

**Current:**
- Random split (70/10/20)
- May have data leakage
- Unrealistic evaluation

**Action:**
- Use temporal split for final evaluation
- Train on past, test on future
- More realistic scenario

**Implementation:**
- Already implemented: `--temporal-split` flag
- Use for final evaluation runs

**Command:**
```bash
python scripts/train.py --model lstm --epochs 200 --batch-size 512 --temporal-split
```

**Expected Results:**
- More realistic performance estimates
- Better understanding of production performance

**Success Criteria:**
- Temporal split shows similar or slightly worse metrics (expected)
- No significant data leakage detected

---

### 4.2 Enhanced Metrics Reporting
**Priority:** 🟢 Medium  
**Effort:** Medium  
**Expected Impact:** Medium

**Action:**
- Add Precision-Recall curves
- Report F1 at different FPR levels (1%, 5%, 10%)
- Add user-level aggregation metrics
- Track detection rate (% incidents detected)

**Implementation Steps:**
1. Add PR curve calculation
2. Add F1@FPR reporting
3. Enhance user-level metrics
4. Add detection rate calculation

**Expected Results:**
- Better understanding of model performance
- More actionable metrics for production

**Success Criteria:**
- PR curves generated
- Multiple threshold points reported
- User-level metrics improved

---

### 4.3 Threshold Analysis and Reporting
**Priority:** 🟢 Medium  
**Effort:** Low  
**Expected Impact:** Medium

**Action:**
- Log optimal threshold value
- Report metrics at multiple thresholds
- Visualize threshold vs metrics curve

**Implementation:**
- Add threshold to history
- Create threshold analysis plot
- Report in final summary

**Expected Results:**
- Better understanding of threshold sensitivity
- Easier threshold tuning

---

## Phase 5: Model Architecture (Lower Priority, Higher Effort)

### 5.1 Model Capacity Check
**Priority:** 🟢 Medium  
**Effort:** Low  
**Expected Impact:** Low-Medium

**Action:**
- Analyze train vs val loss curves
- Check if model is underfitting
- Increase capacity if needed

**Analysis:**
- If train_loss ≈ val_loss but both high → underfitting
- If train_loss << val_loss → overfitting (not current issue)

**If Underfitting:**
- Increase `hidden_dim`: 256 → 512
- Increase `embedding_dim`: 128 → 256
- Add more layers: `num_layers`: 2 → 3

**Expected Results:**
- Better model capacity
- Improved reconstruction quality

**Success Criteria:**
- Train loss decreases further
- Better separation between normal/insider

---

### 5.2 Feature Analysis
**Priority:** 🟢 Medium  
**Effort:** Medium  
**Expected Impact:** Medium

**Action:**
- Analyze per-feature reconstruction errors
- Identify most discriminative features
- Focus model attention on key features

**Implementation:**
- Add feature-level error analysis
- Visualize feature importance
- Potentially add attention mechanism

**Expected Results:**
- Understanding of which features matter
- Potential for feature selection/engineering

---

## Phase 6: Advanced Techniques (Future Work)

### 6.1 Attention Mechanisms
**Priority:** 🔵 Low (Future)  
**Effort:** High  
**Expected Impact:** Medium-High

**Action:**
- Add attention to LSTM encoder
- Focus on anomalous parts of sequences
- Better feature importance

**Implementation:**
- Modify LSTM encoder
- Add attention layer
- Visualize attention weights

---

### 6.2 Contrastive Learning
**Priority:** 🔵 Low (Future)  
**Effort:** High  
**Expected Impact:** High

**Action:**
- Add contrastive loss
- Push normal/insider embeddings apart
- Better separation in latent space

**Implementation:**
- Add contrastive loss term
- Requires labeled data
- More complex training

---

### 6.3 Variational Autoencoder (VAE)
**Priority:** 🔵 Low (Future)  
**Effort:** High  
**Expected Impact:** Medium-High

**Action:**
- Convert to VAE architecture
- Better latent space representation
- Probabilistic anomaly detection

**Implementation:**
- Major architecture change
- Requires significant refactoring

---

## Implementation Timeline

### Week 1: Quick Wins
- [ ] Day 1-2: Reduce batch size, test training
- [ ] Day 3-4: Implement threshold optimization
- [ ] Day 5: Increase early stopping patience
- [ ] Day 6-7: Run experiments, compare results

### Week 2: Class Imbalance
- [ ] Day 1-3: Implement oversampling in batches
- [ ] Day 4-5: Implement weighted loss (if feasible)
- [ ] Day 6-7: Run experiments, compare results

### Week 3: Training & Evaluation
- [ ] Day 1-2: Multi-metric early stopping
- [ ] Day 3-4: Enhanced metrics reporting
- [ ] Day 5-6: Temporal split evaluation
- [ ] Day 7: Final comparison and analysis

### Week 4: Analysis & Optimization
- [ ] Day 1-2: Model capacity analysis
- [ ] Day 3-4: Feature importance analysis
- [ ] Day 5-7: Fine-tuning and final experiments

---

## Success Metrics

### Minimum Viable Improvement (MVP)
- ✅ Session-Level Recall: > 10%
- ✅ User-Level Recall: > 20%
- ✅ Incidents Detected: > 10/70 (14%)
- ✅ F1 Score: > 0.05 (session-level)

### Target Improvement
- 🎯 Session-Level Recall: 20-30%
- 🎯 User-Level Recall: 30-40%
- 🎯 Incidents Detected: 15-20/70 (21-29%)
- 🎯 F1 Score: 0.10-0.15 (session-level)

### Stretch Goals
- 🚀 Session-Level Recall: > 30%
- 🚀 User-Level Recall: > 40%
- 🚀 Incidents Detected: > 20/70 (29%)
- 🚀 F1 Score: > 0.15

---

## Configuration Changes Summary

### config.yaml Updates
```yaml
training:
  batch_size: 512  # Changed from 64 (or 4096 via CLI)
  patience: 20      # Changed from 10
  learning_rate: 0.001  # Keep same
  # Add new parameters:
  warmup_epochs: 20
  oversample_insider: true
  insider_weight: 25.0  # For weighted loss

evaluation:
  threshold_method: f1_optimal  # Changed from percentile
  percentile: 95  # Fallback
  fpr_targets: [0.01, 0.05, 0.10]  # Report F1 at these FPRs
```

### Command-Line Usage
```bash
# Phase 1: Quick wins
python scripts/train.py --model lstm --epochs 200 --batch-size 512

# Phase 2: With oversampling (after implementation)
python scripts/train.py --model lstm --epochs 200 --batch-size 512 --oversample-insider

# Final evaluation: Temporal split
python scripts/train.py --model lstm --epochs 200 --batch-size 512 --temporal-split
```

---

## Risk Mitigation

### Potential Issues
1. **Oversampling may cause overfitting**
   - Mitigation: Monitor train/val gap, use regularization
   
2. **Smaller batch size → slower training**
   - Mitigation: Use gradient accumulation if needed
   
3. **Threshold optimization may overfit to validation set**
   - Mitigation: Use separate threshold validation set or cross-validation
   
4. **Changes may not improve metrics**
   - Mitigation: Test each change independently, keep baseline

---

## Next Steps

1. **Review this plan** - Ensure all items align with goals
2. **Prioritize phases** - Start with Phase 1 (Quick Wins)
3. **Set up experiments** - Create experiment tracking
4. **Implement Phase 1** - Batch size, threshold, patience
5. **Evaluate results** - Compare against baseline
6. **Iterate** - Move to Phase 2 if Phase 1 successful

---

## Notes

- **Baseline**: Current results (recall=0.87%, F1=0.0042, incidents=2/70)
- **Experiments**: Track all experiments with clear naming
- **Version Control**: Tag baseline model and each improvement phase
- **Documentation**: Document all changes and results

---

**Last Updated:** 2026-01-29  
**Status:** Ready for Implementation
