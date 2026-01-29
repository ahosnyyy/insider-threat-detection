# Implementation Checklist - Quick Reference

## Phase 1: Quick Wins ⚡

### 1.1 Batch Size Reduction
- [ ] Change batch size from 4096 → 512 in training command
- [ ] Run training: `python scripts/train.py --model lstm --epochs 200 --batch-size 512`
- [ ] Verify: More epochs before early stopping
- [ ] Verify: Smoother loss curves
- [ ] Test with batch-size 1024 if stable
- [ ] Document results

### 1.2 Threshold Optimization
- [ ] Read `src/training/trainer.py` `_evaluate_test_set()` method
- [ ] Create `find_optimal_threshold()` function
- [ ] Modify evaluation to optimize threshold on validation set
- [ ] Use F1 score as optimization metric
- [ ] Log optimal threshold value
- [ ] Run training and verify improved metrics
- [ ] Document threshold value and improvement

### 1.3 Early Stopping Patience
- [ ] Update `config.yaml`: `patience: 20`
- [ ] Or pass via command line if supported
- [ ] Run training and verify more epochs
- [ ] Document improvement

**Phase 1 Success Criteria:**
- [ ] Recall > 5%
- [ ] F1 > 0.02
- [ ] Training runs for > 20 epochs

---

## Phase 2: Class Imbalance 🔄

### 2.1 Oversampling Insider Sessions
- [ ] Read `src/data/dataset.py` `prepare_training_data()` function
- [ ] Understand current data loading logic
- [ ] Add `oversample_insider` parameter
- [ ] Implement balanced batch creation
- [ ] Ensure insider sessions included in training
- [ ] Test with small dataset first
- [ ] Run full training
- [ ] Verify improved recall

### 2.2 Weighted Loss (Optional)
- [ ] Check if insider labels available during training
- [ ] If yes: Modify `_masked_mse_loss()` in trainer.py
- [ ] Add sample weights based on labels
- [ ] Test weight values: 20, 30, 50
- [ ] Run training and compare results
- [ ] If no labels: Skip this, use oversampling only

**Phase 2 Success Criteria:**
- [ ] Recall > 10%
- [ ] F1 > 0.05
- [ ] Incidents detected > 5/70

---

## Phase 3: Training Strategy 🎯

### 3.1 Learning Rate Scheduling
- [ ] Verify ReduceLROnPlateau is working
- [ ] Check LR values in TensorBoard/logs
- [ ] Add LR to history tracking
- [ ] Document LR schedule behavior

### 3.2 Multi-Metric Early Stopping
- [ ] Modify `EarlyStopping` class in trainer.py
- [ ] Add recall/F1 tracking
- [ ] Implement metric-based stopping
- [ ] Test with validation set
- [ ] Compare with loss-based stopping

### 3.3 Warm-up Period
- [ ] Add `warmup_epochs` parameter
- [ ] Disable early stopping during warm-up
- [ ] Test with warmup_epochs=20
- [ ] Compare results

**Phase 3 Success Criteria:**
- [ ] Better convergence
- [ ] Improved final metrics

---

## Phase 4: Evaluation 📊

### 4.1 Temporal Split
- [ ] Run training with `--temporal-split` flag
- [ ] Compare metrics with random split
- [ ] Document differences
- [ ] Use temporal split for final evaluation

### 4.2 Enhanced Metrics
- [ ] Add Precision-Recall curve calculation
- [ ] Add F1@FPR reporting (1%, 5%, 10%)
- [ ] Enhance user-level metrics
- [ ] Add detection rate (% incidents detected)
- [ ] Update final summary output

### 4.3 Threshold Analysis
- [ ] Log threshold values in history
- [ ] Create threshold vs metrics visualization
- [ ] Report multiple threshold points
- [ ] Document threshold sensitivity

**Phase 4 Success Criteria:**
- [ ] Comprehensive metrics reported
- [ ] Better understanding of model behavior

---

## Phase 5: Model Architecture 🔧

### 5.1 Capacity Check
- [ ] Analyze train vs val loss curves
- [ ] Determine if underfitting
- [ ] If underfitting: Increase hidden_dim or embedding_dim
- [ ] Test with larger model
- [ ] Compare results

### 5.2 Feature Analysis
- [ ] Add per-feature error analysis
- [ ] Visualize feature importance
- [ ] Identify discriminative features
- [ ] Document findings

**Phase 5 Success Criteria:**
- [ ] Model capacity optimized
- [ ] Key features identified

---

## Testing & Validation ✅

### After Each Phase
- [ ] Run training with new changes
- [ ] Compare metrics with baseline
- [ ] Document improvements/regressions
- [ ] Save model checkpoint
- [ ] Update experiment log

### Final Validation
- [ ] Run complete training pipeline
- [ ] Verify all improvements work together
- [ ] Compare final metrics with baseline
- [ ] Document final results
- [ ] Create summary report

---

## Baseline Metrics (Current)
```
Session-Level:
  Recall: 0.87%
  Precision: 0.28%
  F1: 0.0042
  AUC-ROC: 0.7311

User-Level:
  Recall: 4.86%
  Precision: 22.12%
  F1: 0.0797
  AUC-ROC: 0.6820

Incidents Detected: 2/70 (2.9%)
```

## Target Metrics (Phase 1)
```
Session-Level:
  Recall: > 5%
  F1: > 0.02

Incidents Detected: > 5/70
```

## Target Metrics (Phase 2)
```
Session-Level:
  Recall: > 10%
  F1: > 0.05

User-Level:
  Recall: > 20%

Incidents Detected: > 10/70
```

## Target Metrics (Final)
```
Session-Level:
  Recall: 20-30%
  F1: 0.10-0.15

User-Level:
  Recall: 30-40%

Incidents Detected: 15-20/70 (21-29%)
```

---

## Experiment Tracking

### Naming Convention
```
experiment_phase1_batch512_threshold_opt
experiment_phase2_oversample_insider
experiment_phase3_multimetric_stopping
experiment_final_all_improvements
```

### What to Track
- [ ] Experiment name
- [ ] Configuration changes
- [ ] Training metrics (loss, recall, F1)
- [ ] Test metrics
- [ ] Model checkpoint path
- [ ] Training time
- [ ] Notes/observations

---

## Quick Commands Reference

```bash
# Baseline (current)
python scripts/train.py --model lstm --epochs 200 --batch-size 4096

# Phase 1: Batch size + threshold
python scripts/train.py --model lstm --epochs 200 --batch-size 512

# Phase 2: With oversampling (after implementation)
python scripts/train.py --model lstm --epochs 200 --batch-size 512 --oversample-insider

# Final: Temporal split
python scripts/train.py --model lstm --epochs 200 --batch-size 512 --temporal-split

# Compare models
python scripts/compare_models.py models/lstm_autoencoder_best.pt models/lstm_autoencoder_phase2.pt
```

---

**Status Tracking:**
- Phase 1: ⬜ Not Started
- Phase 2: ⬜ Not Started  
- Phase 3: ⬜ Not Started
- Phase 4: ⬜ Not Started
- Phase 5: ⬜ Not Started

**Last Updated:** 2026-01-29
