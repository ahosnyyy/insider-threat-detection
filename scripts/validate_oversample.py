#!/usr/bin/env python
"""
Validate oversampling of session-level insider sessions in the test set.

Checks:
- Without --oversample: test has N normal, M session-level insider (baseline).
- With oversample: test has same N normal, M' insider (M' >= M) such that
  M' / (N + M') ≈ oversample_target_positive_rate.
- Train and val are unchanged when oversampling is enabled.
- Test set is shuffled (insider copies diffused).

Usage:
  python scripts/validate_oversample.py
  python scripts/validate_oversample.py --rate 0.2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.data import get_session_dataframe, prepare_training_data
from src.utils import load_config, load_ground_truth, setup_logging


def main():
    setup_logging()
    cfg = load_config()
    db_path = Path(cfg["data"]["database"])
    answers_dir = Path(cfg["data"].get("answers_dir", "data/raw/answers"))
    answers_dir = Path(answers_dir) if not isinstance(answers_dir, Path) else answers_dir
    target_rate = float(cfg["data"].get("oversample_target_positive_rate", 0.1))

    parser = argparse.ArgumentParser(description="Validate oversampling in prepare_training_data")
    parser.add_argument("--rate", type=float, default=None,
                        help="Target positive rate (default: from config)")
    args = parser.parse_args()
    if args.rate is not None:
        target_rate = args.rate
        if not 0 < target_rate < 1:
            print("Error: --rate must be in (0, 1)")
            return 1

    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        return 1
    if not answers_dir.exists():
        print(f"Error: Answers dir not found: {answers_dir}")
        return 1

    print("Loading session data and ground truth...")
    df = get_session_dataframe(db_path)
    ground_truth = load_ground_truth(answers_dir, dataset="4.2")
    insider_users = ground_truth.get("insider_users", [])
    insider_incidents = ground_truth.get("insider_incidents", [])

    seq_len = cfg["features"]["sequence_length"]
    train_ratio = cfg["training"]["train_split"]
    val_ratio = cfg["training"]["val_ratio"]
    test_ratio = cfg["training"]["test_ratio"]
    seed = cfg["training"]["seed"]

    # 1) Without oversample (use_cache=False for reproducible validation)
    print("\n1) Preparing data WITHOUT oversample (use_cache=False)...")
    data_no = prepare_training_data(
        df,
        sequence_length=seq_len,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        normal_only_train=len(insider_users) > 0,
        insider_users=insider_users,
        insider_incidents=insider_incidents,
        seed=seed,
        use_cache=False,
        oversample=False,
        oversample_target_positive_rate=target_rate,
    )
    n_normal_no = int((data_no["test_labels"] == 0).sum())
    n_insider_no = int((data_no["test_labels"] == 1).sum())
    total_no = len(data_no["test_sequences"])
    rate_no = n_insider_no / total_no if total_no else 0

    print(f"   Test: {total_no:,} total, {n_normal_no:,} normal (session-level), {n_insider_no:,} insider (session-level)")
    print(f"   Session-level positive rate: {rate_no:.2%}")

    # 2) With oversample
    print(f"\n2) Preparing data WITH oversample (target rate={target_rate:.2%}, use_cache=False)...")
    data_yes = prepare_training_data(
        df,
        sequence_length=seq_len,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        normal_only_train=len(insider_users) > 0,
        insider_users=insider_users,
        insider_incidents=insider_incidents,
        seed=seed,
        use_cache=False,
        oversample=True,
        oversample_target_positive_rate=target_rate,
    )
    n_normal_yes = int((data_yes["test_labels"] == 0).sum())
    n_insider_yes = int((data_yes["test_labels"] == 1).sum())
    total_yes = len(data_yes["test_sequences"])
    rate_yes = n_insider_yes / total_yes if total_yes else 0

    print(f"   Test: {total_yes:,} total, {n_normal_yes:,} normal (session-level), {n_insider_yes:,} insider (session-level)")
    print(f"   Session-level positive rate: {rate_yes:.2%}")

    # 3) Validate
    ok = True

    # Train/val unchanged
    if (data_no["train_sequences"].shape != data_yes["train_sequences"].shape or
            data_no["val_sequences"].shape != data_yes["val_sequences"].shape):
        print("\n   FAIL: Train or val shape changed with oversample (should be unchanged).")
        ok = False
    else:
        print("\n   OK: Train and val shapes unchanged with oversample.")

    # Normal count in test unchanged
    if n_normal_yes != n_normal_no:
        print(f"   FAIL: Test normal count changed {n_normal_no:,} -> {n_normal_yes:,} (should be unchanged).")
        ok = False
    else:
        print(f"   OK: Test normal count unchanged ({n_normal_no:,}).")

    # Insider count increased
    if n_insider_yes < n_insider_no:
        print(f"   FAIL: Test insider count decreased {n_insider_no:,} -> {n_insider_yes:,}.")
        ok = False
    else:
        print(f"   OK: Test insider count increased {n_insider_no:,} -> {n_insider_yes:,}.")

    # Actual rate close to target (within ~2% relative or 1 percentage point)
    tol_abs = 0.02
    tol_rel = 0.15
    if abs(rate_yes - target_rate) > tol_abs and abs(rate_yes - target_rate) / max(target_rate, 1e-6) > tol_rel:
        print(f"   FAIL: Actual rate {rate_yes:.2%} is not close to target {target_rate:.2%} (tol abs={tol_abs:.2%}, rel={tol_rel:.0%}).")
        ok = False
    else:
        print(f"   OK: Actual rate {rate_yes:.2%} ~ target {target_rate:.2%}.")

    # Shuffle: insider labels should not be all at the end (simple check: first 20% and last 20% should both contain some insiders)
    labels = data_yes["test_labels"]
    n = len(labels)
    head_ins = int(labels[: max(1, n // 5)].sum())
    tail_ins = int(labels[-max(1, n // 5) :].sum())
    if n_insider_yes >= 2 and (head_ins == 0 or tail_ins == 0):
        print("   WARN: Insiders might not be well diffused (all in one half); shuffle may be weak.")
    else:
        print("   OK: Insiders appear diffused (present in both head and tail of test order).")

    if ok:
        print("\nValidation PASSED.")
        return 0
    print("\nValidation FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
