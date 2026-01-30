#!/usr/bin/env python
"""
Verify insider session counts used in the dataset split.
Checks that 26,193 = number of sessions from ground-truth insider users (user-level).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_session_dataframe
from src.utils import load_config, load_ground_truth


def main():
    cfg = load_config()
    db_path = Path(cfg["data"]["database"])
    answers_dir = Path("data/raw/answers")

    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        return 1

    print("Loading session data...")
    df = get_session_dataframe(db_path)
    print(f"Total sessions in DB: {len(df):,}")

    if not answers_dir.exists():
        print(f"Error: Answers dir not found: {answers_dir}")
        return 1

    print(f"Loading ground truth from {answers_dir} (dataset 4.2)...")
    ground_truth = load_ground_truth(answers_dir, dataset="4.2")
    insider_users = set(ground_truth.get("insider_users", []))
    insider_incidents = ground_truth.get("insider_incidents", [])
    print(f"Insider users (ground truth): {len(insider_users)}")

    # User-level: sessions where user_id is in insider_users
    mask_user = df["user_id"].isin(insider_users)
    n_insider_user_sessions = mask_user.sum()
    print(f"\nSessions from insider users (user-level): {n_insider_user_sessions:,}")
    print(f"  Expected (from prepare_training_data log): 26,193")

    # Session-level: sessions during an incident (optional check)
    if "start_time" in df.columns and insider_incidents:
        import pandas as pd
        df = df.copy()
        df["start_time"] = pd.to_datetime(df["start_time"])
        n_session_insider = 0
        for incident in insider_incidents:
            u = incident.get("user")
            start, end = incident.get("start"), incident.get("end")
            if u is None or start is None or end is None:
                continue
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            n_session_insider += ((df["user_id"] == u) & (df["start_time"] >= start) & (df["start_time"] <= end)).sum()
        print(f"Sessions during incidents (session-level): {n_session_insider:,}")
        print(f"  Expected (from prepare_training_data log): 1,831")

    if n_insider_user_sessions == 26193:
        print("\n26,193 is correct (matches user-level insider session count).")
    else:
        print(f"\nMismatch: got {n_insider_user_sessions:,}, expected 26,193.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
