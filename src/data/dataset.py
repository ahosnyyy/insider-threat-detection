"""
Dataset preparation module.
Moves data preparation logic out of features.py for better separation of concerns.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .features import FeatureExtractor, SequenceBuilder

import hashlib
import json
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

def _compute_config_hash(
    feature_names: List[str],
    sequence_length: int,
    split_ratios: tuple,
    temporal: bool,
    insider_count: int,
    role_features: str = "none",
    role_mapping_file: Optional[Union[str, Path]] = None,
    db_path: Path = Path("data/processed/cert.duckdb"),
    split_by_total: bool = True,
    oversample: bool = False,
    oversample_target_positive_rate: float = 0.1,
) -> str:
    """Compute a unique hash for the dataset configuration."""
    mapping_mtime = 0
    if role_mapping_file:
        p = Path(role_mapping_file)
        if p.exists():
            mapping_mtime = p.stat().st_mtime
    config = {
        "feature_names": sorted(feature_names),
        "sequence_length": sequence_length,
        "split_ratios": split_ratios,
        "temporal": temporal,
        "insider_count": insider_count,
        "role_features": role_features,
        "role_mapping_mtime": mapping_mtime,
        "db_mtime": db_path.stat().st_mtime if db_path.exists() else 0,
        "split_by_total": split_by_total,
        "oversample": oversample,
        "oversample_target_positive_rate": oversample_target_positive_rate,
    }
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def prepare_training_data(
    df: pd.DataFrame,
    sequence_length: int = 100,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    normal_only_train: bool = True,
    insider_users: Optional[List[str]] = None,
    insider_incidents: Optional[List[Dict]] = None,
    seed: int = 42,
    use_cache: bool = True,
    role_features: str = "none",
    role_mapping_file: Optional[Union[str, Path]] = None,
    oversample: bool = False,
    oversample_target_positive_rate: float = 0.1,
) -> Dict[str, Any]:
    """
    Prepare data for autoencoder training with train/val/test split.
    
    Split strategy (70/10/20 of TOTAL data size; train/val normal-only):
    - Train: 70% of (normal + insider) in size, all normal
    - Val: 10% of total in size, all normal
    - Test: remaining 20% = remaining normal + ALL insider sessions
    
    Session-level labels:
    - Uses incident time ranges to label ONLY sessions during malicious activity
    - Normal sessions from insider users are labeled as normal
    
    Args:
        df: Session features DataFrame
        sequence_length: Max sequence length
        train_ratio: Training set ratio (default 0.7)
        val_ratio: Validation set ratio (default 0.1)
        test_ratio: Test set ratio (default 0.2)
        normal_only_train: If True, train/val only on normal users
        insider_users: List of known insider user IDs
        insider_incidents: List of dicts with 'user', 'start', 'end' for session-level labels
        seed: Random seed for reproducibility
        use_cache: If True, load/save to disk cache based on config hash
        role_features: "none" | "roles" | "units". roles=42 one-hot, units=6 from role_mapping_file.
        role_mapping_file: Path to role_units.yaml; required when role_features="units".
        
    Returns:
        Dict with train/val/test sequences, masks, session IDs, labels, and extractor
    """
    # Check cache first
    cache_path = None
    if use_cache:
        try:
            config_hash = _compute_config_hash(
                feature_names=list(df.columns),
                sequence_length=sequence_length,
                split_ratios=(train_ratio, val_ratio, test_ratio),
                temporal=False,
                insider_count=len(insider_users or []),
                role_features=role_features,
                role_mapping_file=role_mapping_file,
                oversample=oversample,
                oversample_target_positive_rate=oversample_target_positive_rate,
            )
            cache_dir = Path("data/processed/cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"random_{config_hash}.pkl"
            
            if cache_path.exists():
                logger.info(f"Loading cached training data from {cache_path}")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except BaseException as e:
            # Some pickle/buffer failures can surface as SystemError with empty message.
            msg = f"{type(e).__name__}: {e}"
            logger.warning(f"Cache load failed for {cache_path}: {msg}")
            # If cache seems corrupted/unreadable, remove it to avoid repeated failures.
            if cache_path is not None:
                try:
                    cache_path.unlink(missing_ok=True)
                    logger.warning(f"Removed unreadable cache file: {cache_path}")
                except Exception as del_e:
                    logger.warning(f"Failed to remove cache file {cache_path}: {type(del_e).__name__}: {del_e}")

    np.random.seed(seed)
    
    logger.info(f"Preparing training data from {len(df):,} sessions")
    logger.info(f"Split: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test")
    
    # Sort by user and time
    df = df.copy()
    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"])
        df = df.sort_values(["user_id", "start_time"]).reset_index(drop=True)
    
    # Identify insider sessions using time ranges (session-level labels)
    insider_user_set = set(insider_users or [])
    
    # Create BOTH labels: user-level and session-level
    def create_labels(row):
        user_id = row["user_id"]
        
        # User-level: 1 if user is in insider list (regardless of timing)
        user_label = 1 if user_id in insider_user_set else 0
        
        # Session-level: 1 only if session falls within incident window
        session_label = 0
        if user_id in insider_user_set and insider_incidents:
            session_time = row.get("start_time")
            if session_time is not None and pd.notna(session_time):
                session_time = pd.to_datetime(session_time)
                
                for incident in insider_incidents:
                    if incident["user"] == user_id:
                        start = incident.get("start")
                        end = incident.get("end")
                        
                        if start is not None and end is not None:
                            if start <= session_time <= end:
                                session_label = 1
                                break
        elif user_id in insider_user_set:
            # Fallback: if no incidents provided, use user-level
            session_label = user_label
        
        return pd.Series({'label_user': user_label, 'label_session': session_label})
    
    labels_df = df.apply(create_labels, axis=1)
    df["label_user"] = labels_df["label_user"]
    df["label_session"] = labels_df["label_session"]
    df["label"] = df["label_session"]  # Default to session-level
    
    n_user_insiders = df["label_user"].sum()
    n_session_insiders = df["label_session"].sum()
    logger.info(f"User-level labels: {n_user_insiders:,} insider sessions ({n_user_insiders/len(df)*100:.2f}%)")
    logger.info(f"Session-level labels: {n_session_insiders:,} insider sessions ({n_session_insiders/len(df)*100:.3f}%)")
    
    # Separate data by USER (not session) - insider users go to test, normal users to train/val/test
    # This ensures proper evaluation: test set has ALL sessions from insider users
    if normal_only_train:
        normal_user_df = df[df["label_user"] == 0].copy()  # Sessions from normal users
    else:
        logger.warning("normal_only_train=False: Including insider users in training data! (Not recommended for Anomaly Detection)")
        normal_user_df = df.copy() # Everyone is treated as 'normal' for training
    
    insider_user_df = df[df["label_user"] == 1].copy()  # ALL sessions from insider users
    
    logger.info(f"Normal user sessions: {len(normal_user_df):,}, Insider user sessions: {len(insider_user_df):,}")
    
    # All unique roles from entire dataset (for role_features='roles' only)
    all_roles = None
    if role_features == "roles" and "role" in df.columns:
        all_roles = sorted(df["role"].dropna().unique().tolist())
    
    role_mapping_path = Path(role_mapping_file) if role_mapping_file else None
    if role_features == "units" and (not role_mapping_path or not role_mapping_path.exists()):
        raise FileNotFoundError("role_features='units' requires role_mapping_file to a valid path")
    
    extractor = FeatureExtractor(
        role_features=role_features,
        role_mapping_path=role_mapping_path,
    )
    if role_features == "roles" and all_roles:
        extractor.fit(normal_user_df, role_categories=all_roles)
    else:
        extractor.fit(normal_user_df)
    
    # Transform normal user data
    normal_features = extractor.transform(normal_user_df)
    
    # Build sequences for normal users
    builder = SequenceBuilder(sequence_length=sequence_length)
    padding_value = extractor.get_padding_value()
    normal_sequences, normal_masks, normal_session_ids = builder.build_user_sequences(
        normal_user_df.reset_index(drop=True), normal_features, padding_value=padding_value
    )
    normal_labels_session = np.zeros(len(normal_sequences), dtype=np.int32)
    normal_labels_user = np.zeros(len(normal_sequences), dtype=np.int32)
    
    logger.info(f"Built {len(normal_sequences):,} normal user sequences")
    
    n_normal = len(normal_sequences)
    
    # Build sequences for ALL insider user sessions (for test set)
    if len(insider_user_df) > 0:
        insider_features = extractor.transform(insider_user_df)
        insider_sequences, insider_masks, insider_session_ids = builder.build_user_sequences(
            insider_user_df.reset_index(drop=True), insider_features, padding_value=padding_value
        )
        
        # Map session_id to labels from the DataFrame
        session_to_label_session = dict(zip(insider_user_df["session_id"], insider_user_df["label_session"]))
        session_to_label_user = dict(zip(insider_user_df["session_id"], insider_user_df["label_user"]))
        
        insider_labels_session = np.array([session_to_label_session.get(sid, 0) for sid in insider_session_ids], dtype=np.int32)
        insider_labels_user = np.array([session_to_label_user.get(sid, 1) for sid in insider_session_ids], dtype=np.int32)
        
        logger.info(f"Built {len(insider_sequences):,} insider user sequences")
        logger.info(f"  Session-level insiders: {insider_labels_session.sum():,}")
        logger.info(f"  User-level insiders: {insider_labels_user.sum():,}")
    else:
        insider_sequences = np.array([]).reshape(0, sequence_length, extractor.get_feature_dim())
        insider_masks = np.array([]).reshape(0, sequence_length)
        insider_session_ids = []
        insider_labels_session = np.array([], dtype=np.int32)
        insider_labels_user = np.array([], dtype=np.int32)
    
    # Split: 70/10/20 of TOTAL data (normal + insider). Train and val are normal-only; test = remaining normal + all insiders.
    n_insider = len(insider_sequences)
    total = n_normal + n_insider
    n_train_target = int(total * train_ratio)
    n_val_target = int(total * val_ratio)
    n_test_target = total - n_train_target - n_val_target
    
    if n_normal < n_train_target + n_val_target:
        logger.warning(
            f"Not enough normal sequences for 70/10/20 of total: need {n_train_target + n_val_target:,} normal, have {n_normal:,}. "
            "Using all normal for train/val; test = all insiders only."
        )
        n_val = min(n_val_target, n_normal)
        n_train = n_normal - n_val
    else:
        n_train = n_train_target
        n_val = n_val_target
    
    indices = np.random.permutation(n_normal)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_normal_idx = indices[n_train + n_val:]
    
    logger.info(f"Split (70/10/20 of total={total:,}): train={n_train:,} (normal), val={n_val:,} (normal), test_normal={len(test_normal_idx):,} + insiders={n_insider:,} = {len(test_normal_idx) + n_insider:,}")
    
    # Combine test set: normal user test portion + ALL insider user sessions
    test_sequences = np.concatenate([normal_sequences[test_normal_idx], insider_sequences])
    test_masks = np.concatenate([normal_masks[test_normal_idx], insider_masks])
    test_labels = np.concatenate([normal_labels_session[test_normal_idx], insider_labels_session])  # Session-level
    test_labels_user = np.concatenate([normal_labels_user[test_normal_idx], insider_labels_user])    # User-level
    test_session_ids = [normal_session_ids[i] for i in test_normal_idx] + insider_session_ids
    
    # Build test_timestamps and test_user_ids for TTD calculation
    session_to_time = {}
    session_to_user = {}
    
    # helper to populate lookups
    def populate_lookups(df_source):
        if "start_time" in df_source.columns and "session_id" in df_source.columns:
            session_to_time.update(dict(zip(df_source["session_id"], df_source["start_time"])))
        if "user_id" in df_source.columns and "session_id" in df_source.columns:
            session_to_user.update(dict(zip(df_source["session_id"], df_source["user_id"])))
            
    populate_lookups(normal_user_df)
    populate_lookups(insider_user_df)
    
    test_timestamps = [session_to_time.get(sid) for sid in test_session_ids]
    test_user_ids = [session_to_user.get(sid) for sid in test_session_ids]
    
    logger.info(f"Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_sequences):,}")
    logger.info(f"Test set (session-level): {(test_labels == 0).sum():,} normal, {(test_labels == 1).sum():,} insider")
    logger.info(f"Test set (user-level): {(test_labels_user == 0).sum():,} normal, {(test_labels_user == 1).sum():,} insider")

    # Optional: oversample session-level insider sessions in test to reach target positive rate (diffuse by shuffling)
    if oversample and 0 < oversample_target_positive_rate < 1:
        n_test_normal = int((test_labels == 0).sum())
        n_insider_session = int((test_labels == 1).sum())
        if n_insider_session > 0:
            target_insiders = oversample_target_positive_rate * n_test_normal / (1 - oversample_target_positive_rate)
            K = max(1, int(np.ceil(target_insiders / n_insider_session)))
            insider_mask = test_labels == 1
            # Duplicate session-level insider rows (K-1) more times
            seq_ins = test_sequences[insider_mask]  # (n_insider_session, ...)
            mask_ins = test_masks[insider_mask]
            lab_ins = test_labels[insider_mask]
            lab_user_ins = test_labels_user[insider_mask]
            sid_ins = [test_session_ids[i] for i in np.where(insider_mask)[0]]
            ts_ins = [test_timestamps[i] for i in np.where(insider_mask)[0]]
            uid_ins = [test_user_ids[i] for i in np.where(insider_mask)[0]]
            # Repeat (K-1) times and append
            for _ in range(K - 1):
                test_sequences = np.concatenate([test_sequences, seq_ins])
                test_masks = np.concatenate([test_masks, mask_ins])
                test_labels = np.concatenate([test_labels, lab_ins])
                test_labels_user = np.concatenate([test_labels_user, lab_user_ins])
                test_session_ids = test_session_ids + sid_ins
                test_timestamps = test_timestamps + ts_ins
                test_user_ids = test_user_ids + uid_ins
            # Shuffle test set (diffuse)
            perm = np.random.permutation(len(test_sequences))
            test_sequences = test_sequences[perm]
            test_masks = test_masks[perm]
            test_labels = test_labels[perm]
            test_labels_user = test_labels_user[perm]
            test_session_ids = [test_session_ids[i] for i in perm]
            test_timestamps = [test_timestamps[i] for i in perm]
            test_user_ids = [test_user_ids[i] for i in perm]
            logger.info(f"Oversampled session-level insiders in test: {n_insider_session:,} -> {n_insider_session * K:,} (target rate={oversample_target_positive_rate:.2%}), test size={len(test_sequences):,}, shuffled (diffused)")
        else:
            logger.warning("Oversample requested but no session-level insider sessions in test; skipping")
    elif oversample:
        logger.warning("Oversample requested but oversample_target_positive_rate must be in (0,1); skipping")

    result = {
        # Training data (normal only)
        "train_sequences": normal_sequences[train_idx],
        "train_masks": normal_masks[train_idx],
        "train_session_ids": [normal_session_ids[i] for i in train_idx],
        "train_labels": normal_labels_session[train_idx],
        # Validation data (normal only, for early stopping)
        "val_sequences": normal_sequences[val_idx],
        "val_masks": normal_masks[val_idx],
        "val_session_ids": [normal_session_ids[i] for i in val_idx],
        "val_labels": normal_labels_session[val_idx],
        # Test data (normal + insider, for evaluation metrics)
        "test_sequences": test_sequences,
        "test_masks": test_masks,
        "test_session_ids": test_session_ids,
        "test_timestamps": test_timestamps,     # For TTD calculation
        "test_user_ids": test_user_ids,         # For TTD calculation
        "test_labels": test_labels,             # Session-level (default)
        "test_labels_user": test_labels_user,   # User-level (for comparison)
        # Metadata
        "feature_extractor": extractor,
        "feature_dim": extractor.get_feature_dim(),
        "feature_names": extractor.get_feature_names(),
        "split_type": "random",
        "n_user_level_insiders": int(n_user_insiders),
        "n_session_level_insiders": int(n_session_insiders),
    }

    if cache_path:
        try:
            logger.info(f"Saving training data to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(f"Cache saved successfully ({cache_size_mb:.1f} MB)")
        except MemoryError:
            logger.warning(f"Failed to save cache: Out of memory (data too large)")
        except OSError as e:
            logger.warning(f"Failed to save cache: {e} (check disk space)")
        except Exception as e:
            logger.warning(f"Failed to save cache: {type(e).__name__}: {e}")
            
    return result


def prepare_training_data_temporal(
    df: pd.DataFrame,
    sequence_length: int = 100,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    insider_users: Optional[List[str]] = None,
    insider_incidents: Optional[List[Dict]] = None,
    use_cache: bool = True,
    role_features: str = "none",
    role_mapping_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Prepare data with TEMPORAL split (no data leakage).
    
    Split is based on time, not random:
    - Train: First 70% of timeline (normal users only)
    - Val: Next 10% of timeline (normal users only)
    - Test: Last 20% of timeline (ALL users including insiders)
    
    This prevents the model from "seeing the future" during training.
    
    Args:
        df: Session features DataFrame
        sequence_length: Max sequence length
        train_ratio: Training set ratio (default 0.7)
        val_ratio: Validation set ratio (default 0.1)
        test_ratio: Test set ratio (default 0.2)
        insider_users: List of known insider user IDs
        insider_incidents: List of dicts with 'user', 'start', 'end' for session-level labels
        use_cache: If True, load/save to disk cache based on config hash
        role_features: "none" | "roles" | "units"
        role_mapping_file: Path to role_units.yaml when role_features="units"
        
    Returns:
        Dict with train/val/test sequences, masks, session IDs, labels, and extractor
    """
    # Check cache first
    cache_path = None
    if use_cache:
        try:
            config_hash = _compute_config_hash(
                feature_names=list(df.columns),
                sequence_length=sequence_length,
                split_ratios=(train_ratio, val_ratio, test_ratio),
                temporal=True,
                insider_count=len(insider_users or []),
                role_features=role_features,
                role_mapping_file=role_mapping_file,
            )
            cache_dir = Path("data/processed/cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"temporal_{config_hash}.pkl"
            
            if cache_path.exists():
                logger.info(f"Loading cached temporal training data from {cache_path}")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except BaseException as e:
            msg = f"{type(e).__name__}: {e}"
            logger.warning(f"Cache load failed for {cache_path}: {msg}")
            if cache_path is not None:
                try:
                    cache_path.unlink(missing_ok=True)
                    logger.warning(f"Removed unreadable cache file: {cache_path}")
                except Exception as del_e:
                    logger.warning(f"Failed to remove cache file {cache_path}: {type(del_e).__name__}: {del_e}")

    logger.info(f"Preparing training data with TEMPORAL split")
    logger.info(f"Split: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test")
    
    # Ensure we have timestamps
    if "start_time" not in df.columns:
        raise ValueError("Temporal split requires 'start_time' column")
    
    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df = df.sort_values("start_time").reset_index(drop=True)
    
    # Find cutoff dates
    min_date = df["start_time"].min()
    max_date = df["start_time"].max()
    total_days = (max_date - min_date).days
    
    train_cutoff = min_date + pd.Timedelta(days=int(total_days * train_ratio))
    val_cutoff = min_date + pd.Timedelta(days=int(total_days * (train_ratio + val_ratio)))
    
    logger.info(f"Date range: {min_date.date()} to {max_date.date()} ({total_days} days)")
    logger.info(f"Train cutoff: {train_cutoff.date()}, Val cutoff: {val_cutoff.date()}")
    
    # Label insider sessions - create BOTH user-level and session-level
    insider_user_set = set(insider_users or [])
    
    def create_labels(row):
        user_id = row["user_id"]
        
        # User-level: 1 if user is in insider list
        user_label = 1 if user_id in insider_user_set else 0
        
        # Session-level: 1 only during incident window
        session_label = 0
        if user_id in insider_user_set and insider_incidents:
            session_time = row["start_time"]
            for incident in insider_incidents:
                if incident["user"] == user_id:
                    start = incident.get("start")
                    end = incident.get("end")
                    if start is not None and end is not None:
                        if start <= session_time <= end:
                            session_label = 1
                            break
        elif user_id in insider_user_set:
            session_label = user_label  # Fallback
        
        return pd.Series({'label_user': user_label, 'label_session': session_label})
    
    labels_df = df.apply(create_labels, axis=1)
    df["label_user"] = labels_df["label_user"]
    df["label_session"] = labels_df["label_session"]
    df["label"] = df["label_session"]  # Default to session-level
    
    n_user_insiders = df["label_user"].sum()
    n_session_insiders = df["label_session"].sum()
    
    # Split by time
    train_df = df[(df["start_time"] < train_cutoff) & (df["label"] == 0)]
    val_df = df[(df["start_time"] >= train_cutoff) & (df["start_time"] < val_cutoff) & (df["label"] == 0)]
    test_df = df[df["start_time"] >= val_cutoff]  # ALL users in test period
    
    logger.info(f"Train period: {len(train_df):,} sessions (normal only)")
    logger.info(f"Val period: {len(val_df):,} sessions (normal only)")
    logger.info(f"Test period: {len(test_df):,} sessions (all users)")
    logger.info(f"  User-level: {(test_df['label_user'] == 1).sum():,} insider, Session-level: {(test_df['label_session'] == 1).sum():,} insider")
    
    all_roles = None
    if role_features == "roles" and "role" in df.columns:
        all_roles = sorted(df["role"].dropna().unique().tolist())
    
    role_mapping_path = Path(role_mapping_file) if role_mapping_file else None
    if role_features == "units" and (not role_mapping_path or not role_mapping_path.exists()):
        raise FileNotFoundError("role_features='units' requires role_mapping_file to a valid path")
    
    extractor = FeatureExtractor(
        role_features=role_features,
        role_mapping_path=role_mapping_path,
    )
    if role_features == "roles" and all_roles:
        extractor.fit(train_df, role_categories=all_roles)
    else:
        extractor.fit(train_df)
    
    # Build sequences
    builder = SequenceBuilder(sequence_length=sequence_length)
    
    # Transform and build for each split
    padding_value = extractor.get_padding_value()
    def build_split(split_df, name):
        if len(split_df) == 0:
            return (
                np.array([]).reshape(0, sequence_length, extractor.get_feature_dim()),
                np.array([]).reshape(0, sequence_length),
                [],
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                [],
                [],
            )
        features = extractor.transform(split_df.reset_index(drop=True))
        seqs, masks, sess_ids = builder.build_user_sequences(
            split_df.reset_index(drop=True), features, padding_value=padding_value
        )
        
        # Map session IDs to both label types and timestamps
        session_to_label = dict(zip(split_df["session_id"], split_df["label_session"]))
        session_to_label_user = dict(zip(split_df["session_id"], split_df["label_user"]))
        session_to_time = dict(zip(split_df["session_id"], split_df["start_time"]))
        session_to_user = dict(zip(split_df["session_id"], split_df["user_id"]))
        
        labels = np.array([session_to_label.get(sid, 0) for sid in sess_ids], dtype=np.int32)
        labels_user = np.array([session_to_label_user.get(sid, 0) for sid in sess_ids], dtype=np.int32)
        timestamps = [session_to_time.get(sid) for sid in sess_ids]
        user_ids = [session_to_user.get(sid) for sid in sess_ids]
        
        logger.info(f"Built {len(seqs):,} {name} sequences")
        return seqs, masks, sess_ids, labels, labels_user, timestamps, user_ids
    
    train_seqs, train_masks, train_ids, train_labels, _, _, _ = build_split(train_df, "train")
    val_seqs, val_masks, val_ids, val_labels, _, _, _ = build_split(val_df, "val")
    test_seqs, test_masks, test_ids, test_labels, test_labels_user, test_timestamps, test_user_ids = build_split(test_df, "test")
    
    logger.info(f"Test labels (session): {(test_labels == 0).sum():,} normal, {(test_labels == 1).sum():,} insider")
    logger.info(f"Test labels (user): {(test_labels_user == 0).sum():,} normal, {(test_labels_user == 1).sum():,} insider")
    
    result = {
        "train_sequences": train_seqs,
        "train_masks": train_masks,
        "train_session_ids": train_ids,
        "train_labels": train_labels,
        "val_sequences": val_seqs,
        "val_masks": val_masks,
        "val_session_ids": val_ids,
        "val_labels": val_labels,
        "test_sequences": test_seqs,
        "test_masks": test_masks,
        "test_session_ids": test_ids,
        "test_timestamps": test_timestamps,     # For TTD calculation
        "test_user_ids": test_user_ids,         # For TTD calculation
        "test_labels": test_labels,             # Session-level (default)
        "test_labels_user": test_labels_user,   # User-level (for comparison)
        "feature_extractor": extractor,
        "feature_dim": extractor.get_feature_dim(),
        "feature_names": extractor.get_feature_names(),
        "split_type": "temporal",
        "train_cutoff": str(train_cutoff.date()),
        "val_cutoff": str(val_cutoff.date()),
        "n_user_level_insiders": int(n_user_insiders),
        "n_session_level_insiders": int(n_session_insiders),
    }

    if cache_path:
        try:
            logger.info(f"Saving temporal training data to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(f"Cache saved successfully ({cache_size_mb:.1f} MB)")
        except MemoryError:
            logger.warning(f"Failed to save cache: Out of memory (data too large)")
        except OSError as e:
            logger.warning(f"Failed to save cache: {e} (check disk space)")
        except Exception as e:
            logger.warning(f"Failed to save cache: {type(e).__name__}: {e}")
            
    return result
