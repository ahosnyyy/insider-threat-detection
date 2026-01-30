#!/usr/bin/env python
"""
Save Feature Extractor Script
Use this to regenerate the feature_extractor.pkl file if it wasn't saved correctly during training.
This avoids re-running the entire training pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
from src.data import FeatureExtractor, get_session_dataframe
from src.utils import setup_logging, load_config

def main():
    setup_logging()
    
    # Load config so we match the same feature settings used in training
    cfg = load_config()
    features_cfg = cfg.get("features", {})
    role_features = features_cfg.get("role_features", "none")
    role_mapping_file = features_cfg.get("role_mapping_file")
    
    db_path = Path(cfg["data"]["database"])
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        return 1
    
    print(f"Loading data from {db_path}...")
    df = get_session_dataframe(db_path)
    print(f"Loaded {len(df):,} sessions")
    
    print("Fitting feature extractor...")
    # Match training-time configuration for role features
    role_mapping_path = Path(role_mapping_file) if role_mapping_file else None
    if role_features == "units" and (not role_mapping_path or not role_mapping_path.exists()):
        raise FileNotFoundError(
            "role_features='units' requires a valid 'role_mapping_file' in config"
        )
    
    extractor = FeatureExtractor(
        role_features=role_features,
        role_mapping_path=role_mapping_path,
    )
    # For role_features='roles', ensure we see all roles to fix the one-hot dimension
    if role_features == "roles" and "role" in df.columns:
        all_roles = sorted(df["role"].dropna().unique().tolist())
        extractor.fit(df, role_categories=all_roles)
    else:
        extractor.fit(df)
    
    output_path = Path("models/feature_extractor.pkl")
    print(f"Saving to {output_path}...")
    extractor.save(output_path)
    
    print("Done! You can now run evaluation/inference.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
