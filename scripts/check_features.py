import sys
from pathlib import Path
# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import duckdb
from src.data.features import FeatureExtractor
from src.utils import load_config

def main():
    cfg = load_config()
    db_path = Path(cfg.get("data", {}).get("database", "data/processed/cert.duckdb"))
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return 1

    # Default to "roles" so we report role categories; use config for role_mapping_file
    role_features = cfg.get("features", {}).get("role_features", "none")
    role_mapping_file = cfg.get("features", {}).get("role_mapping_file", "config/role_units.yaml")
    if role_features == "none":
        role_features = "roles"  # For check script, show roles by default
    role_mapping_path = Path(role_mapping_file)
    if not role_mapping_path.is_absolute():
        role_mapping_path = Path(__file__).parent.parent / role_mapping_path

    print("Loading all sessions to check feature dimensions...")
    con = duckdb.connect(str(db_path))
    df = con.execute("SELECT * FROM session_features").fetchdf()
    con.close()

    print(f"Loaded {len(df):,} sessions.")
    print(f"Role features mode: {role_features}")

    extractor = FeatureExtractor(
        role_features=role_features,
        role_mapping_path=role_mapping_path if role_features == "units" else None,
    )
    extractor.fit(df)

    print("-" * 50)
    print(f"Numeric features count: {len(extractor.feature_names)}")
    print(f"Numeric features list: {extractor.feature_names}")
    print("-" * 50)
    label = "Role units" if role_features == "units" else "Role categories"
    print(f"{label} count: {len(extractor.role_categories) if extractor.role_categories else 0}")
    print(f"{label}: {extractor.role_categories}")
    print("-" * 50)
    print(f"Total feature dimension: {extractor.get_feature_dim()}")
    print("-" * 50)

    return 0

if __name__ == "__main__":
    sys.exit(main())
