import sys
from pathlib import Path
# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import duckdb
from src.data.features import FeatureExtractor

def main():
    db_path = Path("data/processed/cert.duckdb")
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return 1

    print("Loading all sessions to check feature dimensions...")
    con = duckdb.connect(str(db_path))
    # We load only columns needed to save memory if possible, but * is safer to ensure we get everything
    df = con.execute("SELECT * FROM session_features").fetchdf()
    con.close()

    print(f"Loaded {len(df):,} sessions.")
    
    # Cast booleans if needed (though we fixed this in features.py, good to verify)
    # The fix in features.py handles the casting inside fit()
    # Use include_role=True so we report role categories (default is False for training).
    
    extractor = FeatureExtractor(include_role=True)
    extractor.fit(df)

    print("-" * 50)
    print(f"Numeric features count: {len(extractor.feature_names)}")
    print(f"Numeric features list: {extractor.feature_names}")
    print("-" * 50)
    print(f"Role categories count: {len(extractor.role_categories) if extractor.role_categories else 0}")
    print(f"Role categories: {extractor.role_categories}") # List might be long
    print("-" * 50)
    print(f"Total feature dimension: {extractor.get_feature_dim()}")
    print("-" * 50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
