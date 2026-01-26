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
from src.utils import setup_logging

def main():
    setup_logging()
    
    db_path = Path("data/processed/cert.duckdb")
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        return 1
    
    print(f"Loading data from {db_path}...")
    df = get_session_dataframe(db_path)
    print(f"Loaded {len(df):,} sessions")
    
    print("Fitting feature extractor...")
    extractor = FeatureExtractor()
    extractor.fit(df)
    
    output_path = Path("models/feature_extractor.pkl")
    print(f"Saving to {output_path}...")
    extractor.save(output_path)
    
    print("Done! You can now run evaluation/inference.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
