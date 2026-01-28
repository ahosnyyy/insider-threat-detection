#!/usr/bin/env python
"""
Data Ingestion Script
Convert CERT R4.2 CSVs to Parquet and load into DuckDB.

Usage:
    python scripts/ingest_data.py
    python scripts/ingest_data.py --raw-dir data/raw --db-path data/processed/cert.duckdb
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    ingest_cert_data, 
    create_sessions, 
    aggregate_session_events, 
    get_table_stats,
    enrich_with_ldap,
    enrich_with_psychometric,
)
from src.utils import setup_logging, load_config


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Ingest CERT R4.2 data")
    parser.add_argument("--raw-dir", type=Path, default=Path(cfg['data']['raw_path']),
                        help="Directory containing raw CSV files")
    parser.add_argument("--processed-dir", type=Path, default=Path(cfg['data']['processed_path']),
                        help="Directory for processed Parquet files")
    parser.add_argument("--db-path", type=Path, default=Path(cfg['data']['database']),
                        help="Path to DuckDB database")
    args = parser.parse_args()
    
    setup_logging()
    
    # Ensure processed directory exists
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check raw directory exists
    if not args.raw_dir.exists():
        print(f"Error: Raw data directory not found: {args.raw_dir}")
        print("\nPlease download CERT R4.2 dataset:")
        print("  1. Go to: https://doi.org/10.1184/R1/12841247.v1")
        print("  2. Download r4.2.tar.bz2")
        print("  3. Extract to: data/raw/")
        print("\nExpected files in data/raw/:")
        print("  - logon.csv")
        print("  - device.csv")
        print("  - http.csv")
        print("  - email.csv")
        print("  - file.csv")
        print("  - psychometric.csv")
        print("  - LDAP/ (directory with monthly CSVs)")
        return 1
    
    print("=" * 60)
    print("CERT R4.2 Data Ingestion")
    print("=" * 60)
    print(f"Raw directory: {args.raw_dir}")
    print(f"Processed directory: {args.processed_dir}")
    print(f"Database path: {args.db_path}")
    print()
    
    # Step 1: CSV -> Parquet -> DuckDB
    print("Step 1: Ingesting CSV files...")
    stats = ingest_cert_data(args.raw_dir, args.processed_dir, args.db_path)
    
    print("\nIngestion complete:")
    for table, rows in stats.items():
        print(f"  {table}: {rows:,} rows")
    
    # Step 2: Create sessions
    print("\n" + "-" * 60)
    print("Step 2: Creating sessions from logon events...")
    session_count = create_sessions(args.db_path)
    print(f"Created {session_count:,} sessions")
    
    # Step 3: Aggregate session events
    print("\n" + "-" * 60)
    print("Step 3: Aggregating events per session...")
    aggregate_session_events(args.db_path)
    
    # Step 4: Enrich with LDAP
    print("\n" + "-" * 60)
    print("Step 4: Enriching with LDAP data (roles, departments)...")
    enrich_with_ldap(args.db_path)
    
    # Step 5: Enrich with psychometric
    print("\n" + "-" * 60)
    print("Step 5: Enriching with psychometric scores...")
    enrich_with_psychometric(args.db_path)
    
    # Final stats
    print("\n" + "=" * 60)
    print("Final Table Statistics")
    print("=" * 60)
    final_stats = get_table_stats(args.db_path)
    for table, rows in final_stats.items():
        print(f"  {table}: {rows:,} rows")
    
    # Show session feature summary
    import duckdb
    con = duckdb.connect(str(args.db_path))
    feature_count = con.execute("SELECT COUNT(*) FROM session_features").fetchone()[0]
    col_count = len(con.execute("SELECT * FROM session_features LIMIT 1").description)
    con.close()
    
    print(f"\n  session_features: {feature_count:,} rows, {col_count} columns")
    
    print("\n" + "=" * 60)
    print("Data ingestion complete!")
    print(f"Database saved to: {args.db_path}")
    print("\nNext steps:")
    print("  python scripts/train.py --model lstm")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
