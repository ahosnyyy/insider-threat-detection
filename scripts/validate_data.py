import argparse
import logging
from pathlib import Path

import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_data(db_path: Path):
    """Validate DuckDB data quality."""
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return

    con = duckdb.connect(str(db_path))
    
    # 1. Check Tables
    tables = con.execute("SHOW TABLES").fetchall()
    tables = [t[0] for t in tables]
    logger.info(f"Tables found: {tables}")
    
    expected_tables = ['logon', 'device', 'http', 'email', 'file', 'sessions', 'session_features']
    missing = set(expected_tables) - set(tables)
    if missing:
        logger.warning(f"Missing expected tables: {missing}")
        
    # 2. Check Row Counts and Uniqueness
    for table in tables:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info(f"Table '{table}': {count:,} rows")
        
        if count == 0:
            logger.warning(f"Table '{table}' is empty!")
            continue
            
        # Specific Checks
        if table == 'logon':
            # Check ID uniqueness
            unique_ids = con.execute("SELECT COUNT(DISTINCT id) FROM logon").fetchone()[0]
            if unique_ids < count:
                logger.warning(f"CRITICAL: 'logon' table has duplicate IDs! ({unique_ids:,} unique vs {count:,} total)")
                logger.warning("This may cause session collision.")
            else:
                logger.info(f" - ID Uniqueness check passed.")
                
        if table == 'sessions':
            # Check for negative durations
            neg_dur = con.execute("SELECT COUNT(*) FROM sessions WHERE duration_minutes < 0").fetchone()[0]
            if neg_dur > 0:
                logger.warning(f"FOUND {neg_dur} sessions with negative duration!")
                
    con.close()
    logger.info("Validation complete.")

def main():
    parser = argparse.ArgumentParser(description="Validate processed data")
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/cert.duckdb"))
    args = parser.parse_args()
    
    validate_data(args.db_path)

if __name__ == "__main__":
    main()
