"""
Data Ingestion Module
Converts CERT R4.2 CSV files to Parquet and loads into DuckDB.

CERT R4.2 Schema (from readme.txt):
- logon.csv: id, date, user, pc, activity (Logon/Logoff)
- device.csv: id, date, user, pc, activity (Connect/Disconnect)
- http.csv: id, date, user, pc, url, content
- email.csv: id, date, user, pc, to, cc, bcc, from, size, attachments, content
- file.csv: id, date, user, pc, filename, content (file copy to removable media)
- psychometric.csv: employee_name, user_id, O, C, E, A, N
- LDAP/*.csv: employee_name, user_id, email, role, business_unit, functional_unit, department, team, supervisor
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Correct column schemas for CERT R4.2
CSV_SCHEMAS: Dict[str, List[str]] = {
    "logon": ["id", "date", "user", "pc", "activity"],
    "device": ["id", "date", "user", "pc", "activity"],
    "http": ["id", "date", "user", "pc", "url", "content"],
    "email": ["id", "date", "user", "pc", "to", "cc", "bcc", "from", "size", "attachments", "content"],
    "file": ["id", "date", "user", "pc", "filename", "content"],
    "psychometric": ["employee_name", "user_id", "O", "C", "E", "A", "N"],
}

LDAP_SCHEMA = ["employee_name", "user_id", "email", "role", "business_unit", 
               "functional_unit", "department", "team", "supervisor"]


def csv_to_parquet(
    csv_path: Path,
    parquet_path: Path,
    table_name: str,
    chunk_size: int = 100000,
) -> int:
    """
    Convert CSV to Parquet with chunked reading for memory efficiency.
    
    Returns:
        Number of rows processed
    """
    logger.info(f"Converting {csv_path.name} to Parquet...")
    
    # All columns as string initially to avoid type issues
    total_rows = 0
    parquet_writer = None
    
    try:
        for chunk in tqdm(
            pd.read_csv(csv_path, chunksize=chunk_size, dtype=str, low_memory=False),
            desc=f"Processing {table_name}",
        ):
            # Parse dates
            if "date" in chunk.columns:
                chunk["date"] = pd.to_datetime(chunk["date"], format="mixed", errors="coerce")
            
            # Convert numeric columns
            if table_name == "email":
                chunk["size"] = pd.to_numeric(chunk["size"], errors="coerce").fillna(0).astype(int)
                chunk["attachments"] = pd.to_numeric(chunk["attachments"], errors="coerce").fillna(0).astype(int)
            
            if table_name == "psychometric":
                for col in ["O", "C", "E", "A", "N"]:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0).astype(int)
            
            # Convert to PyArrow table
            table = pa.Table.from_pandas(chunk)
            
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(parquet_path, table.schema)
            
            parquet_writer.write_table(table)
            total_rows += len(chunk)
            
    finally:
        if parquet_writer:
            parquet_writer.close()
    
    logger.info(f"Wrote {total_rows:,} rows to {parquet_path}")
    return total_rows


def ingest_ldap(raw_dir: Path, processed_dir: Path) -> int:
    """
    Ingest all LDAP monthly files preserving full history.
    
    Keeps all monthly snapshots to enable:
    - Role change detection
    - Termination detection
    - Temporal role alignment with sessions
    """
    ldap_dir = raw_dir / "LDAP"
    if not ldap_dir.exists():
        logger.warning(f"LDAP directory not found: {ldap_dir}")
        return 0
    
    logger.info("Ingesting LDAP files (preserving full history)...")
    
    all_dfs = []
    for csv_file in sorted(ldap_dir.glob("*.csv")):
        df = pd.read_csv(csv_file, dtype=str)
        # Extract month from filename (e.g., 2010-01.csv)
        month = csv_file.stem
        df["ldap_month"] = month
        # Parse month to date for easier joining
        df["ldap_date"] = pd.to_datetime(month + "-01")
        all_dfs.append(df)
    
    if not all_dfs:
        logger.warning("No LDAP files found")
        return 0
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Sort by user and month for role change detection
    combined = combined.sort_values(["user_id", "ldap_month"]).reset_index(drop=True)
    
    # Detect role changes per user
    combined["prev_role"] = combined.groupby("user_id")["role"].shift(1)
    combined["role_changed"] = (combined["role"] != combined["prev_role"]) & combined["prev_role"].notna()
    combined["prev_department"] = combined.groupby("user_id")["department"].shift(1)
    combined["dept_changed"] = (combined["department"] != combined["prev_department"]) & combined["prev_department"].notna()
    
    # Detect termination (user missing from next month)
    # First, get the max month each user appears in
    last_appearance = combined.groupby("user_id")["ldap_month"].max().reset_index()
    last_appearance.columns = ["user_id", "last_ldap_month"]
    combined = combined.merge(last_appearance, on="user_id")
    
    # Mark if user was terminated (last appearance before final month in dataset)
    final_month = combined["ldap_month"].max()
    combined["is_terminated"] = combined["last_ldap_month"] < final_month
    
    # Also create a "latest only" view for simple lookups
    latest_only = combined.sort_values("ldap_month").drop_duplicates(subset=["user_id"], keep="last")
    
    # Save both versions
    parquet_path = processed_dir / "ldap_history.parquet"
    combined.to_parquet(parquet_path, index=False)
    
    latest_path = processed_dir / "ldap.parquet"
    latest_only.to_parquet(latest_path, index=False)
    
    role_changes = combined["role_changed"].sum()
    terminated = latest_only["is_terminated"].sum()
    
    logger.info(f"LDAP History: {len(combined):,} total records from {len(all_dfs)} months")
    logger.info(f"  - {len(latest_only):,} unique users")
    logger.info(f"  - {role_changes:,} role changes detected")
    logger.info(f"  - {terminated:,} terminated employees")
    
    return len(combined)


def load_parquet_to_duckdb(
    parquet_path: Path,
    db_path: Path,
    table_name: str,
) -> None:
    """Load Parquet file into DuckDB table."""
    logger.info(f"Loading {table_name} into DuckDB...")
    
    con = duckdb.connect(str(db_path))
    
    # Create table from Parquet (use forward slashes for cross-platform compatibility)
    parquet_path_str = str(parquet_path).replace('\\', '/')
    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * FROM read_parquet('{parquet_path_str}')
    """)
    
    # Get row count
    result = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    logger.info(f"Table {table_name}: {result[0]:,} rows")
    
    con.close()


def ingest_cert_data(
    raw_dir: Path,
    processed_dir: Path,
    db_path: Path,
) -> Dict[str, int]:
    """
    Full ingestion pipeline: CSV -> Parquet -> DuckDB
    
    Args:
        raw_dir: Directory containing CERT R4.2 CSV files
        processed_dir: Directory for Parquet files
        db_path: Path to DuckDB database file
        
    Returns:
        Dict mapping table names to row counts
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    # Process main CSV files
    for table_name in CSV_SCHEMAS.keys():
        csv_path = raw_dir / f"{table_name}.csv"
        parquet_path = processed_dir / f"{table_name}.parquet"
        
        if not csv_path.exists():
            logger.warning(f"CSV not found: {csv_path}")
            continue
        
        # Convert to Parquet
        rows = csv_to_parquet(csv_path, parquet_path, table_name)
        stats[table_name] = rows
        
        # Load into DuckDB
        load_parquet_to_duckdb(parquet_path, db_path, table_name)
    
    # Process LDAP separately
    ldap_rows = ingest_ldap(raw_dir, processed_dir)
    if ldap_rows > 0:
        stats["ldap"] = ldap_rows
        load_parquet_to_duckdb(processed_dir / "ldap.parquet", db_path, "ldap")
    
    # Create indexes for performance
    _create_indexes(db_path)
    
    return stats


def _create_indexes(db_path: Path) -> None:
    """Create indexes on commonly queried columns."""
    logger.info("Creating indexes...")
    
    con = duckdb.connect(str(db_path))
    
    indexes = [
        ("logon", "user"),
        ("logon", "date"),
        ("device", "user"),
        ("device", "date"),
        ("http", "user"),
        ("http", "date"),
        ("email", "user"),
        ("email", "date"),
        ("file", "user"),
        ("file", "date"),
    ]
    
    for table, column in indexes:
        try:
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{column} ON {table}({column})")
        except Exception as e:
            logger.debug(f"Could not create index on {table}.{column}: {e}")
    
    con.close()
    logger.info("Indexes created")


def get_table_stats(db_path: Path) -> Dict[str, int]:
    """Get row counts for all tables."""
    con = duckdb.connect(str(db_path))
    
    stats = {}
    tables = ["logon", "device", "http", "email", "file", "psychometric", "ldap"]
    
    for table in tables:
        try:
            result = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            stats[table] = result[0]
        except Exception:
            stats[table] = 0
    
    con.close()
    return stats


if __name__ == "__main__":
    from pathlib import Path
    
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    db_path = processed_dir / "cert.duckdb"
    
    if raw_dir.exists():
        stats = ingest_cert_data(raw_dir, processed_dir, db_path)
        print("\nIngestion complete:")
        for table, rows in stats.items():
            print(f"  {table}: {rows:,} rows")
    else:
        print(f"Raw data directory not found: {raw_dir}")
        print("Please download CERT R4.2 dataset and extract to data/raw/")
