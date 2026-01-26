"""
Sessionization Module
Groups events into user sessions (logon -> logoff).

CERT R4.2 specific:
- logon.csv has activity = Logon/Logoff
- device.csv has activity = Connect/Disconnect
- file.csv entries are file copies to removable media
- email.csv has size (bytes) and attachments count
"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sessions(db_path: Path) -> int:
    """
    Create sessions table by pairing logon/logoff events.
    
    Logic:
    - Session starts with 'Logon' event
    - Session ends with 'Logoff' OR next 'Logon' on same PC OR max 8 hours
    
    Returns:
        Number of sessions created
    """
    logger.info("Creating sessions from logon events...")
    
    con = duckdb.connect(str(db_path))
    
    # Create sessions table
    con.execute("""
        CREATE OR REPLACE TABLE sessions AS
        WITH logon_events AS (
            SELECT 
                id,
                date as start_time,
                "user" as user_id,
                pc,
                activity,
                -- Find next event for same user/pc
                LEAD(date) OVER (PARTITION BY "user", pc ORDER BY date) as next_event_time,
                LEAD(activity) OVER (PARTITION BY "user", pc ORDER BY date) as next_activity
            FROM logon
        ),
        session_bounds AS (
            SELECT
                id as session_id,
                user_id,
                pc,
                start_time,
                CASE 
                    -- If next event is logoff, use it
                    WHEN next_activity = 'Logoff' THEN next_event_time
                    -- If next event is another logon (different session), use it as end marker
                    WHEN next_activity = 'Logon' THEN next_event_time
                    -- If no next event or too far, cap at 8 hours
                    WHEN next_event_time IS NULL THEN start_time + INTERVAL '8 hours'
                    WHEN next_event_time - start_time > INTERVAL '12 hours' THEN start_time + INTERVAL '8 hours'
                    ELSE next_event_time
                END as end_time
            FROM logon_events
            WHERE activity = 'Logon'
        )
        SELECT 
            session_id,
            user_id,
            pc,
            start_time,
            end_time,
            EXTRACT(EPOCH FROM (end_time - start_time)) / 60.0 as duration_minutes,
            EXTRACT(HOUR FROM start_time) as start_hour,
            EXTRACT(DOW FROM start_time) as day_of_week,
            CASE WHEN EXTRACT(DOW FROM start_time) IN (0, 6) THEN 1 ELSE 0 END as is_weekend,
            CASE WHEN EXTRACT(HOUR FROM start_time) < 6 OR EXTRACT(HOUR FROM start_time) >= 18 
                 THEN 1 ELSE 0 END as is_after_hours
        FROM session_bounds
        WHERE end_time > start_time  -- Filter invalid sessions
          AND duration_minutes > 0   -- Ensure positive duration
        ORDER BY user_id, start_time
    """)
    
    # Get count
    result = con.execute("SELECT COUNT(*) FROM sessions").fetchone()
    session_count = result[0]
    
    logger.info(f"Created {session_count:,} sessions")
    
    # Create indexes
    con.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_sessions_time ON sessions(start_time)")
    
    con.close()
    return session_count


def aggregate_session_events(db_path: Path) -> None:
    """
    Aggregate events from all log sources into each session.
    Creates session_features table with counts and aggregations.
    
    Uses CERT R4.2 specific fields:
    - device: Connect/Disconnect activity
    - file: each row is a file copy to removable media
    - email: size (bytes), attachments count
    - http: url, content
    """
    logger.info("Aggregating events per session...")
    
    con = duckdb.connect(str(db_path))
    
    # Device events per session (USB connect/disconnect)
    con.execute("""
        CREATE OR REPLACE TABLE session_device AS
        SELECT 
            s.session_id,
            COUNT(*) as device_event_count,
            COUNT(CASE WHEN d.activity = 'Connect' THEN 1 END) as usb_connect_count,
            COUNT(CASE WHEN d.activity = 'Disconnect' THEN 1 END) as usb_disconnect_count
        FROM sessions s
        LEFT JOIN device d ON d."user" = s.user_id 
            AND d.date >= s.start_time 
            AND d.date < s.end_time
        GROUP BY s.session_id
    """)
    
    # HTTP events per session
    con.execute("""
        CREATE OR REPLACE TABLE session_http AS
        SELECT 
            s.session_id,
            COUNT(*) as http_request_count,
            COUNT(DISTINCT 
                CASE WHEN h.url IS NOT NULL 
                THEN SPLIT_PART(REPLACE(REPLACE(h.url, 'http://', ''), 'https://', ''), '/', 1) 
                END
            ) as unique_domains
        FROM sessions s
        LEFT JOIN http h ON h."user" = s.user_id 
            AND h.date >= s.start_time 
            AND h.date < s.end_time
        GROUP BY s.session_id
    """)
    
    # Email events per session
    # Note: 'from' is a reserved word, need to escape it
    con.execute('''
        CREATE OR REPLACE TABLE session_email AS
        SELECT 
            s.session_id,
            COUNT(*) as email_count,
            -- Sent emails: where the session user is the sender
            COUNT(CASE WHEN e."from" LIKE '%' || s.user_id || '%' THEN 1 END) as emails_sent,
            -- Received emails: where user is in to/cc/bcc
            COUNT(CASE WHEN e."to" LIKE '%' || s.user_id || '%' 
                       OR e.cc LIKE '%' || s.user_id || '%'
                       OR e.bcc LIKE '%' || s.user_id || '%' THEN 1 END) as emails_received,
            COALESCE(SUM(CAST(e.attachments AS INTEGER)), 0) as total_attachments,
            COALESCE(SUM(CAST(e.size AS INTEGER)), 0) as total_email_size,
            -- External recipients (non @dtaa.com)
            COUNT(CASE WHEN e."to" NOT LIKE '%@dtaa.com%' THEN 1 END) as external_email_count
        FROM sessions s
        LEFT JOIN email e ON e."user" = s.user_id 
            AND e.date >= s.start_time 
            AND e.date < s.end_time
        GROUP BY s.session_id
    ''')
    
    # File events per session
    # In CERT R4.2, file.csv entries are file COPIES to removable media
    con.execute("""
        CREATE OR REPLACE TABLE session_file AS
        SELECT 
            s.session_id,
            COUNT(*) as file_copy_count,
            COUNT(CASE WHEN f.filename LIKE '%.exe' THEN 1 END) as exe_copy_count,
            COUNT(CASE WHEN f.filename LIKE '%.doc%' THEN 1 END) as doc_copy_count,
            COUNT(CASE WHEN f.filename LIKE '%.pdf' THEN 1 END) as pdf_copy_count,
            COUNT(CASE WHEN f.filename LIKE '%.zip' OR f.filename LIKE '%.rar' THEN 1 END) as archive_copy_count
        FROM sessions s
        LEFT JOIN file f ON f."user" = s.user_id 
            AND f.date >= s.start_time 
            AND f.date < s.end_time
        GROUP BY s.session_id
    """)
    
    # Combine all into session_features
    con.execute("""
        CREATE OR REPLACE TABLE session_features AS
        SELECT 
            s.*,
            -- Device features
            COALESCE(sd.device_event_count, 0) as device_event_count,
            COALESCE(sd.usb_connect_count, 0) as usb_connect_count,
            COALESCE(sd.usb_disconnect_count, 0) as usb_disconnect_count,
            -- HTTP features
            COALESCE(sh.http_request_count, 0) as http_request_count,
            COALESCE(sh.unique_domains, 0) as unique_domains,
            -- Email features
            COALESCE(se.email_count, 0) as email_count,
            COALESCE(se.emails_sent, 0) as emails_sent,
            COALESCE(se.emails_received, 0) as emails_received,
            COALESCE(se.total_attachments, 0) as total_attachments,
            COALESCE(se.total_email_size, 0) as total_email_size,
            COALESCE(se.external_email_count, 0) as external_email_count,
            -- File copy features (to removable media)
            COALESCE(sf.file_copy_count, 0) as file_copy_count,
            COALESCE(sf.exe_copy_count, 0) as exe_copy_count,
            COALESCE(sf.doc_copy_count, 0) as doc_copy_count,
            COALESCE(sf.pdf_copy_count, 0) as pdf_copy_count,
            COALESCE(sf.archive_copy_count, 0) as archive_copy_count,
            -- Total actions
            COALESCE(sd.device_event_count, 0) + COALESCE(sh.http_request_count, 0) + 
            COALESCE(se.email_count, 0) + COALESCE(sf.file_copy_count, 0) as total_actions,
            -- Action density (actions per minute)
            CASE WHEN s.duration_minutes > 0 
                THEN (COALESCE(sd.device_event_count, 0) + COALESCE(sh.http_request_count, 0) + 
                      COALESCE(se.email_count, 0) + COALESCE(sf.file_copy_count, 0)) / s.duration_minutes
                ELSE 0 
            END as action_density
        FROM sessions s
        LEFT JOIN session_device sd ON s.session_id = sd.session_id
        LEFT JOIN session_http sh ON s.session_id = sh.session_id
        LEFT JOIN session_email se ON s.session_id = se.session_id
        LEFT JOIN session_file sf ON s.session_id = sf.session_id
    """)
    
    result = con.execute("SELECT COUNT(*) FROM session_features").fetchone()
    logger.info(f"Session features table: {result[0]:,} rows")
    
    # Show feature summary
    feature_stats = con.execute("""
        SELECT 
            AVG(duration_minutes) as avg_duration,
            AVG(total_actions) as avg_actions,
            AVG(file_copy_count) as avg_file_copies,
            SUM(CASE WHEN usb_connect_count > 0 THEN 1 ELSE 0 END) as sessions_with_usb
        FROM session_features
    """).fetchone()
    
    logger.info(f"Feature stats: avg_duration={feature_stats[0]:.1f}min, "
                f"avg_actions={feature_stats[1]:.1f}, avg_file_copies={feature_stats[2]:.2f}, "
                f"sessions_with_usb={feature_stats[3]:,}")
    
    con.close()


def enrich_with_ldap(db_path: Path) -> None:
    """
    Add LDAP user information (role, department, role changes, termination) to session features.
    
    Uses temporal matching: session is joined to LDAP snapshot from the same month.
    Adds role change detection features.
    """
    logger.info("Enriching sessions with LDAP data (including role history)...")
    
    con = duckdb.connect(str(db_path))
    
    # Check if LDAP table exists
    try:
        con.execute("SELECT COUNT(*) FROM ldap").fetchone()
    except Exception:
        logger.warning("LDAP table not found, skipping enrichment")
        con.close()
        return
    
    # Check if we have ldap_history table (for temporal matching)
    has_history = False
    ldap_history_path = str(db_path.parent / "ldap_history.parquet").replace('\\', '/')
    try:
        con.execute(f"SELECT COUNT(*) FROM read_parquet('{ldap_history_path}')").fetchone()
        has_history = True
    except Exception:
        pass
    
    if has_history:
        # Temporal matching: join session to LDAP snapshot from same month
        con.execute(f"""
            CREATE OR REPLACE TABLE session_features_enriched AS
            WITH session_months AS (
                SELECT 
                    sf.*,
                    STRFTIME(sf.start_time, '%Y-%m') as session_month
                FROM session_features sf
            ),
            ldap_hist AS (
                SELECT * FROM read_parquet('{ldap_history_path}')
            )
            SELECT 
                sm.* EXCLUDE (session_month),
                l.role,
                l.department,
                l.business_unit,
                l.supervisor,
                CASE WHEN l.role = 'ITAdmin' THEN 1 ELSE 0 END as is_admin,
                -- Role change features
                COALESCE(l.role_changed, FALSE) as role_changed_this_month,
                COALESCE(l.dept_changed, FALSE) as dept_changed_this_month,
                l.prev_role,
                -- Termination status
                COALESCE(l.is_terminated, FALSE) as user_terminated,
                l.last_ldap_month
            FROM session_months sm
            LEFT JOIN ldap_hist l 
                ON sm.user_id = l.user_id 
                AND sm.session_month = l.ldap_month
        """)
    else:
        # Fallback: use latest LDAP only
        con.execute("""
            CREATE OR REPLACE TABLE session_features_enriched AS
            SELECT 
                sf.*,
                l.role,
                l.department,
                l.business_unit,
                CASE WHEN l.role = 'ITAdmin' THEN 1 ELSE 0 END as is_admin,
                COALESCE(l.is_terminated, FALSE) as user_terminated
            FROM session_features sf
            LEFT JOIN ldap l ON sf.user_id = l.user_id
        """)
    
    # Replace original table
    con.execute("DROP TABLE IF EXISTS session_features")
    con.execute("ALTER TABLE session_features_enriched RENAME TO session_features")
    
    # Log statistics
    result = con.execute("SELECT COUNT(DISTINCT role) FROM session_features WHERE role IS NOT NULL").fetchone()
    logger.info(f"Added LDAP data: {result[0]} distinct roles")
    
    if has_history:
        role_changes = con.execute("SELECT SUM(CASE WHEN role_changed_this_month THEN 1 ELSE 0 END) FROM session_features").fetchone()[0]
        terminated = con.execute("SELECT SUM(CASE WHEN user_terminated THEN 1 ELSE 0 END) FROM session_features").fetchone()[0]
        logger.info(f"  - Sessions with role change: {role_changes:,}")
        logger.info(f"  - Sessions by terminated users: {terminated:,}")
    
    con.close()


def enrich_with_psychometric(db_path: Path) -> None:
    """Add psychometric scores (O, C, E, A, N) to session features."""
    logger.info("Enriching sessions with psychometric data...")
    
    con = duckdb.connect(str(db_path))
    
    # Check if psychometric table exists
    try:
        con.execute("SELECT COUNT(*) FROM psychometric").fetchone()
    except Exception:
        logger.warning("Psychometric table not found, skipping enrichment")
        con.close()
        return
    
    # Add psychometric columns
    con.execute("""
        CREATE OR REPLACE TABLE session_features_enriched AS
        SELECT 
            sf.*,
            p."O" as openness,
            p."C" as conscientiousness,
            p."E" as extraversion,
            p."A" as agreeableness,
            p."N" as neuroticism
        FROM session_features sf
        LEFT JOIN psychometric p ON sf.user_id = p.user_id
    """)
    
    # Replace original table
    con.execute("DROP TABLE IF EXISTS session_features")
    con.execute("ALTER TABLE session_features_enriched RENAME TO session_features")
    
    logger.info("Added psychometric scores")
    
    con.close()


def get_session_dataframe(db_path: Path) -> pd.DataFrame:
    """Load session features into pandas DataFrame."""
    con = duckdb.connect(str(db_path))
    df = con.execute("SELECT * FROM session_features").fetchdf()
    con.close()
    return df


if __name__ == "__main__":
    from pathlib import Path
    
    db_path = Path("data/processed/cert.duckdb")
    
    if db_path.exists():
        create_sessions(db_path)
        aggregate_session_events(db_path)
        enrich_with_ldap(db_path)
        enrich_with_psychometric(db_path)
        
        df = get_session_dataframe(db_path)
        print(f"\nSession features shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(df.head())
    else:
        print(f"Database not found: {db_path}")
        print("Run ingest.py first")
