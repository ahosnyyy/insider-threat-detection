"""
Quick validation script for ingested data.
Run after ingest_data.py to verify data integrity.
"""

import duckdb
from pathlib import Path

def validate_data(db_path: str = "data/processed/cert.duckdb"):
    """Run validation queries on the ingested data."""
    
    conn = duckdb.connect(db_path, read_only=True)
    
    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    
    # 1. Check table row counts
    print("\n1. TABLE ROW COUNTS")
    print("-" * 40)
    tables = ['logon', 'device', 'http', 'email', 'file', 'ldap', 'psychometric', 'session_features']
    for table in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {count:,}")
        except Exception as e:
            print(f"  {table}: ERROR - {e}")
    
    # 2. Check LDAP role changes (debug)
    print("\n2. LDAP ROLE CHANGE ANALYSIS")
    print("-" * 40)
    
    # Check if role column varies per user across months
    role_changes_query = """
    SELECT 
        employee_name,
        COUNT(DISTINCT role) as distinct_roles,
        MIN(month) as first_month,
        MAX(month) as last_month
    FROM ldap
    GROUP BY employee_name
    HAVING COUNT(DISTINCT role) > 1
    LIMIT 10
    """
    try:
        results = conn.execute(role_changes_query).fetchall()
        if results:
            print(f"  Found {len(results)} users with multiple roles:")
            for r in results[:5]:
                print(f"    {r[0]}: {r[1]} distinct roles ({r[2]} to {r[3]})")
        else:
            print("  No users found with multiple distinct roles")
            print("  (This may be correct - dataset may not have role changes)")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # 3. Check department changes
    print("\n3. DEPARTMENT CHANGE ANALYSIS")
    print("-" * 40)
    dept_changes_query = """
    SELECT 
        employee_name,
        COUNT(DISTINCT department) as distinct_depts
    FROM ldap
    GROUP BY employee_name
    HAVING COUNT(DISTINCT department) > 1
    LIMIT 10
    """
    try:
        results = conn.execute(dept_changes_query).fetchall()
        if results:
            print(f"  Found {len(results)} users with department changes")
        else:
            print("  No users changed departments")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # 4. Check terminated users
    print("\n4. TERMINATED USERS")
    print("-" * 40)
    terminated_query = """
    SELECT COUNT(DISTINCT user_id) 
    FROM session_features 
    WHERE terminated = TRUE
    """
    try:
        count = conn.execute(terminated_query).fetchone()[0]
        print(f"  Unique terminated users with sessions: {count}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # 5. Sample session features
    print("\n5. SAMPLE SESSION FEATURES (first 3 rows)")
    print("-" * 40)
    sample_query = """
    SELECT 
        session_id,
        user_id,
        duration_minutes,
        total_actions,
        usb_connect_count,
        file_copy_count,
        is_after_hours,
        terminated,
        role
    FROM session_features
    LIMIT 3
    """
    try:
        results = conn.execute(sample_query).fetchall()
        cols = ['session_id', 'user_id', 'duration', 'actions', 'usb', 'files', 'after_hrs', 'terminated', 'role']
        print("  " + " | ".join(f"{c:>10}" for c in cols))
        for r in results:
            print("  " + " | ".join(f"{str(v)[:10]:>10}" for v in r))
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # 6. Feature distribution
    print("\n6. FEATURE STATISTICS")
    print("-" * 40)
    stats_query = """
    SELECT 
        AVG(duration_minutes) as avg_duration,
        AVG(total_actions) as avg_actions,
        AVG(usb_connect_count) as avg_usb,
        AVG(file_copy_count) as avg_files,
        SUM(CASE WHEN is_after_hours THEN 1 ELSE 0 END) as after_hours_sessions,
        SUM(CASE WHEN terminated THEN 1 ELSE 0 END) as terminated_sessions
    FROM session_features
    """
    try:
        r = conn.execute(stats_query).fetchone()
        print(f"  Avg duration: {r[0]:.1f} min")
        print(f"  Avg actions: {r[1]:.1f}")
        print(f"  Avg USB connects: {r[2]:.2f}")
        print(f"  Avg file copies: {r[3]:.2f}")
        print(f"  After-hours sessions: {r[4]:,}")
        print(f"  Terminated user sessions: {r[5]:,}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # 7. Check for NULL/missing values
    print("\n7. NULL VALUE CHECK (critical columns)")
    print("-" * 40)
    null_check = """
    SELECT 
        SUM(CASE WHEN user_id IS NULL THEN 1 ELSE 0 END) as null_user,
        SUM(CASE WHEN session_id IS NULL THEN 1 ELSE 0 END) as null_session,
        SUM(CASE WHEN duration_minutes IS NULL THEN 1 ELSE 0 END) as null_duration,
        SUM(CASE WHEN role IS NULL THEN 1 ELSE 0 END) as null_role
    FROM session_features
    """
    try:
        r = conn.execute(null_check).fetchone()
        print(f"  NULL user_id: {r[0]}")
        print(f"  NULL session_id: {r[1]}")
        print(f"  NULL duration: {r[2]}")
        print(f"  NULL role: {r[3]}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    validate_data()
