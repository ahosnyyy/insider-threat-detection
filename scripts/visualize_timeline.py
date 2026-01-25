import argparse
import logging
from pathlib import Path
from datetime import timedelta

import duckdb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_timeline(session_id: str, db_path: Path, output_dir: Path):
    """Visualize session timeline events."""
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return

    con = duckdb.connect(str(db_path))
    
    # Get session details
    session = con.execute(
        f"SELECT user_id, start_time, end_time FROM sessions WHERE session_id = '{session_id}'"
    ).fetchone()
    
    if not session:
        logger.error(f"Session {session_id} not found.")
        con.close()
        return
        
    user_id, start_time, end_time = session
    logger.info(f"Visualizing session {session_id} for user {user_id}")
    logger.info(f"Time: {start_time} to {end_time}")
    
    # Query raw events
    tables = ['logon', 'device', 'http', 'email', 'file']
    events = []
    
    # Add buffer to margin
    buffer = timedelta(minutes=5)
    query_start = start_time - buffer
    query_end = end_time + buffer
    
    for table in tables:
        # Check if table exists
        try:
            # R4.2 raw tables use 'user' and 'date'
            query = f"""
                SELECT date, '{table}' as type, id
                FROM {table}
                WHERE "user" = '{user_id}' 
                AND date BETWEEN '{query_start}' AND '{query_end}'
            """
            
            # Special columns for context
            if table == 'http':
                query = f"""
                    SELECT date, '{table}' as type, url as info
                    FROM {table}
                    WHERE "user" = '{user_id}' 
                    AND date BETWEEN '{query_start}' AND '{query_end}'
                """
            elif table == 'file':
                query = f"""
                    SELECT date, '{table}' as type, filename as info
                    FROM {table}
                    WHERE "user" = '{user_id}' 
                    AND date BETWEEN '{query_start}' AND '{query_end}'
                """
            elif table == 'email':
                query = f"""
                    SELECT date, '{table}' as type, "to" as info
                    FROM {table}
                    WHERE "user" = '{user_id}' 
                    AND date BETWEEN '{query_start}' AND '{query_end}'
                """
            elif table == 'logon':
                 query = f"""
                    SELECT date, '{table}' as type, activity as info
                    FROM {table}
                    WHERE "user" = '{user_id}' 
                    AND date BETWEEN '{query_start}' AND '{query_end}'
                """
            else:
                 query = f"""
                    SELECT date, '{table}' as type, activity as info
                    FROM {table}
                    WHERE "user" = '{user_id}' 
                    AND date BETWEEN '{query_start}' AND '{query_end}'
                """
                
            df_table = con.execute(query).fetchdf()
            if not df_table.empty:
                events.append(df_table)
                
        except Exception as e:
            logger.warning(f"Could not query table {table}: {e}")
            
    con.close()
    
    if not events:
        logger.warning("No events found for this session timeline (empty session?)")
        return
        
    df_events = pd.concat(events)
    df_events['date'] = pd.to_datetime(df_events['date'])
    df_events = df_events.sort_values('date')
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Create y-axis mapping
    event_types = sorted(df_events['type'].unique())
    y_map = {t: i for i, t in enumerate(event_types)}
    
    # Plot events
    sns.scatterplot(
        data=df_events,
        x='date',
        y='type',
        hue='type',
        s=100,
        palette='deep',
        legend=False
    )
    
    # Add session boundaries
    plt.axvline(start_time, color='green', linestyle='--', label='Session Start')
    plt.axvline(end_time, color='red', linestyle='--', label='Session End')
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    
    plt.title(f"Timeline: Session {session_id} ({user_id})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    save_path = output_dir / f"timeline_{session_id}.png"
    plt.savefig(save_path)
    logger.info(f"Timeline saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize session timeline")
    parser.add_argument("--session-id", type=str, required=True, help="Session ID")
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/cert.duckdb"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visualize_timeline(args.session_id, args.db_path, args.output_dir)

if __name__ == "__main__":
    main()
