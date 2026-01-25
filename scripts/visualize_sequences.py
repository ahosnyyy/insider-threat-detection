#!/usr/bin/env python
"""
Sequence Visualization Script
Visualizes user session sequences over time with activity patterns.

Usage:
    python scripts/visualize_sequences.py
    python scripts/visualize_sequences.py --user U0123
    python scripts/visualize_sequences.py --num-users 5
    python scripts/visualize_sequences.py --output sequences.png
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_user_sessions(db_path: str, user_id: str = None, limit: int = 100) -> pd.DataFrame:
    """Load sessions for a specific user or sample of users."""
    conn = duckdb.connect(db_path, read_only=True)
    
    if user_id:
        query = f"""
        SELECT 
            user_id,
            session_id,
            start_time,
            end_time,
            duration_minutes,
            total_actions,
            http_request_count,
            email_count,
            file_copy_count,
            usb_connect_count,
            is_after_hours,
            is_weekend,
            role,
            user_terminated
        FROM session_features
        WHERE user_id = '{user_id}'
        ORDER BY start_time
        LIMIT {limit}
        """
    else:
        query = f"""
        SELECT 
            user_id,
            session_id,
            start_time,
            end_time,
            duration_minutes,
            total_actions,
            http_request_count,
            email_count,
            file_copy_count,
            usb_connect_count,
            is_after_hours,
            is_weekend,
            role,
            user_terminated
        FROM session_features
        ORDER BY user_id, start_time
        LIMIT {limit}
        """
    
    df = conn.execute(query).fetchdf()
    conn.close()
    
    # Convert timestamps
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    
    return df


def get_interesting_users(db_path: str, num_users: int = 5) -> list:
    """Get users with interesting activity patterns."""
    conn = duckdb.connect(db_path, read_only=True)
    
    # Find users with varied activity
    query = f"""
    SELECT 
        user_id,
        COUNT(*) as session_count,
        SUM(file_copy_count) as total_files,
        SUM(usb_connect_count) as total_usb,
        MAX(user_terminated::INT) as is_terminated
    FROM session_features
    GROUP BY user_id
    HAVING COUNT(*) >= 50
    ORDER BY 
        is_terminated DESC,
        total_files DESC,
        total_usb DESC
    LIMIT {num_users}
    """
    
    result = conn.execute(query).fetchdf()
    conn.close()
    
    return result['user_id'].tolist()


def plot_user_timeline(df: pd.DataFrame, user_id: str, ax: plt.Axes):
    """Plot timeline for a single user."""
    user_df = df[df['user_id'] == user_id].copy()
    
    if len(user_df) == 0:
        ax.text(0.5, 0.5, f"No data for {user_id}", ha='center', va='center')
        return
    
    # Create timeline
    times = user_df['start_time'].values
    actions = user_df['total_actions'].values
    files = user_df['file_copy_count'].values
    usb = user_df['usb_connect_count'].values
    after_hours = user_df['is_after_hours'].values
    
    # Normalize for visibility
    actions_norm = actions / max(actions.max(), 1) * 100
    files_norm = files / max(files.max(), 1) * 50
    usb_norm = usb / max(usb.max(), 1) * 30
    
    # Plot activity bars
    ax.bar(times, actions_norm, width=0.5, alpha=0.6, color='steelblue', label='Actions')
    
    # Overlay file copies
    ax.bar(times, files_norm, width=0.5, alpha=0.8, color='orange', label='File Copies')
    
    # Mark USB events
    usb_mask = usb > 0
    if usb_mask.any():
        ax.scatter(times[usb_mask], usb_norm[usb_mask] + 60, 
                   marker='v', color='red', s=50, label='USB Events', zorder=5)
    
    # Mark after-hours sessions
    after_hours_mask = after_hours.astype(bool)
    if after_hours_mask.any():
        ax.scatter(times[after_hours_mask], np.ones(after_hours_mask.sum()) * 90, 
                   marker='*', color='purple', s=30, alpha=0.7, label='After Hours')
    
    # Formatting
    role = user_df['role'].iloc[0] if 'role' in user_df.columns else 'Unknown'
    terminated = user_df['user_terminated'].iloc[0] if 'user_terminated' in user_df.columns else False
    title_suffix = " [TERMINATED]" if terminated else ""
    
    ax.set_title(f"{user_id} ({role}){title_suffix}", fontsize=10, fontweight='bold')
    ax.set_ylabel('Activity Level')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 120)
    ax.grid(axis='y', alpha=0.3)


def plot_sequence_heatmap(df: pd.DataFrame, user_id: str, ax: plt.Axes):
    """Plot feature heatmap for user sessions."""
    user_df = df[df['user_id'] == user_id].copy()
    
    if len(user_df) == 0:
        return
    
    # Select numeric features for heatmap
    features = ['total_actions', 'http_request_count', 'email_count', 
                'file_copy_count', 'usb_connect_count', 'duration_minutes']
    
    # Normalize each feature
    data = user_df[features].values.T
    data_norm = np.zeros_like(data, dtype=float)
    for i in range(len(features)):
        max_val = data[i].max()
        if max_val > 0:
            data_norm[i] = data[i] / max_val
    
    # Plot heatmap
    im = ax.imshow(data_norm, aspect='auto', cmap='YlOrRd')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([f.replace('_', '\n') for f in features], fontsize=8)
    ax.set_xlabel('Session Index')
    ax.set_title(f'{user_id} - Feature Heatmap', fontsize=10)
    
    return im


def main():
    parser = argparse.ArgumentParser(description="Visualize user session sequences")
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/processed/cert.duckdb",
        help="Path to DuckDB database",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="Specific user ID to visualize",
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=4,
        help="Number of users to visualize (if --user not specified)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/sequence_visualization.png",
        help="Output file path",
    )
    parser.add_argument(
        "--sessions-per-user",
        type=int,
        default=200,
        help="Max sessions to show per user",
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.db_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SEQUENCE VISUALIZATION")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Output: {output_path}")
    
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)
    
    # Get users to visualize
    if args.user:
        users = [args.user]
    else:
        print(f"\nFinding {args.num_users} interesting users...")
        users = get_interesting_users(str(db_path), args.num_users)
    
    print(f"Users to visualize: {users}")
    
    # Load data for all users
    print("\nLoading session data...")
    all_data = []
    for user_id in users:
        df = load_user_sessions(str(db_path), user_id, args.sessions_per_user)
        all_data.append(df)
        print(f"  {user_id}: {len(df)} sessions")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create visualization
    print("\nGenerating visualization...")
    
    num_users = len(users)
    fig = plt.figure(figsize=(16, 4 * num_users))
    
    for i, user_id in enumerate(users):
        # Timeline subplot
        ax1 = fig.add_subplot(num_users, 2, 2*i + 1)
        plot_user_timeline(combined_df, user_id, ax1)
        
        # Heatmap subplot
        ax2 = fig.add_subplot(num_users, 2, 2*i + 2)
        im = plot_sequence_heatmap(combined_df, user_id, ax2)
    
    # Add legend to first subplot
    handles, labels = fig.axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.98))
    
    plt.suptitle('User Session Sequences Over Time', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved to: {output_path}")
    
    # Also save as HTML with session details
    html_path = output_path.with_suffix('.html')
    generate_html_timeline(combined_df, users, html_path)
    print(f"✓ HTML report: {html_path}")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


def generate_html_timeline(df: pd.DataFrame, users: list, output_path: Path):
    """Generate interactive HTML timeline."""
    
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Session Sequence Timeline</title>
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
        .user-section { margin-bottom: 40px; background: #f8f9fa; padding: 20px; border-radius: 8px; }
        table { border-collapse: collapse; width: 100%; margin-top: 10px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 12px; }
        th { background: #3498db; color: white; }
        tr:nth-child(even) { background: #f2f2f2; }
        .highlight { background: #fff3cd !important; }
        .terminated { color: red; font-weight: bold; }
        .stat { display: inline-block; margin: 5px 15px 5px 0; padding: 8px; background: #ecf0f1; border-radius: 4px; }
        .stat-value { font-size: 18px; font-weight: bold; color: #2c3e50; }
        .stat-label { font-size: 11px; color: #7f8c8d; }
    </style>
</head>
<body>
    <h1>📊 Session Sequence Timeline</h1>
    <p>Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
"""
    
    for user_id in users:
        user_df = df[df['user_id'] == user_id].sort_values('start_time')
        
        if len(user_df) == 0:
            continue
        
        role = user_df['role'].iloc[0] if 'role' in user_df.columns else 'Unknown'
        terminated = user_df['user_terminated'].iloc[0] if 'user_terminated' in user_df.columns else False
        
        term_badge = '<span class="terminated">[TERMINATED]</span>' if terminated else ''
        
        # Stats
        total_sessions = len(user_df)
        total_actions = user_df['total_actions'].sum()
        total_files = user_df['file_copy_count'].sum()
        total_usb = user_df['usb_connect_count'].sum()
        after_hours_pct = user_df['is_after_hours'].mean() * 100
        
        html += f"""
    <div class="user-section">
        <h2>{user_id} - {role} {term_badge}</h2>
        <div>
            <div class="stat"><div class="stat-value">{total_sessions}</div><div class="stat-label">Sessions</div></div>
            <div class="stat"><div class="stat-value">{total_actions:,}</div><div class="stat-label">Total Actions</div></div>
            <div class="stat"><div class="stat-value">{total_files:,}</div><div class="stat-label">Files Copied</div></div>
            <div class="stat"><div class="stat-value">{total_usb:,}</div><div class="stat-label">USB Events</div></div>
            <div class="stat"><div class="stat-value">{after_hours_pct:.1f}%</div><div class="stat-label">After Hours</div></div>
        </div>
        <table>
            <tr>
                <th>#</th>
                <th>Start Time</th>
                <th>Duration (min)</th>
                <th>Actions</th>
                <th>HTTP</th>
                <th>Email</th>
                <th>Files</th>
                <th>USB</th>
                <th>After Hrs</th>
            </tr>
"""
        
        for idx, row in user_df.head(50).iterrows():
            highlight = 'highlight' if row['file_copy_count'] > 10 or row['usb_connect_count'] > 2 else ''
            html += f"""
            <tr class="{highlight}">
                <td>{idx}</td>
                <td>{row['start_time'].strftime('%Y-%m-%d %H:%M')}</td>
                <td>{row['duration_minutes']:.0f}</td>
                <td>{row['total_actions']}</td>
                <td>{row['http_request_count']}</td>
                <td>{row['email_count']}</td>
                <td>{row['file_copy_count']}</td>
                <td>{row['usb_connect_count']}</td>
                <td>{'✓' if row['is_after_hours'] else ''}</td>
            </tr>
"""
        
        if len(user_df) > 50:
            html += f"<tr><td colspan='9'><em>... and {len(user_df) - 50} more sessions</em></td></tr>"
        
        html += """
        </table>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


if __name__ == "__main__":
    main()
