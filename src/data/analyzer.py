"""
Data Analysis Module
Generates comprehensive reports on raw and processed data.

Outputs:
- HTML report with visualizations
- Markdown summary
- Sample data exports
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TableStats:
    """Statistics for a single table."""
    name: str
    row_count: int
    column_count: int
    columns: List[str]
    dtypes: Dict[str, str]
    null_counts: Dict[str, int]
    sample: pd.DataFrame
    numeric_stats: Optional[pd.DataFrame] = None


@dataclass
class RawFileStats:
    """Statistics for a raw CSV file."""
    name: str
    file_path: Path
    file_size_mb: float
    row_count: int
    column_count: int
    columns: List[str]
    dtypes: Dict[str, str]
    null_counts: Dict[str, int]
    null_percentages: Dict[str, float]
    sample: pd.DataFrame
    numeric_stats: Optional[Dict[str, Dict[str, float]]] = None
    date_range: Optional[Tuple[str, str]] = None


class RawDataAnalyzer:
    """Analyze raw CSV files before processing."""
    
    RAW_FILES = {
        'logon': 'logon.csv',
        'device': 'device.csv',
        'http': 'http.csv',
        'email': 'email.csv',
        'file': 'file.csv',
        'psychometric': 'psychometric.csv',
    }
    
    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = Path(raw_dir)
        self.stats: Dict[str, RawFileStats] = {}
    
    def analyze_file(self, name: str, filename: str, sample_size: int = 10, 
                     chunk_size: int = 100000) -> Optional[RawFileStats]:
        """Analyze a single CSV file without loading entirely into memory."""
        file_path = self.raw_dir / filename
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
        
        logger.info(f"Analyzing raw file: {filename}")
        
        # File size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Read first chunk for schema and sample
        try:
            sample_df = pd.read_csv(file_path, nrows=sample_size)
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
            return None
        
        columns = sample_df.columns.tolist()
        dtypes = {col: str(sample_df[col].dtype) for col in columns}
        
        # Count rows and null values by iterating chunks
        row_count = 0
        null_counts = {col: 0 for col in columns}
        numeric_stats = {}
        date_col = None
        min_date = None
        max_date = None
        
        # Detect date column
        for col in columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        
        # Process in chunks for large files
        try:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                row_count += len(chunk)
                
                for col in columns:
                    null_counts[col] += chunk[col].isna().sum()
                
                # Track date range
                if date_col and date_col in chunk.columns:
                    chunk_dates = pd.to_datetime(chunk[date_col], errors='coerce')
                    chunk_min = chunk_dates.min()
                    chunk_max = chunk_dates.max()
                    if pd.notna(chunk_min):
                        min_date = chunk_min if min_date is None else min(min_date, chunk_min)
                    if pd.notna(chunk_max):
                        max_date = chunk_max if max_date is None else max(max_date, chunk_max)
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            row_count = -1
        
        # Calculate null percentages
        null_percentages = {}
        for col, count in null_counts.items():
            null_percentages[col] = (count / row_count * 100) if row_count > 0 else 0
        
        # Numeric stats from sample
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            numeric_stats[col] = {
                'min': sample_df[col].min(),
                'max': sample_df[col].max(),
                'mean': sample_df[col].mean(),
            }
        
        # Date range
        date_range = None
        if min_date and max_date:
            date_range = (str(min_date.date()), str(max_date.date()))
        
        return RawFileStats(
            name=name,
            file_path=file_path,
            file_size_mb=file_size_mb,
            row_count=row_count,
            column_count=len(columns),
            columns=columns,
            dtypes=dtypes,
            null_counts=null_counts,
            null_percentages=null_percentages,
            sample=sample_df,
            numeric_stats=numeric_stats,
            date_range=date_range,
        )
    
    def analyze_all(self) -> Dict[str, RawFileStats]:
        """Analyze all raw CSV files."""
        for name, filename in self.RAW_FILES.items():
            stats = self.analyze_file(name, filename)
            if stats:
                self.stats[name] = stats
        
        # Also analyze LDAP files
        ldap_dir = self.raw_dir / "LDAP"
        if ldap_dir.exists():
            ldap_files = list(ldap_dir.glob("*.csv"))
            if ldap_files:
                # Sample first LDAP file
                first_ldap = ldap_files[0]
                sample_df = pd.read_csv(first_ldap, nrows=10)
                total_rows = sum(len(pd.read_csv(f)) for f in ldap_files)
                
                self.stats['ldap'] = RawFileStats(
                    name='ldap',
                    file_path=ldap_dir,
                    file_size_mb=sum(f.stat().st_size for f in ldap_files) / (1024 * 1024),
                    row_count=total_rows,
                    column_count=len(sample_df.columns),
                    columns=sample_df.columns.tolist(),
                    dtypes={col: str(sample_df[col].dtype) for col in sample_df.columns},
                    null_counts={},
                    null_percentages={},
                    sample=sample_df,
                    date_range=(ldap_files[0].stem, ldap_files[-1].stem),
                )
        
        return self.stats
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all raw files."""
        rows = []
        for name, stats in self.stats.items():
            rows.append({
                'File': name,
                'Size (MB)': f"{stats.file_size_mb:.1f}",
                'Rows': f"{stats.row_count:,}",
                'Columns': stats.column_count,
                'Date Range': f"{stats.date_range[0]} to {stats.date_range[1]}" if stats.date_range else "N/A",
            })
        return pd.DataFrame(rows)


class DataAnalyzer:
    """Analyze raw and processed data."""
    
    def __init__(self, db_path: str = "data/processed/cert.duckdb"):
        self.db_path = db_path
        self.conn = None
        self.stats = {}
        
    def connect(self):
        """Connect to DuckDB."""
        self.conn = duckdb.connect(self.db_path, read_only=True)
        return self
    
    def close(self):
        """Close connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def get_table_list(self) -> List[str]:
        """Get list of all tables."""
        result = self.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        return [r[0] for r in result]
    
    def analyze_table(self, table_name: str, sample_size: int = 5) -> TableStats:
        """Analyze a single table."""
        # Row count
        row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        # Column info
        col_info = self.conn.execute(f"DESCRIBE {table_name}").fetchdf()
        columns = col_info['column_name'].tolist()
        dtypes = dict(zip(col_info['column_name'], col_info['column_type']))
        
        # Sample data
        sample = self.conn.execute(f"SELECT * FROM {table_name} LIMIT {sample_size}").fetchdf()
        
        # Null counts (quote column names to handle reserved keywords like 'to')
        null_counts = {}
        for col in columns:
            try:
                null_count = self.conn.execute(
                    f'SELECT COUNT(*) FROM {table_name} WHERE "{col}" IS NULL'
                ).fetchone()[0]
                null_counts[col] = null_count
            except Exception:
                null_counts[col] = -1  # Mark as error
        
        # Numeric statistics
        numeric_cols = [c for c, t in dtypes.items() if 'INT' in t.upper() or 'DOUBLE' in t.upper() or 'FLOAT' in t.upper()]
        numeric_stats = None
        if numeric_cols and row_count > 0:
            stats_query = f"""
            SELECT 
                '{table_name}' as table_name,
                {', '.join([f"AVG({c}) as avg_{c}, MIN({c}) as min_{c}, MAX({c}) as max_{c}" for c in numeric_cols[:10]])}
            FROM {table_name}
            """
            try:
                numeric_stats = self.conn.execute(stats_query).fetchdf()
            except Exception:
                pass
        
        return TableStats(
            name=table_name,
            row_count=row_count,
            column_count=len(columns),
            columns=columns,
            dtypes=dtypes,
            null_counts=null_counts,
            sample=sample,
            numeric_stats=numeric_stats,
        )
    
    def analyze_all_tables(self) -> Dict[str, TableStats]:
        """Analyze all tables."""
        tables = self.get_table_list()
        for table in tables:
            logger.info(f"Analyzing table: {table}")
            self.stats[table] = self.analyze_table(table)
        return self.stats
    
    def get_session_feature_distribution(self) -> pd.DataFrame:
        """Get distribution stats for session features."""
        query = """
        SELECT 
            'duration_minutes' as feature,
            MIN(duration_minutes) as min_val,
            AVG(duration_minutes) as mean_val,
            MEDIAN(duration_minutes) as median_val,
            MAX(duration_minutes) as max_val,
            STDDEV(duration_minutes) as std_val,
            COUNT(*) as count
        FROM session_features
        UNION ALL
        SELECT 'total_actions', MIN(total_actions), AVG(total_actions), 
               MEDIAN(total_actions), MAX(total_actions), STDDEV(total_actions), COUNT(*)
        FROM session_features
        UNION ALL
        SELECT 'http_request_count', MIN(http_request_count), AVG(http_request_count),
               MEDIAN(http_request_count), MAX(http_request_count), STDDEV(http_request_count), COUNT(*)
        FROM session_features
        UNION ALL
        SELECT 'email_count', MIN(email_count), AVG(email_count),
               MEDIAN(email_count), MAX(email_count), STDDEV(email_count), COUNT(*)
        FROM session_features
        UNION ALL
        SELECT 'file_copy_count', MIN(file_copy_count), AVG(file_copy_count),
               MEDIAN(file_copy_count), MAX(file_copy_count), STDDEV(file_copy_count), COUNT(*)
        FROM session_features
        UNION ALL
        SELECT 'usb_connect_count', MIN(usb_connect_count), AVG(usb_connect_count),
               MEDIAN(usb_connect_count), MAX(usb_connect_count), STDDEV(usb_connect_count), COUNT(*)
        FROM session_features
        """
        return self.conn.execute(query).fetchdf()
    
    def get_categorical_distribution(self) -> Dict[str, pd.DataFrame]:
        """Get distribution of categorical variables."""
        distributions = {}
        
        # Role distribution
        distributions['role'] = self.conn.execute("""
            SELECT role, COUNT(*) as count, 
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
            FROM session_features
            GROUP BY role
            ORDER BY count DESC
            LIMIT 20
        """).fetchdf()
        
        # Department distribution
        distributions['department'] = self.conn.execute("""
            SELECT department, COUNT(*) as count,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
            FROM session_features
            GROUP BY department
            ORDER BY count DESC
            LIMIT 15
        """).fetchdf()
        
        # After hours distribution
        distributions['after_hours'] = self.conn.execute("""
            SELECT 
                CASE WHEN is_after_hours THEN 'After Hours' ELSE 'Business Hours' END as time_period,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
            FROM session_features
            GROUP BY is_after_hours
        """).fetchdf()
        
        # Terminated distribution
        distributions['terminated'] = self.conn.execute("""
            SELECT 
                CASE WHEN user_terminated THEN 'Terminated' ELSE 'Active' END as status,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
            FROM session_features
            GROUP BY user_terminated
        """).fetchdf()
        
        return distributions
    
    def get_temporal_patterns(self) -> pd.DataFrame:
        """Analyze temporal patterns in data."""
        return self.conn.execute("""
            SELECT 
                start_hour,
                COUNT(*) as session_count,
                AVG(total_actions) as avg_actions,
                AVG(duration_minutes) as avg_duration
            FROM session_features
            GROUP BY start_hour
            ORDER BY start_hour
        """).fetchdf()
    
    def get_anomaly_candidates(self) -> pd.DataFrame:
        """Find potential anomaly candidates (high activity sessions)."""
        return self.conn.execute("""
            SELECT 
                session_id,
                user_id,
                role,
                duration_minutes,
                total_actions,
                file_copy_count,
                usb_connect_count,
                is_after_hours,
                user_terminated
            FROM session_features
            WHERE 
                file_copy_count > 50
                OR usb_connect_count > 10
                OR (is_after_hours AND total_actions > 500)
                OR user_terminated
            ORDER BY file_copy_count DESC
            LIMIT 50
        """).fetchdf()


def generate_markdown_report(analyzer: DataAnalyzer, output_path: Path) -> str:
    """Generate a Markdown report."""
    
    lines = []
    lines.append("# CERT R4.2 Data Analysis Report")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append(f"\n*Database: {analyzer.db_path}*\n")
    
    # Table Overview
    lines.append("## 1. Table Overview\n")
    lines.append("| Table | Rows | Columns |")
    lines.append("|-------|------|---------|")
    for name, stats in analyzer.stats.items():
        lines.append(f"| {name} | {stats.row_count:,} | {stats.column_count} |")
    
    # Session Features Detail
    if 'session_features' in analyzer.stats:
        sf = analyzer.stats['session_features']
        lines.append("\n## 2. Session Features Table\n")
        lines.append(f"**Total Sessions:** {sf.row_count:,}\n")
        lines.append(f"**Columns ({sf.column_count}):** {', '.join(sf.columns[:15])}...")
        
        # Feature distributions
        lines.append("\n### 2.1 Numeric Feature Statistics\n")
        dist = analyzer.get_session_feature_distribution()
        lines.append("| Feature | Min | Mean | Median | Max | Std |")
        lines.append("|---------|-----|------|--------|-----|-----|")
        for _, row in dist.iterrows():
            lines.append(f"| {row['feature']} | {row['min_val']:.1f} | {row['mean_val']:.1f} | {row['median_val']:.1f} | {row['max_val']:.1f} | {row['std_val']:.1f} |")
    
    # Categorical distributions
    lines.append("\n### 2.2 Role Distribution (Top 10)\n")
    cat_dist = analyzer.get_categorical_distribution()
    lines.append("| Role | Count | % |")
    lines.append("|------|-------|---|")
    for _, row in cat_dist['role'].head(10).iterrows():
        lines.append(f"| {row['role']} | {row['count']:,} | {row['percentage']}% |")
    
    # After hours
    lines.append("\n### 2.3 Session Timing\n")
    lines.append("| Period | Count | % |")
    lines.append("|--------|-------|---|")
    for _, row in cat_dist['after_hours'].iterrows():
        lines.append(f"| {row['time_period']} | {row['count']:,} | {row['percentage']}% |")
    
    # Terminated
    lines.append("\n### 2.4 Employee Status\n")
    lines.append("| Status | Count | % |")
    lines.append("|--------|-------|---|")
    for _, row in cat_dist['terminated'].iterrows():
        lines.append(f"| {row['status']} | {row['count']:,} | {row['percentage']}% |")
    
    # Temporal patterns
    lines.append("\n### 2.5 Hourly Activity Pattern\n")
    temporal = analyzer.get_temporal_patterns()
    lines.append("| Hour | Sessions | Avg Actions | Avg Duration (min) |")
    lines.append("|------|----------|-------------|-------------------|")
    for _, row in temporal.iterrows():
        lines.append(f"| {int(row['start_hour']):02d}:00 | {row['session_count']:,} | {row['avg_actions']:.0f} | {row['avg_duration']:.0f} |")
    
    # Anomaly candidates
    lines.append("\n## 3. Potential Anomaly Candidates (Sample)\n")
    lines.append("Sessions with unusual patterns:\n")
    anomalies = analyzer.get_anomaly_candidates()
    lines.append("| User | Role | Files | USB | Actions | After Hours | Terminated |")
    lines.append("|------|------|-------|-----|---------|-------------|------------|")
    for _, row in anomalies.head(15).iterrows():
        lines.append(f"| {row['user_id']} | {row['role']} | {row['file_copy_count']} | {row['usb_connect_count']} | {row['total_actions']} | {row['is_after_hours']} | {row['user_terminated']} |")
    
    # Data processing summary
    lines.append("\n## 4. Data Processing Applied\n")
    lines.append("""
The following preprocessing is applied during training:

| Step | Method | Details |
|------|--------|---------|
| **Missing Values** | Median Imputation | Missing numeric values filled with column median |
| **Outlier Handling** | Percentile Clipping | Values clipped to 1st-99th percentile |
| **Normalization** | RobustScaler | Uses median/IQR (resistant to outliers) |
| **Role Encoding** | One-Hot | 42 role categories → 42 binary columns |
| **Sequence Padding** | Zero Padding | Sequences padded to fixed length with mask |
""")
    
    # Sample data
    lines.append("\n## 5. Sample Records\n")
    lines.append("### Session Features (First 5 rows)\n")
    lines.append("```")
    if 'session_features' in analyzer.stats:
        sample = analyzer.stats['session_features'].sample
        lines.append(sample.to_string())
    lines.append("```\n")
    
    report = "\n".join(lines)
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report


def generate_html_report(analyzer: DataAnalyzer, output_path: Path) -> str:
    """Generate an HTML report with visualizations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO
    
    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return f'<img src="data:image/png;base64,{img_str}" />'
    
    # Generate plots
    plots = {}
    
    # 1. Hourly activity
    temporal = analyzer.get_temporal_patterns()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(temporal['start_hour'], temporal['session_count'], color='steelblue')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Session Count')
    ax.set_title('Session Activity by Hour')
    ax.set_xticks(range(24))
    plots['hourly'] = fig_to_base64(fig)
    
    # 2. Role distribution
    cat_dist = analyzer.get_categorical_distribution()
    fig, ax = plt.subplots(figsize=(10, 5))
    top_roles = cat_dist['role'].head(15)
    ax.barh(top_roles['role'], top_roles['count'], color='coral')
    ax.set_xlabel('Session Count')
    ax.set_title('Sessions by Role (Top 15)')
    ax.invert_yaxis()
    plots['roles'] = fig_to_base64(fig)
    
    # 3. Feature distributions
    dist = analyzer.get_session_feature_distribution()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    features = dist['feature'].tolist()
    for i, (ax, feat) in enumerate(zip(axes.flat, features)):
        row = dist[dist['feature'] == feat].iloc[0]
        ax.bar(['Min', 'Median', 'Mean', 'Max'], 
               [row['min_val'], row['median_val'], row['mean_val'], row['max_val']],
               color=['green', 'blue', 'orange', 'red'])
        ax.set_title(feat.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plots['features'] = fig_to_base64(fig)
    
    # Build HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CERT R4.2 Data Analysis Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; 
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ display: inline-block; padding: 15px 25px; margin: 5px; 
                   background: #3498db; color: white; border-radius: 5px; text-align: center; }}
        .metric .value {{ font-size: 24px; font-weight: bold; }}
        .metric .label {{ font-size: 12px; opacity: 0.9; }}
        img {{ max-width: 100%; }}
        .highlight {{ background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 15px 0; }}
        pre {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>🔍 CERT R4.2 Data Analysis Report</h1>
    <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    
    <div class="card">
        <h2>📊 Dataset Overview</h2>
        <div>
            {"".join([f'<div class="metric"><div class="value">{stats.row_count:,}</div><div class="label">{name}</div></div>' for name, stats in analyzer.stats.items()])}
        </div>
    </div>
    
    <div class="card">
        <h2>⏰ Hourly Activity Pattern</h2>
        {plots['hourly']}
    </div>
    
    <div class="card">
        <h2>👥 Sessions by Role</h2>
        {plots['roles']}
    </div>
    
    <div class="card">
        <h2>📈 Feature Distributions</h2>
        {plots['features']}
    </div>
    
    <div class="card">
        <h2>📋 Numeric Feature Statistics</h2>
        {dist.to_html(index=False)}
    </div>
    
    <div class="card">
        <h2>🚨 Potential Anomaly Candidates</h2>
        <div class="highlight">
            <strong>High-risk indicators:</strong> High file copies, multiple USB connections, 
            after-hours activity, or terminated employees.
        </div>
        {analyzer.get_anomaly_candidates().head(20).to_html(index=False)}
    </div>
    
    <div class="card">
        <h2>⚙️ Preprocessing Applied</h2>
        <table>
            <tr><th>Step</th><th>Method</th><th>Details</th></tr>
            <tr><td>Missing Values</td><td>Median Imputation</td><td>Filled with column median</td></tr>
            <tr><td>Outliers</td><td>Percentile Clipping</td><td>Clipped to 1st-99th percentile</td></tr>
            <tr><td>Normalization</td><td>RobustScaler</td><td>Uses median and IQR</td></tr>
            <tr><td>Role Encoding</td><td>One-Hot</td><td>42 categories → 42 columns</td></tr>
            <tr><td>Sequences</td><td>Zero Padding</td><td>Padded with attention mask</td></tr>
        </table>
    </div>
    
    <div class="card">
        <h2>📝 Sample Session Records</h2>
        <pre>{analyzer.stats.get('session_features', TableStats('',0,0,[],{},{},pd.DataFrame())).sample.to_string() if 'session_features' in analyzer.stats else 'N/A'}</pre>
    </div>
    
</body>
</html>
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return str(output_path)


def generate_transformation_summary(
    raw_analyzer: RawDataAnalyzer, 
    processed_analyzer: DataAnalyzer,
    output_path: Path
) -> str:
    """Generate a markdown summary of data transformations."""
    
    lines = []
    lines.append("# Data Transformation Summary")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Overview
    lines.append("## 1. Data Pipeline Overview\n")
    lines.append("```")
    lines.append("Raw CSVs → Parquet → DuckDB → Sessions → Features → Sequences")
    lines.append("```\n")
    
    # Raw vs Processed Comparison
    lines.append("## 2. Raw vs Processed Row Counts\n")
    lines.append("| Table | Raw Rows | Processed Rows | Retention |")
    lines.append("|-------|----------|----------------|-----------|")
    
    for name in raw_analyzer.stats:
        raw_count = raw_analyzer.stats[name].row_count
        proc_count = processed_analyzer.stats.get(name, TableStats(name, 0, 0, [], {}, {}, pd.DataFrame())).row_count
        retention = f"{proc_count / raw_count * 100:.1f}%" if raw_count > 0 else "N/A"
        lines.append(f"| {name} | {raw_count:,} | {proc_count:,} | {retention} |")
    
    # Session creation
    if 'session_features' in processed_analyzer.stats:
        sf_count = processed_analyzer.stats['session_features'].row_count
        logon_count = raw_analyzer.stats.get('logon', RawFileStats('', Path('.'), 0, 0, 0, [], {}, {}, {}, pd.DataFrame())).row_count
        lines.append(f"\n**Session Creation:**")
        lines.append(f"- {logon_count:,} logon events → {sf_count:,} sessions")
        lines.append(f"- Aggregation ratio: {logon_count / sf_count:.1f}:1")
    
    # Schema changes
    lines.append("\n## 3. Schema Transformations\n")
    
    for name in ['logon', 'email', 'http', 'device', 'file']:
        if name in raw_analyzer.stats and name in processed_analyzer.stats:
            raw_cols = set(raw_analyzer.stats[name].columns)
            proc_cols = set(processed_analyzer.stats[name].columns)
            
            added = proc_cols - raw_cols
            removed = raw_cols - proc_cols
            
            if added or removed:
                lines.append(f"### {name.title()}")
                if added:
                    lines.append(f"- **Added columns:** {', '.join(sorted(added))}")
                if removed:
                    lines.append(f"- **Removed columns:** {', '.join(sorted(removed))}")
                lines.append("")
    
    # Feature engineering
    lines.append("## 4. Feature Engineering\n")
    lines.append("### Derived Features (session_features table)\n")
    
    derived_features = [
        ("Temporal", ["duration_minutes", "start_hour", "day_of_week", "is_weekend", "is_after_hours"]),
        ("Device", ["device_event_count", "usb_connect_count", "usb_disconnect_count"]),
        ("HTTP", ["http_request_count", "unique_domains"]),
        ("Email", ["email_count", "emails_sent", "emails_received", "total_attachments", "external_email_count"]),
        ("File", ["file_copy_count", "exe_copy_count", "doc_copy_count", "archive_copy_count"]),
        ("Behavioral", ["total_actions", "action_density"]),
        ("LDAP", ["is_admin", "role_changed", "dept_changed", "user_terminated"]),
        ("Psychometric", ["O", "C", "E", "A", "N"]),
    ]
    
    for category, features in derived_features:
        lines.append(f"- **{category}:** {', '.join(features)}")
    
    # Preprocessing
    lines.append("\n## 5. Preprocessing Applied During Training\n")
    lines.append("| Step | Method | Details |")
    lines.append("|------|--------|---------|")
    lines.append("| Missing Values | Median Imputation | NaN → column median |")
    lines.append("| Outliers | Percentile Clipping | Clip to 1st-99th percentile |")
    lines.append("| Normalization | RobustScaler | (x - median) / IQR |")
    lines.append("| Role Encoding | One-Hot | 42 categories → 42 binary columns |")
    lines.append("| Sequences | Zero Padding | Pad to fixed length with mask |")
    
    # Sample data comparison
    lines.append("\n## 6. Sample Data Comparison\n")
    
    if 'logon' in raw_analyzer.stats:
        lines.append("### Raw logon.csv (first 5 rows)")
        lines.append("```")
        lines.append(raw_analyzer.stats['logon'].sample.head().to_string())
        lines.append("```\n")
    
    if 'session_features' in processed_analyzer.stats:
        lines.append("### Processed session_features (first 5 rows, selected columns)")
        lines.append("```")
        sample_cols = ['session_id', 'user_id', 'duration_minutes', 'total_actions', 'file_copy_count']
        sample = processed_analyzer.stats['session_features'].sample
        available_cols = [c for c in sample_cols if c in sample.columns]
        lines.append(sample[available_cols].to_string())
        lines.append("```\n")
    
    report = "\n".join(lines)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report


if __name__ == "__main__":
    print("Use scripts/analyze_data.py to generate reports")
