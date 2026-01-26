#!/usr/bin/env python
"""
Data Analysis Script
Generates comprehensive reports on CERT R4.2 data.

Outputs:
- HTML report with visualizations
- Markdown summary report
- CSV exports of key statistics
- Raw data samples (with --include-raw)
- Transformation summary (with --include-raw)

Usage:
    python scripts/analyze_data.py
    python scripts/analyze_data.py --include-raw
    python scripts/analyze_data.py --output-dir reports
    python scripts/analyze_data.py --format html
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.analyzer import (
    DataAnalyzer, 
    RawDataAnalyzer,
    generate_markdown_report, 
    generate_html_report,
    generate_transformation_summary,
)


def main():
    parser = argparse.ArgumentParser(description="Analyze CERT R4.2 data and generate reports")
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/processed/cert.duckdb",
        help="Path to DuckDB database",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Path to raw CSV files directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["html", "markdown", "all"],
        default="all",
        help="Report format(s) to generate",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include raw CSV analysis (slower but more comprehensive)",
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.db_path)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CERT R4.2 DATA ANALYSIS")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Raw directory: {raw_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Include raw analysis: {args.include_raw}")
    print()
    
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        print("Please run 'python scripts/ingest_data.py' first.")
        sys.exit(1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_analyzer = None
    
    # =========================================
    # STEP 1: Raw Data Analysis (if requested)
    # =========================================
    if args.include_raw:
        print("\n" + "-" * 40)
        print("STEP 1: ANALYZING RAW CSV FILES")
        print("-" * 40)
        
        if not raw_dir.exists():
            print(f"WARNING: Raw directory not found at {raw_dir}")
        else:
            raw_analyzer = RawDataAnalyzer(str(raw_dir))
            raw_analyzer.analyze_all()
            
            print("\nRaw File Summary:")
            summary = raw_analyzer.get_summary()
            print(summary.to_string(index=False))
            
            # Export raw samples
            raw_samples_dir = output_dir / "raw_samples"
            raw_samples_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nExporting raw samples to {raw_samples_dir}/")
            for name, stats in raw_analyzer.stats.items():
                sample_path = raw_samples_dir / f"{name}_raw_sample.csv"
                stats.sample.to_csv(sample_path, index=False)
                print(f"  ✓ {name}_raw_sample.csv")
    
    # =========================================
    # STEP 2: Processed Data Analysis
    # =========================================
    print("\n" + "-" * 40)
    print("STEP 2: ANALYZING PROCESSED DATA (DuckDB)")
    print("-" * 40)
    
    with DataAnalyzer(str(db_path)) as analyzer:
        print("Analyzing tables...")
        analyzer.analyze_all_tables()
        
        # Print summary
        print("\nProcessed Table Summary:")
        for name, stats in analyzer.stats.items():
            print(f"  {name}: {stats.row_count:,} rows, {stats.column_count} columns")
        
        # =========================================
        # STEP 3: Generate Reports
        # =========================================
        print("\n" + "-" * 40)
        print("STEP 3: GENERATING REPORTS")
        print("-" * 40)
        
        if args.format in ["markdown", "all"]:
            md_path = output_dir / f"data_analysis_{timestamp}.md"
            print(f"\nGenerating Markdown report: {md_path}")
            generate_markdown_report(analyzer, md_path)
            print(f"  ✓ Markdown report saved")
        
        if args.format in ["html", "all"]:
            html_path = output_dir / f"data_analysis_{timestamp}.html"
            print(f"\nGenerating HTML report: {html_path}")
            try:
                generate_html_report(analyzer, html_path)
                print(f"  ✓ HTML report saved")
            except ImportError as e:
                print(f"  ⚠ HTML report requires matplotlib: {e}")
        
        # Transformation summary (if raw analysis was done)
        if raw_analyzer:
            transform_path = output_dir / f"transformation_summary_{timestamp}.md"
            print(f"\nGenerating transformation summary: {transform_path}")
            generate_transformation_summary(raw_analyzer, analyzer, transform_path)
            print(f"  ✓ Transformation summary saved")
        
        # =========================================
        # STEP 4: Export Data
        # =========================================
        print("\n" + "-" * 40)
        print("STEP 4: EXPORTING DATA")
        print("-" * 40)
        
        # Export processed samples
        samples_dir = output_dir / "processed_samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nExporting processed samples to {samples_dir}/")
        for name, stats in analyzer.stats.items():
            sample_path = samples_dir / f"{name}_sample.csv"
            stats.sample.to_csv(sample_path, index=False)
            print(f"  ✓ {name}_sample.csv")
        
        # Export feature statistics
        print("\nExporting statistics...")
        feature_stats = analyzer.get_session_feature_distribution()
        feature_stats.to_csv(output_dir / "feature_statistics.csv", index=False)
        print("  ✓ feature_statistics.csv")
        
        # Export anomaly candidates
        anomalies = analyzer.get_anomaly_candidates()
        anomalies.to_csv(output_dir / "anomaly_candidates.csv", index=False)
        print(f"  ✓ anomaly_candidates.csv ({len(anomalies)} records)")
    
    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nReports saved to: {output_dir.absolute()}")
    print("\nFiles generated:")
    
    for item in sorted(output_dir.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(output_dir)
            print(f"  - {rel_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

