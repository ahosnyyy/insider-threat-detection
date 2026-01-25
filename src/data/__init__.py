"""Data processing module for CERT R4.2 dataset."""
from .ingest import ingest_cert_data, get_table_stats
from .sessionize import (
    create_sessions, 
    aggregate_session_events, 
    get_session_dataframe,
    enrich_with_ldap,
    enrich_with_psychometric,
)
from .features import (
    FeatureExtractor, 
    SequenceBuilder, 
    NUMERIC_FEATURES,
    OPTIONAL_FEATURES,
    CATEGORICAL_FEATURES,
)
from .dataset import (
    prepare_training_data,
    prepare_training_data_temporal,
)
from .analyzer import (
    DataAnalyzer, 
    RawDataAnalyzer,
    generate_markdown_report, 
    generate_html_report,
    generate_transformation_summary,
)

__all__ = [
    # Ingestion
    "ingest_cert_data",
    "get_table_stats",
    # Sessionization
    "create_sessions",
    "aggregate_session_events",
    "get_session_dataframe",
    "enrich_with_ldap",
    "enrich_with_psychometric",
    # Feature engineering
    "FeatureExtractor",
    "SequenceBuilder",
    "prepare_training_data",
    "prepare_training_data_temporal",
    "NUMERIC_FEATURES",
    "OPTIONAL_FEATURES",
    "CATEGORICAL_FEATURES",
    # Analysis
    "DataAnalyzer",
    "RawDataAnalyzer",
    "generate_markdown_report",
    "generate_html_report",
    "generate_transformation_summary",
]
