"""
Utility functions.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Path = Path("config/config.yaml")) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: Path) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved to {path}")


def load_json(path: Path) -> Any:
    """Load data from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def load_ground_truth(
    answers_dir: Path, 
    dataset: str = "4.2",
    parse_details: bool = True,
) -> Dict[str, Any]:
    """
    Load CERT ground truth from answers directory.
    
    For R4.2, this parses:
    1. insiders.csv - Master file with user IDs and time ranges
    2. r4.2-X/*.csv - Detail files with exact malicious event timestamps
    
    Args:
        answers_dir: Path to answers directory
        dataset: Dataset version to filter for (default "4.2")
        parse_details: If True, parse detail files for session-level labels
        
    Returns:
        Dict with:
        - 'insider_users': List of insider user IDs
        - 'insider_incidents': List of dicts with user, start, end times
        - 'malicious_events': List of dicts with user, timestamp, event_type
    """
    import pandas as pd
    from datetime import datetime
    
    insider_users = set()
    insider_incidents = []  # Time ranges for each incident
    malicious_events = []   # Individual malicious events
    
    # Load insiders.csv master file
    insiders_file = answers_dir / "insiders.csv"
    if not insiders_file.exists():
        logger.warning(f"insiders.csv not found in {answers_dir}")
        return {
            'insider_users': [],
            'insider_incidents': [],
            'malicious_events': [],
        }
    
    try:
        df = pd.read_csv(insiders_file)
        
        # Filter for specified dataset
        df_filtered = df[df['dataset'].astype(str) == str(dataset)]
        logger.info(f"Found {len(df_filtered)} insider incidents for dataset {dataset}")
        
        for _, row in df_filtered.iterrows():
            user = row['user']
            insider_users.add(user)
            
            # Parse start/end times
            try:
                start = pd.to_datetime(row['start'])
                end = pd.to_datetime(row['end'])
                
                insider_incidents.append({
                    'user': user,
                    'scenario': int(row['scenario']),
                    'details_file': row['details'],
                    'start': start,
                    'end': end,
                })
            except Exception as e:
                logger.warning(f"Could not parse dates for {user}: {e}")
                insider_incidents.append({
                    'user': user,
                    'scenario': int(row['scenario']),
                    'details_file': row['details'],
                    'start': None,
                    'end': None,
                })
        
        logger.info(f"Loaded {len(insider_users)} unique insider users")
        
    except Exception as e:
        logger.error(f"Could not load {insiders_file}: {e}")
        return {
            'insider_users': [],
            'insider_incidents': [],
            'malicious_events': [],
        }
    
    # Parse detail files for exact event timestamps
    if parse_details:
        for scenario_num in [1, 2, 3]:
            detail_dir = answers_dir / f"r{dataset}-{scenario_num}"
            if not detail_dir.exists():
                continue
                
            for detail_file in detail_dir.glob("*.csv"):
                try:
                    # Detail files have no header, variable columns
                    # Format: type,id,timestamp,user,pc,action[,extra...]
                    with open(detail_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split(',')
                            if len(parts) >= 5:
                                event_type = parts[0]
                                timestamp_str = parts[2]
                                user = parts[3]
                                
                                try:
                                    timestamp = pd.to_datetime(timestamp_str)
                                    malicious_events.append({
                                        'user': user,
                                        'timestamp': timestamp,
                                        'event_type': event_type,
                                        'file': detail_file.name,
                                    })
                                except Exception:
                                    pass  # Skip unparseable timestamps
                                    
                except Exception as e:
                    logger.warning(f"Could not parse {detail_file}: {e}")
        
        logger.info(f"Parsed {len(malicious_events)} malicious events from detail files")
    
    return {
        'insider_users': list(insider_users),
        'insider_incidents': insider_incidents,
        'malicious_events': malicious_events,
    }


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )
