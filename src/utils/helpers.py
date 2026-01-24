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


def load_ground_truth(answers_dir: Path) -> Dict[str, List[str]]:
    """
    Load CERT R4.2 ground truth from answers directory.
    
    Returns:
        Dict with 'insider_users' and 'insider_sessions' lists
    """
    insider_users = set()
    insider_sessions = set()
    
    # The answers directory contains files indicating malicious activity
    for file in answers_dir.glob("*.csv"):
        try:
            import pandas as pd
            df = pd.read_csv(file)
            
            if 'user' in df.columns:
                insider_users.update(df['user'].dropna().unique())
            if 'session_id' in df.columns:
                insider_sessions.update(df['session_id'].dropna().unique())
        except Exception as e:
            logger.warning(f"Could not load {file}: {e}")
    
    # Also check for insiders.csv if it exists
    insiders_file = answers_dir / "insiders.csv"
    if insiders_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(insiders_file)
            if 'user_id' in df.columns:
                insider_users.update(df['user_id'].dropna().unique())
            elif 'user' in df.columns:
                insider_users.update(df['user'].dropna().unique())
        except Exception as e:
            logger.warning(f"Could not load {insiders_file}: {e}")
    
    return {
        'insider_users': list(insider_users),
        'insider_sessions': list(insider_sessions),
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
