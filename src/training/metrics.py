"""
Metric calculation utilities.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

def compute_ttd(
    predictions: np.ndarray,
    test_labels: np.ndarray,
    test_session_ids: List[str],
    test_timestamps: List[Union[str, pd.Timestamp]],
    insider_incidents: List[Dict],
    test_user_ids: Optional[List[str]] = None,
) -> Dict[str, Union[float, int, dict]]:
    """
    Compute Time-to-Detect (TTD) metrics.
    
    Args:
        predictions: Binary predictions (0/1) for test set
        test_labels: True labels (0/1) for test set
        test_session_ids: List of session IDs corresponding to predictions
        test_timestamps: List of timestamps corresponding to predictions
        insider_incidents: List of incident dicts with {'user', 'start', 'end'}
        test_user_ids: Optional list of user IDs (if not embedded in session ID)
        
    Returns:
        Dict containing:
            - ttd_hours_mean: Mean time to first detection in hours
            - ttd_hours_median: Median time to first detection
            - ttd_sessions_mean: Mean number of sessions before detection
            - ttd_lag_mean: Mean detection lag as % of incident duration
            - n_incidents_detected: Number of incidents detected
            - n_incidents_total: Total number of incidents
            - pct_first_session: % of detected incidents caught in first session
            - per_incident_ttd: Dict mapping user_id -> TTD hours
    """
    # Check inputs
    if (len(predictions) != len(test_labels) or 
        len(predictions) != len(test_session_ids) or
        not insider_incidents):
        return {
            'ttd_hours_mean': 0.0,
            'ttd_hours_median': 0.0,
            'ttd_sessions_mean': 0.0,
            'ttd_lag_mean': 0.0,
            'n_incidents_detected': 0,
            'n_incidents_total': 0,
            'pct_first_session': 0.0,
            'per_incident_ttd': {},
        }
    
    # Build lookup: session_id -> (timestamp, prediction, label, user)
    session_info = {}
    for i, (sid, pred, label) in enumerate(zip(test_session_ids, predictions, test_labels)):
        # Handle potential None timestamps
        ts = test_timestamps[i]
        if ts is not None:
             # Ensure pandas timestamp for comparison
            if not isinstance(ts, pd.Timestamp):
                ts = pd.to_datetime(ts)
                
            # Determine user_id
            user_id = None
            if test_user_ids and i < len(test_user_ids):
                user_id = test_user_ids[i]
            
            session_info[sid] = {
                'timestamp': ts,
                'prediction': pred,
                'label': label,
                'user_id': user_id,
            }
    
    ttd_hours_list = []
    ttd_sessions_list = []
    ttd_lag_list = []
    per_incident_ttd = {}      # {user_id: ttd_hours}
    first_session_detections = 0
    
    for incident in insider_incidents:
        user = incident.get('user')
        incident_start = incident.get('start')
        incident_end = incident.get('end')
        
        if not (user and incident_start and incident_end):
            continue
            
        # Ensure timestamps
        if not isinstance(incident_start, pd.Timestamp):
            incident_start = pd.to_datetime(incident_start)
        if not isinstance(incident_end, pd.Timestamp):
            incident_end = pd.to_datetime(incident_end)
            
        incident_duration_hours = (incident_end - incident_start).total_seconds() / 3600
        if incident_duration_hours <= 0:
            incident_duration_hours = 1.0  # Avoid division by zero
        
        # Find all sessions for this user during incident
        user_sessions = []
        for sid, info in session_info.items():
            # MATCHING LOGIC: Use explicit user_id if available, fallback to substring
            match = False
            if info['user_id']:
                if info['user_id'] == user:
                    match = True
            elif user in sid:  # Fallback to current logic
                match = True
                
            if match:
                if incident_start <= info['timestamp'] <= incident_end:
                    user_sessions.append(info)
        
        if not user_sessions:
            continue
        
        # Sort by timestamp
        user_sessions.sort(key=lambda x: x['timestamp'])
        
        # Find first detection
        first_detection_idx = None
        for idx, sess in enumerate(user_sessions):
            if sess['prediction'] == 1:
                first_detection_idx = idx
                break
        
        if first_detection_idx is not None:
            first_detection_time = user_sessions[first_detection_idx]['timestamp']
            
            # TTD in hours
            ttd_hours = (first_detection_time - incident_start).total_seconds() / 3600
            ttd_hours = max(0, ttd_hours)  # Can't be negative
            ttd_hours_list.append(ttd_hours)
            
            # Per-incident TTD
            per_incident_ttd[user] = ttd_hours
            
            # TTD in sessions (0-indexed)
            ttd_sessions_list.append(first_detection_idx)
            
            # Track first-session detections (detected at session 0)
            if first_detection_idx == 0:
                first_session_detections += 1
            
            # TTD as lag percentage
            ttd_lag = ttd_hours / incident_duration_hours * 100
            ttd_lag_list.append(min(100, max(0, ttd_lag)))
    
    n_detected = len(ttd_hours_list)
    n_total = len(insider_incidents)
    
    return {
        'ttd_hours_mean': np.mean(ttd_hours_list) if ttd_hours_list else 0.0,
        'ttd_hours_median': np.median(ttd_hours_list) if ttd_hours_list else 0.0,
        'ttd_sessions_mean': np.mean(ttd_sessions_list) if ttd_sessions_list else 0.0,
        'ttd_lag_mean': np.mean(ttd_lag_list) if ttd_lag_list else 0.0,
        'n_incidents_detected': n_detected,
        'n_incidents_total': n_total,
        'pct_first_session': (first_session_detections / n_detected * 100) if n_detected > 0 else 0.0,
        'per_incident_ttd': per_incident_ttd,
    }
