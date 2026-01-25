"""
User Baseline Module (Phase 2)
------------------------------
This module implements profiling and baseline deviation detection for user behavior.

Components:
- VectorStore: Wrapper for FAISS vector database.
- UserProfile: Statistical profiling of user sessions.
- BaselineAnalyzer: Computes deviation scores (cosine, mahalanobis).
"""

from .vector_store import VectorStore
from .profile import UserProfile
from .analyzer import BaselineAnalyzer
