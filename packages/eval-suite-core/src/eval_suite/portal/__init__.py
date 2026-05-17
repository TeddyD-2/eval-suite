"""Submission portal — accepts signed manifests, writes to a
file-backed registry, lists submissions for the public dashboard.

Run: `uvicorn eval_suite.portal.app:app --reload --port 8000`
"""

from .app import create_app
from .storage import Submission, SubmissionStore

__all__ = ["create_app", "Submission", "SubmissionStore"]
