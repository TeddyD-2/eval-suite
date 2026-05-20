"""Submission portal — accepts signed manifests, writes to a
file-backed registry, lists submissions for the public dashboard.

**In plain words.** This is the public-facing website piece. A lab
runs an evaluation locally, signs the resulting manifest, and posts
it to the portal; the portal stores it, lists it on a browsable
page, and renders the profile so anyone can look up "what does the
RT-1 profile look like" without re-running the sweep themselves. This
sub-package is the importable bundle; `portal/app.py` is the actual
web server.

Run: `uvicorn eval_suite.portal.app:app --reload --port 8000`
"""

from .app import create_app
from .storage import Submission, SubmissionStore

__all__ = ["create_app", "Submission", "SubmissionStore"]
