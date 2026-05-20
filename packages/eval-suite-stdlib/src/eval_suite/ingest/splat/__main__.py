"""Module entry-point so `python -m eval_suite.ingest.splat ingest ...` works.

**In plain words.** The hook that turns the splat CLI into a real
command. Just delegates to `cli.py`.
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
