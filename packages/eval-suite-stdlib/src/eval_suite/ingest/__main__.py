"""`python -m eval_suite.ingest` dispatcher entry point.

**In plain words.** This is the file that lets `python -m eval_suite.ingest`
run as a real command. It just hands off to the dispatcher CLI in
`cli.py`.
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
