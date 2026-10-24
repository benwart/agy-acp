"""Module execution entrypoint for python -m agy_acp."""

from __future__ import annotations

import sys

from agy_acp.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
