"""Entry point for the standalone build.

PyInstaller runs its entry script as `__main__` with no package around it, so
`photo_triage/__main__.py` cannot be used directly: its relative imports have
nothing to be relative to. This imports the package properly and calls in.
"""

import sys

from photo_triage.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
