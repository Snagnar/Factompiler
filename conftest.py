"""
Pytest configuration for the Factompiler project.
Ensures that the root directory is in the Python path so imports work correctly.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure bundled Draftsman fork is importable for tests
draftsman_path = project_root / "factorio-draftsman"
if draftsman_path.exists():
    draftsman_str = str(draftsman_path)
    if draftsman_str not in sys.path:
        sys.path.insert(0, draftsman_str)
