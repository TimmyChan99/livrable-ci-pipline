"""
Pytest configuration — ensures the project root is on sys.path
so test files can import agent, state, tools directly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))