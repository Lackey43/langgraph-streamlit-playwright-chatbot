"""
Root-level Streamlit entry point for convenience.

Run with:
    streamlit run streamlit_app.py

This simply imports and executes the real app located in app/streamlit_app.py
so all relative imports (from app.*) continue to work cleanly.
"""
import sys
from pathlib import Path

# Make sure the project root is on sys.path so `import app.xxx` works
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Execute the actual application (all top-level Streamlit code runs here)
import app.streamlit_app  # noqa: F401
