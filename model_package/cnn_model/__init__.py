from pathlib import Path

_version_path = Path(__file__).parent / "VERSION"
__version__ = _version_path.read_text().strip()
