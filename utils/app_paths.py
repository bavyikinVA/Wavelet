"""User-writable application locations for source and frozen distributions."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def user_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "WaveletAnalysis"


def log_file_path() -> Path:
    folder = user_data_dir() / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "wavelet_analysis.log"


def app_resource(name: str) -> Path:
    """Bundled read-only asset; never put mutable workspaces in _MEIPASS."""
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return root / name


def research_downloads_dir() -> Path:
    """Preserve current Downloads behavior, with a writable fallback."""
    downloads = Path.home() / "Downloads"
    if downloads.is_dir():
        return downloads
    documents = Path.home() / "Documents"
    if documents.is_dir():
        return documents
    # On managed Windows systems both profile folders can be redirected.
    directory = user_data_dir() / "research"
    directory.mkdir(parents=True, exist_ok=True)
    return directory
