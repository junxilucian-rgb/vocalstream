"""Portable paths for repository-level experiment scripts.

CLI dataset roots take precedence over MUSDB_ROOT and the configuration file.
Relative configuration defaults are resolved against this repository, so a
script may be invoked from another working directory without changing data.
"""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent


def resolve_dataset_root(cli_root=None, configured_root=None):
    if cli_root is not None:
        return Path(cli_root).expanduser().resolve()
    environment_root = os.environ.get('MUSDB_ROOT')
    if environment_root:
        return Path(environment_root).expanduser().resolve()
    root = Path(configured_root or 'data/musdb18-hq').expanduser()
    return (root if root.is_absolute() else PROJECT / root).resolve()
