import sys
from pathlib import Path

import os
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "suffix-prediction"))


@pytest.fixture(autouse=True, scope="session")
def _session_writable_cwd(tmp_path_factory):
    """
    FiniteStateMachine writes intermediate simpleDFAs/* in the current working directory.
    Use a writable session cwd to avoid permission issues on read-only mounts.
    """
    old_cwd = os.getcwd()
    new_cwd = tmp_path_factory.mktemp("pytest_cwd")
    os.chdir(new_cwd)
    try:
        yield
    finally:
        os.chdir(old_cwd)
