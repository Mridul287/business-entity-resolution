"""
Shared test setup.

Import path
-----------
`src/` is a plain package with no install step, so tests put the repo root on
sys.path and import it as `src.preprocessing...`. `pytest.ini` sets
`pythonpath = .` as well; the insert here keeps tests working when pytest is
invoked from another directory or with a different ini.

Data policy
-----------
No test in this suite reads `dataset/`. The real source files are ~0.5 GB each
and are not committed, so any test that touched them would be slow on a good
day and broken on a fresh clone. Every fixture is synthetic: small DataFrames
built in memory, or small TSVs written to pytest's `tmp_path`. Tests that need
the real data should do it through a marked, explicitly-skipped integration
test, not by default.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repo root."""
    return REPO_ROOT
