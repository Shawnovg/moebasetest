import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: tiny-set learning checks (seconds to a minute on CPU)")
    config.addinivalue_line("markers", "parity: numerical comparison against pinned upstream code in third_party/")


@pytest.fixture(autouse=True)
def _seed():
    try:
        import torch
        torch.manual_seed(0)
    except ImportError:
        pass
