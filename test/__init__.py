from pathlib import Path
import pytest


@pytest.fixture
def plot_dir():
    path = Path('.') / 'test_plots'
    path.mkdir(exist_ok=True)

    yield path
