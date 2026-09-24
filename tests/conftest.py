import glob
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NA2SO4 = os.path.join(ROOT, "data", "Na2SO4")


@pytest.fixture(scope="session")
def na2so4_dir():
    if not glob.glob(os.path.join(NA2SO4, "*.sif")):
        pytest.skip("bundled data/Na2SO4 not present")
    return NA2SO4


@pytest.fixture(scope="session")
def na2so4_files(na2so4_dir):
    return sorted(glob.glob(os.path.join(na2so4_dir, "*.sif")))


@pytest.fixture(scope="session")
def na2so4_dark(na2so4_files):
    return [p for p in na2so4_files if p.endswith("_dark.sif")][0]


@pytest.fixture(scope="session")
def na2so4_scan(na2so4_files):
    return [p for p in na2so4_files if not p.endswith("_dark.sif")]
