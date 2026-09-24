"""Bit-identical regression against outputs captured before the reduce_frame
refactor (see ``tests/make_golden.py``)."""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_golden  # noqa: E402

GOLDEN = os.path.join(HERE, "data", "golden_na2so4.npz")


@pytest.fixture(scope="module")
def current(na2so4_dir):
    return make_golden.compute()


@pytest.fixture(scope="module")
def golden():
    with np.load(GOLDEN) as z:
        return {k: z[k] for k in z.files}


def test_same_keys(current, golden):
    assert set(current) == set(golden)


@pytest.mark.parametrize("key", sorted(np.load(GOLDEN).files))
def test_exact(current, golden, key):
    assert np.array_equal(np.asarray(current[key]), golden[key]), key
