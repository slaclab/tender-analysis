import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL = "Na2SO4_pellet_20pcSucrose_Ka_RIXS_01"


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "tender_analysis.cli", *args],
                          capture_output=True, text=True, timeout=600)


def _read_csv(path):
    with open(path) as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    header = lines[0].strip().split(",")
    data = np.array([[float(v) for v in ln.strip().split(",")] for ln in lines[1:]])
    return header, data


def test_list(na2so4_dir):
    p = _cli("list", na2so4_dir)
    assert p.returncode == 0, p.stderr
    assert LABEL in p.stdout


def test_reduce_rixs_matches_golden(na2so4_dir, tmp_path):
    p = _cli("reduce", na2so4_dir, "--measurement", LABEL, "--out", str(tmp_path),
             "--n", "7")
    assert p.returncode == 0, p.stderr
    path = tmp_path / f"{LABEL}.csv"
    assert p.stdout.strip() == str(path)
    header, data = _read_csv(path)
    assert header[:2] == ["energy_eV", "mu"] and "tfy" in header
    with np.load(os.path.join(HERE, "data", "golden_na2so4.npz")) as g:
        assert np.allclose(data[:, 0], g["rixs_E"])
        assert np.allclose(data[:, 1], g["rixs_HERFD"], rtol=1e-6)
        assert np.allclose(data[:, header.index("tfy")], g["rixs_TFY"], rtol=1e-6)


def test_reduce_unknown_label(na2so4_dir, tmp_path):
    p = _cli("reduce", na2so4_dir, "--measurement", "nope", "--out", str(tmp_path))
    assert p.returncode == 2 and LABEL in p.stderr


def test_batch_parallel(na2so4_dir, tmp_path):
    p = _cli("batch", na2so4_dir, "--out", str(tmp_path), "--max-workers", "2",
             "--central-pix", "1305", "--no-i0", "--thresholds", "60", "100", "170", "2000")
    assert p.returncode == 0, p.stderr + p.stdout
    manifest = json.loads((tmp_path / "analysis" / "run_manifest.json").read_text())
    assert (tmp_path / "analysis" / f"{LABEL}.txt").exists()
    assert "1305" in json.dumps(manifest)
