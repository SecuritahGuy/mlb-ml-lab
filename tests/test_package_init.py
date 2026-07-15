"""Tests for top-level package imports."""

from __future__ import annotations

import subprocess
import sys


def test_package_import_does_not_eagerly_import_train_module():
    result = subprocess.run(
        [sys.executable, "-c", "import sys, mlb_ml_lab; print('mlb_ml_lab.models.train' in sys.modules)"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
