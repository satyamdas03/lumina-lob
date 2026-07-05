"""Smoke test for the throughput benchmark script."""

import subprocess
import sys


def test_benchmark_script_runs() -> None:
    """Run the engine benchmark with a tiny workload and verify it reports rates."""
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.engine_benchmark", "--orders", "1000", "--seed", "7"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Python engine:" in result.stdout
    assert "C++ engine:" in result.stdout
    assert "Speedup:" in result.stdout
