"""Simple smoke tests for attack pipelines.

Runs learned CPA on an existing run and runs the MIA simulator.
"""
import subprocess
import sys
from pathlib import Path

RUN = "runs/cifar10_a0.1_s42_c2_r1"


def test_cpa_learned():
    print("Running learned CPA smoke test...")
    subprocess.check_call([sys.executable, "run_cpa.py", RUN, "--type", "learned"])


def test_mia_sim():
    print("Running MIA simulation...")
    subprocess.check_call([sys.executable, "tools/simulate_mia.py"])


if __name__ == '__main__':
    test_cpa_learned()
    test_mia_sim()
