"""Run a small MIA simulation demo using `membership_scorer_mean_var`.

Prints ROC-AUC for the simulated experiment.
"""

import os
import sys
# Ensure project root is importable when running as a script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.attacks.mia.mia_baselines import simulate_mia_demo


def main():
    res = simulate_mia_demo(n_clients=2, dim=16, n_samples_per_client=30)
    print("MIA simulation result:", res)


if __name__ == "__main__":
    main()
