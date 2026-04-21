import os
import pickle
import argparse
import torch
from src.security.manager import create_security_manager
from src.security.attacks.reconstruction import PrototypeReconstructionAttack
# from src.security.attacks.membership import MembershipInferenceAttack  # Future use

def main():
    parser = argparse.ArgumentParser(description="Security Evaluation Toolkit: Attack Saved Snapshots")
    parser.add_argument("--snapshot", type=str, required=True, help="Path to the saved snapshot .pkl file")
    parser.add_argument("--attack", type=str, choices=["reconstruction", "inference"], default="reconstruction")
    parser.add_argument("--save_dir", type=str, default="outputs/security/eval_results")
    args = parser.parse_args()

    # 1. Load the Adversary Knowledge Snapshot
    if not os.path.exists(args.snapshot):
        print(f"Error: Snapshot not found at {args.snapshot}")
        return

    with open(args.snapshot, "rb") as f:
        snapshot = pickle.load(f)
    print(f"\n>>> Loaded Snapshot for Round {snapshot['round']} <<<")

    # 2. Select the Attack
    attack_module = None
    if args.attack == "reconstruction":
        attack_module = PrototypeReconstructionAttack(
            save_dir=os.path.join(args.save_dir, "visuals"),
            iterations=500
        )
    elif args.attack == "inference":
        # Placeholder for future implementation
        print("MIA Attack Module not implemented yet.")
        return

    # 3. Execute Attack using only Snapshot Data (Honest-but-Curious)
    print(f"Executing {args.attack} attack...")
    results = attack_module.execute(
        model_state=snapshot["model_state"], 
        shared_data={"clients": snapshot["clients"]}
    )

    # 4. Process Results
    print("\n>>> Attack Results <<<")
    # For reconstruction, we might report image paths or final loss
    for client, stats in results.items():
        print(f"{client}: {stats}")

    print(f"\nResults saved to {args.save_dir}")

if __name__ == "__main__":
    main()
