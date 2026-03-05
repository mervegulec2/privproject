from src.data.cifar import Cifar10Config, load_cifar10
from src.data.split import DirichletSplitConfig, dirichlet_split_indices, class_hist, save_split

LABELS = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def main():
    train_ds, _ = load_cifar10(Cifar10Config(root="data"))

    for seed in [42, 123, 999]:
        cfg = DirichletSplitConfig(num_clients=15, alpha=0.3, seed=seed, min_size_per_client=500)
        client_map = dirichlet_split_indices(train_ds, cfg)

        print(f"\n=== Split seed={seed}, alpha={cfg.alpha}, clients={cfg.num_clients} ===")
        total = 0
        for cid in range(cfg.num_clients):
            idx = client_map[cid]
            total += len(idx)
            hist = class_hist(train_ds, idx, num_classes=10)
            present = int((hist > 0).sum())
            top3 = sorted([(LABELS[i], int(hist[i])) for i in range(10)], key=lambda x: -x[1])[:3]
            print(f"Client {cid:02d}: n={len(idx):4d}, classes_present={present:2d}, top3={top3}")

        print("Total assigned:", total)

        out_path = f"outputs/splits/cifar10_dirichlet_a0.3_seed{seed}.npy"
        save_split(client_map, out_path)
        print("Saved split:", out_path)

if __name__ == "__main__":
    main()