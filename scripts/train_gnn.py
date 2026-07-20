"""Train the final GNN on all subjects and save it."""

import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import glob
import numpy as np
import torch
from src.gnn_train import build_subject, train_final_gnn, LABELS


def main():
    data_dir = Path("DATA")
    plv_dir = data_dir / "PLV"

    subjects = []
    for group in ["AD", "HC"]:
        label = LABELS[group]
        npy_files = sorted((plv_dir / group).glob("*.npy"))
        for npy_path in npy_files:
            # Extract subject number, e.g. HC_Sub37_PLV.npy -> 37
            m = re.search(r"Sub(\d+)", npy_path.name)
            sub_num = int(m.group(1))

            # Find the matching .set file for node features
            matches = list((data_dir / group).glob(f"sub-*{sub_num:02d}*.set"))
            if not matches:
                print(f"  skip {npy_path.name}: no matching .set")
                continue
            set_path = matches[0]

            rec = build_subject(npy_path, set_path, label)
            subjects.append(rec)
            print(f"{group} sub-{sub_num:03d}: X={rec['X'].shape}, A={rec['A'].shape}")

    print(f"\nBuilt {len(subjects)} subjects. Training final GNN…")
    model, mu, sd = train_final_gnn(subjects, n_classes=2)

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "mu": mu,
        "sd": sd,
        "n_features": subjects[0]["X"].shape[-1],
    }, out_dir / "gnn.pt")
    print(f"Saved GNN model to {out_dir / 'gnn.pt'}")


if __name__ == "__main__":
    main()