# -*- coding: utf-8 -*-
"""
Expérience 2 (point 4) : perceptron multicouche de référence sur MNIST /
Fashion-MNIST, avec comparaison rigoureuse des activations.

Protocole
- MLP 784 → 256 → 256 → 10 (taille modifiable), Adam, lr 1e-3, batch 128.
- 55 000 images d'entraînement, 5 000 de validation (fixes), test sur 10 000.
- Plusieurs graines par activation ; même graine = même découpage et
  même ordre des batchs pour toutes les activations.
- Activations comparées : relu, prelu, silu, abu (Sütfeld et al. 2018),
  gear (engrenages entraînés), gear_window (entraînés dans une fenêtre
  d'époques), gear_frozen (engrenages aléatoires figés = contrôle).

Lancer (exemples) :
  python exp2_mlp.py --dataset fashion --seeds 5 --epochs 30
  python exp2_mlp.py --dataset mnist --acts relu gear --seeds 3
  python exp2_mlp.py --fake-data --epochs 2 --seeds 1      (test sans téléchargement)
Les résultats (JSON par run + résumé + graphiques) vont dans results/mlp_<dataset>/.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (set_seed, pick_device, save_json, load_runs, count_params,
                    summarize, plot_curves, plot_theta_trajectories)
from gears import (make_activation, split_params, GearSchedule, gear_modules,
                   snapshot_thetas, ACTIVATION_CHOICES)


class MLP(nn.Module):
    def __init__(self, act, hidden=(256, 256), in_dim=784, n_classes=10, cycle=None):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            kw = {"functions": cycle} if cycle and act in ("abu", "gear", "gear_window", "gear_frozen") else {}
            layers.append(make_activation(act, h, **kw))
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


def load_data(args):
    from torchvision import datasets, transforms
    if args.fake_data:
        tr = datasets.FakeData(2000, (1, 28, 28), 10, transforms.ToTensor(), random_offset=0)
        te = datasets.FakeData(500, (1, 28, 28), 10, transforms.ToTensor(), random_offset=10_000)
        to_tensors = lambda ds: (torch.stack([ds[i][0] for i in range(len(ds))]),
                                 torch.tensor([int(ds[i][1]) for i in range(len(ds))]))
        Xtr, Ytr = to_tensors(tr)
        Xte, Yte = to_tensors(te)
        n_val = 200
    else:
        cls = datasets.MNIST if args.dataset == "mnist" else datasets.FashionMNIST
        tr = cls(args.data_dir, train=True, download=True)
        te = cls(args.data_dir, train=False, download=True)
        Xtr, Ytr = tr.data.float().div(255).unsqueeze(1), tr.targets
        Xte, Yte = te.data.float().div(255).unsqueeze(1), te.targets
        n_val = 5000
    mean, std = Xtr.mean(), Xtr.std()
    Xtr, Xte = (Xtr - mean) / std, (Xte - mean) / std
    g = torch.Generator().manual_seed(0)              # découpage validation fixe
    perm = torch.randperm(len(Xtr), generator=g)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    return (Xtr[tr_idx], Ytr[tr_idx]), (Xtr[val_idx], Ytr[val_idx]), (Xte, Yte)


@torch.no_grad()
def evaluate(model, X, Y, device, bs=2000):
    model.eval()
    loss, correct = 0.0, 0
    for i in range(0, len(X), bs):
        x, y = X[i:i + bs].to(device), Y[i:i + bs].to(device)
        out = model(x)
        loss += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
    model.train()
    return loss / len(X), correct / len(X)


def train_one(act, seed, data, args, device):
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = data
    set_seed(seed)
    model = MLP(act, args.hidden, cycle=args.cycle).to(device)
    others, thetas = split_params(model)
    groups = [{"params": others, "lr": args.lr}]
    if thetas:
        groups.append({"params": thetas, "lr": args.lr_theta, "weight_decay": 0.0})
    opt = torch.optim.Adam(groups)
    sched = GearSchedule(model, *args.gear_window) if act == "gear_window" else None

    hist = {k: [] for k in ("train_loss", "val_loss", "val_acc", "epoch_time", "gear_active")}
    theta_hist, track = [], None
    if gear_modules(model):
        n_th = len(snapshot_thetas(model))
        g = torch.Generator().manual_seed(123)
        track = torch.randperm(n_th, generator=g)[:args.track_neurons]   # mêmes neurones suivis pour toutes les graines
        theta_hist.append(snapshot_thetas(model, raw=True)[track].tolist())
        theta_init_mod1 = snapshot_thetas(model)

    gen = torch.Generator().manual_seed(seed)
    for epoch in range(args.epochs):
        on = sched.step(epoch) if sched else bool(thetas)
        t0 = time.time()
        perm = torch.randperm(len(Xtr), generator=gen)
        run_loss = 0.0
        for i in range(0, len(Xtr), args.batch):
            idx = perm[i:i + args.batch]
            x, y = Xtr[idx].to(device), Ytr[idx].to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(idx)
        hist["epoch_time"].append(time.time() - t0)
        hist["train_loss"].append(run_loss / len(Xtr))
        vl, va = evaluate(model, Xva, Yva, device)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        hist["gear_active"].append(on)
        if track is not None:
            theta_hist.append(snapshot_thetas(model, raw=True)[track].tolist())
        print(f"  [{act} s{seed}] époque {epoch + 1:>3}/{args.epochs}  "
              f"train {hist['train_loss'][-1]:.4f}  val {va * 100:.2f}%"
              + ("  (θ actifs)" if on and thetas else ""), flush=True)

    tl, ta = evaluate(model, Xte, Yte, device)
    run = {"activation": act, "seed": seed, "dataset": args.dataset, "test_loss": tl, "test_acc": ta,
           "n_params": count_params(model), "history": hist, "args": vars(args)}
    if track is not None:
        run["theta_history"] = theta_hist
        run["theta_index"] = track.tolist()
        run["cycle"] = gear_modules(model)[0].names
        dom = [d for g_ in gear_modules(model) for d in g_.dominant_function()]
        run["dominant_counts"] = {n: dom.count(n) for n in run["cycle"]}
        moved = (snapshot_thetas(model) - theta_init_mod1).abs()
        moved = torch.minimum(moved, 1 - moved)          # distance sur le cercle
        run["theta_moved_mean"] = moved.mean().item()
    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "fashion"], default="fashion")
    p.add_argument("--acts", nargs="+", default=["relu", "prelu", "silu", "abu", "gear", "gear_window", "gear_frozen"],
                   choices=ACTIVATION_CHOICES)
    p.add_argument("--cycle", nargs="+", default=["relu", "silu", "tanh", "identity"],
                   help="cycle de fonctions des engrenages (et fonctions d'ABU)")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-theta", type=float, default=1e-3)
    p.add_argument("--gear-window", type=int, nargs=2, default=[5, 15],
                   help="époques (début, fin, à partir de 0) où θ est entraîné pour gear_window")
    p.add_argument("--track-neurons", type=int, default=12)
    p.add_argument("--target", type=float, default=None,
                   help="précision de validation cible pour mesurer la vitesse de convergence (ex. 0.88)")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--fake-data", action="store_true", help="données aléatoires, pour tester le script")
    args = p.parse_args()
    args.out = args.out or f"results/mlp_{'fake' if args.fake_data else args.dataset}"
    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    print(f"Appareil : {device}")

    data = load_data(args)
    for act in args.acts:
        for seed in range(args.seeds):
            path = os.path.join(args.out, f"run_{act}_s{seed}.json")
            if os.path.exists(path):
                print(f"  déjà fait : {path}")
                continue
            save_json(train_one(act, seed, data, args, device), path)

    runs = [r for r in load_runs(args.out) if r["activation"] in args.acts]
    target = args.target if args.target is not None else (0.88 if args.dataset == "fashion" else 0.98)
    table = summarize(runs, "test_acc", "val_acc", target=target)
    print("\n" + table)
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
        for r in runs:
            if "dominant_counts" in r:
                f.write(f"\n{r['activation']} s{r['seed']} : fonctions dominantes {r['dominant_counts']}, "
                        f"déplacement moyen de θ {r['theta_moved_mean']:.3f} tour")
    plot_curves(runs, ["train_loss", "val_loss", "val_acc"], os.path.join(args.out, "curves.png"),
                title=f"MLP — {args.dataset}")
    plot_theta_trajectories(runs, os.path.join(args.out, "theta.png"), title=f"MLP — {args.dataset}")
    print(f"\nRésultats : {args.out}/summary.txt, curves.png, theta_*.png")


if __name__ == "__main__":
    main()
