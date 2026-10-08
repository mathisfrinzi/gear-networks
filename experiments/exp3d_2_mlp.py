# -*- coding: utf-8 -*-
"""
Expérience 3D-2 : engrenages sphériques dans un MLP sur Fashion-MNIST / MNIST.

Même protocole que exp2_mlp.py (MLP 784 → 256 → 256 → 10, Adam lr 1e-3,
batch 128, 55 000 / 5 000 / 10 000 images), mais les variantes sont
comparées à nombre de fonctions égal (les 6 fonctions de la sphère :
relu, silu, tanh, identity, elu, sin) :

  relu           référence fixe
  gear           cercle à 4 fonctions (la version de exp2, pour mémoire)
  gear6          cercle, 6 fonctions          1 paramètre par neurone
  sphere         sphère, barycentrique        2 degrés de liberté par neurone
  sphere_kernel  sphère, à noyau (lisse)      2 degrés de liberté par neurone
  abu6           ABU, 6 fonctions             6 paramètres par neurone
  sphere_frozen  sphère, directions aléatoires figées (contrôle)

Lancer :
  python experiments/exp3d_2_mlp.py --seeds 3 --epochs 20           (~20 min sur CPU)
  python experiments/exp3d_2_mlp.py --fake-data --epochs 1 --seeds 1   (test sans téléchargement)
Résultats : results/sphere_mlp_<dataset>/ (un JSON par run, summary.txt, figures).
Les runs déjà faits sont ignorés : relancer reprend où on en était.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # racine du dépôt
sys.path.insert(0, HERE)                    # pour réutiliser exp2_mlp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gears.utils import (repo_path, set_seed, pick_device, save_json, load_runs, count_params,
                         summarize, plot_curves, plot_sphere_paths)
from gears import (make_activation, split_params, position_modules, SphereGearActivation,
                   GearActivation, snapshot_directions, DEFAULT_SPHERE)
from exp2_mlp import load_data, evaluate

ACTS = ("relu", "gear", "gear6", "sphere", "sphere_kernel", "abu6", "sphere_frozen",
        "gear_relu_init", "sphere_relu_init")


class MLP(nn.Module):
    def __init__(self, act, hidden=(256, 256), in_dim=784, n_classes=10):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), make_activation(act, h)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


def structure(model):
    """Nombre de neurones sur 1, 2 ou 3+ fonctions actives (poids > 0,02), par couche."""
    out = []
    for m in position_modules(model):
        if isinstance(m, SphereGearActivation):
            k = m.n_active(0.02)
        else:
            p = torch.remainder(m.theta.detach(), 1.0)[:, None]
            d = ((p - m.starts[None] + 0.5) % 1.0 - 0.5).abs().min(1).values
            k = torch.where(d < 0.02, 1, 2)
        dom = m.dominant_function()
        out.append({"1 fonction": int((k == 1).sum()), "2 fonctions": int((k == 2).sum()),
                    "3+ fonctions": int((k >= 3).sum()),
                    "dominant": {n: dom.count(n) for n in m.names}})
    return out


def train_one(act, seed, data, args, device):
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = data
    set_seed(seed)
    model = MLP(act, args.hidden).to(device)
    others, thetas = split_params(model)
    groups = [{"params": others, "lr": args.lr}]
    if thetas:
        groups.append({"params": thetas, "lr": args.lr_theta, "weight_decay": 0.0})
    opt = torch.optim.Adam(groups)

    hist = {k: [] for k in ("train_loss", "val_loss", "val_acc", "epoch_time")}
    track = None
    sphere = any(isinstance(m, SphereGearActivation) for m in position_modules(model))
    if sphere:
        n = len(snapshot_directions(model))
        track = torch.randperm(n, generator=torch.Generator().manual_seed(123))[:args.track_neurons]
        paths = [snapshot_directions(model)[track]]
        u0 = snapshot_directions(model)

    gen = torch.Generator().manual_seed(seed)
    for epoch in range(args.epochs):
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
        if track is not None:
            paths.append(snapshot_directions(model)[track])
        print(f"  [{act} s{seed}] époque {epoch + 1:>3}/{args.epochs}  "
              f"train {hist['train_loss'][-1]:.4f}  val {va * 100:.2f}%", flush=True)

    tl, ta = evaluate(model, Xte, Yte, device)
    run = {"activation": act, "seed": seed, "dataset": args.dataset, "test_loss": tl, "test_acc": ta,
           "n_params": count_params(model), "history": hist, "args": vars(args)}
    if position_modules(model):
        run["structure"] = structure(model)
    if track is not None:
        run["sphere_path"] = torch.stack(paths).tolist()          # (époques + 1, n_suivis, 3)
        cos = (snapshot_directions(model) * u0).sum(1).clamp(-1, 1)
        run["angle_moved_deg"] = torch.rad2deg(torch.acos(cos)).mean().item()
    return run


def write_summary(runs, args):
    target = args.target if args.target is not None else (0.88 if args.dataset == "fashion" else 0.98)
    lines = [summarize(runs, "test_acc", "val_acc", target=target)]
    lines.append("\nPosition finale des neurones (moyenne sur les graines, toutes couches) :")
    for act in ACTS:
        rs = [r for r in runs if r["activation"] == act and "structure" in r]
        if not rs:
            continue
        agg = {}
        for r in rs:
            for L in r["structure"]:
                for k in ("1 fonction", "2 fonctions", "3+ fonctions"):
                    agg[k] = agg.get(k, 0) + L[k]
        tot = max(sum(agg.values()), 1)
        extra = ""
        if "angle_moved_deg" in rs[0]:
            extra = f"   déplacement moyen {np.mean([r['angle_moved_deg'] for r in rs]):.1f}°"
        lines.append(f"  {act:<14} " + "   ".join(f"{k} {100 * c / tot:>3.0f} %" for k, c in agg.items()) + extra)
    lines.append("\nFonction dominante finale, par couche (moyenne sur les graines) :")
    for act in ACTS:
        rs = [r for r in runs if r["activation"] == act and "structure" in r]
        if not rs:
            continue
        for li in range(len(rs[0]["structure"])):
            agg = {}
            for r in rs:
                for k, c in r["structure"][li]["dominant"].items():
                    agg[k] = agg.get(k, 0) + c
            tot = max(sum(agg.values()), 1)
            lines.append(f"  {act:<14} couche {li + 1} : "
                         + ", ".join(f"{k} {100 * c / tot:.0f} %" for k, c in agg.items()))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "fashion"], default="fashion")
    p.add_argument("--acts", nargs="+", default=list(ACTS), choices=ACTS)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-theta", type=float, default=1e-3)
    p.add_argument("--track-neurons", type=int, default=12)
    p.add_argument("--target", type=float, default=None)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--fake-data", action="store_true")
    args = p.parse_args()
    args.out = args.out or f"results/sphere_mlp_{'fake' if args.fake_data else args.dataset}"
    args.out, args.data_dir = repo_path(args.out), repo_path(args.data_dir)
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
    text = write_summary(runs, args)
    print("\n" + text)
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    plot_curves(runs, ["train_loss", "val_loss", "val_acc"], os.path.join(args.out, "curves.png"),
                title=f"MLP {args.dataset} — cercle, sphère, ABU à 6 fonctions")
    for act in ("sphere", "sphere_kernel"):
        r0 = next((r for r in runs if r["activation"] == act and r["seed"] == 0 and "sphere_path" in r), None)
        if r0:
            P = np.array(r0["sphere_path"])
            g = SphereGearActivation(1, DEFAULT_SPHERE)
            plot_sphere_paths([P[:, i] for i in range(P.shape[1])], g.positions.numpy(), g.names,
                              os.path.join(args.out, f"{act}_paths.png"),
                              title=f"MLP {args.dataset} — {act} : {P.shape[1]} neurones suivis (graine 0)")
    print(f"\nRésultats : {args.out}/summary.txt, curves.png, sphere_paths.png")


if __name__ == "__main__":
    main()
