# -*- coding: utf-8 -*-
"""
Expérience 1 : problèmes jouets (ET, XOR, somme bruitée, vague), avec une vraie
rétropropagation, et suivi de la convergence des engrenages θ pour plusieurs
initialisations des poids.

Pour chaque tâche et chaque graine (= une initialisation des poids) :
  - fixed       : θ figés sur leur position initiale (pas d'engrenage)
  - gear        : θ entraînés tout le long
  - gear_window : θ entraînés seulement dans une fenêtre d'itérations
                  (θ ouverts pendant une plage d'itérations seulement)

Les θ initiaux sont IDENTIQUES d'une graine à l'autre (neurone i placé sur la
fonction i du cycle) : seules les initialisations des poids changent. On voit
donc si les engrenages convergent vers la même configuration quels que soient
les poids de départ.

Lancer :  python experiments/exp1_toy.py            (≈ 1 à 3 min sur CPU)
          python experiments/exp1_toy.py --quick    (test rapide)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # racine du dépôt

import numpy as np
import torch
import torch.nn as nn

from gears.utils import set_seed, save_json, repo_path
from gears import GearActivation, GearSchedule, split_params, snapshot_thetas


# ---------------------------------------------------------------------------
# Tâches
# ---------------------------------------------------------------------------
def make_task(name, rng):
    if name == "and":       # sorties [-1,-1,-1,1] : un ET logique
        X = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)
        Y = torch.tensor([[-1], [-1], [-1], [1]], dtype=torch.float32)
        return X, Y, True
    if name == "xor":
        X = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)
        Y = torch.tensor([[-1], [1], [1], [-1]], dtype=torch.float32)
        return X, Y, True
    if name == "sum_noisy":  # y = x1 + x2 + bruit
        X = torch.tensor([[x / 100, y / 100] for x in range(100) for y in range(10)], dtype=torch.float32)
        Y = (X.sum(1, keepdim=True) + torch.from_numpy(rng.normal(0, 0.1, (len(X), 1))).float())
        return X, Y, False
    if name == "wave":       # non linéaire : y = sin(2π x1) · x2 + bruit
        X = torch.from_numpy(rng.uniform(-1, 1, (1000, 2))).float()
        Y = torch.sin(2 * np.pi * X[:, :1]) * X[:, 1:] + torch.from_numpy(rng.normal(0, 0.05, (1000, 1))).float()
        return X, Y, False
    raise ValueError(name)


class ToyNet(nn.Module):
    def __init__(self, hidden, cycle, tanh_out):
        super().__init__()
        self.l1 = nn.Linear(2, hidden)
        self.act = GearActivation(hidden, cycle, init=0.0)
        with torch.no_grad():   # neurone i démarre exactement sur la fonction i % n
            self.act.theta.copy_(self.act.starts[torch.arange(hidden) % len(cycle)])
        self.l2 = nn.Linear(hidden, 1)
        self.tanh_out = tanh_out

    def forward(self, x):
        y = self.l2(self.act(self.l1(x)))
        return torch.tanh(y) if self.tanh_out else y


def run(task, variant, seed, args):
    rng = np.random.default_rng(1234)          # mêmes données pour toutes les graines
    X, Y, tanh_out = make_task(task, rng)
    set_seed(seed)                             # la graine ne change que les poids
    net = ToyNet(args.hidden, args.cycle, tanh_out)
    if variant == "fixed":
        net.act.theta.requires_grad_(False)
    others, thetas = split_params(net)
    groups = [{"params": others, "lr": args.lr}]
    if thetas:
        groups.append({"params": thetas, "lr": args.lr_theta})
    opt = torch.optim.Adam(groups)
    sched = None
    if variant == "gear_window":
        sched = GearSchedule(net, *args.window)

    losses, theta_hist = [], []
    for it in range(args.iters):
        if sched:
            sched.step(it)
        opt.zero_grad()
        loss = ((net(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if it % args.log_every == 0:
            theta_hist.append(snapshot_thetas(net, raw=True).tolist())
    theta_hist.append(snapshot_thetas(net, raw=True).tolist())
    return {"task": task, "variant": variant, "seed": seed, "loss": losses,
            "theta_history": theta_hist, "dominant": net.act.dominant_function()}


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------
def plot_losses(results, task, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for variant in ("fixed", "gear", "gear_window"):
        arr = np.array([r["loss"] for r in results if r["task"] == task and r["variant"] == variant])
        if len(arr) == 0:
            continue
        x = np.arange(arr.shape[1])
        med = np.median(arr, 0)
        ax.plot(x, med, label=f"{variant} (médiane)")
        ax.fill_between(x, arr.min(0), arr.max(0), alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("itération")
    ax.set_ylabel("MSE")
    ax.set_title(f"{task} : perte d'entraînement ({arr.shape[0]} initialisations des poids)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_thetas(results, task, variant, cycle, log_every, path, window=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = [r for r in results if r["task"] == task and r["variant"] == variant]
    H = len(rs[0]["theta_history"][0])
    cols = 4
    rows = int(np.ceil(H / cols))
    n = len(cycle)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows), squeeze=False)
    for i in range(H):
        ax = axes[i // cols][i % cols]
        lo, hi = np.inf, -np.inf
        for r in rs:
            th = np.array(r["theta_history"])[:, i]
            x = np.arange(len(th)) * log_every
            ax.plot(x, th, lw=1.2, label=f"poids n°{r['seed']}")
            lo, hi = min(lo, th.min()), max(hi, th.max())
        for k in range(int(np.floor(lo * n)) - 1, int(np.ceil(hi * n)) + 2):
            y = k / n
            if lo - 0.1 <= y <= hi + 0.1:
                ax.axhline(y, color="gray", lw=0.6, ls="--")
                ax.text(x[-1], y, " " + cycle[k % n], fontsize=7, color="dimgray", va="center")
        if window:
            ax.axvspan(*window, color="orange", alpha=0.12)
        ax.set_title(f"neurone {i}", fontsize=9)
        ax.tick_params(labelsize=7)
    for j in range(H, rows * cols):
        axes[j // cols][j % cols].axis("off")
    axes[0][0].legend(fontsize=6)
    fig.suptitle(f"{task} — {variant} : θ au fil des itérations (lignes pointillées = fonctions du cycle)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["and", "xor", "sum_noisy", "wave"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--cycle", nargs="+", default=["relu", "tanh", "sigmoid", "identity"])
    p.add_argument("--iters", type=int, default=3500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lr-theta", type=float, default=1e-2)
    p.add_argument("--window", type=int, nargs=2, default=[500, 1500],
                   help="itérations pendant lesquelles θ est entraîné (gear_window)")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out", default="results/toy")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.iters, args.seeds, args.window = 300, 2, [50, 150]
    args.out = repo_path(args.out)
    os.makedirs(args.out, exist_ok=True)

    results = []
    for task in args.tasks:
        for variant in ("fixed", "gear", "gear_window"):
            for seed in range(args.seeds):
                results.append(run(task, variant, seed, args))
        print(f"\n== {task} : MSE finale (médiane [min, max] sur {args.seeds} initialisations)")
        for variant in ("fixed", "gear", "gear_window"):
            fin = [r["loss"][-1] for r in results if r["task"] == task and r["variant"] == variant]
            print(f"   {variant:<12} {np.median(fin):.4g}  [{min(fin):.3g}, {max(fin):.3g}]")
        for r in results:
            if r["task"] == task and r["variant"] == "gear":
                print(f"   gear, poids n°{r['seed']} -> fonctions dominantes : {r['dominant']}")
        plot_losses(results, task, os.path.join(args.out, f"{task}_loss.png"))
        plot_thetas(results, task, "gear", args.cycle, args.log_every,
                    os.path.join(args.out, f"{task}_theta_gear.png"))
        plot_thetas(results, task, "gear_window", args.cycle, args.log_every,
                    os.path.join(args.out, f"{task}_theta_window.png"), window=args.window)
    save_json({"args": vars(args), "results": results}, os.path.join(args.out, "toy_results.json"))
    print(f"\nGraphiques et résultats dans {args.out}/")


if __name__ == "__main__":
    main()
