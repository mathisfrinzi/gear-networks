# -*- coding: utf-8 -*-
"""
Expérience 4 : partir de ReLU et ouvrir les engrenages par intervalles.

Idée : au lieu d'entraîner les θ en continu depuis une position aléatoire,
on part d'un réseau ReLU (tous les θ sur ReLU) et on alterne :
  θ figés (le réseau converge comme un réseau ReLU)
  → θ entraînés (chaque neurone peut s'écarter de ReLU)
  → θ refigés (les poids se réadaptent) → ...

Variantes (réseau 2 → H → 1, cycle relu → silu → tanh → identity) :
  relu              référence fixe
  gear              θ au milieu des segments, entraînés en continu
  gear_relu_init    θ sur ReLU au départ, entraînés en continu
  gear_alt          θ sur ReLU, alternance fixe : OFF blocs figés, ON blocs entraînés
  gear_alt_plateau  θ sur ReLU, ouverture quand la perte d'entraînement se
                    stabilise, fermeture quand les θ ne bougent plus, arrêt
                    quand une ouverture ne fait plus bouger les θ

Unité de temps : un bloc de 50 itérations (batch complet) ; 60 blocs, comme
les 60 époques du ResNet, avec le même calendrier par défaut (10 figés / 5
entraînés). Le calendrier déclenché n'utilise que la perte d'entraînement.

Tâches : wave (y = sin(2π·x1)·x2) et ripple (y = sin(3π‖x‖)), où
l'activation change nettement le résultat (voir exp3d_1_toy.py).

Lancer :  python experiments/exp4_alternance.py            (~10 min sur CPU)
          python experiments/exp4_alternance.py --quick    (test rapide)
Résultats : results/alternance/ (summary.txt, *_loss.png, JSON)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # racine du dépôt
sys.path.insert(0, HERE)                    # pour réutiliser les tâches de exp3d_1_toy

import numpy as np
import torch
import torch.nn as nn

from gears.utils import set_seed, save_json, repo_path
from gears import make_activation, make_schedule, split_params, DEFAULT_CYCLE
from exp3d_1_toy import make_task

VARIANTS = ("relu", "gear", "gear_relu_init", "gear_alt", "gear_alt_plateau")


class ToyNet(nn.Module):
    def __init__(self, variant, hidden):
        super().__init__()
        self.l1 = nn.Linear(2, hidden)          # créées avant l'activation : mêmes poids
        self.l2 = nn.Linear(hidden, 1)          # initiaux pour toutes les variantes
        kind = "gear_relu_init" if variant in ("gear_alt", "gear_alt_plateau") else variant
        self.act = make_activation(kind, hidden, functions=DEFAULT_CYCLE)
        if variant == "gear":                   # comme exp3d_1 : milieu du segment i % 4
            with torch.no_grad():
                self.act.theta.copy_((torch.arange(hidden) % 4 + 0.5) / 4)

    def forward(self, x):
        return self.l2(self.act(self.l1(x)))


def run(task, variant, seed, args):
    (Xtr, Ytr), (Xva, Yva), _ = make_task(task)
    set_seed(seed)
    net = ToyNet(variant, args.hidden)
    others, thetas = split_params(net)
    groups = [{"params": others, "lr": args.lr}]
    if thetas:
        groups.append({"params": thetas, "lr": args.lr_theta})
    opt = torch.optim.Adam(groups)
    sched = make_schedule(variant, net, alt=args.alt,
                          plateau=dict(patience=int(args.plateau[0]), min_delta=args.plateau[1],
                                       theta_tol=args.plateau[2], stop_tol=args.plateau[3], min_on=2))
    theta0 = net.act.theta.detach().clone() if thetas else None

    block_losses, active = [], []
    for b in range(args.blocks):
        last = block_losses[-1] if block_losses else None
        on = sched.step(b, last) if sched else bool(thetas)
        active.append(bool(on and thetas))
        tot = 0.0
        for _ in range(args.block_iters):
            opt.zero_grad()
            loss = ((net(Xtr) - Ytr) ** 2).mean()
            loss.backward()
            opt.step()
            tot += loss.item()
        block_losses.append(tot / args.block_iters)
    with torch.no_grad():
        val = ((net(Xva) - Yva) ** 2).mean().item()
        train = ((net(Xtr) - Ytr) ** 2).mean().item()
    out = {"task": task, "variant": variant, "seed": seed, "val_mse": val, "train_mse": train,
           "block_loss": block_losses, "active": active}
    if thetas:
        dom = net.act.dominant_function()
        out["dominant"] = {n: dom.count(n) for n in net.act.names}
        d = (net.act.theta.detach() - theta0).abs()
        out["theta_moved"] = d.mean().item()
    if hasattr(sched, "events"):
        out["events"], out["cycles"] = sched.events, sched.cycles
    return out


def plot_task(results, task, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for v in VARIANTS:
        arr = np.array([r["block_loss"] for r in results if r["task"] == task and r["variant"] == v])
        x = (np.arange(arr.shape[1]) + 1) * args.block_iters
        ax.plot(x, np.median(arr, 0), label=v, lw=1.5, ls="--" if v == "relu" else "-")
    r0 = next(r for r in results if r["task"] == task and r["variant"] == "gear_alt")
    for b, on in enumerate(r0["active"]):
        if on:
            ax.axvspan(b * args.block_iters, (b + 1) * args.block_iters, color="orange", alpha=0.08, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("itération (orange : θ ouverts pour gear_alt)")
    ax.set_ylabel("MSE d'entraînement (médiane)")
    ax.set_title(task)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary(results, tasks, n_seeds):
    L = []
    for task in tasks:
        L.append(f"\n== {task} : MSE de validation ×1e3, médiane [min, max] sur {n_seeds} graines ==")
        by = {v: {r["seed"]: r for r in results if r["task"] == task and r["variant"] == v} for v in VARIANTS}
        for v in VARIANTS:
            vals = np.array([r["val_mse"] for r in by[v].values()]) * 1e3
            line = f"  {v:<18} {np.median(vals):8.3f}  [{vals.min():.3f}, {vals.max():.3f}]"
            if v != "relu":
                wins = sum(by[v][s]["val_mse"] < by["relu"][s]["val_mse"] for s in by[v])
                line += f"   < relu sur {wins}/{len(by[v])}"
            if v in ("gear_alt", "gear_alt_plateau"):
                wins = sum(by[v][s]["val_mse"] < by["gear_relu_init"][s]["val_mse"] for s in by[v])
                line += f"   < gear_relu_init sur {wins}/{len(by[v])}"
                frac = np.mean([np.mean(r["active"]) for r in by[v].values()])
                line += f"   θ ouverts {100 * frac:.0f} % du temps"
            L.append(line)
        rs = list(by["gear_alt_plateau"].values())
        n_open = [sum(1 for _, p in r["events"] if p == "on") for r in rs]
        first = [next((t for t, p in r["events"] if p == "on"), None) for r in rs]
        done = sum(1 for r in rs if any(p == "done" for _, p in r["events"]))
        L.append(f"  gear_alt_plateau : ouvertures par run {n_open}, première ouverture au bloc {first}, "
                 f"arrêt définitif atteint dans {done}/{len(rs)} runs")
        for v in VARIANTS[1:]:
            agg = {}
            for r in by[v].values():
                for k, c in r["dominant"].items():
                    agg[k] = agg.get(k, 0) + c
            tot = sum(agg.values())
            moved = np.mean([r["theta_moved"] for r in by[v].values()])
            L.append(f"  {v:<18} fonction dominante : "
                     + ", ".join(f"{k} {100 * c / tot:.0f} %" for k, c in agg.items())
                     + f"   déplacement moyen de θ {moved:.3f} tour")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["wave", "ripple"])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--blocks", type=int, default=60)
    p.add_argument("--block-iters", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lr-theta", type=float, default=1e-2)
    p.add_argument("--alt", type=int, nargs=2, default=[10, 5], metavar=("OFF", "ON"))
    p.add_argument("--plateau", type=float, nargs=4, default=[2, 0.05, 1e-3, 5e-3],
                   metavar=("PATIENCE", "MIN_DELTA", "THETA_TOL", "STOP_TOL"))
    p.add_argument("--out", default="results/alternance")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.seeds, args.blocks, args.block_iters = 2, 12, 20
        args.alt = [2, 1]
    args.out = repo_path(args.out)
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    results = []
    for task in args.tasks:
        for v in VARIANTS:
            for seed in range(args.seeds):
                results.append(run(task, v, seed, args))
        print(f"  {task} terminé", flush=True)
        plot_task(results, task, args, os.path.join(args.out, f"{task}_loss.png"))
    text = summary(results, args.tasks, args.seeds)
    print(text)
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Réseau 2 → {args.hidden} → 1, {args.blocks} blocs de {args.block_iters} itérations, "
                f"alternance {args.alt[0]}/{args.alt[1]}, plateau {args.plateau}\n" + text + "\n")
    save_json({"args": vars(args), "results": results}, os.path.join(args.out, "alternance_results.json"))
    print(f"\nRésultats dans {args.out}/")


if __name__ == "__main__":
    main()
