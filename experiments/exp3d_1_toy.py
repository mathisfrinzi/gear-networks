# -*- coding: utf-8 -*-
"""
Expérience 3D-1 : engrenages sphériques sur des tâches jouets.

Question : à nombre de fonctions égal, donner 2 degrés de liberté à chaque
neurone (sphère) plutôt qu'un seul (cercle) change-t-il quelque chose ?

Variantes comparées (même réseau 2 → H → 1, mêmes poids initiaux pour une
graine donnée, seule l'activation change) :
  relu           référence fixe
  gear           cercle, 4 fonctions (relu → silu → tanh → identity)
  gear6          cercle, les 6 fonctions de la sphère (+ elu, sin)
  sphere         sphère (octaèdre), barycentrique : au plus 3 fonctions actives
  sphere_kernel  sphère, poids à noyau (lisse, toutes les fonctions actives)
  sphere_frozen  sphère, directions figées à leur position initiale (contrôle)

Positions initiales identiques pour toutes les graines : neurone i au milieu
du segment i (cercle) ou au centre de la face i de l'octaèdre (sphère). Au
départ, chaque neurone mélange donc 2 (cercle) ou 3 (sphère) fonctions à
parts égales ; seuls les poids changent d'une graine à l'autre.

Tâches de régression (bruit σ = 0,05, 1 000 points d'entraînement et 1 000 de
validation) et XOR :
  xor     XOR à 4 points (sortie tanh)
  wave    y = sin(2π·x1)·x2
  bump    y = exp(−4‖x‖²)
  ripple  y = sin(3π‖x‖)              (périodique : sin devrait aider)

Lancer :  python experiments/exp3d_1_toy.py           (~5 min sur CPU)
          python experiments/exp3d_1_toy.py --quick   (test rapide)
Résultats : results/sphere_toy/ (summary.txt, *_loss.png, *_sphere.png, JSON)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # racine du dépôt

import numpy as np
import torch
import torch.nn as nn

from gears.utils import set_seed, save_json, repo_path, plot_sphere_paths
from gears import (GearActivation, SphereGearActivation, DEFAULT_CYCLE, DEFAULT_SPHERE,
                   split_params)

VARIANTS = ("relu", "gear", "gear6", "sphere", "sphere_kernel", "sphere_frozen")
EPS_NODE = 0.02     # cercle : « sur un nœud » si à moins de 0,02 tour
TOL_ACTIVE = 0.02   # sphère : une fonction compte comme active si son poids > 0,02


# ---------------------------------------------------------------------------
# Tâches
# ---------------------------------------------------------------------------
def make_task(name, n=1000, noise=0.05):
    rng = np.random.default_rng(1234)                 # mêmes données pour toutes les graines
    if name == "xor":
        X = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)
        Y = torch.tensor([[-1], [1], [1], [-1]], dtype=torch.float32)
        return (X, Y), (X, Y), True
    X = torch.from_numpy(rng.uniform(-1, 1, (2 * n, 2))).float()
    r = X.norm(dim=1, keepdim=True)
    if name == "wave":
        Y = torch.sin(2 * np.pi * X[:, :1]) * X[:, 1:]
    elif name == "bump":
        Y = torch.exp(-4 * r ** 2)
    elif name == "ripple":
        Y = torch.sin(3 * np.pi * r)
    else:
        raise ValueError(name)
    Y = Y + torch.from_numpy(rng.normal(0, noise, (2 * n, 1))).float()
    return (X[:n], Y[:n]), (X[n:], Y[n:]), False


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------
def make_act(variant, hidden):
    if variant == "relu":
        return nn.ReLU()
    if variant in ("gear", "gear6"):
        fns = DEFAULT_CYCLE if variant == "gear" else DEFAULT_SPHERE
        g = GearActivation(hidden, fns, init=0.0)
        n = len(fns)
        with torch.no_grad():                         # neurone i au milieu du segment i % n
            g.theta.copy_((torch.arange(hidden) % n + 0.5) / n)
        return g
    mode = "kernel" if variant == "sphere_kernel" else "barycentric"
    g = SphereGearActivation(hidden, DEFAULT_SPHERE, mode=mode)
    with torch.no_grad():                             # neurone i au centre de la face i % 8
        centers = g.positions[g.faces].mean(1)
        g.theta.copy_(centers[torch.arange(hidden) % len(centers)])
        g.renormalize()
    if variant == "sphere_frozen":
        g.theta.requires_grad_(False)
    return g


class ToyNet(nn.Module):
    def __init__(self, variant, hidden, tanh_out):
        super().__init__()
        # couches linéaires créées avant l'activation : pour une graine donnée,
        # les poids initiaux sont identiques pour toutes les variantes
        self.l1 = nn.Linear(2, hidden)
        self.l2 = nn.Linear(hidden, 1)
        self.act = make_act(variant, hidden)
        self.tanh_out = tanh_out

    def forward(self, x):
        y = self.l2(self.act(self.l1(x)))
        return torch.tanh(y) if self.tanh_out else y


# ---------------------------------------------------------------------------
# Un run
# ---------------------------------------------------------------------------
def structure(act):
    """Où sont les neurones à la fin : sur une fonction pure, entre deux, entre trois."""
    if isinstance(act, SphereGearActivation):
        k = act.n_active(TOL_ACTIVE)
        return {"1 fonction": int((k == 1).sum()), "2 fonctions": int((k == 2).sum()),
                "3+ fonctions": int((k >= 3).sum())}
    if isinstance(act, GearActivation):
        p = torch.remainder(act.theta.detach(), 1.0)[:, None]
        d = ((p - act.starts[None] + 0.5) % 1.0 - 0.5).abs().min(1).values
        return {"1 fonction": int((d < EPS_NODE).sum()), "2 fonctions": int((d >= EPS_NODE).sum()),
                "3+ fonctions": 0}
    return {}


def run(task, variant, seed, args):
    (Xtr, Ytr), (Xva, Yva), tanh_out = make_task(task)
    set_seed(seed)
    net = ToyNet(variant, args.hidden, tanh_out)
    others, thetas = split_params(net)
    groups = [{"params": others, "lr": args.lr}]
    if thetas:
        groups.append({"params": thetas, "lr": args.lr_theta})
    opt = torch.optim.Adam(groups)

    is_sphere = isinstance(net.act, SphereGearActivation)
    losses, path = [], []
    for it in range(args.iters):
        if is_sphere and it % args.log_every == 0:
            path.append(net.act.directions().detach().clone())
        opt.zero_grad()
        loss = ((net(Xtr) - Ytr) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        val = ((net(Xva) - Yva) ** 2).mean().item()
    out = {"task": task, "variant": variant, "seed": seed, "loss": losses[::args.log_every] + [losses[-1]],
           "train_mse": losses[-1], "val_mse": val, "structure": structure(net.act)}
    if hasattr(net.act, "dominant_function"):
        dom = net.act.dominant_function()
        out["dominant"] = {n: dom.count(n) for n in net.act.names}
    if is_sphere:
        path.append(net.act.directions().detach().clone())
        out["sphere_path"] = torch.stack(path).tolist()           # (T, H, 3)
        out["u_norm"] = net.act.theta.detach().norm(dim=1).mean().item()
    return out


# ---------------------------------------------------------------------------
# Graphiques et résumé
# ---------------------------------------------------------------------------
def plot_losses(results, task, log_every, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for variant in VARIANTS:
        arr = np.array([r["loss"] for r in results if r["task"] == task and r["variant"] == variant])
        if len(arr) == 0:
            continue
        x = np.arange(arr.shape[1]) * log_every
        ax.plot(x, np.median(arr, 0), label=variant, lw=1.5,
                ls="--" if variant in ("relu", "sphere_frozen") else "-")
    ax.set_yscale("log")
    ax.set_xlabel("itération")
    ax.set_ylabel("MSE d'entraînement (médiane sur les graines)")
    ax.set_title(f"{task}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary(results, tasks, n_seeds):
    lines = []
    for task in tasks:
        key = "train_mse" if task == "xor" else "val_mse"
        lines.append(f"\n== {task} : MSE de {'entraînement' if task == 'xor' else 'validation'} "
                     f"(médiane [min, max] sur {n_seeds} graines) ==")
        ref = {r["seed"]: r[key] for r in results if r["task"] == task and r["variant"] == "gear6"}
        for v in VARIANTS:
            rs = [r for r in results if r["task"] == task and r["variant"] == v]
            vals = [r[key] for r in rs]
            line = f"  {v:<14} {np.median(vals):.4g}  [{min(vals):.3g}, {max(vals):.3g}]"
            if v not in ("gear6",) and ref:
                wins = sum(r[key] < ref[r["seed"]] for r in rs)
                line += f"   meilleur que gear6 sur {wins}/{len(rs)} graines"
            lines.append(line)
        lines.append("  Position finale des neurones (total sur les graines) :")
        for v in VARIANTS[1:]:
            agg = {}
            for r in results:
                if r["task"] == task and r["variant"] == v:
                    for k, c in r["structure"].items():
                        agg[k] = agg.get(k, 0) + c
            tot = max(sum(agg.values()), 1)
            lines.append(f"    {v:<14} " + "   ".join(f"{k} {100 * c / tot:>3.0f} %" for k, c in agg.items()))
        dom = {}
        for r in results:
            if r["task"] == task and r["variant"] == "sphere":
                for k, c in r["dominant"].items():
                    dom[k] = dom.get(k, 0) + c
        tot = max(sum(dom.values()), 1)
        lines.append("  sphere, fonction dominante finale : "
                     + ", ".join(f"{k} {100 * c / tot:.0f} %" for k, c in dom.items()))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["xor", "wave", "bump", "ripple"])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lr-theta", type=float, default=1e-2)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out", default="results/sphere_toy")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.iters, args.seeds = 200, 2
    args.out = repo_path(args.out)
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    results = []
    for task in args.tasks:
        for variant in VARIANTS:
            for seed in range(args.seeds):
                results.append(run(task, variant, seed, args))
        print(f"  {task} terminé", flush=True)
        plot_losses(results, task, args.log_every, os.path.join(args.out, f"{task}_loss.png"))
        r0 = next(r for r in results if r["task"] == task and r["variant"] == "sphere" and r["seed"] == 0)
        P = np.array(r0["sphere_path"])                                  # (T, H, 3)
        g = SphereGearActivation(1, DEFAULT_SPHERE)
        plot_sphere_paths([P[:, i] for i in range(P.shape[1])], g.positions.numpy(), g.names,
                          os.path.join(args.out, f"{task}_sphere.png"),
                          title=f"{task} — trajectoires des {P.shape[1]} neurones sur la sphère (graine 0)")

    text = summary(results, args.tasks, args.seeds)
    print(text)
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Réseau 2 → {args.hidden} → 1, {args.iters} itérations, Adam lr {args.lr}, "
                f"lr_theta {args.lr_theta}, {args.seeds} graines\n" + text + "\n")
    for r in results:
        r.pop("sphere_path", None) if r["seed"] > 0 else None
    save_json({"args": vars(args), "results": results}, os.path.join(args.out, "sphere_toy_results.json"))
    print(f"\nRésultats dans {args.out}/")


if __name__ == "__main__":
    main()
