# -*- coding: utf-8 -*-
"""
Diagnostic du « collage aux nœuds » des engrenages.

Question
--------
Avec l'interpolation linéaire entre fonctions voisines, le gradient par rapport
à θ est constant par morceaux et change de valeur aux jonctions (les « nœuds »,
c'est-à-dire les fonctions pures : relu, silu, tanh...). On observe que de
nombreux θ viennent se fixer exactement sur un nœud. Deux explications :

  (a) Sélection : la perte forme une pointe en V sur le nœud (comme la
      parcimonie du Lasso). La fonction pure est réellement la meilleure pour
      ce neurone : comportement sain.
  (b) Piège : la perte continuerait à baisser de l'autre côté, mais
      l'optimisation n'y arrive pas.

Analyses
--------
Banc d'essai : MLP 2 → H → H → 1 sur la tâche « wave »
(y = sin(2π·x1)·x2 + bruit). Aucun téléchargement, CPU, quelques minutes.

  1. Dynamique      Fraction de θ à moins de EPS d'un nœud au fil de
                    l'entraînement, comparée au niveau dû au hasard.
  2. Profils        Pour quelques neurones, un seul θ varie de 0 à 1 (le reste
                    est figé) et on trace la perte d'entraînement.
  3. Classement     Pour chaque θ collé, forme de la perte autour du nœud :
                    pointe en V, pente qui traverse le nœud, ou maximum local.
  4. Perturbation   Les θ collés sont écartés de ±PERTURB, puis l'entraînement
                    reprend. Reviennent-ils sur leur nœud ? Un contrôle sans
                    perturbation mesure combien se décollent d'eux-mêmes.
  5. Balayage       Taux d'apprentissage des θ (lr_theta = 0 : θ figés).

Utilisation
-----------
    python diagnostics/diag_nodes.py                 # version complète (~3 min, CPU)
    python diagnostics/diag_nodes.py --quick         # vérification de bon fonctionnement (~30 s)
    python diagnostics/diag_nodes.py --seeds 0 1 2 3 4 --iters 5000 --out results/mon_essai

Le script peut aussi être lancé depuis un éditeur (bouton ▶) : les réglages par
défaut sont ceux de `Config`.

Sorties (dans `--out`, par défaut results/diag_nodes/)
------------------------------------------------------
    summary.txt          tableau du balayage et résultats des analyses 3 et 4
    dynamics.png         analyse 1
    profiles.png         analyse 2
    sweep_lr_theta.png   analyse 5

Résultats de référence : docs/resultats_preliminaires.md (section « Collage
aux nœuds »).
"""
import argparse
import copy
import math
import os
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # racine du dépôt
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gears.utils import set_seed  # noqa: E402
from gears import GearActivation, gear_modules  # noqa: E402

Data = Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]

RECORD_EVERY = 10  # une position de θ enregistrée toutes les 10 itérations


# ---------------------------------------------------------------------------
# Réglages
# ---------------------------------------------------------------------------
@dataclass
class Config:
    cycle: Tuple[str, ...] = ("relu", "silu", "tanh", "identity")
    hidden: int = 64                    # largeur des deux couches cachées
    iters: int = 3000                   # itérations d'entraînement (batch complet)
    lr: float = 1e-3                    # Adam, poids et biais
    lr_theta_main: float = 1e-2         # lr des θ pour les analyses 1 à 4
    sweep_lr_theta: List[float] = field(
        default_factory=lambda: [0.0, 1e-4, 1e-3, 1e-2, 1e-1])  # 0.0 = θ figés (contrôle)
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    eps: float = 0.02                   # « collé » = à moins de eps d'un nœud
    perturb: float = 0.03               # écart imposé aux θ collés (analyse 4)
    perturb_iters: int = 500
    n_profiles: int = 8                 # neurones tracés dans profiles.png
    out: str = "results/diag_nodes"

    @property
    def chance(self) -> float:
        """Fraction de θ « collés » attendue si les θ étaient uniformes sur [0, 1)."""
        return min(1.0, 2 * self.eps * len(self.cycle))

    def quick(self) -> "Config":
        return replace(self, iters=400, seeds=[0, 1], perturb_iters=100,
                       sweep_lr_theta=[0.0, 1e-2], out="results/diag_nodes_test")


# ---------------------------------------------------------------------------
# Données et modèle
# ---------------------------------------------------------------------------
def make_data(n_train: int = 800, n_val: int = 400, noise: float = 0.05) -> Data:
    """Tâche « wave » : y = sin(2π·x1)·x2 + bruit gaussien, x dans [-1, 1]²."""
    n = n_train + n_val
    rng = np.random.default_rng(1234)
    X = torch.from_numpy(rng.uniform(-1, 1, (n, 2))).float()
    Y = torch.sin(2 * math.pi * X[:, :1]) * X[:, 1:]
    Y = Y + torch.from_numpy(rng.normal(0, noise, (n, 1))).float()
    return (X[:n_train], Y[:n_train]), (X[n_train:], Y[n_train:])


class Net(nn.Module):
    """MLP 2 → H → H → 1 avec une GearActivation après chaque couche cachée."""

    def __init__(self, hidden: int, cycle: Sequence[str]):
        super().__init__()
        self.l1, self.a1 = nn.Linear(2, hidden), GearActivation(hidden, cycle)
        self.l2, self.a2 = nn.Linear(hidden, hidden), GearActivation(hidden, cycle)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.out(self.a2(self.l2(self.a1(self.l1(x)))))


def mse(net: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return ((net(X) - Y) ** 2).mean()


# ---------------------------------------------------------------------------
# Outils sur les θ
# ---------------------------------------------------------------------------
def all_thetas(net: nn.Module) -> torch.Tensor:
    """Tous les θ du réseau, concaténés (copie)."""
    return torch.cat([g.theta.detach().flatten() for g in gear_modules(net)]).clone()


def set_thetas(net: nn.Module, vec: torch.Tensor) -> None:
    i = 0
    with torch.no_grad():
        for g in gear_modules(net):
            n = g.theta.numel()
            g.theta.copy_(vec[i:i + n])
            i += n


def locate(net: nn.Module, k: int):
    """Module et indice local du k-ième θ du réseau."""
    for g in gear_modules(net):
        if k < g.theta.numel():
            return g, k
        k -= g.theta.numel()
    raise IndexError(k)


def nearest_node(theta: torch.Tensor, starts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Distance (sur le cercle) au nœud le plus proche, et indice de ce nœud."""
    p = torch.remainder(theta, 1.0)[:, None]
    d = ((p - starts[None, :] + 0.5) % 1.0 - 0.5).abs()
    dist, idx = d.min(1)
    return dist, idx


def stuck_fraction(theta: torch.Tensor, starts: torch.Tensor, eps: float) -> float:
    return (nearest_node(theta, starts)[0] < eps).float().mean().item()


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------
def make_optimizer(net: nn.Module, lr: float, lr_theta: float) -> torch.optim.Optimizer:
    """Adam avec un taux d'apprentissage séparé pour les θ. lr_theta = 0 : θ figés."""
    thetas = [g.theta for g in gear_modules(net)]
    ids = {id(t) for t in thetas}
    others = [p for p in net.parameters() if id(p) not in ids]
    groups = [{"params": others, "lr": lr}]
    if lr_theta > 0:
        groups.append({"params": thetas, "lr": lr_theta})
    for t in thetas:
        t.requires_grad_(lr_theta > 0)
    return torch.optim.Adam(groups)


def train(net: nn.Module, data: Data, lr: float, lr_theta: float, iters: int) -> torch.Tensor:
    """Entraîne en batch complet. Renvoie les θ enregistrés, de forme (n_records, n_theta)."""
    (Xtr, Ytr), _ = data
    opt = make_optimizer(net, lr, lr_theta)
    traj = [all_thetas(net)]
    for it in range(iters):
        opt.zero_grad()
        mse(net, Xtr, Ytr).backward()
        opt.step()
        if (it + 1) % RECORD_EVERY == 0:
            traj.append(all_thetas(net))
    return torch.stack(traj)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
@torch.no_grad()
def loss_with_theta(net: nn.Module, X: torch.Tensor, Y: torch.Tensor, k: int, value: float) -> float:
    """Perte d'entraînement si le k-ième θ vaut `value` (tout le reste inchangé)."""
    g, j = locate(net, k)
    old = g.theta[j].item()
    g.theta[j] = value
    out = mse(net, X, Y).item()
    g.theta[j] = old
    return out


V_SHAPE = "pointe en V (minimum sur le nœud)"
SLOPE = "pente qui traverse le nœud (pas un minimum)"
LOCAL_MAX = "maximum local sur le nœud"


def classify_stuck(net: nn.Module, data: Data, starts: torch.Tensor,
                   eps: float, delta: float = 0.01) -> Dict[str, int]:
    """Pour chaque θ collé, forme de la perte autour de son nœud (±delta)."""
    (Xtr, Ytr), _ = data
    th = all_thetas(net)
    dist, idx = nearest_node(th, starts)
    counts = {V_SHAPE: 0, SLOPE: 0, LOCAL_MAX: 0}
    for k in torch.nonzero(dist < eps).flatten().tolist():
        # θ n'est pas replié : on place le nœud dans le même « tour » que θ
        offset = (th[k].item() - starts[idx[k]].item() + 0.5) % 1.0 - 0.5
        node = th[k].item() - offset
        l0 = loss_with_theta(net, Xtr, Ytr, k, node)
        lm = loss_with_theta(net, Xtr, Ytr, k, node - delta)
        lp = loss_with_theta(net, Xtr, Ytr, k, node + delta)
        if lm > l0 and lp > l0:
            counts[V_SHAPE] += 1
        elif lm < l0 and lp < l0:
            counts[LOCAL_MAX] += 1
        else:
            counts[SLOPE] += 1
    return counts


def perturbation_test(net: nn.Module, data: Data, starts: torch.Tensor,
                      cfg: Config, seed: int) -> Optional[dict]:
    """Écarte les θ collés de ±perturb, reprend l'entraînement et regarde s'ils reviennent.

    Contrôle : même reprise sans perturbation (restent-ils collés de toute façon ?).
    """
    th0 = all_thetas(net)
    dist, idx = nearest_node(th0, starts)
    stuck = torch.nonzero(dist < cfg.eps).flatten()
    if len(stuck) == 0:
        return None
    gen = torch.Generator().manual_seed(100 + seed)
    signs = torch.randint(0, 2, (len(stuck),), generator=gen).float() * 2 - 1

    res: dict = {}
    for name, perturb in (("perturbés", True), ("contrôle sans perturbation", False)):
        n2 = copy.deepcopy(net)
        if perturb:
            th = th0.clone()
            th[stuck] += signs * cfg.perturb
            set_thetas(n2, th)
        set_seed(1000 + seed)
        train(n2, data, cfg.lr, cfg.lr_theta_main, cfg.perturb_iters)
        d2, i2 = nearest_node(all_thetas(n2), starts)
        back = ((d2[stuck] < cfg.eps) & (i2[stuck] == idx[stuck])).float().mean().item()
        other = ((d2[stuck] < cfg.eps) & (i2[stuck] != idx[stuck])).float().mean().item()
        res[name] = {"revenus sur le même nœud": back,
                     "collés à un autre nœud": other,
                     "restés hors d'un nœud": 1 - back - other}
    res["n"] = len(stuck)
    return res


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------
def plot_dynamics(trajs: Dict[int, torch.Tensor], starts: torch.Tensor, cfg: Config, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for seed, traj in trajs.items():
        steps = np.arange(len(traj)) * RECORD_EVERY
        frac = [stuck_fraction(t, starts, cfg.eps) for t in traj]
        axes[0].plot(steps, frac, label=f"graine {seed}")
        speed = (traj[1:] - traj[:-1]).abs().median(1).values / RECORD_EVERY
        axes[1].plot(steps[1:], speed.numpy(), label=f"graine {seed}")
    axes[0].axhline(cfg.chance, color="gray", ls="--", label=f"hasard ({cfg.chance:.0%})")
    axes[0].set_title(f"fraction de θ à moins de {cfg.eps} d'un nœud")
    axes[0].set_ylim(0, 1)
    axes[1].set_yscale("log")
    axes[1].set_title("vitesse médiane des θ (|Δθ| par itération)")
    for ax in axes:
        ax.set_xlabel("itération")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_profiles(net: nn.Module, data: Data, starts: torch.Tensor, cfg: Config, path: str) -> None:
    """Perte en faisant varier un seul θ : moitié de neurones collés, moitié libres."""
    (Xtr, Ytr), _ = data
    th = all_thetas(net)
    dist, _ = nearest_node(th, starts)
    order = torch.argsort(dist)
    n_stuck = int((dist < cfg.eps).sum())
    chosen = order[:min(cfg.n_profiles // 2, n_stuck)].tolist()
    rest = order[n_stuck:].tolist()
    gen = torch.Generator().manual_seed(0)
    pick = torch.randperm(len(rest), generator=gen)[:cfg.n_profiles - len(chosen)].tolist()
    chosen += [rest[i] for i in pick]

    cols = 4
    rows = math.ceil(len(chosen) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows), squeeze=False)
    grid = np.linspace(0, 1, 201)
    for n, k in enumerate(chosen):
        ax = axes[n // cols][n % cols]
        base = math.floor(th[k].item())
        losses = [loss_with_theta(net, Xtr, Ytr, k, base + v) for v in grid]
        ax.plot(grid, losses, lw=1.3)
        for s, name in zip(starts.tolist(), cfg.cycle):
            ax.axvline(s, color="gray", lw=0.6, ls="--")
            ax.text(s, max(losses), " " + name, fontsize=7, color="dimgray", va="top")
        ax.axvline(th[k].item() - base, color="red", lw=1.2)
        tag = "collé" if dist[k] < cfg.eps else "libre"
        ax.set_title(f"neurone {k} ({tag}) — rouge = θ appris", fontsize=8)
        ax.set_yscale("log")
        ax.tick_params(labelsize=7)
    for j in range(len(chosen), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Perte d'entraînement en faisant varier un seul θ (tout le reste figé)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sweep(table: dict, cfg: Config, path: str) -> None:
    lrs = sorted(table)
    xs = [max(lr, 1e-5) for lr in lrs]
    labels = ["figés" if lr == 0 else f"{lr:g}" for lr in lrs]
    panels = (("val", "MSE de validation"),
              ("stuck", f"fraction de θ collés (<{cfg.eps})"),
              ("moved", "déplacement moyen des θ (tours)"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (key, title) in zip(axes, panels):
        m = [np.mean(table[lr][key]) for lr in lrs]
        s = [np.std(table[lr][key]) for lr in lrs]
        ax.errorbar(xs, m, yerr=s, marker="o", capsize=3)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("lr des θ")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------
def run_sweep(cfg: Config, data: Data, starts: torch.Tensor):
    """Entraîne un réseau par (lr_theta, graine). Garde ceux de lr_theta_main."""
    (_, _), (Xva, Yva) = data
    table: dict = {}
    models: Dict[int, nn.Module] = {}
    trajs: Dict[int, torch.Tensor] = {}
    for lr_theta in cfg.sweep_lr_theta:
        table[lr_theta] = {"val": [], "stuck": [], "moved": []}
        for seed in cfg.seeds:
            set_seed(seed)
            net = Net(cfg.hidden, cfg.cycle)
            traj = train(net, data, cfg.lr, lr_theta, cfg.iters)
            with torch.no_grad():
                val = mse(net, Xva, Yva).item()
            stuck = stuck_fraction(traj[-1], starts, cfg.eps)
            moved = (traj[-1] - traj[0]).abs().mean().item()
            table[lr_theta]["val"].append(val)
            table[lr_theta]["stuck"].append(stuck)
            table[lr_theta]["moved"].append(moved)
            print(f"  lr_theta={lr_theta:<7g} graine {seed} : MSE val {val:.5f}, "
                  f"collés {stuck:.0%}, déplacement {moved:.3f} tour", flush=True)
            if lr_theta == cfg.lr_theta_main:
                models[seed], trajs[seed] = net, traj
    return table, models, trajs


def main(cfg: Optional[Config] = None) -> None:
    cfg = cfg or Config()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    t_start = time.time()
    out = cfg.out if os.path.isabs(cfg.out) else os.path.join(ROOT, cfg.out)
    os.makedirs(out, exist_ok=True)

    data = make_data()
    starts = GearActivation(1, cfg.cycle).starts
    lines: List[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    say(f"Tâche wave, MLP 2→{cfg.hidden}→{cfg.hidden}→1, "
        f"cycle {' → '.join(cfg.cycle)} → {cfg.cycle[0]}")
    say(f"{cfg.iters} itérations, graines {cfg.seeds}. "
        f"Niveau de collage dû au hasard : {cfg.chance:.0%}\n")

    table, models, trajs = run_sweep(cfg, data, starts)

    say("\n== Balayage de lr_theta (moyenne ± écart-type sur les graines) ==")
    say(f"{'lr_theta':>9}  {'MSE validation':>20}  {'θ collés':>12}  {'déplacement θ':>14}")
    for lr in sorted(table):
        v, s, m = table[lr]["val"], table[lr]["stuck"], table[lr]["moved"]
        name = "figés" if lr == 0 else f"{lr:g}"
        say(f"{name:>9}  {np.mean(v):>11.5f} ± {np.std(v):.5f}  "
            f"{100 * np.mean(s):>8.0f} %   {np.mean(m):>11.3f} tour")
    plot_sweep(table, cfg, os.path.join(out, "sweep_lr_theta.png"))

    if not models:
        say(f"\nlr_theta_main={cfg.lr_theta_main} n'est pas dans le balayage : "
            "analyses 1 à 4 sautées.")
    else:
        plot_dynamics(trajs, starts, cfg, os.path.join(out, "dynamics.png"))
        plot_profiles(models[cfg.seeds[0]], data, starts, cfg, os.path.join(out, "profiles.png"))

        say(f"\n== Forme de la perte autour des nœuds (θ collés, lr_theta={cfg.lr_theta_main}) ==")
        total: Dict[str, int] = {}
        for net in models.values():
            for k, v in classify_stuck(net, data, starts, cfg.eps).items():
                total[k] = total.get(k, 0) + v
        n_tot = sum(total.values())
        for k, v in total.items():
            say(f"  {k:<46} {v:>4}  ({100 * v / max(n_tot, 1):.0f} %)")

        say(f"\n== Perturbation de ±{cfg.perturb} puis {cfg.perturb_iters} itérations ==")
        agg: dict = {}
        n_pert = 0
        for seed, net in models.items():
            r = perturbation_test(net, data, starts, cfg, seed)
            if r is None:
                continue
            n_pert += r.pop("n")
            for cond, vals in r.items():
                for k, v in vals.items():
                    agg.setdefault(cond, {}).setdefault(k, []).append(v)
        say(f"  ({n_pert} θ collés au total)")
        for cond, vals in agg.items():
            say(f"  {cond} :")
            for k, v in vals.items():
                say(f"      {k:<28} {100 * np.mean(v):>5.0f} %")

        say("\n== Comment lire ==")
        say("- Majorité de « pointes en V » ET θ perturbés qui reviennent sur leur nœud : le collage")
        say("  est un vrai optimum (effet de parcimonie). Rien à corriger ; on peut même figer chaque")
        say("  neurone sur sa fonction en fin d'entraînement (GearActivation.harden).")
        say("- Beaucoup de « pentes qui traversent » ou de θ qui ne reviennent pas : l'optimisation")
        say("  se bloque sur les nœuds sans que ce soit optimal. Il faut alors tester des remèdes")
        say("  (voir docs/ROADMAP.md).")
        say("- Le balayage indique si un lr_theta trop faible fige les θ (déplacement proche de 0).")

    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nTerminé en {(time.time() - t_start) / 60:.1f} min. Résultats dans {out}/ : "
          "summary.txt, dynamics.png, profiles.png, sweep_lr_theta.png")


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    d = Config()
    p = argparse.ArgumentParser(description="Diagnostic du collage aux nœuds des engrenages.")
    p.add_argument("--quick", action="store_true", help="version courte (~30 s) pour vérifier que tout marche")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help=f"graines (défaut : {d.seeds})")
    p.add_argument("--iters", type=int, default=None, help=f"itérations d'entraînement (défaut : {d.iters})")
    p.add_argument("--hidden", type=int, default=None, help=f"largeur des couches cachées (défaut : {d.hidden})")
    p.add_argument("--cycle", nargs="+", default=None, help=f"fonctions du cycle (défaut : {' '.join(d.cycle)})")
    p.add_argument("--lr-theta-main", type=float, default=None,
                   help=f"lr des θ pour les analyses 1 à 4 (défaut : {d.lr_theta_main})")
    p.add_argument("--out", default=None, help=f"dossier de sortie (défaut : {d.out})")
    a = p.parse_args(argv)

    cfg = d.quick() if a.quick else d
    overrides = {"seeds": a.seeds, "iters": a.iters, "hidden": a.hidden, "out": a.out,
                 "lr_theta_main": a.lr_theta_main,
                 "cycle": tuple(a.cycle) if a.cycle else None}
    return replace(cfg, **{k: v for k, v in overrides.items() if v is not None})


if __name__ == "__main__":
    main(parse_args())
