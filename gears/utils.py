# -*- coding: utf-8 -*-
"""Outils communs aux expériences et diagnostics : graines, journaux, résumé, graphiques."""
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # racine du dépôt


def repo_path(path):
    """Chemin relatif -> relatif à la racine du dépôt (results/, data/), d'où que l'on lance."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def load_runs(folder):
    runs = []
    for name in sorted(os.listdir(folder)):
        if name.startswith("run_") and name.endswith(".json"):
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                runs.append(json.load(f))
    return runs


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def mean_std(values):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def first_epoch_reaching(curve, threshold, higher_is_better=True):
    for i, v in enumerate(curve):
        if (v >= threshold) if higher_is_better else (v <= threshold):
            return i + 1
    return None


def summarize(runs, metric_final="test_acc", curve="val_acc", target=None):
    """Tableau texte : moyenne ± écart-type par activation, sur les graines."""
    by_act = defaultdict(list)
    for r in runs:
        by_act[r["activation"]].append(r)
    lines = []
    header = f"{'activation':<13}{'n':>3}  {metric_final + ' finale':>20}  {'meilleure ' + curve:>22}"
    if target is not None:
        header += f"  {'époques → ' + str(target):>16}"
    header += f"  {'params':>10}  {'s/époque':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    rows = []
    for act, rs in by_act.items():
        fin = [r[metric_final] for r in rs]
        best = [max(r["history"][curve]) for r in rs]
        m, s = mean_std(fin)
        bm, bs = mean_std(best)
        row = f"{act:<13}{len(rs):>3}  {100*m:>12.2f} ± {100*s:<5.2f}  {100*bm:>14.2f} ± {100*bs:<5.2f}"
        if target is not None:
            ep = [first_epoch_reaching(r["history"][curve], target) for r in rs]
            reached = [e for e in ep if e is not None]
            em = f"{np.mean(reached):.1f} ({len(reached)}/{len(rs)})" if reached else "jamais"
            row += f"  {em:>16}"
        t = np.mean([np.mean(r["history"]["epoch_time"]) for r in rs])
        row += f"  {rs[0]['n_params']:>10}  {t:>9.1f}"
        rows.append((m, row))
    for _, row in sorted(rows, key=lambda x: -x[0]):
        lines.append(row)
    return "\n".join(lines)


def plot_curves(runs, keys, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_act = defaultdict(list)
    for r in runs:
        by_act[r["activation"]].append(r)
    fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 4))
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        for act, rs in sorted(by_act.items()):
            L = min(len(r["history"][key]) for r in rs)
            arr = np.array([r["history"][key][:L] for r in rs])
            x = np.arange(1, L + 1)
            m = arr.mean(0)
            ax.plot(x, m, label=act, lw=1.6)
            if len(rs) > 1:
                ax.fill_between(x, arr.min(0), arr.max(0), alpha=0.15)
        ax.set_xlabel("époque")
        ax.set_title(key)
        if "loss" in key:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_theta_trajectories(runs, path, max_neurons=12, title=""):
    """Trajectoires de θ (non repliées) au fil des époques, pour les runs à engrenages.
    Une couleur par graine ; les lignes horizontales marquent les fonctions du cycle."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [r for r in runs if r.get("theta_history") and r["activation"] != "gear_frozen"]
    if not runs:
        return
    by_act = defaultdict(list)
    for r in runs:
        by_act[r["activation"]].append(r)
    for act, rs in by_act.items():
        n_neur = min(max_neurons, len(rs[0]["theta_history"][0]))
        cols = 4
        rows = math.ceil(n_neur / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), squeeze=False)
        names = rs[0]["cycle"]
        n_fn = len(names)
        for i in range(n_neur):
            ax = axes[i // cols][i % cols]
            lo, hi = np.inf, -np.inf
            for r in rs:
                th = np.array(r["theta_history"])[:, i]
                ax.plot(np.arange(len(th)), th, lw=1.2, label=f"graine {r['seed']}")
                lo, hi = min(lo, th.min()), max(hi, th.max())
            for k in range(int(np.floor(lo * n_fn)) - 1, int(np.ceil(hi * n_fn)) + 2):
                y = k / n_fn
                if lo - 0.15 <= y <= hi + 0.15:
                    ax.axhline(y, color="gray", lw=0.5, ls="--")
                    ax.text(0, y, names[k % n_fn], fontsize=7, color="gray", va="bottom")
            ax.set_title(f"neurone {rs[0]['theta_index'][i]}", fontsize=9)
        for j in range(n_neur, rows * cols):
            axes[j // cols][j % cols].axis("off")
        axes[0][0].legend(fontsize=6)
        fig.suptitle(f"{title} — {act} : trajectoires de θ (une couleur par graine)")
        fig.tight_layout()
        fig.savefig(path.replace(".png", f"_{act}.png"), dpi=120)
        plt.close(fig)


def _azimuthal(u):
    """Projection azimutale équivalente (Lambert) centrée sur le pôle de l'hémisphère de u."""
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    r = np.sqrt(np.clip(2 * (1 - np.abs(u[..., 2])), 0, None))
    h = np.hypot(u[..., 0], u[..., 1])
    h = np.where(h < 1e-12, 1.0, h)
    return u[..., 0] / h * r, u[..., 1] / h * r


def plot_sphere_paths(paths, positions, names, path, title="", labels=None):
    """Trajectoires de directions sur la sphère, en deux disques :
    hémisphère nord (z ≥ 0) et sud (z < 0), vus d'en haut, projection équivalente.

    paths     : liste de tableaux (T, 3) ; une couleur par trajectoire
    positions : sommets (n, 3) des fonctions ; names : leurs noms
    Le bord des disques est l'équateur ; les arêtes passant par un pôle sont
    des segments radiaux. o = départ, ● = arrivée.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = np.asarray(positions, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    R = np.sqrt(2)
    cmap = plt.get_cmap("tab20" if len(paths) > 10 else "tab10")
    for h, ax in enumerate(axes):
        north = h == 0
        t = np.linspace(0, 2 * np.pi, 300)
        ax.plot(R * np.cos(t), R * np.sin(t), color="gray", lw=0.8)
        pole = [i for i, p in enumerate(pos) if abs(p[2]) > 0.99 and (p[2] > 0) == north]
        equator = [i for i, p in enumerate(pos) if abs(p[2]) < 1e-6]
        if pole:
            for i in equator:
                x, y = _azimuthal(pos[i])
                ax.plot([0, x], [0, y], color="gray", lw=0.6, ls="--")
        for i, p in enumerate(pos):
            if abs(p[2]) < 1e-6 or (p[2] > 0) == north:
                x, y = _azimuthal(p)
                ax.plot(x, y, "s", color="black", ms=5)
                ax.annotate(names[i], (x, y), textcoords="offset points",
                            xytext=(6 * np.sign(x) if abs(x) > 0.1 else 6, 6 * np.sign(y) if abs(y) > 0.1 else 6),
                            fontsize=10, fontweight="bold", ha="center")
        for k, P in enumerate(paths):
            P = np.asarray(P, dtype=float)
            x, y = _azimuthal(P)
            keep = (P[:, 2] >= 0) if north else (P[:, 2] < 0)
            x, y = np.where(keep, x, np.nan), np.where(keep, y, np.nan)
            c = cmap(k % cmap.N)
            ax.plot(x, y, lw=1.1, color=c, alpha=0.85,
                    label=(labels[k] if labels and h == 0 else None))
            if keep[0]:
                ax.plot(x[0], y[0], "o", mfc="none", color=c, ms=5)
            if keep[-1]:
                ax.plot(x[-1], y[-1], "o", color=c, ms=5)
        ax.set_aspect("equal")
        ax.set_xlim(-1.65, 1.65)
        ax.set_ylim(-1.65, 1.65)
        ax.axis("off")
        ax.set_title("hémisphère nord (z ≥ 0)" if north else "hémisphère sud (z < 0)", fontsize=10)
    if labels:
        axes[0].legend(fontsize=7, loc="lower left", ncol=2)
    fig.suptitle(title + "\n(bord = équateur ; o départ, ● arrivée)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
