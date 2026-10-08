# -*- coding: utf-8 -*-
"""
Expérience 3 (point 5) : ResNet-20 sur CIFAR-10, avec activations à engrenages.

Référence : He, Zhang, Ren, Sun, « Deep Residual Learning for Image
Recognition », CVPR 2016, section 4.2. ResNet-20 (0,27 M paramètres) y
obtient 8,75 % d'erreur sur le test de CIFAR-10, soit 91,25 % de précision.

Protocole reproduit (par défaut) :
- 3 étages de 16/32/64 canaux, 3 blocs résiduels par étage, raccourcis
  d'identité avec zéro-padding (« option A » de l'article) ;
- SGD, lr 0,1, momentum 0,9, weight decay 1e-4, batch 128 ;
- 182 époques (≈ 64 000 itérations), lr divisé par 10 aux époques 91 et 136 ;
- augmentation : padding 4 + recadrage aléatoire 32×32 + retournement horizontal.

Avec engrenages, chaque ReLU du réseau est remplacée par une activation à
engrenage avec un θ PAR CANAL (16, 32 ou 64 θ par couche, ~ 1 000 θ au total).
Les θ n'ont pas de weight decay et ont leur propre taux d'apprentissage.

Durée indicative d'un run de 182 époques : ~ 30-60 min sur un GPU récent
(Colab T4 : plutôt 1 h 30 à 2 h ; l'activation à engrenage est plus lente que
ReLU car elle évalue toutes les fonctions du cycle).

Lancer (exemples) :
  python experiments/exp3_resnet.py --act relu --seed 0
  python experiments/exp3_resnet.py --act gear --seed 0
  python experiments/exp3_resnet.py --act gear --epochs 60          (version courte, jalons à 50 % et 75 %)
  python experiments/exp3_resnet.py --act gear --fake-data --epochs 1 --max-batches 5   (test)
  python experiments/exp3_resnet.py --summarize                      (tableau + graphiques de tous les runs)
Reprise automatique depuis le dernier checkpoint si le run est interrompu.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # racine du dépôt
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gears.utils import (repo_path, plot_sphere_paths, set_seed, pick_device, save_json, load_runs, count_params,
                    summarize, plot_curves, plot_theta_trajectories)
from gears import (make_activation, make_schedule, split_params, GearSchedule, gear_modules, position_modules,
                   sphere_modules, snapshot_thetas, snapshot_directions, layout_positions,
                   ACTIVATION_CHOICES)


# ---------------------------------------------------------------------------
# ResNet-20 (He et al. 2016, CIFAR) avec activation interchangeable
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, act_fn):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.act1 = act_fn(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act2 = act_fn(cout)
        self.stride, self.pad = stride, cout - cin

    def shortcut(self, x):
        # option A : sous-échantillonnage + zéro-padding des canaux (pas de paramètres)
        if self.stride != 1 or self.pad:
            x = x[:, :, ::self.stride, ::self.stride]
            x = F.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))
        return x

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + self.shortcut(x))


class ResNetCifar(nn.Module):
    def __init__(self, act_fn, n=3, num_classes=10):     # n=3 → ResNet-20 (6n+2 couches)
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.act = act_fn(16)
        blocks, cin = [], 16
        for stage, cout in enumerate((16, 32, 64)):
            for b in range(n):
                stride = 2 if (stage > 0 and b == 0) else 1
                blocks.append(BasicBlock(cin, cout, stride, act_fn))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def loaders(args, start_epoch=0):
    from torchvision import datasets, transforms
    norm = transforms.Normalize(MEAN, STD)
    t_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(), norm])
    t_test = transforms.Compose([transforms.ToTensor(), norm])
    if args.fake_data:
        tr = datasets.FakeData(1024, (3, 32, 32), 10, t_train, random_offset=0)
        te = datasets.FakeData(256, (3, 32, 32), 10, t_test, random_offset=10_000)
    else:
        tr = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=t_train)
        te = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=t_test)
    # ordre des batchs : dépend de la graine et de l'époque de reprise
    g = torch.Generator().manual_seed(1000 * args.seed + start_epoch)
    kw = dict(num_workers=args.workers, pin_memory=torch.cuda.is_available(),
              persistent_workers=args.workers > 0)
    return (torch.utils.data.DataLoader(tr, args.batch, shuffle=True, drop_last=False, generator=g, **kw),
            torch.utils.data.DataLoader(te, 500, shuffle=False, **kw))


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp):
            out = model(x)
        loss += F.cross_entropy(out.float(), y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    model.train()
    return loss / n, correct / n


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------
def train(args):
    device = pick_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    set_seed(args.seed)
    cycle = args.cycle
    act_fn = lambda c: make_activation(args.act, c, functions=cycle) \
        if args.act in ("abu", "gear", "gear_window", "gear_frozen", "gear_relu_init", "gear_alt", "gear_alt_plateau") else make_activation(args.act, c)
    model = ResNetCifar(act_fn, n=args.depth_n).to(device)
    print(f"Appareil : {device} | AMP : {amp} | ResNet-{6 * args.depth_n + 2} | activation : {args.act} "
          f"| paramètres : {count_params(model):,}")

    others, thetas = split_params(model)
    # pas de weight decay sur les θ ni sur les coefficients ABU
    abu = [p for n_, p in model.named_parameters() if n_.endswith(".alpha")]
    abu_ids = {id(p) for p in abu}
    decay = [p for p in others if id(p) not in abu_ids]
    groups = [{"params": decay, "lr": args.lr, "weight_decay": args.wd}]
    if abu:
        groups.append({"params": abu, "lr": args.lr, "weight_decay": 0.0})
    if thetas:
        groups.append({"params": thetas, "lr": args.lr_theta, "weight_decay": 0.0})
    opt = torch.optim.SGD(groups, momentum=0.9, nesterov=False)
    milestones = args.milestones or [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones, 0.1)
    gear_sched = make_schedule(args.act, model, window=args.gear_window, alt=args.gear_alt,
                               plateau=dict(patience=int(args.plateau[0]), min_delta=args.plateau[1], theta_tol=args.plateau[2], stop_tol=args.plateau[3], min_on=2))
    scaler = torch.amp.GradScaler(enabled=amp)

    tag = f"{args.act}_s{args.seed}"
    ckpt_path = os.path.join(args.out, f"ckpt_{tag}.pt")
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time", "gear_active")}
    theta_hist, start_epoch = [], 0
    # suivi de quelques neurones : θ (cercle) ou direction u (sphère)
    track, snap = None, None
    if gear_modules(model):
        snap = lambda: snapshot_thetas(model, raw=True)[track].tolist()
        n_track = len(snapshot_thetas(model))
    elif sphere_modules(model):
        snap = lambda: snapshot_directions(model)[track].tolist()
        n_track = len(snapshot_directions(model))
    if snap is not None:
        g = torch.Generator().manual_seed(123)
        track = torch.randperm(n_track, generator=g)[:args.track_neurons]
        theta_hist.append(snap())

    if os.path.exists(ckpt_path) and not args.no_resume:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        lr_sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        hist, theta_hist, start_epoch = ck["hist"], ck["theta_hist"], ck["epoch"]
        torch.set_rng_state(ck["rng"].cpu())
        if gear_sched is not None and ck.get("gear_sched"):
            gear_sched.load_state_dict(ck["gear_sched"])
        print(f"Reprise à l'époque {start_epoch}")

    train_loader, test_loader = loaders(args, start_epoch)
    for epoch in range(start_epoch, args.epochs):
        last = hist["train_loss"][-1] if hist["train_loss"] else None
        on = gear_sched.step(epoch, last) if gear_sched else bool(thetas)
        t0 = time.time()
        tot_loss, tot_correct, n = 0.0, 0, 0
        for b, (x, y) in enumerate(train_loader):
            if args.max_batches and b >= args.max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp):
                out = model(x)
                loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_loss += loss.item() * len(y)
            tot_correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        lr_sched.step()
        hist["epoch_time"].append(time.time() - t0)
        hist["train_loss"].append(tot_loss / n)
        hist["train_acc"].append(tot_correct / n)
        vl, va = evaluate(model, test_loader, device, amp)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        hist["lr"].append(opt.param_groups[0]["lr"])
        hist["gear_active"].append(on)
        if track is not None:
            theta_hist.append(snap())
        print(f"[{tag}] époque {epoch + 1:>3}/{args.epochs}  train {hist['train_loss'][-1]:.3f} "
              f"({100 * hist['train_acc'][-1]:.1f}%)  test {100 * va:.2f}%  "
              f"{hist['epoch_time'][-1]:.0f}s" + ("  θ actifs" if on and thetas else ""), flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": lr_sched.state_dict(),
                    "scaler": scaler.state_dict(), "hist": hist, "theta_hist": theta_hist,
                    "epoch": epoch + 1, "rng": torch.get_rng_state(),
                    "gear_sched": gear_sched.state_dict() if gear_sched is not None else None}, ckpt_path)

    # NB : comme dans l'article, la courbe « val » est celle du jeu de test CIFAR-10.
    # Ne réglez donc pas lr_theta ou le cycle en regardant cette courbe (voir README).
    run = {"activation": args.act, "seed": args.seed, "test_loss": hist["val_loss"][-1],
           "test_acc": hist["val_acc"][-1], "n_params": count_params(model), "history": hist,
           "args": vars(args)}
    if hasattr(gear_sched, "events"):
        run["schedule_events"], run["schedule_cycles"] = gear_sched.events, gear_sched.cycles
    if track is not None:
        key = "theta_history" if gear_modules(model) else "sphere_path"
        run[key], run["theta_index"] = theta_hist, track.tolist()
        run["cycle"] = position_modules(model)[0].names
        run["dominant_per_layer"] = []
        for name, m in model.named_modules():
            if getattr(m, "is_gear", False):
                d = m.dominant_function()
                run["dominant_per_layer"].append({"layer": name, **{f: d.count(f) for f in m.names}})
    save_json(run, os.path.join(args.out, f"run_{tag}.json"))
    print(f"Précision finale sur le test : {100 * run['test_acc']:.2f}%  (référence ReLU de l'article : 91,25%)")
    return run


def do_summary(args):
    runs = load_runs(args.out)
    if not runs:
        print("Aucun run trouvé dans", args.out)
        return
    table = summarize(runs, "test_acc", "val_acc", target=0.90)
    print(table)
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
        for r in runs:
            for L in r.get("dominant_per_layer", []):
                f.write(f"\n{r['activation']} s{r['seed']} {L}")
    plot_curves(runs, ["train_loss", "val_loss", "val_acc"], os.path.join(args.out, "curves.png"),
                title="ResNet-20 — CIFAR-10")
    plot_theta_trajectories(runs, os.path.join(args.out, "theta.png"), title="ResNet-20 — CIFAR-10")
    for r in runs:
        if r.get("sphere_path") and r["seed"] == 0 and r["activation"] != "sphere_frozen":
            P = np.array(r["sphere_path"])
            pos = layout_positions("octa").numpy()
            plot_sphere_paths([P[:, i] for i in range(P.shape[1])], pos, r["cycle"],
                              os.path.join(args.out, f"{r['activation']}_paths.png"),
                              title=f"ResNet-20 — {r['activation']} : neurones suivis (graine 0)")
    print(f"\nRésumé : {args.out}/summary.txt, curves.png, theta_*.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--act", default="relu", choices=ACTIVATION_CHOICES)
    p.add_argument("--cycle", nargs="+", default=["relu", "silu", "tanh", "identity"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth-n", type=int, default=3, help="3 → ResNet-20, 5 → ResNet-32, 9 → ResNet-56")
    p.add_argument("--epochs", type=int, default=182)
    p.add_argument("--milestones", type=int, nargs="*", default=None,
                   help="époques de division du lr (par défaut 50 %% et 75 %% ; 91 136 pour 182 époques)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--lr-theta", type=float, default=0.01)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--gear-window", type=int, nargs=2, default=[20, 90],
                   help="époques où θ est entraîné pour --act gear_window")
    p.add_argument("--gear-alt", type=int, nargs=2, default=[10, 5], metavar=("OFF", "ON"),
                   help="gear_alt : époques θ figés, puis époques θ entraînés, en boucle")
    p.add_argument("--plateau", type=float, nargs=4, default=[2, 0.05, 1e-3, 5e-3],
                   metavar=("PATIENCE", "MIN_DELTA", "THETA_TOL", "STOP_TOL"),
                   help="gear_alt_plateau : ouverture quand la perte d'entraînement baisse de moins de "
                        "MIN_DELTA (relatif, par époque) sur PATIENCE époques ; fermeture quand θ bouge de "
                        "moins de THETA_TOL par époque ; arrêt si une phase entière bouge de moins de STOP_TOL")
    p.add_argument("--track-neurons", type=int, default=12)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out", default="results/resnet20_cifar10")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--max-batches", type=int, default=0, help="limite de batchs par époque (tests)")
    p.add_argument("--fake-data", action="store_true")
    p.add_argument("--no-resume", action="store_true", help="ignorer un checkpoint existant")
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    if args.fake_data and args.out == "results/resnet20_cifar10":
        args.out = "results/resnet_fake"
    args.out, args.data_dir = repo_path(args.out), repo_path(args.data_dir)
    os.makedirs(args.out, exist_ok=True)
    if args.summarize:
        do_summary(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
