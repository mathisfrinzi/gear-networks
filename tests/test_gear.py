# -*- coding: utf-8 -*-
"""Tests du module d'engrenages. Lancer : python tests/test_gear.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from gears import (GearActivation, ABUActivation, make_activation, GearSchedule,
                   split_params, snapshot_thetas)

torch.manual_seed(0)
z = torch.linspace(-3, 3, 101).repeat(4, 1).T.contiguous()   # (101, 4)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok -", msg)


def test_nodes_are_pure_functions():
    g = GearActivation(4, ("relu", "tanh", "sigmoid", "identity"), init=0.0)
    with torch.no_grad():
        for k, ref in enumerate([F.relu, torch.tanh, torch.sigmoid, lambda x: x]):
            g.theta.fill_(k / 4)
            check(torch.allclose(g(z), ref(z), atol=1e-6), f"θ={k}/4 donne exactement la fonction {k}")


def test_midpoint_is_average():
    g = GearActivation(4, ("relu", "tanh"), init=0.25)
    check(torch.allclose(g(z), 0.5 * F.relu(z) + 0.5 * torch.tanh(z), atol=1e-6),
          "θ au milieu d'un segment = moyenne des deux voisines")


def test_periodicity():
    g1 = GearActivation(4, ("relu", "tanh", "silu"), init=0.37)
    g2 = GearActivation(4, ("relu", "tanh", "silu"), init=2.37)
    g3 = GearActivation(4, ("relu", "tanh", "silu"), init=-0.63)
    check(torch.allclose(g1(z), g2(z), atol=1e-5) and torch.allclose(g1(z), g3(z), atol=1e-5),
          "θ et θ±entier donnent la même activation (cycle fermé)")


def test_continuity_in_theta():
    g = GearActivation(1, ("relu", "tanh", "silu"))
    x = torch.linspace(-3, 3, 50).unsqueeze(1)
    worst = 0.0
    with torch.no_grad():
        for b in [0.0, 1 / 3, 2 / 3, 1.0]:
            g.theta.fill_(b - 1e-5); a = g(x)
            g.theta.fill_(b + 1e-5); c = g(x)
            worst = max(worst, (a - c).abs().max().item())
    check(worst < 1e-3, f"f_θ continue en θ aux jonctions (écart max {worst:.1e}), y compris au bouclage")


def test_unequal_lengths():
    # segments de longueurs relatives 2, 4, 3, 3 (soit 2/12, 4/12, 3/12, 3/12)
    g = GearActivation(1, ("identity", "tanh", "sigmoid", "identity"), lengths=[2, 4, 3, 3], init=1 / 12)
    x = torch.linspace(-2, 2, 20).unsqueeze(1)
    check(torch.allclose(g(x), 0.5 * x + 0.5 * torch.tanh(x), atol=1e-6),
          "longueurs de segments inégales respectées")


def test_theta_gradient_analytic():
    # ∂f/∂θ = (f_{k+1} - f_k) / longueur_k  -> ici (tanh - relu) / (1/3)
    g = GearActivation(4, ("relu", "tanh", "silu"), init=0.1)
    y = g(z).sum()
    y.backward()
    expected = ((torch.tanh(z) - F.relu(z)) * 3).sum(0)
    check(torch.allclose(g.theta.grad, expected, atol=1e-4),
          "gradient autograd de θ = formule analytique (f_k+1 - f_k)/longueur")


def test_gradcheck():
    g = GearActivation(3, ("relu", "tanh", "silu"), init="uniform").double()
    with torch.no_grad():
        g.theta.copy_(torch.tensor([0.11, 0.52, 0.83], dtype=torch.double))  # loin des jonctions
    x = torch.randn(5, 3, dtype=torch.double)
    from torch.func import functional_call
    ok = torch.autograd.gradcheck(lambda th: functional_call(g, {"theta": th}, (x,)),
                                  (g.theta.detach().clone().requires_grad_(),), eps=1e-6, atol=1e-5)
    check(ok, "gradcheck numérique sur θ")


def test_gradient_descent_direction():
    # Cible = tanh pur ; on part de relu (θ=0.1 vers tanh à 1/3). La perte doit baisser.
    g = GearActivation(1, ("relu", "tanh", "silu"), init=0.1)
    x = torch.linspace(-3, 3, 200).unsqueeze(1)
    target = torch.tanh(x)
    opt = torch.optim.SGD(g.parameters(), lr=0.01)
    losses = []
    for _ in range(300):
        opt.zero_grad()
        loss = ((g(x) - target) ** 2).mean()
        loss.backward(); opt.step(); losses.append(loss.item())
    th = torch.remainder(g.theta, 1).item()
    check(losses[-1] < 1e-3 * losses[0] and abs(th - 1 / 3) < 0.02,
          f"descente : θ converge vers tanh (θ={th:.3f}, attendu 0.333), perte {losses[0]:.2e} → {losses[-1]:.2e}")


def test_conv_channels():
    g = GearActivation(8, feature_dim=1)
    x = torch.randn(2, 8, 5, 5)
    y = g(x)
    check(y.shape == x.shape and g.theta.shape == (8,), "fonctionne sur (B, C, H, W) avec un θ par canal")


def test_abu_and_factory():
    a = ABUActivation(4, ("relu", "tanh"))
    check(torch.allclose(a(z), 0.5 * F.relu(z) + 0.5 * torch.tanh(z)), "ABU initialisé à 1/n")
    for k in ("relu", "prelu", "silu", "tanh", "abu", "gear", "gear_frozen", "gear_window"):
        m = make_activation(k, 4)
        check(m(z).shape == z.shape, f"make_activation('{k}')")
    fz = make_activation("gear_frozen", 4)
    others, thetas = split_params(torch.nn.Sequential(fz))
    check(len(thetas) == 0, "gear_frozen : θ non entraînable")


def test_schedule():
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), make_activation("gear_window", 4))
    s = GearSchedule(m, start=5, end=10)
    check(not s.step(0) and m[1].theta.requires_grad is False, "fenêtre : θ figé avant le début")
    check(s.step(7) and m[1].theta.requires_grad is True, "fenêtre : θ actif pendant")
    check(not s.step(11), "fenêtre : θ figé après la fin")
    check(snapshot_thetas(m).shape == (4,), "snapshot_thetas")


def test_harden():
    g = GearActivation(3, ("relu", "tanh", "silu"))
    with torch.no_grad():
        g.theta.copy_(torch.tensor([0.05, 0.30, 0.95]))
    g.harden()
    check(g.dominant_function() == ["relu", "tanh", "relu"] and
          torch.allclose(torch.remainder(g.theta, 1), torch.tensor([0.0, 1 / 3, 0.0])),
          "harden() fixe chaque neurone sur sa fonction dominante")


def test_relu_init_equals_relu():
    for kind in ("gear_relu_init", "sphere_relu_init"):
        a = make_activation(kind, 4)
        check(torch.allclose(a(z), F.relu(z), atol=1e-6), f"{kind} : identique à ReLU à l'itération 0")
        check(a.theta.requires_grad, f"{kind} : θ entraînés")
        out = a(z.clone().requires_grad_(False))
        a.theta.grad = None
        (a(z) ** 2).sum().backward()
        check(a.theta.grad is not None and a.theta.grad.abs().sum() > 0,
              f"{kind} : le gradient atteint θ dès le départ")
    try:
        make_activation("gear_relu_init", 4, functions=("tanh", "relu"))
        check(False, "cycle sans relu en premier doit échouer")
    except ValueError:
        check(True, "gear_relu_init refuse un cycle qui ne commence pas par relu")


def test_cycle_schedule():
    from gears import GearCycleSchedule
    net = torch.nn.Sequential(torch.nn.Linear(3, 4), make_activation("gear_alt", 4))
    s = GearCycleSchedule(net, off=3, on=2)
    pattern = [s.step(t) for t in range(10)]
    check(pattern == [False] * 3 + [True] * 2 + [False] * 3 + [True] * 2,
          "alternance fixe : 3 unités figées, 2 entraînées, en boucle")
    check(torch.allclose(net[1](z), F.relu(z)), "gear_alt démarre sur ReLU")


def test_plateau_schedule():
    from gears import GearPlateauSchedule
    net = torch.nn.Sequential(torch.nn.Linear(3, 4), make_activation("gear_alt_plateau", 4))
    g = net[1]
    s = GearPlateauSchedule(net, patience=2, min_delta=0.05, theta_tol=1e-3, stop_tol=5e-3)
    phases = []
    # pertes qui baissent vite puis se stabilisent : ouverture attendue
    for t, loss in enumerate([None, 1.0, 0.6, 0.4, 0.39, 0.385]):
        phases.append(s.step(t, loss))
    check(phases[:4] == [False] * 4 and phases[-1], "s'ouvre quand la perte se stabilise")
    # les θ bougent beaucoup, puis plus du tout : fermeture
    with torch.no_grad():
        g.theta += 0.1
    check(s.step(6, 0.38), "reste ouvert tant que les θ bougent")
    check(not s.step(7, 0.38), "se referme quand les θ ne bougent plus")
    check(s.phase == "off" and len(s.cycles) == 1, "cycle enregistré, retour en phase figée")
    # nouveau plateau, phase « on » sans mouvement : arrêt définitif
    for t, loss in enumerate([0.378, 0.377, 0.377], start=8):
        s.step(t, loss)
    check(s.phase == "on", "rouvre au plateau suivant")
    s.step(11, 0.377)
    check(s.phase == "done" and not s.step(12, 0.3), "arrêt définitif si les θ ne bougent plus du tout")
    s2 = GearPlateauSchedule(net)
    s2.load_state_dict(s.state_dict())
    check(s2.phase == "done" and s2.events == s.events, "état sauvegardé et rechargé (reprise sur checkpoint)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print(f"\n{len(tests)} tests passés.")
