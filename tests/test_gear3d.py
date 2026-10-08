# -*- coding: utf-8 -*-
"""Tests des engrenages sphériques. Lancer : python tests/test_gear3d.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from gears import (SphereGearActivation, GearActivation, DEFAULT_SPHERE, BASE_FUNCTIONS,
                   layout_positions, hull_faces, make_activation, split_params, GearSchedule)

torch.manual_seed(0)
z = torch.linspace(-3, 3, 101).repeat(5, 1).T.contiguous()   # (101, 5)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok -", msg)


def test_polyhedra():
    for layout, n, n_faces in (("tetra", 4, 4), ("octa", 6, 8), ("icosa", 12, 20)):
        f = hull_faces(layout_positions(layout))
        check(len(f) == n_faces, f"{layout} : {n} sommets, {n_faces} faces")


def test_weights_are_barycentric():
    g = SphereGearActivation(500, DEFAULT_SPHERE)
    w = g.mixing_weights()
    check(torch.allclose(w.sum(1), torch.ones(500), atol=1e-5), "poids de somme 1")
    check(bool((w >= 0).all()), "poids positifs")
    check(int((w > 1e-6).sum(1).max()) <= 3, "au plus 3 fonctions actives")
    # le point reconstruit est bien sur le rayon u
    u = g.directions()
    proj = w @ g.positions
    check(torch.allclose(proj / proj.norm(dim=1, keepdim=True), u, atol=1e-5),
          "Σ w_i p_i est colinéaire à u (coordonnées barycentriques exactes)")


def test_vertices_are_pure_functions():
    g = SphereGearActivation(5, DEFAULT_SPHERE)
    with torch.no_grad():
        for k, name in enumerate(DEFAULT_SPHERE):
            g.theta.copy_(g.positions[k].expand(5, 3))
            check(torch.allclose(g(z), BASE_FUNCTIONS[name](z), atol=1e-5),
                  f"sommet {k} = {name} exactement")


def test_equator_matches_circle():
    """Sur l'équateur de l'octaèdre, milieu d'arête = moyenne des deux voisines,
    comme le milieu de segment de l'engrenage cercle."""
    g = SphereGearActivation(5, DEFAULT_SPHERE)
    c = GearActivation(5, DEFAULT_SPHERE[:4], init=1 / 8)        # milieu relu → silu
    with torch.no_grad():
        g.theta.copy_(torch.tensor([1.0, 1.0, 0.0]).expand(5, 3))
    check(torch.allclose(g(z), c(z), atol=1e-5), "milieu de l'arête relu–silu = engrenage cercle au milieu")
    check(torch.allclose(g(z), 0.5 * F.relu(z) + 0.5 * F.silu(z), atol=1e-5), "= (relu + silu) / 2")


def test_face_center():
    g = SphereGearActivation(5, DEFAULT_SPHERE)
    with torch.no_grad():
        g.theta.copy_(torch.tensor([1.0, 1.0, 1.0]).expand(5, 3))   # face relu, silu, elu
    w = g.mixing_weights()[0]
    expected = torch.tensor([1 / 3, 1 / 3, 0, 0, 1 / 3, 0])
    check(torch.allclose(w, expected, atol=1e-5), "centre de face = 1/3 sur chacune des 3 fonctions")


def test_scale_invariance_and_gradient():
    g = SphereGearActivation(5, DEFAULT_SPHERE)
    out = g(z)
    with torch.no_grad():
        g.theta.mul_(3.7)
    check(torch.allclose(g(z), out, atol=1e-5), "seule la direction de u compte")
    g.theta.grad = None
    (g(z) ** 2).sum().backward()
    gr = g.theta.grad
    check(gr is not None and gr.abs().sum() > 0, "le gradient atteint u")
    radial = (gr * g.directions()).sum(1).abs().max()
    check(radial < 1e-3 * gr.norm(), "gradient orthogonal à u (la norme de u ne dérive pas)")


def test_gradient_at_vertices_and_edges():
    """Un neurone exactement sur un sommet ou une arête doit pouvoir repartir."""
    g = SphereGearActivation(6, DEFAULT_SPHERE)
    with torch.no_grad():
        for k in range(6):
            g.theta[k] = g.positions[k]
    z6 = torch.linspace(-3, 3, 101).repeat(6, 1).T.contiguous()
    g.theta.grad = None
    (g(z6) ** 2).sum().backward()
    check(bool((g.theta.grad.norm(dim=1) > 1e-3).all()), "gradient non nul sur chacun des 6 sommets")
    with torch.no_grad():
        g.theta.copy_(torch.tensor([1.0, 1.0, 0.0]).expand(6, 3) / 2 ** 0.5)
    g.theta.grad = None
    (g(z6) ** 2).sum().backward()
    check(bool((g.theta.grad.norm(dim=1) > 1e-3).all()), "gradient non nul au milieu d'une arête")


def test_kernel_mode():
    g = SphereGearActivation(50, DEFAULT_SPHERE, mode="kernel", kappa=4.0)
    w = g.mixing_weights()
    check(torch.allclose(w.sum(1), torch.ones(50), atol=1e-5) and bool((w > 0).all()),
          "noyau : poids > 0 de somme 1")
    with torch.no_grad():
        g.theta.copy_(g.positions[0].expand(50, 3))
    check(g.mixing_weights()[0, 0] > 0.9, "noyau : sur un sommet, sa fonction domine")


def test_conv_and_factory():
    for kind in ("sphere", "sphere_kernel", "sphere_frozen", "gear6", "abu6"):
        a = make_activation(kind, 8, feature_dim=1)
        y = a(torch.randn(2, 8, 5, 5))
        check(y.shape == (2, 8, 5, 5), f"make_activation('{kind}') sur une entrée (B, C, H, W)")
    a = make_activation("sphere_frozen", 8)
    check(not a.theta.requires_grad, "sphere_frozen : directions figées")


def test_training_utils():
    net = torch.nn.Sequential(torch.nn.Linear(3, 8), make_activation("sphere", 8), torch.nn.Linear(8, 1))
    others, thetas = split_params(net)
    check(len(thetas) == 1 and thetas[0].shape == (8, 3), "split_params isole les directions sphériques")
    s = GearSchedule(net, 2, 4)
    s.step(0)
    check(not net[1].theta.requires_grad, "GearSchedule fige la sphère hors fenêtre")
    s.step(3)
    check(net[1].theta.requires_grad, "GearSchedule l'active dans la fenêtre")


def test_harden_and_counts():
    g = SphereGearActivation(200, DEFAULT_SPHERE)
    dom = g.dominant_function()
    g.harden()
    check(g.dominant_function() == dom and bool((g.n_active() == 1).all()),
          "harden place chaque neurone sur sa fonction dominante")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print(f"\n{len(tests)} tests passés.")
