# -*- coding: utf-8 -*-
"""
Activations à engrenages (« gear activations »).

Chaque neurone (ou canal, pour une convolution) possède un angle θ ∈ [0, 1)
qui indique sa position sur un cycle fermé de fonctions d'activation
f_0 → f_1 → ... → f_{n-1} → f_0. Entre deux fonctions voisines, l'activation
est une interpolation linéaire :

    p = θ mod 1,   p dans le segment k,   t = (p - début_k) / longueur_k
    f_θ(z) = (1 - t) · f_k(z) + t · f_{k+1}(z)

C'est la même idée que RearFunction dans main.py (2025), réécrite pour
PyTorch : l'autograd calcule le vrai gradient par rapport à θ
(∂f/∂θ = (f_{k+1} - f_k) / longueur_k), et la descente de gradient
se fait dans le bon sens.

Le module contient aussi les points de comparaison nécessaires :
- ABU (Adaptive Blending Units, Sütfeld et al. 2018) : un coefficient libre
  par fonction et par neurone ;
- des activations classiques (ReLU, PReLU, SiLU...).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Catalogue des fonctions de base
# ---------------------------------------------------------------------------
BASE_FUNCTIONS = {
    "relu": F.relu,
    "leaky_relu": lambda z: F.leaky_relu(z, 0.01),
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "silu": F.silu,          # = swish
    "gelu": F.gelu,
    "elu": F.elu,
    "identity": lambda z: z,
    "sin": torch.sin,
    "softplus": F.softplus,
}

DEFAULT_CYCLE = ("relu", "silu", "tanh", "identity")


def _resolve(functions):
    """Accepte des noms (str) ou des callables."""
    out = []
    for f in functions:
        if isinstance(f, str):
            if f not in BASE_FUNCTIONS:
                raise ValueError(f"Fonction inconnue : {f}. Choix : {list(BASE_FUNCTIONS)}")
            out.append(BASE_FUNCTIONS[f])
        elif callable(f):
            out.append(f)
        else:
            raise TypeError(f"{f!r} n'est ni un nom ni une fonction")
    return out


def _broadcast_shape(x, feature_dim, n_features):
    """Forme pour diffuser un paramètre (n_features,) sur l'entrée x."""
    shape = [1] * x.dim()
    shape[feature_dim] = n_features
    return shape


# ---------------------------------------------------------------------------
# Activation à engrenage
# ---------------------------------------------------------------------------
class GearActivation(nn.Module):
    """
    Paramètres
    ----------
    num_features : nombre de neurones (MLP) ou de canaux (CNN).
                   Mettre 1 pour un engrenage partagé par toute la couche.
    functions    : liste de noms ou de callables, dans l'ordre du cycle.
    lengths      : longueur relative de chaque segment f_k → f_{k+1}
                   (comme les poids de RearFunction). Par défaut : égales.
    feature_dim  : dimension de l'entrée qui porte les neurones/canaux
                   (1 pour (B, N) et pour (B, C, H, W)).
    init         : 'uniform' (θ aléatoire dans [0,1)), 'node:<k>' (θ placé
                   exactement sur la fonction k), ou un flottant.

    Notes
    -----
    - θ est stocké sans borne ; on prend sa partie fractionnaire à chaque
      passe avant. Le cycle est donc réellement périodique : sortir par la
      dernière fonction ramène à la première.
    - Pour la lisibilité, toutes les fonctions sont évaluées puis pondérées
      (au plus deux poids non nuls par neurone). Une version optimisée
      n'évaluerait que les deux voisines ; voir `harden()` pour l'inférence.
    """

    def __init__(self, num_features, functions=DEFAULT_CYCLE, lengths=None,
                 feature_dim=1, init="uniform"):
        super().__init__()
        self.names = [f if isinstance(f, str) else getattr(f, "__name__", "f")
                      for f in functions]
        self.fns = _resolve(functions)
        n = len(self.fns)
        if n < 2:
            raise ValueError("Il faut au moins deux fonctions dans le cycle.")
        if lengths is None:
            lengths = [1.0] * n
        if len(lengths) != n:
            raise ValueError("lengths doit avoir autant d'éléments que functions.")
        lengths = torch.tensor([abs(float(l)) for l in lengths])
        lengths = lengths / lengths.sum()
        starts = torch.cat([torch.zeros(1), torch.cumsum(lengths, 0)[:-1]])
        self.register_buffer("lengths", lengths)
        self.register_buffer("starts", starts)
        self.n_fn = n
        self.num_features = num_features
        self.feature_dim = feature_dim

        if init == "uniform":
            theta = torch.rand(num_features)
        elif isinstance(init, str) and init.startswith("node:"):
            k = int(init.split(":")[1]) % n
            theta = torch.full((num_features,), float(starts[k]))
        else:
            theta = torch.full((num_features,), float(init))
        self.theta = nn.Parameter(theta)

    # -- poids de mélange --------------------------------------------------
    def mixing_weights(self):
        """Matrice (num_features, n_fn) : au plus deux poids non nuls par ligne."""
        p = torch.remainder(self.theta, 1.0)
        # segment k tel que starts[k] <= p < starts[k+1]
        k = torch.bucketize(p.detach(), self.starts, right=True) - 1
        k = k.clamp(0, self.n_fn - 1)
        t = (p - self.starts[k]) / self.lengths[k]           # dépend de θ → gradient
        w = torch.zeros(self.num_features, self.n_fn, device=p.device, dtype=p.dtype)
        w = w.scatter(1, k.unsqueeze(1), (1 - t).unsqueeze(1))
        w = w.scatter_add(1, ((k + 1) % self.n_fn).unsqueeze(1), t.unsqueeze(1))
        return w

    def forward(self, x):
        w = self.mixing_weights().to(x.dtype)                 # (F, n)
        shape = _broadcast_shape(x, self.feature_dim, self.num_features)
        out = 0
        for j, f in enumerate(self.fns):
            out = out + w[:, j].view(shape) * f(x)
        return out

    # -- outils ------------------------------------------------------------
    @torch.no_grad()
    def dominant_function(self):
        """Nom de la fonction de poids maximal pour chaque neurone."""
        idx = self.mixing_weights().argmax(1)
        return [self.names[i] for i in idx.tolist()]

    @torch.no_grad()
    def harden(self):
        """Place chaque θ sur la fonction la plus proche (fin d'entraînement)."""
        w = self.mixing_weights()
        k = w.argmax(1)
        self.theta.copy_(self.starts[k])

    def extra_repr(self):
        return f"{self.num_features}, cycle={'→'.join(self.names)}→{self.names[0]}"


# ---------------------------------------------------------------------------
# Point de comparaison : Adaptive Blending Units (Sütfeld et al., 2018)
# ---------------------------------------------------------------------------
class ABUActivation(nn.Module):
    """Combinaison linéaire libre de n fonctions, n coefficients par neurone,
    initialisés à 1/n (comme dans l'article)."""

    def __init__(self, num_features, functions=DEFAULT_CYCLE, feature_dim=1):
        super().__init__()
        self.names = list(functions)
        self.fns = _resolve(functions)
        self.num_features = num_features
        self.feature_dim = feature_dim
        n = len(self.fns)
        self.alpha = nn.Parameter(torch.full((num_features, n), 1.0 / n))

    def forward(self, x):
        shape = _broadcast_shape(x, self.feature_dim, self.num_features)
        out = 0
        for j, f in enumerate(self.fns):
            out = out + self.alpha[:, j].view(shape) * f(x)
        return out


# ---------------------------------------------------------------------------
# Fabrique d'activations utilisée par les expériences
# ---------------------------------------------------------------------------
ACTIVATION_CHOICES = ("relu", "prelu", "silu", "tanh", "abu",
                      "gear", "gear_frozen", "gear_window")


def make_activation(kind, num_features, functions=DEFAULT_CYCLE, feature_dim=1,
                    gear_init="uniform"):
    """
    kind :
      relu / silu / tanh  : activation fixe
      prelu               : PReLU par neurone/canal
      abu                 : Adaptive Blending Units
      gear                : engrenages entraînés tout le long
      gear_window         : engrenages entraînés seulement dans une fenêtre
                            (la fenêtre est gérée par GearSchedule)
      gear_frozen         : engrenages initialisés au hasard puis figés
                            -> contrôle : le gain vient-il de l'entraînement
                               de θ ou seulement de la diversité des fonctions ?
    """
    if kind == "relu":
        return nn.ReLU()
    if kind == "silu":
        return nn.SiLU()
    if kind == "tanh":
        return nn.Tanh()
    if kind == "prelu":
        return nn.PReLU(num_features)
    if kind == "abu":
        return ABUActivation(num_features, functions, feature_dim)
    if kind in ("gear", "gear_window", "gear_frozen"):
        g = GearActivation(num_features, functions, feature_dim=feature_dim, init=gear_init)
        if kind == "gear_frozen":
            g.theta.requires_grad_(False)
        return g
    raise ValueError(f"Activation inconnue : {kind}. Choix : {ACTIVATION_CHOICES}")


def gear_modules(model):
    return [m for m in model.modules() if isinstance(m, GearActivation)]


def split_params(model):
    """Sépare les θ des autres paramètres (lr et weight decay différents)."""
    theta_ids = {id(m.theta) for m in gear_modules(model)}
    thetas = [p for p in model.parameters() if id(p) in theta_ids and p.requires_grad]
    others = [p for p in model.parameters() if id(p) not in theta_ids and p.requires_grad]
    return others, thetas


class GearSchedule:
    """Active l'entraînement des θ uniquement pendant [start, end] (en époques
    ou en itérations, selon ce qu'on passe à step). end=None : jusqu'à la fin.
    Reproduit l'option interval_gear de main.py."""

    def __init__(self, model, start=0, end=None):
        self.gears = [g for g in gear_modules(model) if g.theta.requires_grad]
        self.start, self.end = start, end

    def active(self, t):
        return t >= self.start and (self.end is None or t <= self.end)

    def step(self, t):
        on = self.active(t)
        for g in self.gears:
            g.theta.requires_grad_(on)
            if not on:
                g.theta.grad = None
        return on
      
@torch.no_grad()
def snapshot_thetas(model, raw=False):
    """Concatène tous les θ du modèle, pour le suivi de convergence.
    raw=False : θ mod 1 (position sur le cycle) ; raw=True : θ non replié,
    pratique pour tracer des trajectoires continues."""
    gs = gear_modules(model)
    if not gs:
        return torch.empty(0)
    if raw:
        return torch.cat([g.theta.detach().flatten().cpu() for g in gs])
    return torch.cat([torch.remainder(g.theta, 1.0).flatten().cpu() for g in gs])

