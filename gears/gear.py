# -*- coding: utf-8 -*-
"""
Activations à engrenages (« gear activations »).

Chaque neurone (ou canal, pour une convolution) possède un angle θ ∈ [0, 1)
qui indique sa position sur un cycle fermé de fonctions d'activation
f_0 → f_1 → ... → f_{n-1} → f_0. Entre deux fonctions voisines, l'activation
est une interpolation linéaire :

    p = θ mod 1,   p dans le segment k,   t = (p - début_k) / longueur_k
    f_θ(z) = (1 - t) · f_k(z) + t · f_{k+1}(z)

L'autograd de PyTorch calcule le vrai gradient par rapport à θ
(∂f/∂θ = (f_{k+1} - f_k) / longueur_k).

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
                   (poids relatifs des segments). Par défaut : égales.
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

    is_gear = True

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
                      "gear", "gear_frozen", "gear_window",
                      # mêmes 6 fonctions que la sphère (voir gear3d.py)
                      "abu6", "gear6", "sphere", "sphere_frozen", "sphere_kernel",
                      # départ exactement sur ReLU (identique à la référence à l'itération 0)
                      "gear_relu_init", "sphere_relu_init",
                      # départ sur ReLU + alternance figé / entraîné (voir GearCycleSchedule,
                      # GearPlateauSchedule ; le calendrier est créé par le script d'expérience)
                      "gear_alt", "gear_alt_plateau")


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
    Départ sur ReLU :
      gear_relu_init      : tous les θ placés sur ReLU (premier élément du
                            cycle, qui doit être « relu »), puis entraînés.
                            Le réseau démarre identique à la référence ReLU et
                            ne s'en écarte que si le gradient le juge utile.
      sphere_relu_init    : même chose pour la sphère (sommet « relu »).
      gear_alt            : départ sur ReLU, puis alternance fixe θ figés /
                            θ entraînés (GearCycleSchedule)
      gear_alt_plateau    : départ sur ReLU, alternance déclenchée par la
                            stabilisation de la perte d'entraînement
                            (GearPlateauSchedule)
    Variantes à 6 fonctions (catalogue DEFAULT_SPHERE, l'argument functions
    est ignoré) :
      abu6 / gear6        : ABU et engrenage cercle sur les 6 fonctions
      sphere              : engrenage sphérique (octaèdre, barycentrique)
      sphere_frozen       : directions aléatoires figées (contrôle)
      sphere_kernel       : engrenage sphérique à noyau (lisse)
    """
    if kind in ("gear_relu_init", "gear_alt", "gear_alt_plateau"):
        if not functions or functions[0] != "relu":
            raise ValueError(f"{kind} : le cycle doit commencer par « relu ».")
        return GearActivation(num_features, functions, feature_dim=feature_dim, init="node:0")
    if kind in ("abu6", "gear6", "sphere", "sphere_frozen", "sphere_kernel", "sphere_relu_init"):
        from .gear3d import DEFAULT_SPHERE, SphereGearActivation
        if kind == "abu6":
            return ABUActivation(num_features, DEFAULT_SPHERE, feature_dim)
        if kind == "gear6":
            return GearActivation(num_features, DEFAULT_SPHERE, feature_dim=feature_dim, init=gear_init)
        mode = "kernel" if kind == "sphere_kernel" else "barycentric"
        init = "node:0" if kind == "sphere_relu_init" else "uniform"       # sommet 0 = relu
        g = SphereGearActivation(num_features, DEFAULT_SPHERE, mode=mode, feature_dim=feature_dim, init=init)
        if kind == "sphere_frozen":
            g.theta.requires_grad_(False)
        return g
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
    """Engrenages cercle uniquement (θ scalaire par neurone)."""
    return [m for m in model.modules() if isinstance(m, GearActivation)]


def position_modules(model):
    """Tous les engrenages, cercle et sphère (attribut `theta` = position)."""
    return [m for m in model.modules() if getattr(m, "is_gear", False)]


def split_params(model):
    """Sépare les θ (cercle et sphère) des autres paramètres (lr et weight decay différents)."""
    theta_ids = {id(m.theta) for m in position_modules(model)}
    thetas = [p for p in model.parameters() if id(p) in theta_ids and p.requires_grad]
    others = [p for p in model.parameters() if id(p) not in theta_ids and p.requires_grad]
    return others, thetas


class GearSchedule:
    """Active l'entraînement des θ uniquement pendant [start, end] (en époques
    ou en itérations, selon ce qu'on passe à step). end=None : jusqu'à la fin.

    Tous les calendriers ont la même interface : `step(t, loss=None)` est
    appelé au début de l'unité t (époque ou bloc d'itérations), avec la perte
    d'entraînement de l'unité précédente ; il renvoie True si les θ sont
    entraînés pendant l'unité t. Les poids, eux, sont toujours entraînés."""

    def __init__(self, model, start=0, end=None):
        self.gears = [g for g in position_modules(model) if g.theta.requires_grad]
        self.start, self.end = start, end

    def active(self, t):
        return t >= self.start and (self.end is None or t <= self.end)

    def _apply(self, on):
        for g in self.gears:
            g.theta.requires_grad_(on)
            if not on:
                g.theta.grad = None
        return on

    def step(self, t, loss=None):
        return self._apply(self.active(t))

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


class GearCycleSchedule(GearSchedule):
    """Alternance fixe : θ figés pendant `off` unités, puis entraînés pendant
    `on` unités, et ainsi de suite. On commence figé. Avec des θ placés sur
    ReLU au départ (gear_alt), le réseau s'entraîne d'abord comme un réseau
    ReLU, puis les engrenages s'ouvrent par intervalles."""

    def __init__(self, model, off, on, start=0):
        super().__init__(model, start, None)
        if off < 0 or on <= 0:
            raise ValueError("off >= 0 et on > 0")
        self.off, self.on = off, on

    def active(self, t):
        return t >= self.start and (t - self.start) % (self.off + self.on) >= self.off


class GearPlateauSchedule(GearSchedule):
    """Alternance déclenchée par la stabilisation.

    Phase « off » (θ figés) : on attend que la perte d'entraînement se
      stabilise, c'est-à-dire que sa baisse relative moyenne sur les
      `patience` dernières unités passe sous `min_delta`. On ouvre alors.
    Phase « on » (θ entraînés) : au moins `min_on` unités, au plus `max_on`.
      On referme quand les θ ne bougent presque plus : déplacement moyen sur
      la dernière unité < `theta_tol`.
    Arrêt : si, sur toute une phase « on », les θ ont bougé de moins de
      `stop_tol` en moyenne, on ne rouvre plus jamais (« plus d'évolution »).

    Seule la perte d'ENTRAÎNEMENT est utilisée : jamais la validation ni le
    test. Le déplacement des θ se mesure en tours (cercle) ou en norme du
    vecteur u (sphère), moyenné sur tous les neurones.
    """

    def __init__(self, model, patience=2, min_delta=0.05, theta_tol=2e-3, stop_tol=5e-3,
                 min_on=1, max_on=None):
        super().__init__(model, 0, None)
        self.patience, self.min_delta = patience, min_delta
        self.theta_tol, self.stop_tol = theta_tol, stop_tol
        self.min_on, self.max_on = min_on, max_on
        self.phase = "off"
        self.losses = []
        self.n_on = 0
        self.snap0 = self.prev = None
        self.cycles = []          # déplacement total de chaque phase « on »
        self.events = []          # (unité, nouvelle phase)

    def _thetas(self):
        return torch.cat([g.theta.detach().flatten().cpu() for g in self.gears]).clone()

    def _switch(self, t, phase):
        self.phase = phase
        self.events.append((t, phase))
        self.losses = []
        if phase == "on":
            self.snap0 = self.prev = self._thetas()
            self.n_on = 0

    def step(self, t, loss=None):
        if loss is not None and self.phase != "done":
            self.losses.append(float(loss))
        if self.phase == "off" and len(self.losses) > self.patience:
            a, b = self.losses[-1 - self.patience], self.losses[-1]
            rate = (a - b) / max(abs(a), 1e-12) / self.patience     # baisse relative par unité
            if rate < self.min_delta:
                self._switch(t, "on")
        elif self.phase == "on" and t > 0:
            self.n_on += 1
            cur = self._thetas()
            move = (cur - self.prev).abs().mean().item()
            self.prev = cur
            closing = self.n_on >= self.min_on and (
                move < self.theta_tol or (self.max_on is not None and self.n_on >= self.max_on))
            if closing:
                total = (cur - self.snap0).abs().mean().item()
                self.cycles.append(total)
                self._switch(t, "done" if total < self.stop_tol else "off")
        return self._apply(self.phase == "on")

    def state_dict(self):
        return {"phase": self.phase, "losses": self.losses, "n_on": self.n_on,
                "snap0": self.snap0, "prev": self.prev, "cycles": self.cycles, "events": self.events}

    def load_state_dict(self, state):
        for k, v in state.items():
            setattr(self, k, v)


def make_schedule(kind, model, window=None, alt=None, plateau=None):
    """Calendrier d'entraînement des θ associé à une variante (None si θ toujours entraînés).
    window = (début, fin) ; alt = (off, on) ; plateau = dict d'options de GearPlateauSchedule."""
    if kind == "gear_window":
        return GearSchedule(model, *window)
    if kind == "gear_alt":
        return GearCycleSchedule(model, *alt)
    if kind == "gear_alt_plateau":
        return GearPlateauSchedule(model, **(plateau or {}))
    return None


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
