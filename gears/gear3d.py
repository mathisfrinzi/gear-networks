# -*- coding: utf-8 -*-
"""
Engrenages sphériques : chaque neurone se déplace sur une sphère de fonctions.

Avec les engrenages « cercle » (gear.py), chaque neurone a un angle θ et ne
peut mélanger que deux fonctions voisines sur un cycle dont l'ordre est
arbitraire. Ici, chaque neurone porte une direction u ∈ S² (2 degrés de
liberté). Les n fonctions sont placées aux sommets d'un polyèdre inscrit dans
la sphère (tétraèdre : 4, octaèdre : 6, icosaèdre : 12). Chaque fonction a
alors plusieurs voisines.

Deux façons de transformer la direction u en poids de mélange :

  « barycentrique » (par défaut) — la généralisation directe du cercle
      Le rayon issu du centre dans la direction u traverse une face
      triangulaire (i, j, k) du polyèdre. Les poids sont les coordonnées
      barycentriques du point de traversée :
          u = λ_i p_i + λ_j p_j + λ_k p_k,   λ ≥ 0,   w = λ / Σλ
      Au plus 3 fonctions actives ; exactement 1 sur un sommet, 2 sur une
      arête. Les poids sont linéaires par morceaux, comme sur le cercle : on
      retrouve donc le même effet de sélection (collage) aux sommets et aux
      arêtes.

  « noyau » — variante lisse
      w_i = softmax_i(κ · ⟨û, p_i⟩), û = u / |u|.
      Toutes les fonctions sont actives (poids jamais exactement nuls) ; pas
      de point anguleux, donc pas de collage. κ règle la netteté.

Paramétrage : u est stocké comme un vecteur 3D libre (paramètre `theta`, de
forme (num_features, 3)) et seule sa direction compte. Pas de singularité aux
pôles, contrairement à un paramétrage (latitude, longitude). Les poids
barycentriques sont invariants par changement d'échelle de u, donc le gradient
est orthogonal à u : la norme de u ne change qu'au second ordre. Avec Adam,
elle grossit lentement (1,2 en moyenne après 3 000 itérations sur les tâches
jouets), ce qui réduit un peu le pas angulaire ; `renormalize()` la ramène à 1.

Disposition par défaut (octaèdre) : l'équateur reprend le cycle du cercle
(relu → silu → tanh → identity), et les deux pôles ajoutent elu et sin.
Un engrenage sphérique qui reste sur l'équateur se comporte donc comme un
engrenage cercle (à la paramétrisation de la vitesse près).
"""
import itertools

import torch
import torch.nn as nn

from .gear import _broadcast_shape, _resolve

DEFAULT_SPHERE = ("relu", "silu", "tanh", "identity", "elu", "sin")


# ---------------------------------------------------------------------------
# Polyèdres
# ---------------------------------------------------------------------------
def layout_positions(layout):
    """Sommets (n, 3), normalisés, d'un polyèdre régulier."""
    if layout == "tetra":
        p = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    elif layout == "octa":
        # équateur dans l'ordre du cycle, puis pôle nord, pôle sud
        p = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    elif layout == "icosa":
        phi = (1 + 5 ** 0.5) / 2
        p = []
        for a, b in itertools.product((-1, 1), (-phi, phi)):
            p += [(0, a, b), (a, b, 0), (b, 0, a)]
    else:
        raise ValueError(f"Disposition inconnue : {layout}. Choix : tetra, octa, icosa")
    p = torch.tensor(p, dtype=torch.float64)
    return p / p.norm(dim=1, keepdim=True)


LAYOUT_SIZES = {"tetra": 4, "octa": 6, "icosa": 12}


def hull_faces(points, tol=1e-7):
    """Faces triangulaires de l'enveloppe convexe de points sur la sphère.

    Force brute en O(n⁴), suffisant pour n ≤ 20. Chaque face est orientée vers
    l'extérieur. L'enveloppe doit contenir le centre (sinon certaines
    directions ne traverseraient aucune face).
    """
    P = points.double()
    n = len(P)
    faces = []
    for i, j, k in itertools.combinations(range(n), 3):
        nrm = torch.linalg.cross(P[j] - P[i], P[k] - P[i])
        if nrm.norm() < 1e-9:
            continue
        side = (P - P[i]) @ nrm
        if (side <= tol).all():
            faces.append((i, j, k))
        elif (side >= -tol).all():
            faces.append((i, k, j))
    for (i, j, k) in faces:
        nrm = torch.linalg.cross(P[j] - P[i], P[k] - P[i])
        if float(nrm @ P[i]) <= tol:
            raise ValueError("L'enveloppe des fonctions ne contient pas le centre de la sphère.")
    if not faces:
        raise ValueError("Impossible de trianguler ces positions.")
    return faces


# ---------------------------------------------------------------------------
# Activation sphérique
# ---------------------------------------------------------------------------
class SphereGearActivation(nn.Module):
    """
    Paramètres
    ----------
    num_features : nombre de neurones (MLP) ou de canaux (CNN).
    functions    : noms ou callables, un par sommet du polyèdre.
    layout       : 'tetra' (4 fonctions), 'octa' (6), 'icosa' (12), ou un
                   tenseur (n, 3) de positions libres. Par défaut : déduit du
                   nombre de fonctions.
    mode         : 'barycentric' (au plus 3 fonctions actives) ou 'kernel'.
    kappa        : netteté du noyau (mode 'kernel').
    feature_dim  : dimension de l'entrée qui porte les neurones/canaux.
    init         : 'uniform' (direction aléatoire), 'node:<k>' (sur la
                   fonction k), ou un tenseur (3,) / (num_features, 3).
    """

    is_gear = True

    def __init__(self, num_features, functions=DEFAULT_SPHERE, layout=None, mode="barycentric",
                 kappa=4.0, feature_dim=1, init="uniform"):
        super().__init__()
        self.names = [f if isinstance(f, str) else getattr(f, "__name__", "f") for f in functions]
        self.fns = _resolve(functions)
        n = len(self.fns)
        if layout is None:
            layout = {v: k for k, v in LAYOUT_SIZES.items()}.get(n)
            if layout is None:
                raise ValueError(f"Pas de disposition par défaut pour {n} fonctions : "
                                 "passez layout (tenseur de positions).")
        if isinstance(layout, str):
            if LAYOUT_SIZES[layout] != n:
                raise ValueError(f"'{layout}' attend {LAYOUT_SIZES[layout]} fonctions, pas {n}.")
            pos = layout_positions(layout)
            self.layout = layout
        else:
            pos = torch.as_tensor(layout, dtype=torch.float64)
            pos = pos / pos.norm(dim=1, keepdim=True)
            if pos.shape != (n, 3):
                raise ValueError("layout doit être de forme (n_fonctions, 3).")
            self.layout = "custom"
        if mode not in ("barycentric", "kernel"):
            raise ValueError("mode : 'barycentric' ou 'kernel'")

        faces = hull_faces(pos)
        M = torch.stack([pos[list(f)].T for f in faces])          # (F, 3, 3), colonnes = sommets
        self.register_buffer("positions", pos.float())
        self.register_buffer("faces", torch.tensor(faces, dtype=torch.long))
        self.register_buffer("face_inv", torch.linalg.inv(M).float())
        self.n_fn = n
        self.mode = mode
        self.kappa = float(kappa)
        self.num_features = num_features
        self.feature_dim = feature_dim

        if isinstance(init, str) and init == "uniform":
            u = torch.randn(num_features, 3)
        elif isinstance(init, str) and init.startswith("node:"):
            u = self.positions[int(init.split(":")[1]) % n].repeat(num_features, 1)
        else:
            u = torch.as_tensor(init, dtype=torch.float32).expand(num_features, 3).clone()
        self.theta = nn.Parameter(u / u.norm(dim=1, keepdim=True))

    # -- poids de mélange --------------------------------------------------
    def directions(self):
        return self.theta / self.theta.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def mixing_weights(self):
        """Matrice (num_features, n_fn) ; lignes de somme 1."""
        # calcul en float32 même sous précision mixte (AMP) : le choix de la face
        # et les coordonnées barycentriques doivent rester précis
        with torch.autocast(self.theta.device.type, enabled=False):
            return self._mixing_weights()

    def _mixing_weights(self):
        if self.mode == "kernel":
            return torch.softmax(self.kappa * self.directions() @ self.positions.T, dim=1)
        u = self.theta
        lam = torch.einsum("fab,nb->nfa", self.face_inv, u)          # (N, F, 3)
        face = lam.detach().min(2).values.argmax(1)                   # face traversée par le rayon
        lam = lam[torch.arange(len(u), device=u.device), face]        # (N, 3), garde le gradient
        # valeur écrêtée à 0 (bruit numérique sur les arêtes), mais gradient conservé :
        # un clamp_min ordinaire coupe le gradient exactement sur un sommet, où deux
        # des trois coordonnées valent 0, et le neurone ne pourrait plus en repartir
        lam = lam + (lam.clamp_min(0) - lam).detach()
        w_local = lam / lam.sum(1, keepdim=True).clamp_min(1e-12)
        w = torch.zeros(len(u), self.n_fn, device=u.device, dtype=u.dtype)
        return w.scatter_add(1, self.faces[face], w_local)

    def forward(self, x):
        w = self.mixing_weights().to(x.dtype)
        shape = _broadcast_shape(x, self.feature_dim, self.num_features)
        out = 0
        for j, f in enumerate(self.fns):
            out = out + w[:, j].view(shape) * f(x)
        return out

    # -- outils ------------------------------------------------------------
    @torch.no_grad()
    def dominant_function(self):
        idx = self.mixing_weights().argmax(1)
        return [self.names[i] for i in idx.tolist()]

    @torch.no_grad()
    def n_active(self, tol=0.02):
        """Nombre de fonctions de poids > tol pour chaque neurone :
        1 = sur un sommet, 2 = sur une arête, 3 = dans une face (mode barycentrique)."""
        return (self.mixing_weights() > tol).sum(1)

    @torch.no_grad()
    def harden(self):
        """Place chaque neurone sur sa fonction dominante (fin d'entraînement)."""
        self.theta.copy_(self.positions[self.mixing_weights().argmax(1)])

    @torch.no_grad()
    def renormalize(self):
        """Ramène |u| à 1 (facultatif : seule la direction compte)."""
        self.theta.copy_(self.directions())

    def extra_repr(self):
        return (f"{self.num_features}, {self.layout} : {', '.join(self.names)}, mode={self.mode}"
                + (f", kappa={self.kappa}" if self.mode == "kernel" else ""))


def sphere_modules(model):
    return [m for m in model.modules() if isinstance(m, SphereGearActivation)]


@torch.no_grad()
def snapshot_directions(model):
    """Directions unitaires de tous les neurones sphériques, concaténées (N, 3)."""
    ms = sphere_modules(model)
    if not ms:
        return torch.empty(0, 3)
    return torch.cat([m.directions().cpu() for m in ms])


def to_lonlat(u):
    """Directions (N, 3) -> longitude, latitude en degrés (pour les cartes)."""
    u = u / u.norm(dim=-1, keepdim=True)
    lon = torch.rad2deg(torch.atan2(u[..., 1], u[..., 0]))
    lat = torch.rad2deg(torch.asin(u[..., 2].clamp(-1, 1)))
    return lon, lat
