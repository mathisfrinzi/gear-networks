# Réseaux de neurones à engrenages

*Gear activations : chaque neurone apprend sa propre fonction d'activation en
tournant sur un cycle de fonctions, avec un seul paramètre.*

> **English summary.** Each neuron carries a single learnable angle θ that
> places it on a closed cycle of activation functions
> (relu → silu → tanh → identity → relu). Between two neighbouring functions,
> the activation is their linear interpolation. Weights, biases **and** θ are
> trained jointly by backpropagation (PyTorch). This is a constrained,
> one-parameter variant of learnable activation mixtures such as Adaptive
> Blending Units. Status: personal research project, preliminary results
> (toy tasks, MLP on Fashion-MNIST, ResNet-20 on CIFAR-10 in progress).

---

## L'idée

![Phases d'entraînement](assets/schema_phases.png)

Dans un réseau classique, on choisit les fonctions d'activation à la main,
puis on n'optimise que les poids. Ici, chaque neurone possède un « engrenage » :
un angle θ ∈ [0, 1) qui le place sur un cycle de fonctions. Si θ tombe entre
les fonctions f_k et f_{k+1} :

```
f_θ(z) = (1 − t)·f_k(z) + t·f_{k+1}(z),   t = position de θ dans le segment k
∂f_θ/∂θ = (f_{k+1}(z) − f_k(z)) / longueur du segment
```

![Propagation](assets/schema_propagation.png)

On optimise les poids w, les biais b **et** les positions θ des engrenages.

## Positionnement par rapport à l'existant

Apprendre un mélange de fonctions d'activation n'est pas nouveau :

- Adaptive Blending Units — Sütfeld et al., 2018 ([arXiv:1806.10064](https://arxiv.org/abs/1806.10064))
- *Learning Combinations of Activation Functions* — Manessi & Rozza, 2018 ([arXiv:1801.09403](https://arxiv.org/abs/1801.09403))
- *Activation Ensembles* — Harmon & Klabjan ([arXiv:1702.07790](https://arxiv.org/abs/1702.07790))
- SmartMixed, 2025 ([arXiv:2510.22450](https://arxiv.org/abs/2510.22450))
- Revue : Apicella et al., *Neural Networks* 138, 2021 ([arXiv:2005.00817](https://arxiv.org/abs/2005.00817))

Ce qui distingue les engrenages :

| | ABU / mélanges libres | Engrenages |
|---|---|---|
| Paramètres d'activation par neurone | n (un par fonction) | **1**, quel que soit n |
| Fonctions actives par neurone | toutes | **au plus 2** |
| Structure | combinaison libre | parcours d'un **cycle** (θ périodique) |
| Entraînement | en continu | possible dans une **fenêtre** d'époques seulement |

## Résultats préliminaires

Ce sont des premiers résultats, sur peu de graines, à prendre comme tels.
Détails : [docs/resultats_preliminaires.md](docs/resultats_preliminaires.md).

### Tâches jouets (5 initialisations des poids)

| Tâche | θ figés | engrenages | engrenages (fenêtre) |
|---|---|---|---|
| ET logique | 1,6e-5 | 6,7e-6 | 4,4e-6 |
| XOR | 4,0e-5 | 1,6e-5 | 1,2e-5 |
| y = x₁ + x₂ + bruit | 0,0103 | 0,0103 | 0,0103 |
| y = sin(2πx₁)·x₂ + bruit | 0,0120 | 0,0088 | 0,0112 |

*MSE finale médiane.* Avec les mêmes θ de départ, les θ finaux dépendent
fortement de l'initialisation des poids : il n'y a pas une configuration
optimale unique.

![Trajectoires de θ](assets/toy_wave_theta.png)

### MLP sur Fashion-MNIST (3 graines, 20 époques)

| Activation | Test (%) | Époques pour 88 % | s / époque (CPU) |
|---|---|---|---|
| SiLU | 89,29 ± 0,26 | 3,3 | 0,9 |
| ReLU | 89,01 ± 0,33 | 3,7 | 0,9 |
| engrenages figés (contrôle) | 88,94 ± 0,25 | 4,0 | 1,4 |
| **engrenages** | 88,91 ± 0,13 | 3,7 | 1,7 |
| PReLU | 88,81 ± 0,14 | 5,7 | 1,1 |
| ABU | 88,74 ± 0,30 | 8,0 | 1,5 |
| engrenages (fenêtre) | 88,70 ± 0,18 | 4,0 | 1,5 |

Aucune différence significative de précision : sur un MLP simple,
l'activation compte peu. Les engrenages convergent aussi vite que ReLU et deux
fois plus vite qu'ABU, avec 4 fois moins de paramètres d'activation. Constat
stable sur les 3 graines : le réseau abandonne presque totalement l'identité
(≈ 25 % des neurones y démarrent, 0 à 4 sur 512 y restent).

### ResNet-20 sur CIFAR-10 (en cours)

Protocole de He et al. (2016), raccourci à 60 époques.

| Activation | Graine 0 |
|---|---|
| ReLU | 90,41 % |
| engrenages figés (contrôle) | 89,71 % |
| engrenages | *en cours* |

Une seule graine pour l'instant : un écart inférieur à ~0,3 point n'est pas
interprétable.

## Lancer le code

```bash
pip install -r requirements.txt
python tests/test_gear.py                  # 12 tests
python tests/test_legacy_equivalence.py    # équivalence avec la version 2025
```

Tous les scripts se lancent aussi depuis VS Code avec le bouton ▶, avec leurs
réglages par défaut.

| Script | Contenu | Matériel |
|---|---|---|
| `exp1_toy.py` | tâches jouets, trajectoires de θ pour 5 initialisations | CPU, ~3 min |
| `exp2_mlp.py` | MLP sur MNIST / Fashion-MNIST, 7 activations, plusieurs graines | CPU possible |
| `exp3_resnet.py` | ResNet-20 sur CIFAR-10 | GPU |
| `run_resnet.py` | enchaîne les runs ResNet (réglages en tête de fichier) | GPU |
| `diag_nodes.py` | diagnostic du collage aux nœuds | CPU, ~3 min |

Pour reproduire le tableau MLP :
`python exp2_mlp.py --dataset fashion --seeds 3 --epochs 20 --gear-window 3 10`.

Pour le ResNet, un GPU est nécessaire (par exemple Google Colab : activez le
GPU, envoyez le dossier, puis `!python run_resnet.py`). Un checkpoint est écrit
à chaque époque, et relancer le script reprend un run interrompu.

### Variantes comparées

| Nom | Description | Rôle |
|---|---|---|
| `relu`, `silu` | activation fixe | référence |
| `prelu` | 1 paramètre par neurone | même nombre de paramètres que les engrenages |
| `abu` | Adaptive Blending Units | méthode existante la plus proche |
| `gear` | engrenages entraînés | la méthode |
| `gear_window` | θ entraînés dans une fenêtre d'époques | « structure linéaire, puis non linéaire » |
| `gear_frozen` | θ aléatoires, jamais entraînés | **contrôle** : le gain vient-il de l'apprentissage des θ ? |

## Structure du dépôt

```
gears/          module GearActivation, ABUActivation, outils
tests/          tests unitaires et équivalence avec la version 2025
exp1_toy.py     exp2_mlp.py     exp3_resnet.py     run_resnet.py     diag_nodes.py
common.py       graines, journaux, tableaux, graphiques
legacy/         version NumPy d'origine (2025) et problèmes identifiés
docs/           résultats préliminaires, feuille de route
assets/         figures du README
```

## Historique

L'idée date d'août 2025 : la première implémentation NumPy est conservée dans
[legacy/](legacy/), avec la liste des problèmes corrigés lors de la réécriture
en PyTorch.





