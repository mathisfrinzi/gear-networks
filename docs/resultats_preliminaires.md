# Résultats préliminaires (octobre 2026)

À prendre comme des tests de bon fonctionnement, pas comme des conclusions.

## Exp. 1 — jouets (5 initialisations des poids, `experiments/exp1_toy.py`)

| Tâche | fixed | gear | gear_window |
|---|---|---|---|
| ET (jeu de main.py) | 1.6e-5 | 6.7e-6 | 4.4e-6 |
| XOR | 4.0e-5 | 1.6e-5 | 1.2e-5 |
| somme bruitée (main.py) | 0.0103 | 0.0103 | 0.0103 |
| vague sin(2πx₁)·x₂ | 0.0120 | 0.0088 | 0.0112 |

MSE finale médiane. Les engrenages accélèrent un peu la convergence sur les tâches non linéaires ; sur la somme bruitée, tout le monde atteint le plancher du bruit (σ² = 0,01).

**Convergence des θ** : avec les mêmes θ de départ, les θ finaux **dépendent fortement de l'initialisation des poids**. Il n'y a pas une configuration « optimale » unique vers laquelle tout converge. Les θ ont aussi tendance à **se coller sur une fonction pure** (relu, tanh...) et à y rester : c'est l'effet attendu d'un gradient constant par morceaux, qui change de valeur aux jonctions.

## Exp. 2 — MLP Fashion-MNIST (3 graines, 20 époques, `experiments/exp2_mlp.py --dataset fashion --seeds 3 --epochs 20 --gear-window 3 10`)

| Activation | Test (%) | Meilleure val. (%) | Époques → 88 % | s/époque |
|---|---|---|---|---|
| silu | 89.29 ± 0.26 | 89.43 | 3.3 | 0.9 |
| relu | 89.01 ± 0.33 | 89.27 | 3.7 | 0.9 |
| gear_frozen | 88.94 ± 0.25 | 89.15 | 4.0 | 1.4 |
| gear | 88.91 ± 0.13 | 89.41 | 3.7 | 1.7 |
| prelu | 88.81 ± 0.14 | 89.48 | 5.7 | 1.1 |
| abu | 88.74 ± 0.30 | 89.33 | 8.0 | 1.5 |
| gear_window | 88.70 ± 0.18 | 89.21 | 4.0 | 1.5 |

- **Aucune différence significative** : tous les écarts tiennent dans ~0,5 point, de l'ordre de l'écart-type entre graines. Sur un MLP simple, l'activation compte peu ; c'est attendu.
- `gear` converge aussi vite que relu, et plus vite qu'ABU (3,7 contre 8 époques pour 88 %), avec 4 fois moins de paramètres d'activation qu'ABU.
- **Constat qualitatif robuste** (identique sur les 3 graines) : le réseau **abandonne presque totalement l'identité**. Environ 25 % des neurones y démarrent, et 0 à 4 neurones sur 512 y restent. Les θ se déplacent vers silu et tanh. Les θ apprennent donc quelque chose de cohérent à l'échelle de la population, même si chaque neurone suit sa propre trajectoire.

## Exp. 3 — ResNet-20 CIFAR-10 (en cours, Colab, GPU T4)

Architecture à 269 722 paramètres, conforme aux 0,27 M de l'article. Protocole raccourci à 60 époques (lr divisé par 10 aux époques 30 et 45).

| Activation | Graine 0 | Durée |
|---|---|---|
| relu | 90,41 % | ≈ 25 min (22 s/époque) |
| gear_frozen | 89,71 % | 29,5 min (28 s/époque) |
| gear | en cours | |

La référence de l'article (91,25 %) correspond à 182 époques ; perdre un peu moins d'un point à 60 époques est attendu. Avec une seule graine, l'écart relu / gear_frozen (0,7 point) reste indicatif.

## Exp. 4 — Collage aux nœuds (`diagnostics/diag_nodes.py`, tâche wave, 3 graines)

MLP 2 → 64 → 64 → 1, Adam, 3 000 itérations. Niveau de collage dû au hasard : 16 % (θ à moins de 0,02 d'un nœud).

| lr_theta | MSE validation | θ collés | Déplacement moyen des θ |
|---|---|---|---|
| figés | 0,00409 ± 0,00053 | 16 % | 0 |
| 1e-4 | 0,00377 ± 0,00058 | 26 % | 0,03 tour |
| 1e-3 | 0,00338 ± 0,00022 | 68 % | 0,11 tour |
| 1e-2 | 0,00345 ± 0,00019 | 41 % | 0,23 tour |
| 1e-1 | 0,01358 ± 0,00455 | 34 % | 2,27 tours |

Forme de la perte autour des 156 θ collés (lr_theta = 1e-2) : 55 % en pointe (minimum sur le nœud), 30 % pente qui traverse le nœud, 15 % maximum local.

Après un écart de ±0,03 et 500 itérations : 69 % reviennent sur leur nœud, 31 % restent ailleurs. Sans perturbation, 81 % restent collés et 19 % se décollent d'eux-mêmes.

**Lecture** : le collage est surtout un effet de sélection (la fonction pure est réellement optimale pour ce neurone), pas un blocage permanent. Les θ se collent et se décollent au cours de l'entraînement. Un lr_theta trop faible fige les θ : point à vérifier sur le ResNet, où les θ sont entraînés par SGD.

## Figures

- Tâche jouet « wave » : [perte](../assets/toy_wave_loss.png), [trajectoires de θ](../assets/toy_wave_theta.png)
- MLP Fashion-MNIST : [courbes d'apprentissage](../assets/mlp_fashion_curves.png), [trajectoires de θ](../assets/mlp_fashion_theta.png)
- Collage aux nœuds : [dynamique](../assets/diag_dynamics.png), [profils de perte](../assets/diag_profiles.png), [balayage de lr_theta](../assets/diag_sweep_lr_theta.png)
