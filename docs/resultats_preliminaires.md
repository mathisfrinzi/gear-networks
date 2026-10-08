# Résultats détaillés (octobre 2026)

À prendre comme des tests de bon fonctionnement, pas comme des conclusions.

## Exp. 1 — jouets (5 initialisations des poids, `experiments/exp1_toy.py`)

| Tâche | fixed | gear | gear_window |
|---|---|---|---|
| ET logique | 1.6e-5 | 6.7e-6 | 4.4e-6 |
| XOR | 4.0e-5 | 1.6e-5 | 1.2e-5 |
| somme bruitée y = x₁ + x₂ | 0.0103 | 0.0103 | 0.0103 |
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

## Exp. 3 — ResNet-20 CIFAR-10 (Colab, GPU T4, 3 graines)

Architecture à 269 722 paramètres, conforme aux 0,27 M de l'article. Protocole raccourci à 60 époques (lr divisé par 10 aux époques 30 et 45).

| Activation | Graine 0 | Graine 1 | Graine 2 | Moyenne ± écart-type | Durée |
|---|---|---|---|---|---|
| relu | 90,41 % | 90,54 % | 90,69 % | 90,55 ± 0,14 | 21 s/époque |
| gear | 90,18 % | 90,27 % | 90,33 % | 90,26 ± 0,08 | 32 s/époque |
| gear_frozen | 89,71 % | 89,65 % | 90,19 % | 89,85 ± 0,30 | 27 s/époque |

La référence de l'article (91,25 %) correspond à 182 époques ; perdre un peu moins d'un point à 60 époques est attendu.

Écarts appariés (même graine) :

| | Graine 0 | Graine 1 | Graine 2 | Moyenne ± écart-type |
|---|---|---|---|---|
| gear − relu | −0,23 | −0,27 | −0,36 | −0,29 ± 0,07 |
| gear − gear_frozen | +0,47 | +0,62 | +0,14 | +0,41 ± 0,25 |
| gear_frozen − relu | −0,70 | −0,89 | −0,50 | −0,70 ± 0,20 |

- **Cas « gear > gear_frozen mais ≤ relu »** de la feuille de route : les θ corrigent en partie les mauvais choix initiaux, sans battre la référence. gear est sous relu dans les 3 graines. L'avance sur gear_frozen est de même signe dans les 3 graines mais fragile (+0,14 dans la graine 2, écart-type 0,25 sur 3 graines).
- Les θ ont bougé (0,12, 0,07 et 0,11 tour en moyenne sur les neurones suivis) : ce n'est donc pas un `lr_theta` trop faible.
- Perte d'entraînement finale (moyenne) : relu 0,104, gear 0,116, gear_frozen 0,140 ; perte de test : relu 0,319, gear 0,330, gear_frozen 0,330. Les engrenages entraînés optimisent moins bien que ReLU, ils ne sur-apprennent pas.
- **Choix selon la position (reproduit sur les 3 graines)** : première activation de chaque bloc résiduel, relu 46 %, 40 %, 39 % des neurones ; seconde activation, 21 %, 21 %, 24 %. Silu, tanh et identité y sont plus fréquents (identité : 8 %, 6 %, 8 % contre 15 %, 19 %, 16 %). Les 336 neurones de chaque position sont comptés.

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

## Exp. 5 — Engrenages sphériques, tâches jouets (`experiments/exp3d_1_toy.py`, 10 graines)

Réseau 2 → 16 → 1, Adam lr 1e-2 (θ et directions : 1e-2), 3 000 itérations en
batch complet. Mêmes poids initiaux pour toutes les variantes à graine égale ;
neurones placés au départ au milieu d'un segment (cercle) ou au centre d'une
face de l'octaèdre (sphère). MSE de validation (XOR : entraînement), médiane
[min, max] ×10⁻³ :

| Variante | XOR | wave | bump | ripple |
|---|---|---|---|---|
| relu | 0,021 | 14,4 [6,6 ; 60] | 3,42 | 293 [261 ; 360] |
| gear (4 fonctions) | 0,008 | 8,8 [7,3 ; 10,1] | 2,85 | 205 [147 ; 271] |
| gear6 | 0,006 | 5,7 [4,7 ; 9,6] | 2,97 | 211 [144 ; 299] |
| sphere | 0,004 | 5,4 [2,6 ; 9,6] | 2,99 | 151 [93 ; 290] |
| sphere_kernel | 0,005 | 4,6 [2,6 ; 6,6] | 2,86 | 113 [70 ; 246] |
| sphere_frozen | 0,008 | 4,6 [3,2 ; 8,8] | 2,83 | 207 [150 ; 307] |

Comparaisons appariées (même graine) :

| | XOR | wave | bump | ripple |
|---|---|---|---|---|
| sphere meilleur que sphere_frozen | 9/10 | 3/10 | 4/10 | 9/10 |
| sphere_kernel meilleur que sphere_frozen | 9/10 | 4/10 | 5/10 | 10/10 |
| sphere meilleur que gear6 | 9/10 | 5/10 | 7/10 | 9/10 |
| gear6 meilleur que gear | 8/10 | 10/10 | 3/10 | 5/10 |

Position finale des neurones (sphère barycentrique, toutes graines) : sur une
fonction pure 14 à 49 % selon la tâche, sur une arête 22 à 43 %, dans une face
29 à 59 %. Pour le cercle : 44 à 88 % sur un nœud.

**Lecture**
- Entraîner la direction aide sur `ripple` et XOR, pas sur `wave` ni `bump`.
  Sur `wave`, le mélange figé de 3 fonctions suffit.
- Ajouter des fonctions au cercle (gear6 contre gear) aide sur `wave`
  (10/10), pas ailleurs.
- La sphère colle moins aux sommets que le cercle aux nœuds, mais colle aux
  arêtes. Sur `ripple`, l'arête relu–sin est très fréquentée.
- La norme de u grossit un peu avec Adam (1,19 en moyenne, 2,3 au maximum),
  ce qui réduit le pas angulaire. Sans effet visible ici ; à surveiller.

## Exp. 6 — Engrenages sphériques, MLP Fashion-MNIST (`experiments/exp3d_2_mlp.py --seeds 3 --epochs 20`)

Même protocole que l'exp. 2 (lr_theta 1e-3). Les runs relu et gear
reproduisent exactement ceux de l'exp. 2.

| Activation | Test (%) | Époques → 88 % | s/époque | Paramètres |
|---|---|---|---|---|
| relu | 89,01 ± 0,33 | 3,7 | 0,9 | 269 322 |
| gear | 88,91 ± 0,13 | 3,7 | 1,7 | 269 834 |
| gear6 | 88,90 ± 0,03 | 5,7 | 2,0 | 269 834 |
| sphere | 88,90 ± 0,14 | 3,7 | 2,3 | 270 858 |
| abu6 | 88,78 ± 0,29 | 4,3 | 1,8 | 272 394 |
| sphere_kernel | 88,70 ± 0,11 | 5,7 | 2,1 | 270 858 |
| sphere_frozen | 88,55 ± 0,17 | 4,3 | 1,8 | 270 858 |

- Aucune différence significative.
- sphere contre sphere_frozen, graine par graine : +0,37, +0,06, +0,60 point.
  Cohérent avec les jouets, mais 3 graines ne suffisent pas.
- Directions déplacées de 20° en moyenne. Fonction dominante finale : sin
  26 à 31 %, identité 1 à 2 % (17 % au départ). Le cercle à 6 fonctions fait
  pareil (sin 23 à 35 %).

## Exp. 7 — Départ exactement sur ReLU, MLP Fashion-MNIST (`experiments/exp3d_2_mlp.py --acts relu gear gear_relu_init sphere_relu_init`)

Tous les θ (ou toutes les directions) démarrent sur ReLU, puis sont entraînés
comme d'habitude. Le réseau est identique à la référence ReLU à l'itération 0
(test unitaire, écart de sortie 0,0). 3 graines, 20 époques :

| Activation | Test (%) | Écart à relu, par graine |
|---|---|---|
| relu | 89,01 ± 0,33 | — |
| gear_relu_init | 88,94 ± 0,10 | +0,12 / +0,16 / −0,49 |
| gear (θ aléatoires) | 88,91 ± 0,13 | +0,30 / −0,08 / −0,52 |
| sphere_relu_init | 88,90 ± 0,64 | −0,60 / +0,33 / −0,07 |

- Aucune différence significative : sur ce MLP, tout se vaut, donc le test ne
  discrimine pas. Il faut le refaire sur le ResNet (`docs/ROADMAP.md`, 1 bis).
- Partis de ReLU, les neurones du cercle ne vont que vers silu (71 % en
  couche 1, jamais tanh ni identité) : ils ne s'éloignent que de leur voisine
  immédiate. En couche 2, 87 % restent sur ReLU.
- Les directions de la sphère se sont déplacées de 12° en moyenne et ReLU
  reste dominante pour 100 % des neurones.
- Correctif découvert en écrivant le test : la sphère ne pouvait pas repartir
  d'un sommet exact (gradient nul). Les exp. 5 et 6 ont été calculées avant ce
  correctif ; elles ont été **relancées après** et sont identiques : les 240
  runs jouets (4 tâches × 6 variantes × 10 graines, MSE d'entraînement et de
  validation) et les 6 runs MLP `sphere` / `sphere_kernel` (précision de test
  et courbes de validation). Cause : les neurones démarrent à des positions
  aléatoires ou au centre d'une face, et ne tombent jamais exactement sur un
  sommet pendant l'entraînement. Le défaut n'affectait que les départs exacts
  sur un sommet (`sphere_relu_init`, `harden()`), non utilisés avant le
  correctif.

## Exp. 8 — Partir de ReLU et ouvrir les engrenages par intervalles

Idée : laisser le réseau converger comme un réseau ReLU (θ tous sur ReLU,
figés), ouvrir les engrenages, les refiger, et recommencer jusqu'à ce que les
θ ne bougent plus. Deux calendriers :
- `gear_alt` : fixe, 10 unités figées puis 5 ouvertes, en boucle ;
- `gear_alt_plateau` : ouverture quand la perte d'**entraînement** baisse de
  moins de 5 % par unité sur 2 unités ; fermeture quand les θ bougent de moins
  de 0,001 tour par unité ; arrêt définitif si une ouverture entière les fait
  bouger de moins de 0,005 tour.

### Tâches jouets (`experiments/exp4_alternance.py`, 10 graines)

Réseau 2 → 16 → 1, 60 blocs de 50 itérations (comme les 60 époques du
ResNet). MSE de validation médiane [min, max] ×10⁻³ :

| Variante | wave | ripple |
|---|---|---|
| relu | 14,4 [6,6 ; 60,0] | 293 [261 ; 360] |
| gear (θ mélangés au départ, continu) | 8,8 [7,3 ; 10,1] | 205 [147 ; 271] |
| gear_relu_init (départ ReLU, continu) | 8,5 [7,0 ; 12,3] | 230 [166 ; 291] |
| gear_alt (fixe 10/5) | 9,6 [6,7 ; 12,1] | 248 [214 ; 786] |
| gear_alt_plateau | 10,3 [6,7 ; 13,5] | 231 [164 ; 311] |

- **L'alternance n'aide pas** par rapport à l'entraînement continu depuis
  ReLU : gear_alt fait mieux que gear_relu_init sur 5/10 graines (wave) et
  2/10 (ripple) ; gear_alt_plateau sur 5/10 et 5/10.
- Chaque ouverture provoque un saut de la perte, puis une redescente
  (figure `assets/alternance_ripple_loss.png`). Une graine de gear_alt
  diverge sur ripple (786).
- Le calendrier déclenché s'ouvre tôt (blocs 3 à 18) puis se referme
  rarement, car les θ continuent de bouger : il se comporte presque comme
  gear_relu_init (θ ouverts 73 % et 81 % du temps). Arrêt définitif dans 1/10
  et 2/10 runs.
- Partir de ReLU retient les neurones sur ReLU (74 % à 90 % restent dominés
  par ReLU, contre 52 % à 65 % depuis un départ mélangé). Sur ripple, le
  départ mélangé fait un peu mieux (6/10 graines, rapport médian 0,90).

### MLP Fashion-MNIST (`experiments/exp2_mlp.py --acts relu gear gear_relu_init gear_alt gear_alt_plateau`, 3 graines, 20 époques)

Alternance fixe 4 époques figées / 2 ouvertes (6 époques ouvertes sur 20).

| Activation | Test (%) | Écart à relu, par graine |
|---|---|---|
| gear_alt_plateau | 89,26 ± 0,21 | +0,30 / +0,36 / +0,09 |
| gear_alt | 89,25 ± 0,13 | +0,35 / +0,47 / −0,10 |
| relu | 89,01 ± 0,33 | — |
| gear_relu_init | 88,94 ± 0,10 | +0,12 / +0,16 / −0,49 |
| gear | 88,91 ± 0,13 | +0,30 / −0,08 / −0,52 |

- Les deux alternances sont en tête, mais l'écart (+0,25 point) reste sous le
  critère (2 écarts-types) avec 3 graines. Et beaucoup de variantes ont été
  essayées : un écart de cette taille peut sortir du lot par hasard.
- gear_alt_plateau ne s'est ouvert qu'une fois, à l'époque 13, dans les 3
  graines : en pratique, « ReLU puis engrenages sur les 7 dernières époques ».
- Résultat inverse de celui des jouets, où l'alternance n'apporte rien. À
  trancher sur le ResNet (`docs/ROADMAP.md`, 1 ter).

## Pistes ouvertes par ces premiers résultats

1. Tester sur une tâche où l'activation compte davantage (réseaux étroits ou profonds, régression de fonctions, ResNet).
2. Mesurer la perte de précision de `harden()` (chaque neurone figé sur sa fonction dominante) : le diagnostic des nœuds suggère qu'elle devrait être faible.
3. Étudier l'ordre du cycle (`--cycle`) : relu → silu → tanh → identité n'est qu'un choix parmi d'autres.

## Figures

- Tâche jouet « wave » : [perte](../assets/toy_wave_loss.png), [trajectoires de θ](../assets/toy_wave_theta.png)
- MLP Fashion-MNIST : [courbes d'apprentissage](../assets/mlp_fashion_curves.png), [trajectoires de θ](../assets/mlp_fashion_theta.png)
- Alternance : [perte sur ripple](../assets/alternance_ripple_loss.png)
- Sphère, jouets : [trajectoires (ripple)](../assets/sphere_toy_ripple_paths.png), [perte (ripple)](../assets/sphere_toy_ripple_loss.png)
- Sphère, MLP : [courbes](../assets/sphere_mlp_curves.png), [trajectoires](../assets/sphere_mlp_paths.png)
- Collage aux nœuds : [dynamique](../assets/diag_dynamics.png), [profils de perte](../assets/diag_profiles.png), [balayage de lr_theta](../assets/diag_sweep_lr_theta.png)
