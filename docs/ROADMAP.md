# Critères de décision et pistes non explorées

*Le projet a été conclu en octobre 2026. Ce document garde les critères fixés
**avant** de voir les résultats, pour éviter de choisir après coup ce qui
arrange, et les tests qui n'ont pas été faits (en particulier sur le ResNet :
`gear_relu_init`, `gear_alt`, `gear_alt_plateau`, sphère).*

## Règles générales

- Au moins 3 graines par configuration ; on rapporte moyenne ± écart-type.
- Toujours comparer `gear` à `gear_frozen` (θ aléatoires jamais entraînés),
  `abu` (Adaptive Blending Units) et `relu`.
- Un gain ne compte que s'il dépasse les deux références de plus de 2
  écarts-types (et d'au moins ~0,3 point sur CIFAR-10).
- Aucun réglage d'hyperparamètre sur le jeu de test. Sur le ResNet, la courbe
  « val » est le test CIFAR-10 : on y règle rien. `lr_theta` peut être réglé
  sur un critère de dynamique (les θ bougent-ils ?), pas sur la précision.
- Les résultats neutres ou négatifs sont publiés comme les autres.

## 1. ResNet-20 / CIFAR-10 (3 graines faites : `relu`, `gear_frozen`, `gear`)

`relu`, `gear_frozen`, `gear` sur 3 graines, 60 époques.
- `gear` ≈ `gear_frozen` : l'apprentissage des θ n'apporte rien ici. Vérifier
  d'abord que les θ ont bougé (`theta_history` dans le JSON) : sinon, c'est
  `lr_theta` qui est trop faible, pas l'idée qui échoue.
- `gear` > `gear_frozen` mais ≤ `relu` : les θ corrigent les mauvais choix
  initiaux, sans battre la référence.
- `gear` > `relu` : à confirmer sur 182 époques et plus de graines.

État (3 graines) : `gear` 90,26 ± 0,08, `relu` 90,55 ± 0,14, `gear_frozen`
89,85 ± 0,30. `gear` < `relu` dans les 3 graines (−0,29 en moyenne) ; `gear` >
`gear_frozen` de 0,41 en moyenne, mais seulement de 0,14 dans la graine 2 :
on est dans le deuxième cas, sans battre la référence. Suite : `gear_relu_init`
(1 bis), qui dit si le déficit vient du départ ou de l'optimisation de θ. La
sphère n'est pas lancée sur ResNet tant que ce test n'est pas fait.

### 1 bis. Départ sur ReLU (`gear_relu_init`)

Les θ initiaux aléatoires font démarrer le ResNet loin de la référence (le
contrôle figé perd 0,7 point). Variante : tous les θ sur ReLU au départ, puis
entraînés. Le réseau est alors identique à ReLU à l'itération 0 (test), et ne
s'en écarte que si le gradient le juge utile.
- `gear_relu_init` ≈ `relu` (à moins de ~0,3 point) : l'optimisation de θ ne
  perd rien mais ne gagne rien non plus.
- `gear_relu_init` < `relu` de plus de ~0,3 point : l'optimisation de θ est
  elle-même le problème, pas le départ.
- `gear_relu_init` > `relu` de plus de 2 écarts-types et ~0,3 point : résultat
  à confirmer sur 182 époques.
Commande : mettre `ACTIVATIONS = ["gear_relu_init"]` dans
`experiments/run_resnet.py`, 3 graines, 60 époques (≈ 32 s/époque sur T4).

### 1 ter. Alternance depuis ReLU (`gear_alt`, `gear_alt_plateau`)

Jouets : pas mieux que l'entraînement continu depuis ReLU. MLP : +0,25 point
sur ReLU, non significatif (exp. 8). Test sur le ResNet, 3 graines, 60 époques :
- `gear_alt` (10 époques figées / 5 ouvertes) et `gear_alt_plateau` ;
- comparer à `relu` et à `gear_relu_init`, avec les critères de la section 1.
  Si l'alternance ne bat pas `gear_relu_init`, on l'abandonne.
Commande : `ACTIVATIONS = ["gear_relu_init", "gear_alt", "gear_alt_plateau"]`
dans `experiments/run_resnet.py`.

## 2. Collage aux nœuds (`diagnostics/diag_nodes.py`)

Les θ se fixent souvent sur une fonction pure. Est-ce un optimum (perte en
pointe sur le nœud, comme la parcimonie du Lasso) ou un piège ?
- Diagnostic : fraction de θ collés, profils de perte, classement des formes
  de perte autour des nœuds, test de perturbation.
- Remèdes, seulement si c'est un piège : bruit sur θ, gradient moyenné aux
  jonctions, poids à noyau avec température. Le lissage de type smoothstep
  est à éviter : il annule la dérivée aux nœuds.

## 3. Ablations

- Ordre du cycle (3 permutations), taille du catalogue (4, 6, 10 fonctions).
- Fenêtre d'entraînement des θ : début, milieu, fin (`gear_window`).
- Un θ par couche ou par neurone.
- `harden()` en fin d'entraînement : perte de précision quand chaque neurone
  est figé sur sa fonction dominante.
- Implémentation n'évaluant que les deux fonctions voisines : vitesse réelle
  face à ABU.
- Tâches où l'activation compte davantage : réseaux étroits, régression de
  fonctions.

## 4. Engrenages sphériques (implémentés : `gears/gear3d.py`)

Deux degrés de liberté par neurone : un point sur la sphère S² au lieu du
cercle.
- Paramétrage par un vecteur 3D normalisé (pas de singularité aux pôles).
- Deux familles de poids : barycentrique sur une triangulation (octaèdre, 6
  fonctions ; icosaèdre, 12) avec 3 fonctions actives, ou à noyau
  (w_i ∝ exp(κ·⟨u, p_i⟩)), sans collage mais avec toutes les fonctions actives.
- Avantage attendu : chaque fonction a plusieurs voisines, ce qui lève
  l'arbitraire de l'ordre du cycle.
- Placement des fonctions : à la main par familles, puis par similarité de
  forme (distance L2 + MDS), puis positions apprenables partagées par couche.
- Contrôles à nombre de fonctions égal : cercle (1 paramètre), sphère (2),
  ABU (n), PReLU.

État : octaèdre barycentrique et noyau implémentés ; jouets et MLP faits
(voir docs/resultats_preliminaires.md, exp. 5 et 6).
- ResNet-20 : `sphere`, `sphere_frozen`, `gear6` sur 3 graines
  (`experiments/exp3d_3_resnet.py`). Mêmes critères que la section 1, en
  comparant aussi à `gear6`.
- `ripple` : vérifier que le gain tient avec plus de neurones et sur d'autres
  fonctions périodiques, avant d'en parler comme d'un résultat.
- Norme de u : comparer avec `renormalize()` après chaque pas.
- Placement par similarité de forme (MDS), puis positions apprenables.
