# Version originale (août 2025)

`main.py` est la première implémentation de l'idée, en NumPy, conservée telle
quelle pour l'historique. Elle contient la classe `RearFunction` : une
activation qui parcourt un cycle de fonctions selon un « pitch » (l'actuel θ).

La réécriture PyTorch (`gears/gear.py`) reproduit exactement la même
activation : le test `tests/test_legacy_equivalence.py` vérifie que les deux
donnent le même résultat à la précision machine près.

## Problèmes identifiés lors de la réécriture (2026)

Ils expliquent pourquoi les premiers résultats de cette version ne
permettaient pas de conclure :

1. **Signe du pitch inversé** : les poids étaient mis à jour avec `-alpha`,
   mais le pitch avec `+alpha`. Les engrenages faisaient une montée de
   gradient (ils augmentaient l'erreur).
2. **Pas de vraie rétropropagation** : chaque couche recevait directement
   l'erreur de sortie, sans repasser par les poids et les dérivées des
   couches suivantes.
3. **Poids initialisés à zéro** : les neurones d'une même couche ne se
   distinguaient que par leur activation.
4. Petits bugs : dérivée de `cos` de mauvais signe,
   `get_anti_overfitting_version_cost_function` ne renvoie rien,
   `validation_set_in != None` échoue sur un tableau NumPy, moyenne
   géométrique indéfinie pour des valeurs négatives, matrice identité
   entraînée sur la couche d'entrée.
5. Le premier jeu de test (sorties −1, −1, −1, 1) est un ET logique, pas un
   XOR.

L'autograd de PyTorch règle les points 1 et 2 ; les autres sont corrigés dans
la nouvelle version.
