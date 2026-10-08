# -*- coding: utf-8 -*-
"""
Expérience 3D-3 : engrenages sphériques dans un ResNet-20 sur CIFAR-10 (GPU).

Lanceur, sur le modèle de run_resnet.py : enchaîne les runs de
exp3_resnet.py pour les variantes à 6 fonctions. Les résultats vont dans le
même dossier que les runs relu / gear / gear_frozen, pour que le tableau
récapitulatif compare tout ensemble.

  sphere          sphère (octaèdre), barycentrique
  sphere_frozen   directions aléatoires figées (contrôle)
  gear6           cercle avec les mêmes 6 fonctions (contrôle à n égal)

Utilisation (Colab, GPU activé), depuis la racine du dépôt :
    !python experiments/exp3d_3_resnet.py
Commencez avec TEST_RAPIDE = True (≈ 1 min, données factices), puis False.
Relancer reprend où on en était (runs finis ignorés, checkpoint à chaque époque).

Compter environ 35 à 40 min par run de 60 époques sur un T4 (6 fonctions
évaluées par activation, contre 4 pour gear).
"""
import os
import subprocess
import sys
import time

# ============================ RÉGLAGES ======================================
TEST_RAPIDE = True            # True : test de 1 min sur données factices

ACTIVATIONS = ["sphere", "sphere_frozen", "gear6"]   # ordre = ordre de lancement
# autres choix : "sphere_kernel", "abu6", "sphere_relu_init"
GRAINES = [0, 1, 2]
EPOCHS = 60                   # même protocole que les runs relu / gear
WORKERS = 0
# ============================================================================

DOSSIER = os.path.dirname(os.path.abspath(__file__))     # experiments/
RACINE = os.path.dirname(DOSSIER)                         # racine du dépôt
EXP3 = os.path.join("experiments", "exp3_resnet.py")
SORTIE = "results/resnet_test" if TEST_RAPIDE else "results/resnet20_cifar10"


def commande(act, seed):
    cmd = [sys.executable, EXP3, "--act", act, "--seed", str(seed),
           "--epochs", str(1 if TEST_RAPIDE else EPOCHS),
           "--workers", str(WORKERS), "--out", SORTIE]
    if TEST_RAPIDE:
        cmd += ["--fake-data", "--max-batches", "2", "--no-resume"]
    return cmd


def main():
    os.chdir(RACINE)
    try:
        import torch
        print(f"PyTorch {torch.__version__} | GPU disponible : {torch.cuda.is_available()}")
        if not torch.cuda.is_available() and not TEST_RAPIDE:
            print("\n/!\\ Aucun GPU détecté : utilisez Google Colab avec un GPU.\n")
    except ImportError:
        print("PyTorch n'est pas installé : " + sys.executable + " -m pip install torch torchvision")
        return

    taches = [(a, s) for s in GRAINES for a in ACTIVATIONS]   # graine 0 pour tout le monde d'abord
    print(f"{len(taches)} runs prévus ({'TEST RAPIDE' if TEST_RAPIDE else str(EPOCHS) + ' époques'}), "
          f"résultats dans {SORTIE}/\n")
    t_debut = time.time()
    for i, (act, seed) in enumerate(taches, 1):
        if os.path.exists(os.path.join(SORTIE, f"run_{act}_s{seed}.json")) and not TEST_RAPIDE:
            print(f"[{i}/{len(taches)}] {act} graine {seed} : déjà fait, ignoré")
            continue
        print(f"\n[{i}/{len(taches)}] === {act} | graine {seed} ===", flush=True)
        t0 = time.time()
        if subprocess.call(commande(act, seed), cwd=RACINE) != 0:
            print(f"\nÉCHEC du run {act} graine {seed}. Lisez l'erreur juste au-dessus.")
            return
        print(f"    terminé en {(time.time() - t0) / 60:.1f} min", flush=True)

    print("\n=== Tableau récapitulatif (tous les runs du dossier) ===")
    subprocess.call([sys.executable, EXP3, "--summarize", "--out", SORTIE], cwd=RACINE)
    print(f"\nTout est fini en {(time.time() - t_debut) / 60:.1f} min.")
    if TEST_RAPIDE:
        print("\nLe test rapide a fonctionné. Passez TEST_RAPIDE = False pour lancer les vrais runs.")


if __name__ == "__main__":
    main()
