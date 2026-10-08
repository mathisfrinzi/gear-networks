# -*- coding: utf-8 -*-
"""
Lance les runs ResNet-20 / CIFAR-10 depuis un simple fichier Python.

Utilisation dans VS Code : ouvrez ce fichier (dans experiments/), puis cliquez sur le bouton ▶ (Run
Python File) en haut à droite. Rien à taper dans un terminal.

1. Réglez la section « RÉGLAGES » ci-dessous.
2. Commencez par TEST_RAPIDE = True (≈ 1 min, données factices) pour vérifier
   que tout fonctionne, puis passez à False pour les vrais runs.
3. Les runs déjà terminés sont ignorés : relancer ce fichier reprend où
   vous en étiez (et un run interrompu reprend à sa dernière époque).
"""
import os
import subprocess
import sys
import time

# ============================ RÉGLAGES ======================================
TEST_RAPIDE = True            # True : test de 1 min sur données factices

ACTIVATIONS = ["relu", "gear_frozen", "gear"]   # ordre = ordre de lancement
# autres choix possibles : "gear_relu_init" (θ tous sur ReLU au départ),
# "gear_alt" (départ ReLU, 10 époques figées / 5 entraînées en boucle),
# "gear_alt_plateau" (départ ReLU, ouverture quand la perte d'entraînement se stabilise),
# "abu", "prelu", "gear_window", "silu"
GRAINES = [0, 1, 2]
EPOCHS = 60                   # 182 = protocole complet de l'article
WORKERS = 0                   # 0 = le plus sûr (surtout sous Windows) ;
                              # essayez 2 ou 4 ensuite pour aller plus vite
# ============================================================================

DOSSIER = os.path.dirname(os.path.abspath(__file__))     # experiments/
RACINE = os.path.dirname(DOSSIER)                         # racine du dépôt
EXP3 = os.path.join("experiments", "exp3_resnet.py")
SORTIE = "results/resnet_test" if TEST_RAPIDE else "results/resnet20_cifar10"


def commande(act, seed):
    cmd = [sys.executable, EXP3,
           "--act", act, "--seed", str(seed),
           "--epochs", str(1 if TEST_RAPIDE else EPOCHS),
           "--workers", str(WORKERS), "--out", SORTIE]
    if TEST_RAPIDE:
        cmd += ["--fake-data", "--max-batches", "2", "--no-resume"]
    return cmd


def main():
    os.chdir(RACINE)                       # tout se passe depuis la racine du dépôt
    try:
        import torch
        print(f"Python : {sys.executable}")
        print(f"PyTorch {torch.__version__} | GPU disponible : {torch.cuda.is_available()}")
        if not torch.cuda.is_available() and not TEST_RAPIDE:
            print("\n/!\\ Aucun GPU détecté : un ResNet-20 sera très lent sur CPU "
                  "(plusieurs heures par run). Utilisez plutôt Google Colab avec un GPU.\n")
    except ImportError:
        print("PyTorch n'est pas installé dans cet interpréteur Python.")
        print("Installez-le avec :  " + sys.executable + " -m pip install torch torchvision matplotlib numpy")
        return

    taches = [(a, s) for s in GRAINES for a in ACTIVATIONS]   # graine 0 pour tout le monde d'abord
    print(f"{len(taches)} runs prévus ({'TEST RAPIDE' if TEST_RAPIDE else str(EPOCHS) + ' époques'}), "
          f"résultats dans {SORTIE}/\n")

    t_debut = time.time()
    for i, (act, seed) in enumerate(taches, 1):
        fichier = os.path.join(RACINE, SORTIE, f"run_{act}_s{seed}.json")
        if os.path.exists(fichier) and not TEST_RAPIDE:
            print(f"[{i}/{len(taches)}] {act} graine {seed} : déjà fait, ignoré")
            continue
        print(f"\n[{i}/{len(taches)}] === {act} | graine {seed} ===", flush=True)
        t0 = time.time()
        code = subprocess.call(commande(act, seed), cwd=RACINE)
        if code != 0:
            print(f"\nÉCHEC du run {act} graine {seed} (code {code}).")
            print("Lisez l'erreur affichée juste au-dessus. Si elle parle de « workers » "
                  "ou de « multiprocessing », vérifiez que WORKERS = 0.")
            return
        print(f"    terminé en {(time.time() - t0) / 60:.1f} min", flush=True)

    print("\n=== Tableau récapitulatif ===")
    subprocess.call([sys.executable, EXP3, "--summarize", "--out", SORTIE], cwd=RACINE)
    print(f"\nTout est fini en {(time.time() - t_debut) / 60:.1f} min. "
          f"Résumé et graphiques : {SORTIE}/summary.txt, curves.png, theta_*.png")
    if TEST_RAPIDE:
        print("\nLe test rapide a fonctionné. Passez TEST_RAPIDE = False pour lancer les vrais runs.")


if __name__ == "__main__":
    main()
