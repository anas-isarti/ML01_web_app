Ce dossier est un repo git qui contient deux mini-projets mélangés (UFO et MNIST). Je veux le nettoyer pour qu'il ne contienne QUE le projet UFO. Procède méthodiquement et montre-moi un plan AVANT d'agir.

Liste d'abord tout le contenu du dossier (y compris fichiers cachés) pour qu'on voie l'état réel.
Identifie et SUPPRIME tous les fichiers/dossiers liés à MNIST ou inutiles à l'UFO :

mnist_app.py, mnist_train.py, mnist_cnn.pt, mnist_training_report.txt
les fichiers de l'autre auteur liés à MNIST : app.py, backend.py, frontend.py, 8.png
le dossier data/ (dataset MNIST téléchargé)
les Dockerfile liés à MNIST s'il y en a (Dockerfile qui copie backend.py/mnist.hdf5)
tout fichier mnist*.hdf5, mnist*.npz, *.onnx, convert_to_onnx.py, quick_train_mnist.py, train_mnist*.py, train_and_export.py s'ils existent
Avant de supprimer quoi que ce soit, MONTRE-MOI la liste exacte des fichiers que tu comptes supprimer et attends ma validation.


CONSERVE uniquement les fichiers UFO et la config :

ufo_app.py, ufo_train.py, ufo_backend.py, ufo_frontend.py
ufo_model.pkl, ufo_label_encoder.pkl, ufo_training_report.txt
Dockerfile.ufo (si présent)
.gitignore, README.md, consigne.md


Réécris requirements.txt pour qu'il contienne EXACTEMENT ceci (versions UFO uniquement, sans torch, sans pillow, sans streamlit-drawable-canvas) :

streamlit==1.47.1
scikit-learn==1.8.0
numpy==2.4.4
scipy==1.17.1
joblib==1.5.3
pandas==2.3.3

Réécris runtime.txt pour qu'il contienne uniquement : 3.12
Vérifie que ufo_app.py charge ufo_model.pkl et ufo_label_encoder.pkl par des chemins relatifs simples (sans préfixe de dossier) et qu'il n'importe rien lié à MNIST/torch. Signale toute incohérence sans la corriger.
Vérifie la syntaxe : python -c "import ast; ast.parse(open('ufo_app.py').read()); print('SYNTAX OK')".

Montre-moi : le plan de suppression (étape 2) AVANT d'agir, puis après ma validation, le résultat de chaque étape, le requirements.txt final, le runtime.txt, et le SYNTAX OK. Ne fais AUCUNE commande git — je m'en occupe moi-même après.