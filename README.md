\# Détection et vérification d'une pièce de 100 FCFA



\## 1. Présentation du projet



Cette application permet d'analyser une photographie d'une pièce de monnaie de 100 FCFA afin de détecter la pièce et de vérifier sa ressemblance avec une pièce de référence.



Elle combine des techniques de vision par ordinateur et de traitement d'images pour examiner les caractéristiques visuelles de la pièce.



\## 2. Technologies utilisées



\### Backend

\- Python

\- FastAPI

\- YOLO (détection d'objets)

\- OpenCV (traitement d'images)

\- ORB (extraction et comparaison de caractéristiques)

\- Tesseract OCR



\### Frontend

\- Next.js

\- React

\- TypeScript



\## 3. Prérequis



Avant de lancer le projet, installer les logiciels suivants :



\- Python (version compatible avec les dépendances du projet)

\- Node.js et npm

\- Git

\- Tesseract OCR



Tesseract OCR doit être installé à cet emplacement :



`C:\\Program Files\\Tesseract-OCR\\tesseract.exe`



Si Tesseract est installé ailleurs, adapter son chemin dans le fichier `api.py`.



\## 4. Récupération du projet



Ouvrir PowerShell et exécuter :



```powershell

git clone https://github.com/Stesie123/Detection_100_FCFA.git

```



Remplacer `URL\_DU\_DEPOT` par l'adresse du dépôt GitHub.



Entrer ensuite dans le dossier :



```powershell

cd Detection\_100\_FCFA

```



\## 5. Installation du backend



Depuis le dossier principal du projet, créer un environnement virtuel :



```powershell

py -m venv .venv

```



Activer l'environnement :



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Mettre pip à jour :



```powershell

python -m pip install --upgrade pip

```



Installer les dépendances :



```powershell

pip install -r requirements.txt

```



\## 6. Démarrage du backend



Toujours depuis le dossier principal, exécuter :



```powershell

uvicorn api:app --reload

```



Le backend sera accessible à l'adresse :



http://127.0.0.1:8000



Documentation de l'API :



http://127.0.0.1:8000/docs



\## 7. Installation du frontend



Ouvrir un deuxième terminal PowerShell.



Entrer dans le dossier frontend :



```powershell

cd Detection\_100\_FCFA\\frontend

```



Installer les dépendances :



```powershell

npm install

```



\## 8. Démarrage du frontend



Dans le même terminal, exécuter :



```powershell

npm run dev

```



L'application sera accessible dans le navigateur à l'adresse :



http://localhost:3000



\## 9. Ordre de lancement



Pour utiliser l'application, les deux serveurs doivent fonctionner simultanément :



1\. Démarrer le backend dans un premier terminal.

2\. Démarrer le frontend dans un deuxième terminal.

3\. Ouvrir http://localhost:3000 dans le navigateur.



\*\*Important :\*\* Ne pas fermer les terminaux pendant l'utilisation de l'application.



\## 10. Fichiers importants



\- `api.py` : point d'entrée du backend et routes de l'API.

\- `frontend/` : interface utilisateur.

\- `reference\_100\_fcfa/` : caractéristiques et images de référence.

\- `resultats/detection\_100\_fcfa/weights/best.pt` : modèle YOLO entraîné utilisé pour la détection.

\- `yolo11n.pt` : modèle YOLO de base.

\- `requirements.txt` : dépendances Python du backend.

