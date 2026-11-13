from ultralytics import YOLO
from pathlib import Path

BASE = Path(__file__).parent

MODELE = BASE / "resultats" / "detection_100_fcfa" / "weights" / "best.pt"

modele = YOLO(str(MODELE))

image = input("Chemin complet de la photo à tester : ")

resultats = modele.predict(
    source=image,
    conf=0.25,
    save=True,
    show=True
)

for resultat in resultats:
    if len(resultat.boxes) > 0:
        print("Pièce de 100 FCFA détectée !")
    else:
        print("Aucune pièce détectée.")

print("Test terminé.")