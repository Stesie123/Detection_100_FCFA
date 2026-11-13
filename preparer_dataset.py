import os
import cv2
import shutil
from pathlib import Path

BASE = Path(__file__).parent

SOURCE = BASE / "dataset" / "images_augmentees"
IMAGES = BASE / "dataset_yolo" / "images" / "train"
LABELS = BASE / "dataset_yolo" / "labels" / "train"

IMAGES.mkdir(parents=True, exist_ok=True)
LABELS.mkdir(parents=True, exist_ok=True)

fichiers = list(SOURCE.glob("*.jpg"))

if not fichiers:
    raise FileNotFoundError("Aucune image trouvée.")

for fichier in fichiers:
    image = cv2.imread(str(fichier))

    if image is None:
        continue

    hauteur, largeur = image.shape[:2]

    # Détection approximative du contour de la pièce
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flou = cv2.GaussianBlur(gris, (5, 5), 0)
    _, seuil = cv2.threshold(
        flou, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        seuil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("Contour introuvable :", fichier.name)
        continue

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    # Coordonnées YOLO normalisées
    cx = (x + w / 2) / largeur
    cy = (y + h / 2) / hauteur
    nw = w / largeur
    nh = h / hauteur

    shutil.copy2(fichier, IMAGES / fichier.name)

    with open(LABELS / (fichier.stem + ".txt"), "w") as f:
        f.write(f"0 {cx} {cy} {nw} {nh}\n")

    print("Préparé :", fichier.name)

# Configuration YOLO
configuration = BASE / "dataset_yolo" / "data.yaml"

with open(configuration, "w") as f:
    f.write(
        f"path: {BASE / 'dataset_yolo'}\n"
        "train: images/train\n"
        "val: images/train\n"
        "names:\n"
        "  0: piece_100_fcfa\n"
    )

print("\nPréparation terminée.")
print("Images :", len(list(IMAGES.glob("*.jpg"))))
print("Annotations :", len(list(LABELS.glob("*.txt"))))