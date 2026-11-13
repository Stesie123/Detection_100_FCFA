import cv2
from pathlib import Path

BASE = Path(__file__).parent
IMAGES = BASE / "dataset_yolo" / "images" / "train"
LABELS = BASE / "dataset_yolo" / "labels" / "train"
SORTIE = BASE / "verification_annotations"

SORTIE.mkdir(exist_ok=True)

fichiers = sorted(IMAGES.glob("*.jpg"))[:6]

for fichier in fichiers:
    image = cv2.imread(str(fichier))
    hauteur, largeur = image.shape[:2]

    annotation = LABELS / (fichier.stem + ".txt")

    with open(annotation, "r") as f:
        for ligne in f:
            classe, cx, cy, w, h = map(float, ligne.split())

            x1 = int((cx - w / 2) * largeur)
            y1 = int((cy - h / 2) * hauteur)
            x2 = int((cx + w / 2) * largeur)
            y2 = int((cy + h / 2) * hauteur)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imwrite(str(SORTIE / fichier.name), image)

print("Vérification créée dans le dossier verification_annotations.")