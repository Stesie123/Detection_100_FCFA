
from pathlib import Path
from ultralytics import YOLO
import random
import shutil

BASE = Path(__file__).parent
SOURCE_IMAGES = BASE / "dataset_yolo" / "images" / "train"
SOURCE_LABELS = BASE / "dataset_yolo" / "labels" / "train"

DATASET = BASE / "dataset_final"

# Préparer les dossiers
for dossier in ["images/train", "images/val",
                "labels/train", "labels/val"]:
    (DATASET / dossier).mkdir(parents=True, exist_ok=True)

# Séparer les images
images = list(SOURCE_IMAGES.glob("*.jpg"))
random.seed(42)
random.shuffle(images)

limite = int(len(images) * 0.8)
train_images = images[:limite]
val_images = images[limite:]

def copier_images(liste, partie):
    for image in liste:
        label = SOURCE_LABELS / (image.stem + ".txt")

        shutil.copy2(image, DATASET / "images" / partie / image.name)
        shutil.copy2(label, DATASET / "labels" / partie / label.name)

copier_images(train_images, "train")
copier_images(val_images, "val")

# Créer la configuration
yaml = BASE / "data_final.yaml"

with open(yaml, "w", encoding="utf-8") as f:
    f.write(
        f"path: {DATASET.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: piece_100_fcfa\n"
    )

print("Images d'entraînement :", len(train_images))
print("Images de validation :", len(val_images))

# Charger YOLO pré-entraîné
modele = YOLO("yolo11n.pt")

# Entraîner
modele.train(
    data=str(yaml),
    epochs=30,
    imgsz=416,
    batch=4,
    workers=0,
    device="cpu",
    project=str(BASE / "resultats"),
    name="detection_100_fcfa"
)

print("Entraînement terminé !")
