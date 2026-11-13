
import cv2
import numpy as np
import json
from pathlib import Path

# ==========================================
# 1. CHEMINS
# ==========================================

BASE = Path(__file__).parent

IMAGE_PATH = BASE / "dataset" / "images" / "100_reference.jpeg"

DOSSIER_REF = BASE / "reference_100_fcfa"

MASQUE_CENTRE_PATH = DOSSIER_REF / "masque_centre_gris.png"
MASQUE_COURONNE_PATH = DOSSIER_REF / "masque_couronne_doree.png"

DOSSIER_REF.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. CHARGEMENT DES FICHIERS
# ==========================================

image = cv2.imread(str(IMAGE_PATH))

masque_centre = cv2.imread(
    str(MASQUE_CENTRE_PATH),
    cv2.IMREAD_GRAYSCALE
)

masque_couronne = cv2.imread(
    str(MASQUE_COURONNE_PATH),
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(
        f"Image de référence introuvable : {IMAGE_PATH}"
    )

if masque_centre is None:
    raise FileNotFoundError(
        f"Masque du centre introuvable : {MASQUE_CENTRE_PATH}"
    )

if masque_couronne is None:
    raise FileNotFoundError(
        f"Masque de la couronne introuvable : {MASQUE_COURONNE_PATH}"
    )

if (
    image.shape[:2] != masque_centre.shape
    or image.shape[:2] != masque_couronne.shape
):
    raise ValueError(
        "Les dimensions des masques ne correspondent pas à l'image."
    )

print("Images et masques chargés.")

# ==========================================
# 3. PRETRAITEMENT DE LA LUMINOSITE
# ==========================================

# Conversion HSV
hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV
)

# Amélioration du canal V uniquement.
# Les informations de teinte et de saturation sont conservées.

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

hsv_ameliore = hsv.copy()

hsv_ameliore[:, :, 2] = clahe.apply(
    hsv[:, :, 2]
)

# ==========================================
# 4. PRETRAITEMENT POUR ORB
# ==========================================

gris = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

gris_ameliore = clahe.apply(gris)

# ==========================================
# 5. CALCUL DES HISTOGRAMMES
# ==========================================

def calculer_histogramme(masque):

    histogramme = cv2.calcHist(
        [hsv_ameliore],
        [0, 1],
        masque,
        [30, 32],
        [0, 180, 0, 256]
    )

    cv2.normalize(
        histogramme,
        histogramme,
        alpha=1,
        norm_type=cv2.NORM_L1
    )

    return histogramme.flatten().astype(np.float32)


histogramme_centre = calculer_histogramme(
    masque_centre
)

histogramme_couronne = calculer_histogramme(
    masque_couronne
)

# ==========================================
# 6. EXTRACTION ORB PAR ZONE
# ==========================================

orb = cv2.ORB_create(
    nfeatures=1000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    patchSize=31,
    fastThreshold=10
)

points_centre, descripteurs_centre = orb.detectAndCompute(
    gris_ameliore,
    masque_centre
)

points_couronne, descripteurs_couronne = orb.detectAndCompute(
    gris_ameliore,
    masque_couronne
)

# ==========================================
# 7. VERIFICATION DES DESCRIPTEURS
# ==========================================

if descripteurs_centre is None or len(descripteurs_centre) < 2:
    raise ValueError(
        "Pas assez de caractéristiques ORB dans le centre de la référence."
    )

if descripteurs_couronne is None or len(descripteurs_couronne) < 2:
    raise ValueError(
        "Pas assez de caractéristiques ORB dans la couronne de la référence."
    )

# ==========================================
# 8. SAUVEGARDE DES SIGNATURES
# ==========================================

np.savez_compressed(
    DOSSIER_REF / "signatures_reference.npz",
    histogramme_centre=histogramme_centre,
    histogramme_couronne=histogramme_couronne
)

np.save(
    DOSSIER_REF / "orb_centre.npy",
    descripteurs_centre
)

np.save(
    DOSSIER_REF / "orb_couronne.npy",
    descripteurs_couronne
)

# ==========================================
# 9. SAUVEGARDE DU RESUME
# ==========================================

resume = {
    "reference": "100 FCFA bimetallique",

    "pretraitement": {
        "luminosite": "CLAHE sur le canal V HSV",
        "orb": "CLAHE sur image en niveaux de gris",
        "clip_limit": 2.0,
        "tile_grid_size": [8, 8]
    },

    "histogrammes": {
        "methode": "Histogramme HSV 2D",
        "canaux": ["Teinte H", "Saturation S"],
        "bins": [30, 32],
        "taille_signature_centre": int(
            len(histogramme_centre)
        ),
        "taille_signature_couronne": int(
            len(histogramme_couronne)
        )
    },

    "orb": {
        "methode": "ORB",
        "nombre_maximum_points": 1000,
        "points_centre": int(len(points_centre)),
        "points_couronne": int(len(points_couronne)),
        "descripteurs_centre": int(len(descripteurs_centre)),
        "descripteurs_couronne": int(len(descripteurs_couronne))
    }
}

with open(
    DOSSIER_REF / "resume_signatures.json",
    "w",
    encoding="utf-8"
) as fichier:

    json.dump(
        resume,
        fichier,
        indent=4,
        ensure_ascii=False
    )

# ==========================================
# 10. SAUVEGARDE DES IMAGES AMELIOREES
# ==========================================

cv2.imwrite(
    str(DOSSIER_REF / "reference_gris_amelioree.png"),
    gris_ameliore
)

cv2.imwrite(
    str(DOSSIER_REF / "reference_luminosite_amelioree.png"),
    hsv_ameliore[:, :, 2]
)

# ==========================================
# 11. AFFICHAGE DES RESULTATS
# ==========================================

print("\n================================")
print("SIGNATURES EXTRAITES AVEC SUCCES")
print("================================")

print("\nPrétraitement : CLAHE")

print(
    "Histogramme centre :",
    len(histogramme_centre),
    "valeurs"
)

print(
    "Histogramme couronne :",
    len(histogramme_couronne),
    "valeurs"
)

print(
    "Points ORB centre :",
    len(points_centre)
)

print(
    "Points ORB couronne :",
    len(points_couronne)
)

print("\nFichiers enregistrés dans :")
print(DOSSIER_REF)
