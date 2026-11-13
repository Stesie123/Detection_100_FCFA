
import cv2
import json
import numpy as np
from pathlib import Path

# ==================================================
# 1. CHEMINS
# ==================================================

BASE = Path(__file__).parent

IMAGE_SOURCE = BASE / "dataset" / "images" / "100_reference.jpeg"

DOSSIER_SORTIE = BASE / "reference_100_fcfa"
DOSSIER_SORTIE.mkdir(exist_ok=True)

# ==================================================
# 2. CHARGEMENT DE L'IMAGE
# ==================================================

image = cv2.imread(str(IMAGE_SOURCE))

if image is None:
    raise FileNotFoundError(
        f"Image introuvable : {IMAGE_SOURCE}"
    )

hauteur, largeur = image.shape[:2]
dimension_min = min(hauteur, largeur)

print("Image chargée avec succès.")
print(f"Dimensions : {largeur} x {hauteur}")

# ==================================================
# 3. PRETRAITEMENT
# ==================================================

gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

flou = cv2.GaussianBlur(
    gris,
    (9, 9),
    2
)

# ==================================================
# 4. DETECTION DU CERCLE EXTERIEUR
# ==================================================

cercles = cv2.HoughCircles(
    flou,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=dimension_min / 2,
    param1=100,
    param2=35,
    minRadius=int(dimension_min * 0.15),
    maxRadius=int(dimension_min * 0.49)
)

if cercles is None:
    raise ValueError("Aucun cercle détecté.")

cercles = np.round(cercles[0]).astype(int)

cx, cy, rayon = max(
    cercles,
    key=lambda cercle: cercle[2]
)

# Ajustement déjà validé visuellement
rayon = int(rayon * 1.04)

print("\nPièce détectée.")
print(f"Centre : ({cx}, {cy})")
print(f"Rayon : {rayon} pixels")

# ==================================================
# 5. MASQUE EXTERIEUR DE LA PIECE
# ==================================================

masque = np.zeros(
    (hauteur, largeur),
    dtype=np.uint8
)

cv2.circle(
    masque,
    (cx, cy),
    rayon,
    255,
    thickness=-1
)

contours, _ = cv2.findContours(
    masque,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

contour = max(
    contours,
    key=cv2.contourArea
)

aire_contour = float(cv2.contourArea(contour))
perimetre = float(cv2.arcLength(contour, True))

x, y, w, h = cv2.boundingRect(contour)

# ==================================================
# 6. CARACTERISTIQUES GEOMETRIQUES
# ==================================================

circularite = (
    (4 * np.pi * aire_contour) / (perimetre ** 2)
    if perimetre > 0 else 0
)

rapport_largeur_hauteur = (
    float(w / h) if h > 0 else 0
)

rayon_equivalent = float(
    np.sqrt(aire_contour / np.pi)
)

# ==================================================
# 7. ANALYSE DES COULEURS
# ==================================================

hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV
)

lab = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2LAB
)

pixels_hsv = hsv[masque == 255]
pixels_lab = lab[masque == 255]

if len(pixels_hsv) == 0:
    raise ValueError("Masque de pièce vide.")

couleur_hsv_moyenne = np.mean(
    pixels_hsv,
    axis=0
).tolist()

couleur_lab_moyenne = np.mean(
    pixels_lab,
    axis=0
).tolist()

# ==================================================
# 8. CALCUL DES DISTANCES AU CENTRE
# ==================================================

Y, X = np.ogrid[:hauteur, :largeur]

distance = np.sqrt(
    (X - cx) ** 2 + (Y - cy) ** 2
)

distance_normalisee = distance / rayon

# ==================================================
# 9. SEGMENTATION DU CENTRE GRIS
# ==================================================

# Seuil provisoire déterminé à partir du profil radial.
# Il devra être vérifié visuellement.

masque_centre = (
    (distance_normalisee <= 0.68)
    & (masque == 255)
).astype(np.uint8) * 255

# ==================================================
# 10. SEGMENTATION DE LA COURONNE DOREE
# ==================================================

masque_couronne = (
    (distance_normalisee >= 0.72)
    & (distance_normalisee <= 1.0)
    & (masque == 255)
).astype(np.uint8) * 255

# ==================================================
# 11. ZONE DE TRANSITION
# ==================================================

masque_transition = (
    (distance_normalisee > 0.68)
    & (distance_normalisee < 0.72)
    & (masque == 255)
).astype(np.uint8) * 255

# ==================================================
# 12. FONCTION D'ANALYSE D'UNE ZONE
# ==================================================

def analyser_zone(nom, masque_zone):

    pixels_hsv = hsv[masque_zone == 255]
    pixels_lab = lab[masque_zone == 255]
    pixels_gris = gris[masque_zone == 255]

    if len(pixels_hsv) == 0:
        return {
            "nom": nom,
            "nombre_pixels": 0
        }

    return {
        "nom": nom,
        "nombre_pixels": int(len(pixels_hsv)),
        "couleur_moyenne_HSV": np.mean(
            pixels_hsv,
            axis=0
        ).tolist(),
        "couleur_moyenne_LAB": np.mean(
            pixels_lab,
            axis=0
        ).tolist(),
        "intensite_moyenne": float(
            np.mean(pixels_gris)
        ),
        "ecart_type": float(
            np.std(pixels_gris)
        )
    }

# ==================================================
# 13. ANALYSE SEPAREE DES ZONES
# ==================================================

caracteristiques_centre = analyser_zone(
    "Centre gris",
    masque_centre
)

caracteristiques_couronne = analyser_zone(
    "Couronne exterieure",
    masque_couronne
)

caracteristiques_transition = analyser_zone(
    "Zone de transition",
    masque_transition
)

# ==================================================
# 14. PROFIL RADIAL DES COULEURS
# ==================================================

profil_radial = []

bornes = np.linspace(0, 1, 11)

for i in range(len(bornes) - 1):

    debut = bornes[i]
    fin = bornes[i + 1]

    zone = (
        (distance_normalisee >= debut)
        & (distance_normalisee < fin)
        & (masque == 255)
    )

    pixels_zone = hsv[zone]

    if len(pixels_zone) > 0:

        profil_radial.append({
            "rayon_debut": round(float(debut), 2),
            "rayon_fin": round(float(fin), 2),
            "couleur_moyenne_HSV": np.mean(
                pixels_zone,
                axis=0
            ).tolist()
        })

# ==================================================
# 15. ANALYSE DES TEXTURES
# ==================================================

pixels_gris = gris[masque == 255]

texture_moyenne = float(
    np.mean(pixels_gris)
)

texture_ecart_type = float(
    np.std(pixels_gris)
)

bords = cv2.Canny(
    gris,
    50,
    150
)

nombre_pixels_bords = int(
    np.count_nonzero(bords[masque == 255])
)

# ==================================================
# 16. EXTRACTION DES POINTS ORB
# ==================================================

orb = cv2.ORB_create(
    nfeatures=500
)

points, descripteurs = orb.detectAndCompute(
    gris,
    masque
)

nombre_points_orb = (
    len(points) if points is not None else 0
)

if descripteurs is not None:

    np.save(
        DOSSIER_SORTIE / "descripteurs_orb.npy",
        descripteurs
    )

# ==================================================
# 17. ENREGISTREMENT DES CARACTERISTIQUES
# ==================================================

caracteristiques = {

    "reference": "Piece de 100 FCFA bimetallique",

    "image_source": IMAGE_SOURCE.name,

    "dimensions_image": {
        "largeur": largeur,
        "hauteur": hauteur
    },

    "geometrie": {
        "centre_x": int(cx),
        "centre_y": int(cy),
        "rayon": int(rayon),
        "aire_contour": aire_contour,
        "perimetre": perimetre,
        "circularite": circularite,
        "rapport_largeur_hauteur": rapport_largeur_hauteur,
        "rayon_equivalent": rayon_equivalent
    },

    "couleurs": {
        "moyenne_HSV": couleur_hsv_moyenne,
        "moyenne_LAB": couleur_lab_moyenne,
        "profil_radial": profil_radial
    },

    "zones": {
        "centre_gris": caracteristiques_centre,
        "couronne_doree": caracteristiques_couronne,
        "transition": caracteristiques_transition
    },

    "texture": {
        "intensite_moyenne": texture_moyenne,
        "ecart_type": texture_ecart_type,
        "nombre_pixels_bords": nombre_pixels_bords
    },

    "points_caracteristiques": {
        "methode": "ORB",
        "nombre_points": nombre_points_orb
    },

    "seuils_segmentation": {
        "centre_maximum": 0.68,
        "transition_debut": 0.68,
        "transition_fin": 0.72,
        "couronne_debut": 0.72
    }
}

with open(
    DOSSIER_SORTIE / "caracteristiques.json",
    "w",
    encoding="utf-8"
) as fichier:

    json.dump(
        caracteristiques,
        fichier,
        indent=4,
        ensure_ascii=False
    )

# ==================================================
# 18. SAUVEGARDE DES MASQUES
# ==================================================

cv2.imwrite(
    str(DOSSIER_SORTIE / "masque_reference.png"),
    masque
)

cv2.imwrite(
    str(DOSSIER_SORTIE / "masque_centre_gris.png"),
    masque_centre
)

cv2.imwrite(
    str(DOSSIER_SORTIE / "masque_couronne_doree.png"),
    masque_couronne
)

cv2.imwrite(
    str(DOSSIER_SORTIE / "masque_transition.png"),
    masque_transition
)

# ==================================================
# 19. CREATION DE L'IMAGE DES ZONES
# ==================================================

image_zones = image.copy()

# Centre gris : bleu
image_zones[masque_centre == 255] = (
    255, 100, 0
)

# Couronne dorée : jaune
image_zones[masque_couronne == 255] = (
    0, 220, 255
)

# Transition : rouge
image_zones[masque_transition == 255] = (
    0, 0, 255
)

# Mélange avec l'image originale
image_zones = cv2.addWeighted(
    image,
    0.45,
    image_zones,
    0.55,
    0
)

cv2.imwrite(
    str(DOSSIER_SORTIE / "zones_reference.png"),
    image_zones
)

# ==================================================
# 20. IMAGE DU CONTOUR EXTERIEUR
# ==================================================

image_contour = image.copy()

cv2.circle(
    image_contour,
    (cx, cy),
    rayon,
    (0, 255, 0),
    3
)

cv2.imwrite(
    str(DOSSIER_SORTIE / "contour_reference.png"),
    image_contour
)

# ==================================================
# 21. IMAGE DE LA PIECE ISOLEE
# ==================================================

image_isolee = cv2.bitwise_and(
    image,
    image,
    mask=masque
)

cv2.imwrite(
    str(DOSSIER_SORTIE / "piece_isolee.png"),
    image_isolee
)

# ==================================================
# 22. BILAN
# ==================================================

print("\n===================================")
print("SEGMENTATION DE LA PIECE TERMINEE")
print("===================================")

print(f"Circularité : {circularite:.4f}")
print(f"Rayon : {rayon} pixels")
print(f"Points ORB : {nombre_points_orb}")

print("\nCentre gris :")
print(f"Pixels : {caracteristiques_centre.get('nombre_pixels', 0)}")

print("\nCouronne dorée :")
print(f"Pixels : {caracteristiques_couronne.get('nombre_pixels', 0)}")

print("\nZone de transition :")
print(f"Pixels : {caracteristiques_transition.get('nombre_pixels', 0)}")

print("\nRésultats enregistrés dans :")
print(DOSSIER_SORTIE)
