
import cv2
import numpy as np
from pathlib import Path

# ==================================================
# 1. CHEMINS
# ==================================================

BASE = Path(__file__).parent

IMAGE_TEST = Path.home() / "Desktop" / "pi.png"

DOSSIER_REF = BASE / "reference_100_fcfa"

SIGNATURES_PATH = DOSSIER_REF / "signatures_reference.npz"

ORB_CENTRE_PATH = DOSSIER_REF / "orb_centre.npy"
ORB_COURONNE_PATH = DOSSIER_REF / "orb_couronne.npy"

# ==================================================
# 2. VERIFICATION DES FICHIERS
# ==================================================

if not IMAGE_TEST.exists():
    raise FileNotFoundError(
        f"Image test introuvable : {IMAGE_TEST}"
    )

if not SIGNATURES_PATH.exists():
    raise FileNotFoundError(
        "Signatures de référence introuvables."
    )

# ==================================================
# 3. CHARGEMENT DES SIGNATURES
# ==================================================

signatures = np.load(str(SIGNATURES_PATH))

hist_reference_centre = signatures["histogramme_centre"]
hist_reference_couronne = signatures["histogramme_couronne"]

orb_reference_centre = (
    np.load(str(ORB_CENTRE_PATH))
    if ORB_CENTRE_PATH.exists()
    else None
)

orb_reference_couronne = (
    np.load(str(ORB_COURONNE_PATH))
    if ORB_COURONNE_PATH.exists()
    else None
)

# ==================================================
# 4. CHARGEMENT DE L'IMAGE TEST
# ==================================================

image = cv2.imread(str(IMAGE_TEST))

if image is None:
    raise ValueError(
        "Impossible de lire l'image pi.JPG."
    )

hauteur, largeur = image.shape[:2]
dimension_min = min(hauteur, largeur)

print("Image test chargée.")
print(f"Dimensions : {largeur} x {hauteur}")

# ==================================================
# 5. DETECTION DU CERCLE
# ==================================================

gris = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

flou = cv2.GaussianBlur(
    gris,
    (9, 9),
    2
)

cercles = cv2.HoughCircles(
    flou,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=dimension_min / 2,
    param1=100,
    param2=35,
    minRadius=int(dimension_min * 0.10),
    maxRadius=int(dimension_min * 0.49)
)

if cercles is None:
    raise ValueError(
        "Aucun cercle détecté dans pi.JPG. "
        "Vérifie que la pièce est bien visible."
    )

cercles = np.round(cercles[0]).astype(int)

cx, cy, rayon = max(
    cercles,
    key=lambda cercle: cercle[2]
)

rayon = int(rayon * 1.04)

print("\nCercle détecté.")
print(f"Centre : ({cx}, {cy})")
print(f"Rayon : {rayon} pixels")

# ==================================================
# 6. CREATION DES MASQUES
# ==================================================

Y, X = np.ogrid[:hauteur, :largeur]

distance = np.sqrt(
    (X - cx) ** 2 + (Y - cy) ** 2
)

distance_normalisee = distance / rayon

masque_piece = (
    distance_normalisee <= 1.0
).astype(np.uint8) * 255

masque_centre = (
    (distance_normalisee <= 0.68)
    & (masque_piece == 255)
).astype(np.uint8) * 255

masque_couronne = (
    (distance_normalisee >= 0.72)
    & (distance_normalisee <= 1.0)
    & (masque_piece == 255)
).astype(np.uint8) * 255

# ==================================================
# 7. EXTRACTION DES HISTOGRAMMES
# ==================================================

hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV
)

def calculer_histogramme(masque):

    histogramme = cv2.calcHist(
        [hsv],
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

    return histogramme.flatten()


hist_test_centre = calculer_histogramme(
    masque_centre
)

hist_test_couronne = calculer_histogramme(
    masque_couronne
)

# ==================================================
# 8. COMPARAISON DES COULEURS
# ==================================================

similarite_centre = cv2.compareHist(
    hist_reference_centre.astype(np.float32),
    hist_test_centre.astype(np.float32),
    cv2.HISTCMP_CORREL
)

similarite_couronne = cv2.compareHist(
    hist_reference_couronne.astype(np.float32),
    hist_test_couronne.astype(np.float32),
    cv2.HISTCMP_CORREL
)

# Transformation en pourcentage indicatif
score_centre = max(
    0,
    min(100, similarite_centre * 100)
)

score_couronne = max(
    0,
    min(100, similarite_couronne * 100)
)

# ==================================================
# 9. EXTRACTION ORB SUR L'IMAGE TEST
# ==================================================

orb = cv2.ORB_create(
    nfeatures=500
)

points_centre, descripteurs_centre = orb.detectAndCompute(
    gris,
    masque_centre
)

points_couronne, descripteurs_couronne = orb.detectAndCompute(
    gris,
    masque_couronne
)

# ==================================================
# 10. COMPARAISON ORB
# ==================================================

def comparer_orb(reference, test):

    if reference is None or test is None:
        return {
            "correspondances": 0,
            "bonnes_correspondances": 0,
            "taux": 0.0
        }

    if len(reference) < 2 or len(test) < 2:
        return {
            "correspondances": 0,
            "bonnes_correspondances": 0,
            "taux": 0.0
        }

    outil = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False
    )

    paires = outil.knnMatch(
        reference,
        test,
        k=2
    )

    bonnes = []

    for paire in paires:

        if len(paire) < 2:
            continue

        meilleur, deuxieme = paire

        if meilleur.distance < 0.75 * deuxieme.distance:
            bonnes.append(meilleur)

    nombre_reference = len(reference)

    taux = (
        len(bonnes) / nombre_reference * 100
        if nombre_reference > 0 else 0
    )

    return {
        "correspondances": len(paires),
        "bonnes_correspondances": len(bonnes),
        "taux": taux
    }


resultat_orb_centre = comparer_orb(
    orb_reference_centre,
    descripteurs_centre
)

resultat_orb_couronne = comparer_orb(
    orb_reference_couronne,
    descripteurs_couronne
)

# ==================================================
# 11. AFFICHAGE DES RESULTATS
# ==================================================

print("\n======================================")
print("COMPARAISON AVEC LA REFERENCE 100 FCFA")
print("======================================")

print("\nSIMILARITE DES COULEURS")

print(
    f"Centre gris : {score_centre:.2f} %"
)

print(
    f"Couronne dorée : {score_couronne:.2f} %"
)

print("\nCOMPARAISON ORB")

print(
    "Centre :",
    resultat_orb_centre["bonnes_correspondances"],
    "bonnes correspondances"
)

print(
    "Couronne :",
    resultat_orb_couronne["bonnes_correspondances"],
    "bonnes correspondances"
)

print("\nNombre de points ORB détectés dans l'image test :")

print(
    "Centre :",
    len(points_centre) if points_centre is not None else 0
)

print(
    "Couronne :",
    len(points_couronne) if points_couronne is not None else 0
)

# ==================================================
# 12. VISUALISATION
# ==================================================

visualisation = image.copy()

cv2.circle(
    visualisation,
    (cx, cy),
    rayon,
    (0, 255, 0),
    3
)

cv2.imwrite(
    str(DOSSIER_REF / "visualisation_test_pi.png"),
    visualisation
)

cv2.imwrite(
    str(DOSSIER_REF / "masque_test_centre.png"),
    masque_centre
)

cv2.imwrite(
    str(DOSSIER_REF / "masque_test_couronne.png"),
    masque_couronne
)

print("\nImages de vérification enregistrées.")

print(
    "\nAttention : ces scores mesurent des ressemblances, "
    "pas l'authenticité d'une pièce."
)
