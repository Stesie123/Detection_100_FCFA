
import cv2
import numpy as np
import pytesseract
import re
import unicodedata

from pathlib import Path
from difflib import SequenceMatcher

# ==================================================
# 1. CHEMINS
# ==================================================

BASE = Path(__file__).parent

IMAGE_TEST = Path.home() / "Desktop" / "pi.png"

IMAGE_REFERENCE = BASE / "dataset" / "images" / "100_reference.jpeg"

DOSSIER_REF = BASE / "reference_100_fcfa"

SIGNATURES_PATH = DOSSIER_REF / "signatures_reference.npz"

ORB_CENTRE_PATH = DOSSIER_REF / "orb_centre.npy"
ORB_COURONNE_PATH = DOSSIER_REF / "orb_couronne.npy"

# Chemin de Tesseract installé sur Windows
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ==================================================
# 2. PARAMETRES DE DECISION
# ==================================================

# Pondération des critères
POIDS_CENTRE = 0.2125
POIDS_COURONNE = 0.2975
POIDS_ORB_CENTRE = 0.17
POIDS_ORB_COURONNE = 0.17
POIDS_TEXTE = 0.15

# Seuil provisoire
SEUIL_RESEMBLANCE = 60

# ==================================================
# 3. VERIFICATION DES FICHIERS
# ==================================================

fichiers = [
    IMAGE_TEST,
    IMAGE_REFERENCE,
    SIGNATURES_PATH,
    ORB_CENTRE_PATH,
    ORB_COURONNE_PATH
]

for fichier in fichiers:
    if not fichier.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {fichier}"
        )

# ==================================================
# 4. CHARGEMENT DES SIGNATURES
# ==================================================

signatures = np.load(str(SIGNATURES_PATH))

hist_reference_centre = signatures[
    "histogramme_centre"
].astype(np.float32)

hist_reference_couronne = signatures[
    "histogramme_couronne"
].astype(np.float32)

orb_reference_centre = np.load(
    str(ORB_CENTRE_PATH)
)

orb_reference_couronne = np.load(
    str(ORB_COURONNE_PATH)
)

# ==================================================
# 5. FONCTIONS OCR
# ==================================================

def preparer_image_ocr(image):

    gris_ocr = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Agrandissement pour faciliter la lecture
    gris_ocr = cv2.resize(
        gris_ocr,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # Amélioration du contraste
    gris_ocr = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gris_ocr)

    # Seuillage adaptatif
    seuil = cv2.adaptiveThreshold(
        gris_ocr,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return gris_ocr, seuil


def lire_texte(image):

    gris_ocr, seuil = preparer_image_ocr(image)

    configurations = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 11",
        "--oem 3 --psm 12"
    ]

    textes = []

    for configuration in configurations:

        try:
            texte = pytesseract.image_to_string(
                seuil,
                lang="eng",
                config=configuration
            )

            texte = texte.strip()

            if texte:
                textes.append(texte)

        except pytesseract.TesseractError:
            continue

    if not textes:
        return ""

    # On retient la lecture qui contient le plus de caractères
    meilleur_texte = max(
        textes,
        key=len
    )

    return meilleur_texte


def normaliser_texte(texte):

    texte = texte.upper()

    texte = unicodedata.normalize(
        "NFD",
        texte
    )

    texte = "".join(
        caractere
        for caractere in texte
        if unicodedata.category(caractere) != "Mn"
    )

    # Conservation des lettres et des chiffres
    texte = re.sub(
        r"[^A-Z0-9]",
        "",
        texte
    )

    return texte


def comparer_textes(texte_reference, texte_test):

    reference = normaliser_texte(
        texte_reference
    )

    test = normaliser_texte(
        texte_test
    )

    if not reference or not test:
        return None

    score = SequenceMatcher(
        None,
        reference,
        test
    ).ratio() * 100

    return float(score)


# ==================================================
# 6. CHARGEMENT DE L'IMAGE DE REFERENCE
# ==================================================

image_reference = cv2.imread(
    str(IMAGE_REFERENCE)
)

if image_reference is None:
    raise ValueError(
        "Impossible de lire l'image de référence."
    )

# Lecture du texte de référence
texte_reference = lire_texte(
    image_reference
)

print("\nINSCRIPTIONS DE LA REFERENCE")
print("----------------------------------------")

if texte_reference:
    print(texte_reference)
else:
    print("Aucun texte reconnu sur la référence.")

# ==================================================
# 7. CHARGEMENT DE L'IMAGE TEST
# ==================================================

image = cv2.imread(
    str(IMAGE_TEST)
)

if image is None:
    raise ValueError(
        "Impossible de lire l'image de test."
    )

hauteur, largeur = image.shape[:2]
dimension_min = min(hauteur, largeur)

gris = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV
)

print("\nImage test chargée.")
print(f"Dimensions : {largeur} x {hauteur}")

# ==================================================
# 8. DETECTION DE LA PIECE
# ==================================================

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
    print("\nDECISION : ANALYSE IMPOSSIBLE")
    print("Aucune pièce circulaire détectée.")
    raise SystemExit

cercles = np.round(
    cercles[0]
).astype(int)

cx, cy, rayon = max(
    cercles,
    key=lambda cercle: cercle[2]
)

rayon = int(rayon * 1.04)

print("\nPièce détectée.")
print(f"Centre : ({cx}, {cy})")
print(f"Rayon : {rayon} pixels")

# ==================================================
# 9. CREATION DES MASQUES
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

if (
    np.count_nonzero(masque_centre) == 0
    or np.count_nonzero(masque_couronne) == 0
):
    raise ValueError(
        "Les zones de comparaison sont vides."
    )

# ==================================================
# 10. EXTRACTION DE LA PIECE POUR L'OCR
# ==================================================

x1 = max(0, cx - rayon)
y1 = max(0, cy - rayon)

x2 = min(largeur, cx + rayon)
y2 = min(hauteur, cy + rayon)

piece_ocr = image[
    y1:y2,
    x1:x2
].copy()

masque_ocr = masque_piece[
    y1:y2,
    x1:x2
]

# Fond blanc autour de la pièce
piece_ocr[masque_ocr == 0] = 255

texte_test = lire_texte(
    piece_ocr
)

print("\nINSCRIPTIONS DE LA PIECE TEST")
print("----------------------------------------")

if texte_test:
    print(texte_test)
else:
    print("Aucun texte reconnu sur la pièce test.")

# ==================================================
# 11. HISTOGRAMMES HSV
# ==================================================

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

    return histogramme.flatten().astype(
        np.float32
    )


hist_test_centre = calculer_histogramme(
    masque_centre
)

hist_test_couronne = calculer_histogramme(
    masque_couronne
)

# ==================================================
# 12. COMPARAISON DES COULEURS
# ==================================================

def comparer_couleurs(reference, test):

    correlation = cv2.compareHist(
        reference,
        test,
        cv2.HISTCMP_CORREL
    )

    score = (
        (correlation + 1) / 2
    ) * 100

    return float(
        np.clip(score, 0, 100)
    )


score_centre = comparer_couleurs(
    hist_reference_centre,
    hist_test_centre
)

score_couronne = comparer_couleurs(
    hist_reference_couronne,
    hist_test_couronne
)

# ==================================================
# 13. EXTRACTION ORB
# ==================================================

orb = cv2.ORB_create(
    nfeatures=500
)

points_centre, descripteurs_centre = (
    orb.detectAndCompute(
        gris,
        masque_centre
    )
)

points_couronne, descripteurs_couronne = (
    orb.detectAndCompute(
        gris,
        masque_couronne
    )
)

# ==================================================
# 14. COMPARAISON ORB
# ==================================================

def comparer_orb(reference, test):

    if reference is None or test is None:
        return 0.0, 0

    if len(reference) < 2 or len(test) < 2:
        return 0.0, 0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False
    )

    paires = matcher.knnMatch(
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

    taux = (
        len(bonnes) / len(reference)
    ) * 100

    return float(
        min(taux, 100)
    ), len(bonnes)


score_orb_centre, bonnes_centre = comparer_orb(
    orb_reference_centre,
    descripteurs_centre
)

score_orb_couronne, bonnes_couronne = comparer_orb(
    orb_reference_couronne,
    descripteurs_couronne
)

# ==================================================
# 15. COMPARAISON DES INSCRIPTIONS
# ==================================================

score_texte = comparer_textes(
    texte_reference,
    texte_test
)

# ==================================================
# 16. CALCUL DU SCORE GLOBAL
# ==================================================

criteres = [
    (score_centre, POIDS_CENTRE),
    (score_couronne, POIDS_COURONNE),
    (score_orb_centre, POIDS_ORB_CENTRE),
    (score_orb_couronne, POIDS_ORB_COURONNE),
    (score_texte, POIDS_TEXTE)
]

# Si l'OCR ne reconnaît pas de texte exploitable,
# son poids est réparti entre les autres critères.
poids_actifs = sum(
    poids
    for score, poids in criteres
    if score is not None
)

score_global = sum(
    score * poids
    for score, poids in criteres
    if score is not None
) / poids_actifs

# ==================================================
# 17. DECISION
# ==================================================

if score_global >= SEUIL_RESEMBLANCE:

    decision = "FORTE RESSEMBLANCE"

else:

    decision = "FAIBLE RESSEMBLANCE"

# ==================================================
# 18. AFFICHAGE DES RESULTATS
# ==================================================

print("\n========================================")
print("ANALYSE FINALE DE LA PIECE")
print("========================================")

print("\n1. COMPARAISON DES COULEURS")

print(
    f"Centre gris : {score_centre:.2f}"
)

print(
    f"Couronne dorée : {score_couronne:.2f}"
)

print("\n2. COMPARAISON DES DETAILS VISUELS")

print(
    f"ORB centre : {score_orb_centre:.2f} "
    f"({bonnes_centre} correspondances)"
)

print(
    f"ORB couronne : {score_orb_couronne:.2f} "
    f"({bonnes_couronne} correspondances)"
)

print("\n3. RECONNAISSANCE DES INSCRIPTIONS")

print("Texte de référence :")
print(texte_reference if texte_reference else "Non reconnu")

print("\nTexte de la pièce test :")
print(texte_test if texte_test else "Non reconnu")

if score_texte is not None:

    print(
        f"\nRessemblance textuelle : {score_texte:.2f} / 100"
    )

else:

    print(
        "\nRessemblance textuelle : indisponible"
    )

print("\n4. SCORE GLOBAL")

print(
    f"Score : {score_global:.2f} / 100"
)

print(
    f"Seuil provisoire : {SEUIL_RESEMBLANCE} / 100"
)

print("\n========================================")
print("DECISION :", decision)
print("========================================")

print(
    "\nAttention : les scores sont des indicateurs "
    "de ressemblance avec une seule photographie "
    "de référence. Ils ne certifient pas "
    "l'authenticité de la monnaie."
)

print(
    "\nLa reconnaissance OCR peut commettre des erreurs, "
    "notamment lorsque les inscriptions sont petites, "
    "usées, inclinées ou peu visibles."
)

# ==================================================
# 19. SAUVEGARDE DE LA VISUALISATION
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
    str(
        DOSSIER_REF / "decision_visualisation.png"
    ),
    visualisation
)

print(
    "\nVisualisation enregistrée dans :",
    DOSSIER_REF / "decision_visualisation.png"
)
