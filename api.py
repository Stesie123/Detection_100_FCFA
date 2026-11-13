
import cv2
import numpy as np
import pytesseract
import re
import unicodedata

from pathlib import Path
from difflib import SequenceMatcher
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE = Path(__file__).parent

DOSSIER_REF = BASE / "reference_100_fcfa"

IMAGE_REFERENCE = BASE / "dataset" / "images" / "100_reference.jpeg"

SIGNATURES_PATH = DOSSIER_REF / "signatures_reference.npz"
ORB_CENTRE_PATH = DOSSIER_REF / "orb_centre.npy"
ORB_COURONNE_PATH = DOSSIER_REF / "orb_couronne.npy"

TAILLE_MAX_IMAGE = 10 * 1024 * 1024

SEUIL_RESEMBLANCE = 60

RAYON_MINIMUM = 40
DESCRIPTEURS_MINIMUM = 10

POIDS_CENTRE = 0.2125
POIDS_COURONNE = 0.2975
POIDS_ORB_CENTRE = 0.17
POIDS_ORB_COURONNE = 0.17
POIDS_TEXTE = 0.15

# Paramètres CLAHE : identiques à ceux de extraire_signatures.py
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Paramètres ORB : identiques à ceux de extraire_signatures.py
ORB_PARAMETRES = {
    "nfeatures": 1000,
    "scaleFactor": 1.2,
    "nlevels": 8,
    "edgeThreshold": 15,
    "patchSize": 31,
    "fastThreshold": 10
}

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ==================================================
# 2. CREATION DE L'API
# ==================================================

app = FastAPI(
    title="API Reconnaissance 100 FCFA",
    description="API d'analyse visuelle d'une pièce de 100 FCFA.",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==================================================
# 3. PRETRAITEMENT DES IMAGES
# ==================================================

def ameliorer_image_orb(image):
    """
    Améliore le contraste de l'image en niveaux de gris
    avant l'extraction des caractéristiques ORB.
    """

    gris = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )

    return clahe.apply(gris)


def ameliorer_image_hsv(image):
    """
    Applique CLAHE sur le canal de luminosité V.
    Les histogrammes de comparaison utilisent H et S,
    comme dans le script d'extraction des références.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    canal_h, canal_s, canal_v = cv2.split(hsv)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )

    canal_v = clahe.apply(canal_v)

    return cv2.merge([
        canal_h,
        canal_s,
        canal_v
    ])


# ==================================================
# 4. CHARGEMENT DES REFERENCES
# ==================================================

def charger_references():

    fichiers = [
        IMAGE_REFERENCE,
        SIGNATURES_PATH,
        ORB_CENTRE_PATH,
        ORB_COURONNE_PATH
    ]

    for fichier in fichiers:

        if not fichier.exists():

            raise RuntimeError(
                f"Fichier de référence introuvable : {fichier}"
            )

    signatures = np.load(
        str(SIGNATURES_PATH)
    )

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

    image_reference = cv2.imread(
        str(IMAGE_REFERENCE)
    )

    if image_reference is None:

        raise RuntimeError(
            "Impossible de lire l'image de référence."
        )

    if len(orb_reference_centre) < 2:

        raise RuntimeError(
            "La référence ne contient pas assez de caractéristiques ORB au centre."
        )

    if len(orb_reference_couronne) < 2:

        raise RuntimeError(
            "La référence ne contient pas assez de caractéristiques ORB dans la couronne."
        )

    return {
        "image": image_reference,
        "hist_centre": hist_reference_centre,
        "hist_couronne": hist_reference_couronne,
        "orb_centre": orb_reference_centre,
        "orb_couronne": orb_reference_couronne
    }


# ==================================================
# 5. FONCTIONS OCR
# ==================================================

def preparer_image_ocr(image):

    gris = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gris = cv2.resize(
        gris,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    gris = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gris)

    seuil = cv2.adaptiveThreshold(
        gris,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return seuil


def lire_texte(image):

    seuil = preparer_image_ocr(image)

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

    return max(
        textes,
        key=len
    )


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

    return re.sub(
        r"[^A-Z0-9]",
        "",
        texte
    )


def comparer_textes(reference, test):

    reference = normaliser_texte(reference)
    test = normaliser_texte(test)

    if not reference or not test:
        return None

    return float(
        SequenceMatcher(
            None,
            reference,
            test
        ).ratio() * 100
    )


# ==================================================
# 6. COMPARAISON HSV
# ==================================================

def comparer_couleurs(reference, test):

    correlation = cv2.compareHist(
        reference,
        test,
        cv2.HISTCMP_CORREL
    )

    score = ((correlation + 1) / 2) * 100

    return float(
        np.clip(score, 0, 100)
    )


# ==================================================
# 7. COMPARAISON ORB
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


# ==================================================
# 8. DECISION DE COMPATIBILITE
# ==================================================

def determiner_decision(
    score_global,
    rayon,
    nombre_descripteurs
):

    if rayon < RAYON_MINIMUM:

        return {
            "decision": "INDETERMINE",
            "compatible": None,
            "motif": (
                "La pièce est trop petite dans l'image. "
                "Prenez une photo plus rapprochée et plus nette."
            )
        }

    if nombre_descripteurs < DESCRIPTEURS_MINIMUM:

        return {
            "decision": "INDETERMINE",
            "compatible": None,
            "motif": (
                "L'image ne contient pas suffisamment "
                "de caractéristiques visuelles exploitables."
            )
        }

    if score_global >= SEUIL_RESEMBLANCE:

        return {
            "decision": "COMPATIBLE",
            "compatible": True,
            "motif": (
                "La pièce présente une ressemblance suffisante "
                "avec la référence disponible."
            )
        }

    return {
        "decision": "NON COMPATIBLE",
        "compatible": False,
        "motif": (
            "La pièce présente une ressemblance insuffisante "
            "avec la référence disponible."
        )
    }


# ==================================================
# 9. ANALYSE D'UNE IMAGE
# ==================================================

def analyser_image(image, references):

    hauteur, largeur = image.shape[:2]

    dimension_min = min(
        hauteur,
        largeur
    )

    gris = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Amélioration du contraste pour ORB

    gris_ameliore = ameliorer_image_orb(
        image
    )

    # Préparation HSV

    hsv = ameliorer_image_hsv(
        image
    )

    # Détection circulaire sur l'image originale

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
            "Aucune pièce circulaire détectée. "
            "Essayez une photo plus nette."
        )

    cercles = np.round(
        cercles[0]
    ).astype(int)

    cx, cy, rayon = max(
        cercles,
        key=lambda cercle: cercle[2]
    )

    rayon = int(rayon * 1.04)

    # Création des masques

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
            "Impossible d'isoler les zones de la pièce."
        )

    # Extraction de la pièce pour l'OCR

    x1 = max(
        0,
        cx - rayon
    )

    y1 = max(
        0,
        cy - rayon
    )

    x2 = min(
        largeur,
        cx + rayon
    )

    y2 = min(
        hauteur,
        cy + rayon
    )

    piece_ocr = image[
        y1:y2,
        x1:x2
    ].copy()

    masque_ocr = masque_piece[
        y1:y2,
        x1:x2
    ]

    piece_ocr[masque_ocr == 0] = 255

    texte_reference = lire_texte(
        references["image"]
    )

    texte_test = lire_texte(
        piece_ocr
    )

    # ==================================================
    # HISTOGRAMMES HSV
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

    score_centre = comparer_couleurs(
        references["hist_centre"],
        hist_test_centre
    )

    score_couronne = comparer_couleurs(
        references["hist_couronne"],
        hist_test_couronne
    )

    # ==================================================
    # EXTRACTION ORB
    # ==================================================

    orb = cv2.ORB_create(
        **ORB_PARAMETRES
    )

    points_centre, descripteurs_centre = (
        orb.detectAndCompute(
            gris_ameliore,
            masque_centre
        )
    )

    points_couronne, descripteurs_couronne = (
        orb.detectAndCompute(
            gris_ameliore,
            masque_couronne
        )
    )

    score_orb_centre, bonnes_centre = comparer_orb(
        references["orb_centre"],
        descripteurs_centre
    )

    score_orb_couronne, bonnes_couronne = comparer_orb(
        references["orb_couronne"],
        descripteurs_couronne
    )

    # Nombre de caractéristiques détectées

    nombre_descripteurs_centre = (
        len(descripteurs_centre)
        if descripteurs_centre is not None
        else 0
    )

    nombre_descripteurs_couronne = (
        len(descripteurs_couronne)
        if descripteurs_couronne is not None
        else 0
    )

    nombre_descripteurs = (
        nombre_descripteurs_centre
        + nombre_descripteurs_couronne
    )

    # ==================================================
    # COMPARAISON DES INSCRIPTIONS
    # ==================================================

    score_texte = comparer_textes(
        texte_reference,
        texte_test
    )

    # ==================================================
    # CALCUL DU SCORE GLOBAL
    # ==================================================

    criteres = [
        (score_centre, POIDS_CENTRE),
        (score_couronne, POIDS_COURONNE),
        (score_orb_centre, POIDS_ORB_CENTRE),
        (score_orb_couronne, POIDS_ORB_COURONNE),
        (score_texte, POIDS_TEXTE)
    ]

    poids_actifs = sum(
        poids
        for score, poids in criteres
        if score is not None
    )

    if poids_actifs == 0:

        raise ValueError(
            "Impossible de calculer le score de comparaison."
        )

    score_global = sum(
        score * poids
        for score, poids in criteres
        if score is not None
    ) / poids_actifs

    score_global = float(
        score_global
    )

    # ==================================================
    # DECISION
    # ==================================================

    resultat_decision = determiner_decision(
        score_global,
        rayon,
        nombre_descripteurs
    )

    # ==================================================
    # RESULTAT ENVOYE AU FRONTEND
    # ==================================================

    return {
        "decision": resultat_decision["decision"],
        "compatible": resultat_decision["compatible"],
        "motif_decision": resultat_decision["motif"],

        "denomination": "100 FCFA",

        "score_global": round(
            score_global,
            2
        ),

        "seuil_provisoire": SEUIL_RESEMBLANCE,

        "qualite_analyse": {
            "rayon_piece": int(rayon),
            "descripteurs_centre": nombre_descripteurs_centre,
            "descripteurs_couronne": nombre_descripteurs_couronne,
            "total_descripteurs": nombre_descripteurs
        },

        "couleurs": {
            "centre": round(score_centre, 2),
            "couronne": round(score_couronne, 2)
        },

        "details_visuels": {
            "orb_centre": round(score_orb_centre, 2),
            "orb_couronne": round(score_orb_couronne, 2),
            "correspondances_centre": bonnes_centre,
            "correspondances_couronne": bonnes_couronne
        },

        "inscriptions": {
            "reference": texte_reference,
            "piece_test": texte_test,
            "score": (
                round(score_texte, 2)
                if score_texte is not None
                else None
            )
        },

        "detection": {
            "centre_x": int(cx),
            "centre_y": int(cy),
            "rayon": int(rayon),
            "largeur": largeur,
            "hauteur": hauteur
        },

        "avertissement": (
            "Cette analyse mesure une ressemblance avec une seule "
            "photographie de référence. Elle ne certifie pas "
            "l'authenticité de la monnaie. Une pièce non compatible "
            "n'est pas nécessairement une fausse pièce."
        ),

        "nom_fichier": None
    }


# ==================================================
# 10. ROUTES DE L'API
# ==================================================

@app.get("/")
def accueil():

    return {
        "message": "API de reconnaissance 100 FCFA opérationnelle",
        "version": "1.2.0"
    }


@app.get("/api/health")
def verifier_api():

    try:

        charger_references()

        return {
            "status": "ok",
            "message": "API et fichiers de référence disponibles."
        }

    except Exception as erreur:

        raise HTTPException(
            status_code=500,
            detail=str(erreur)
        )


@app.post("/api/recognition/analyze")
async def reconnaitre_piece(
    file: UploadFile = File(...)
):

    try:

        if (
            not file.content_type
            or not file.content_type.startswith("image/")
        ):

            raise HTTPException(
                status_code=400,
                detail="Veuillez envoyer un fichier image."
            )

        contenu = await file.read(
            TAILLE_MAX_IMAGE + 1
        )

        if not contenu:

            raise HTTPException(
                status_code=400,
                detail="Le fichier envoyé est vide."
            )

        if len(contenu) > TAILLE_MAX_IMAGE:

            raise HTTPException(
                status_code=413,
                detail="L'image dépasse la limite de 10 Mo."
            )

        tableau = np.frombuffer(
            contenu,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            tableau,
            cv2.IMREAD_COLOR
        )

        if image is None:

            raise ValueError(
                "Impossible de lire cette image."
            )

        references = charger_references()

        resultat = analyser_image(
            image,
            references
        )

        resultat["nom_fichier"] = file.filename

        return resultat

    except HTTPException:
        raise

    except ValueError as erreur:

        raise HTTPException(
            status_code=422,
            detail=str(erreur)
        )

    except Exception as erreur:

        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant l'analyse : {str(erreur)}"
        )

    finally:

        await file.close()
