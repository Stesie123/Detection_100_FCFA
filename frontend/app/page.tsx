
"use client";

import { useState } from "react";

type Resultat = {
  denomination?: string;
  decision: string;
  compatible: boolean | null;
  motif_decision?: string;
  score_global: number;
  seuil_provisoire: number;
  couleurs: {
    centre: number;
    couronne: number;
  };
  avertissement: string;
  nom_fichier?: string;
};

export default function Home() {
  const [fichier, setFichier] = useState<File | null>(null);
  const [apercu, setApercu] = useState("");
  const [resultat, setResultat] = useState<Resultat | null>(null);
  const [chargement, setChargement] = useState(false);
  const [erreur, setErreur] = useState("");

  function importerPhoto(event: React.ChangeEvent<HTMLInputElement>) {
    const image = event.target.files?.[0];

    if (!image) return;

    if (!["image/jpeg", "image/png"].includes(image.type)) {
      setErreur("Veuillez sélectionner une image JPG, JPEG ou PNG.");
      return;
    }

    if (image.size > 10 * 1024 * 1024) {
      setErreur("L'image ne doit pas dépasser 10 Mo.");
      return;
    }

    if (apercu) URL.revokeObjectURL(apercu);

    setFichier(image);
    setApercu(URL.createObjectURL(image));
    setResultat(null);
    setErreur("");
  }

  async function analyser() {
    if (!fichier) return;

    setChargement(true);
    setErreur("");
    setResultat(null);

    try {
      const donnees = new FormData();
      donnees.append("file", fichier);

      const reponse = await fetch(
        "http://127.0.0.1:8000/api/recognition/analyze",
        {
          method: "POST",
          body: donnees,
        }
      );

      const data = await reponse.json();

      if (!reponse.ok) {
        throw new Error(data.detail || "Erreur pendant l'analyse.");
      }

      setResultat(data);
    } catch (error) {
      setErreur(
        error instanceof Error
          ? error.message
          : "Impossible de contacter le serveur."
      );
    } finally {
      setChargement(false);
    }
  }

  function nouvelleAnalyse() {
    if (apercu) URL.revokeObjectURL(apercu);

    setFichier(null);
    setApercu("");
    setResultat(null);
    setErreur("");
  }

  const statut = resultat?.compatible;

  const couleurStatut =
    statut === true
      ? "border-green-200 bg-green-50 text-green-800"
      : statut === false
        ? "border-red-200 bg-red-50 text-red-800"
        : "border-amber-200 bg-amber-50 text-amber-800";

  const titreStatut =
    statut === true
      ? "COMPATIBLE AVEC LA RÉFÉRENCE"
      : statut === false
        ? "NON COMPATIBLE AVEC LA RÉFÉRENCE"
        : "ANALYSE INDÉTERMINÉE";

  return (
    <main className="min-h-screen bg-gray-100 px-4 py-12 text-gray-800">
      <div className="mx-auto max-w-3xl">

        <header className="mb-10 text-center">
          <h1 className="text-3xl font-bold text-blue-900">
            RECONNAISSANCE DES PIÈCES BEAC
          </h1>

          <p className="mt-3 text-gray-500">
            Identification et analyse visuelle des pièces CFA
          </p>
        </header>

        <div className="grid gap-6 md:grid-cols-2">

          {/* Sélection de l'image */}
          <section className="rounded-xl bg-white p-6 shadow-sm">
            <h2 className="mb-5 text-lg font-semibold">
              1. Sélectionner une image
            </h2>

            <label className="flex min-h-48 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-gray-300 bg-gray-50 p-4 text-center transition hover:border-blue-500">

              {apercu ? (
                <img
                  src={apercu}
                  alt="Aperçu de la pièce"
                  className="max-h-44 rounded-lg object-contain"
                />
              ) : (
                <>
                  <span className="mb-3 text-4xl text-blue-600">+</span>

                  <span className="font-medium">
                    Déposer une image ou choisir un fichier
                  </span>

                  <span className="mt-2 text-xs text-gray-400">
                    JPG · JPEG · PNG
                  </span>
                </>
              )}

              <input
                type="file"
                accept="image/jpeg,image/png"
                capture="environment"
                onChange={importerPhoto}
                className="hidden"
              />
            </label>

            {fichier && (
              <p className="mt-3 truncate text-center text-xs text-gray-500">
                {fichier.name}
              </p>
            )}

            <button
              onClick={analyser}
              disabled={!fichier || chargement}
              className="mt-5 w-full rounded-lg bg-blue-800 px-4 py-3 font-semibold text-white transition hover:bg-blue-900 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {chargement ? "Analyse en cours..." : "Analyser la pièce"}
            </button>

            {fichier && (
              <button
                onClick={nouvelleAnalyse}
                disabled={chargement}
                className="mt-3 w-full rounded-lg border border-gray-300 px-4 py-2 text-sm text-gray-600 hover:bg-gray-50 disabled:opacity-40"
              >
                Nouvelle analyse
              </button>
            )}

            {erreur && (
              <p className="mt-4 text-center text-sm text-red-600">
                {erreur}
              </p>
            )}
          </section>

          {/* Résultat */}
          <section className="rounded-xl bg-white p-6 shadow-sm">
            <h2 className="mb-5 text-lg font-semibold">
              2. Résultat de l'analyse
            </h2>

            {!resultat && !chargement && (
              <div className="flex min-h-48 items-center justify-center text-center text-sm text-gray-400">
                Le résultat apparaîtra ici après l'analyse.
              </div>
            )}

            {chargement && (
              <div className="flex min-h-48 items-center justify-center text-center text-sm text-blue-700">
                Analyse de l'image en cours...
              </div>
            )}

            {resultat && (
              <div className="space-y-4">

                <div className="rounded-lg bg-blue-50 p-4">
                  <p className="text-sm text-gray-500">
                    Dénomination
                  </p>

                  <p className="mt-1 text-2xl font-bold text-blue-900">
                    {resultat.denomination || "100 FCFA"}
                  </p>
                </div>

                <div className={`rounded-lg border p-4 ${couleurStatut}`}>
                  <p className="text-xs font-medium">
                    Résultat de la comparaison
                  </p>

                  <p className="mt-2 text-lg font-bold">
                    {titreStatut}
                  </p>
                </div>

                <div>
                  <p className="text-sm text-gray-500">
                    Score global de ressemblance
                  </p>

                  <p className="text-xl font-bold">
                    {Number(resultat.score_global).toFixed(2)}/100
                  </p>
                </div>

                <div className="rounded-lg border border-gray-200 p-4">
                  <p className="text-sm font-semibold text-gray-700">
                    Avertissement
                  </p>

                  <p className="mt-2 text-sm leading-5 text-gray-500">
                    {resultat.avertissement}
                  </p>
                </div>

                <p className="text-xs leading-5 text-gray-400">
                  Une faible ressemblance ne prouve pas qu'une pièce est
                  fausse. Cette analyse ne constitue pas une authentification
                  officielle de la monnaie.
                </p>

              </div>
            )}
          </section>

        </div>
      </div>
    </main>
  );
}
