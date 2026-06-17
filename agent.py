# agent.py

import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def calculer_reponse_directe(question, df):

    question_lower = question.lower()

    # CA moyen par vendeur — DOIT être avant "meilleur vendeur"
    if ("moyen" in question_lower or "moyenne" in question_lower) and "vendeur" in question_lower:
        if "vendeur" in df.columns and "ca" in df.columns:
            result = df.groupby("vendeur")["ca"].mean().round(2).sort_values(ascending=False)
            texte = "CA moyen par transaction pour chaque vendeur :\n\n"
            for vendeur, moyenne in result.items():
                texte += f"  - {vendeur} : {moyenne:,.2f}€\n"
            return texte

    # Meilleur vendeur par CA total
    if "meilleur vendeur" in question_lower or "top vendeur" in question_lower:
        if "vendeur" in df.columns and "ca" in df.columns:
            result = df.groupby("vendeur")["ca"].sum().sort_values(ascending=False)
            top = result.index[0]
            return f"Classement des vendeurs par CA total :\n\n{result.to_string()}\n\nMeilleur vendeur : **{top}** avec **{result[top]:,.0f}€**"

    # CA par produit
    if "produit" in question_lower and ("ca" in question_lower or "vente" in question_lower or "meilleur" in question_lower):
        if "produit" in df.columns and "ca" in df.columns:
            result = df.groupby("produit")["ca"].sum().sort_values(ascending=False)
            top = result.index[0]
            return f"CA total par produit :\n\n{result.to_string()}\n\nMeilleur produit : **{top}** avec **{result[top]:,.0f}€**"

    # CA par région
    if "région" in question_lower or "region" in question_lower:
        if "region" in df.columns and "ca" in df.columns:
            result = df.groupby("region")["ca"].sum().sort_values(ascending=False)
            top = result.index[0]
            return f"CA total par région :\n\n{result.to_string()}\n\nMeilleure région : **{top}** avec **{result[top]:,.0f}€**"

    # CA par mois
    if "mois" in question_lower or "évolution" in question_lower or "evolution" in question_lower:
        if "date" in df.columns and "ca" in df.columns:
            df_copy = df.copy()
            df_copy["date"] = pd.to_datetime(df_copy["date"])
            result = df_copy.groupby(df_copy["date"].dt.to_period("M"))["ca"].sum()
            return f"CA total par mois :\n\n{result.to_string()}"

    # Valeurs manquantes
    if "manquant" in question_lower or "null" in question_lower or "vide" in question_lower:
        manquants = df.isnull().sum()
        total = manquants.sum()
        if total == 0:
            return "Aucune valeur manquante dans le fichier."
        return f"Valeurs manquantes par colonne :\n\n{manquants[manquants > 0].to_string()}"

    # CA total
    if "ca total" in question_lower or "chiffre d'affaires total" in question_lower:
        if "ca" in df.columns:
            total = df["ca"].sum()
            return f"Chiffre d'affaires total : **{total:,.0f}€**"

    return None



def construire_systeme(contexte_csv):
    return f"""Tu es un assistant expert en analyse de données.
Tu aides l'utilisateur à explorer et comprendre son fichier CSV.

Voici les données disponibles :
{contexte_csv}

Règles importantes :
- Réponds toujours en français
- Sois précis et concis
- Si la question porte sur des chiffres, donne des valeurs exactes
- Pour suggérer un graphique, utilise TOUJOURS ce format exact en fin de réponse :
  GRAPHIQUE: type — colonne_x — colonne_y
  Pour une moyenne, ajoute _mean au type :
  GRAPHIQUE: bar_mean — vendeur — ca
  Exemples :
  GRAPHIQUE: bar — vendeur — ca
  GRAPHIQUE: bar_mean — vendeur — ca
  GRAPHIQUE: line — date — ca
  GRAPHIQUE: pie — categorie — ca
- La colonne_y est OBLIGATOIRE sauf pour histogram
- Si tu ne peux pas répondre avec les données disponibles, dis-le clairement"""


def poser_question(question, contexte_csv, historique, df):
    """
    1. Essaie de répondre directement avec pandas
    2. Si pas de réponse directe, envoie au LLM
    """

    # Étape 1 : calcul direct pandas
    reponse_directe = calculer_reponse_directe(question, df)

    if reponse_directe:
        # On demande quand même au LLM d'ajouter un graphique si pertinent
        messages = [
            {"role": "system", "content": construire_systeme(contexte_csv)},
            {"role": "user", "content": f"""L'utilisateur demande : {question}
Voici la réponse calculée directement depuis les données :
{reponse_directe}

Confirme cette réponse en une phrase et suggère un graphique si pertinent."""}
        ]
    else:
        # Étape 2 : on laisse le LLM répondre
        messages = [{"role": "system", "content": construire_systeme(contexte_csv)}]
        for echange in historique:
            messages.append({"role": echange["role"], "content": echange["content"]})
        messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1500,
        messages=messages
    )

    return response.choices[0].message.content


def extraire_graphique(reponse):
    for ligne in reponse.split("\n"):
        if "GRAPHIQUE:" in ligne:
            try:
                contenu = ligne[ligne.index("GRAPHIQUE:") + len("GRAPHIQUE:"):].strip()
                parties = [p.strip() for p in contenu.split("—")]
                if len(parties) >= 2:
                    col_y = parties[2] if len(parties) > 2 else None
                    if parties[1] == "date" and not col_y:
                        col_y = "ca"

                    # Détecte si c'est une moyenne
                    agregation = "mean" if "mean" in parties[0] or len(parties) > 3 and "mean" in parties[3] else "sum"
                    type_graphique = parties[0].replace("_mean", "").strip()

                    return {
                        "type": type_graphique,
                        "col_x": parties[1],
                        "col_y": col_y,
                        "agregation": agregation
                    }
            except Exception:
                return None
    return None


def nettoyer_reponse(reponse):
    """
    Retire toute mention de GRAPHIQUE: de la réponse.
    On cherche dans tout le texte, pas seulement en début de ligne.
    """
    lignes = reponse.split("\n")
    lignes_propres = []

    for ligne in lignes:
        if "GRAPHIQUE:" in ligne:           # ← cherche n'importe où dans la ligne
            # Si la ligne contient autre chose que GRAPHIQUE:, on garde le reste
            partie_texte = ligne[:ligne.index("GRAPHIQUE:")].strip()
            if partie_texte:                # Si il y a du texte avant GRAPHIQUE:
                lignes_propres.append(partie_texte)
            # Sinon on ignore toute la ligne
        else:
            lignes_propres.append(ligne)

    return "\n".join(lignes_propres).strip()
