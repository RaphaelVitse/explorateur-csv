# analyse.py

import pandas as pd


def charger_csv(fichier):
    """
    Charge un fichier CSV uploadé par Streamlit.
    Streamlit retourne un objet BytesIO, pas un chemin de fichier.
    """
    try:
        df = pd.read_csv(fichier)       # pandas lit directement l'objet BytesIO
        return df, None                 # On retourne le DataFrame et aucune erreur
    except Exception as e:
        return None, str(e)             # En cas d'erreur, on retourne le message


def analyser_dataframe(df):
    """
    Génère un résumé complet du DataFrame pour le donner au LLM.
    Le LLM a besoin de connaître la structure des données
    pour répondre intelligemment aux questions.
    """

    infos = {}

    # Dimensions du fichier
    infos["nombre_lignes"] = df.shape[0]        # shape retourne (lignes, colonnes)
    infos["nombre_colonnes"] = df.shape[1]

    # Noms et types de chaque colonne
    # dtypes retourne un dict {nom_colonne: type}
    # astype(str) convertit les types en texte lisible ("int64" → "int64")
    infos["colonnes"] = {
        col: str(df[col].dtype) for col in df.columns
    }

    # Valeurs manquantes par colonne
    # isnull() → True/False par cellule
    # sum() → compte les True (= valeurs manquantes)
    infos["valeurs_manquantes"] = df.isnull().sum().to_dict()

    # Aperçu des 3 premières lignes en texte
    # to_string() convertit le DataFrame en texte lisible
    infos["apercu"] = df.head(3).to_string()

    # Statistiques pour les colonnes numériques
    # describe() calcule count, mean, std, min, max...
    # include="number" → uniquement les colonnes numériques
    numeriques = df.describe(include="number")
    if not numeriques.empty:            # S'il y a au moins une colonne numérique
        infos["stats_numeriques"] = numeriques.to_string()

    # Liste des valeurs uniques pour les colonnes catégorielles
    # (colonnes avec peu de valeurs différentes — utile pour les filtres)
    infos["valeurs_uniques"] = {}
    for col in df.select_dtypes(include="object").columns:  # "object" = texte
        nb_uniques = df[col].nunique()          # Nombre de valeurs distinctes
        if nb_uniques <= 20:                    # On liste seulement si moins de 20 valeurs
            infos["valeurs_uniques"][col] = df[col].unique().tolist()

    return infos


def formater_contexte_pour_llm(df, infos):
    """
    Transforme les infos du DataFrame en texte structuré
    que le LLM va recevoir comme contexte.
    """

    texte = f"""Voici les informations sur le fichier CSV chargé :

Dimensions : {infos['nombre_lignes']} lignes × {infos['nombre_colonnes']} colonnes

Colonnes et types :
"""
    for col, dtype in infos["colonnes"].items():
        manquants = infos["valeurs_manquantes"].get(col, 0)
        texte += f"  - {col} ({dtype})"
        if manquants > 0:
            texte += f" — {manquants} valeurs manquantes"
        texte += "\n"

    texte += f"\nAperçu des premières lignes :\n{infos['apercu']}\n"

    if "stats_numeriques" in infos:
        texte += f"\nStatistiques numériques :\n{infos['stats_numeriques']}\n"

    if infos["valeurs_uniques"]:
        texte += "\nValeurs uniques (colonnes catégorielles) :\n"
        for col, valeurs in infos["valeurs_uniques"].items():
            texte += f"  - {col} : {valeurs}\n"

    return texte
