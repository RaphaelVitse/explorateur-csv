# app.py
import pandas as pd
import streamlit as st
import plotly.express as px
from analyse import charger_csv, analyser_dataframe, formater_contexte_pour_llm
from agent import poser_question, extraire_graphique, nettoyer_reponse
import time  # Ajoute cet import en haut de app.py

def afficher_graphique(df, type_graphique, col_x, col_y, agregation="sum"):
    try:
        if col_x == "date" and col_y == "ca":
            df_plot = df.copy()
            df_plot["date"] = pd.to_datetime(df_plot["date"])
            df_plot = df_plot.groupby(
                df_plot["date"].dt.to_period("M").astype(str)
            )["ca"].sum().reset_index()
            df_plot.columns = ["mois", "ca"]
            fig = px.line(df_plot, x="mois", y="ca", markers=True)
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{time.time()}")
            return

        if col_x not in df.columns:
            st.warning(f"Colonne '{col_x}' introuvable.")
            return

        if type_graphique == "bar":
            if col_y and col_y in df.columns:
                # On choisit sum ou mean selon le paramètre agregation
                if agregation == "mean":
                    df_plot = df.groupby(col_x)[col_y].mean().round(2).reset_index()
                else:
                    df_plot = df.groupby(col_x)[col_y].sum().reset_index()
                fig = px.bar(df_plot, x=col_x, y=col_y)
            else:
                df_plot = df[col_x].value_counts().reset_index()
                df_plot.columns = [col_x, "count"]
                fig = px.bar(df_plot, x=col_x, y="count")

        elif type_graphique == "line":
            if col_y and col_y in df.columns:
                if agregation == "mean":
                    df_plot = df.groupby(col_x)[col_y].mean().round(2).reset_index()
                else:
                    df_plot = df.groupby(col_x)[col_y].sum().reset_index()
                fig = px.line(df_plot, x=col_x, y=col_y, markers=True)
            else:
                st.warning("Colonne Y manquante.")
                return

        elif type_graphique == "pie":
            if col_y and col_y in df.columns:
                if agregation == "mean":
                    df_plot = df.groupby(col_x)[col_y].mean().round(2).reset_index()
                else:
                    df_plot = df.groupby(col_x)[col_y].sum().reset_index()
                fig = px.pie(df_plot, names=col_x, values=col_y)
            else:
                df_plot = df[col_x].value_counts().reset_index()
                df_plot.columns = [col_x, "count"]
                fig = px.pie(df_plot, names=col_x, values="count")

        elif type_graphique == "scatter":
            if col_y and col_y in df.columns:
                fig = px.scatter(df, x=col_x, y=col_y)
            else:
                st.warning("Colonne Y manquante.")
                return

        elif type_graphique == "histogram":
            fig = px.histogram(df, x=col_x)

        else:
            return

        st.plotly_chart(fig, use_container_width=True, key=f"chart_{time.time()}")

    except Exception as e:
        st.warning(f"Impossible de générer le graphique : {e}")

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Explorateur CSV",
    page_icon="📊",
    layout="wide"                       # Pleine largeur pour les graphiques
)

st.title("📊 Explorateur de données CSV")
st.caption("Uploade ton fichier CSV et pose des questions en langage naturel.")

# ── Initialisation de la session Streamlit ────────────────────────────────────
# st.session_state persiste les données entre les interactions
# Sans ça, chaque clic recharge tout depuis zéro
if "historique" not in st.session_state:
    st.session_state.historique = []    # Historique des échanges

if "contexte_csv" not in st.session_state:
    st.session_state.contexte_csv = None  # Contexte du CSV pour le LLM

if "df" not in st.session_state:
    st.session_state.df = None          # Le DataFrame pandas

# ── Sidebar : upload et infos du fichier ─────────────────────────────────────
with st.sidebar:
    st.header("Fichier CSV")

    fichier = st.file_uploader(
        label="Uploade ton CSV",
        type=["csv"],                   # On accepte uniquement les .csv
        help="Fichier CSV avec en-têtes en première ligne"
    )

    if fichier is not None:             # Si un fichier a été uploadé

        # On charge le CSV uniquement si c'est un nouveau fichier
        # fichier.name contient le nom du fichier
        if st.session_state.df is None or fichier.name != st.session_state.get("nom_fichier"):

            with st.spinner("Analyse du fichier..."):
                df, erreur = charger_csv(fichier)

                if erreur:
                    st.error(f"Erreur : {erreur}")
                else:
                    # On sauvegarde dans la session
                    st.session_state.df = df
                    st.session_state.nom_fichier = fichier.name
                    st.session_state.historique = []    # Reset historique

                    # On analyse et génère le contexte pour le LLM
                    infos = analyser_dataframe(df)
                    st.session_state.contexte_csv = formater_contexte_pour_llm(df, infos)

                    st.success(f"Fichier chargé !")

    # Infos du fichier chargé
    if st.session_state.df is not None:
        df = st.session_state.df
        st.divider()
        st.metric("Lignes", df.shape[0])        # metric() affiche un grand chiffre
        st.metric("Colonnes", df.shape[1])
        st.write("**Colonnes :**")
        for col in df.columns:
            st.write(f"- {col} ({df[col].dtype})")

        # Bouton pour réinitialiser
        if st.button("Nouveau fichier", use_container_width=True):
            st.session_state.df = None
            st.session_state.contexte_csv = None
            st.session_state.historique = []
            st.rerun()                  # Recharge la page


# ── Zone principale ───────────────────────────────────────────────────────────
if st.session_state.df is None:

    # Message d'accueil si pas de fichier chargé
    st.info("Upload un fichier CSV dans la barre latérale pour commencer.")

    # Exemples de questions pour guider l'utilisateur
    st.subheader("Exemples de questions que tu pourras poser :")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - Quel est le produit le plus vendu ?
        - Montre-moi l'évolution du CA par mois
        - Quelles sont les valeurs manquantes ?
        """)
    with col2:
        st.markdown("""
        - Quelle région performe le mieux ?
        - Y a-t-il des anomalies dans les données ?
        - Quel est le panier moyen par client ?
        """)

else:
    df = st.session_state.df

    # Onglets : Aperçu des données | Chat
    tab1, tab2 = st.tabs(["Aperçu des données", "Chat avec l'agent"])

    # ── Onglet 1 : Aperçu ────────────────────────────────────────────────────
    with tab1:
        st.subheader("Premières lignes")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Statistiques")
        st.dataframe(df.describe(), use_container_width=True)

    # ── Onglet 2 : Chat ──────────────────────────────────────────────────────
    with tab2:

        # Affichage de l'historique des échanges
        for echange in st.session_state.historique:
            with st.chat_message(echange["role"]):  # "user" ou "assistant"
                st.markdown(echange["content"])

                # Si l'échange contient un graphique, on le réaffiche
                if "graphique" in echange:
                    g = echange["graphique"]
                    afficher_graphique(df, g["type"], g["col_x"], g["col_y"])

        # Champ de saisie de la question
        question = st.chat_input("Pose une question sur tes données...")

        if question:

            # Affiche la question de l'utilisateur
            with st.chat_message("user"):
                st.markdown(question)

            # Appel au LLM
            with st.chat_message("assistant"):
                with st.spinner("Analyse en cours..."):
                    reponse_brute = poser_question(
                        question,
                        st.session_state.contexte_csv,
                        st.session_state.historique,
                        df                              # ← df est déjà disponible car on est dans le bloc "else"
)

                # Détection d'un graphique dans la réponse
                params_graphique = extraire_graphique(reponse_brute)

                # Affichage de la réponse texte (sans la ligne GRAPHIQUE:)
                reponse_propre = nettoyer_reponse(reponse_brute)
                st.markdown(reponse_propre)

                # Affichage du graphique si demandé
                if params_graphique:
                    afficher_graphique(
                        df,
                        params_graphique["type"],
                        params_graphique["col_x"],
                        params_graphique["col_y"]
                    )

            # Sauvegarde dans l'historique
            echange_user = {"role": "user", "content": question}
            echange_assistant = {
                "role": "assistant",
                "content": reponse_propre
            }
            if params_graphique:
                echange_assistant["graphique"] = params_graphique

            st.session_state.historique.append(echange_user)
            st.session_state.historique.append(echange_assistant)
