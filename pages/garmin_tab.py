import streamlit as st
import pandas as pd
import io
import gpxpy
from src.parser import open_gpx, GPXData
import authentification as auth


def format_garmin_df(df):
    df_display = df.copy()

    # Formater la date en JJ/MM/AAAA
    df_display["startTimeLocal"] = pd.to_datetime(df_display["startTimeLocal"]).dt.strftime("%d/%m/%Y")

    # Arrondir distance et dénivelé
    df_display["distance_km"] = df_display["distance_km"].round(1)
    df_display["elev_gain"] = df_display["elev_gain"].round(1)

    # Formater la durée en h:mm
    def format_duration(minutes):
        h = int(minutes // 60)
        m = int(minutes % 60)
        return f"{h}:{m:02d}"

    df_display["duration_minutes"] = df_display["duration_minutes"].apply(format_duration)

    # Renommer les colonnes pour affichage
    df_display = df_display.rename(columns={
        "startTimeLocal": "Date",
        "activityName": "Activité",
        "distance_km": "Distance (km)",
        "duration_minutes": "Durée",
        "elev_gain": "Dénivelé positif (m)",
        "Sélectionner": "Sélectionner"
    })

    # Garder seulement les colonnes utiles
    return df_display[["Date", "Activité", "Distance (km)", "Durée", "Dénivelé positif (m)", "Sélectionner"]]


def render_garmin_tab():
    st.subheader("Connexion Garmin Connect")

    if "garmin" not in st.session_state:
        auth.garmin_auth()

    else:
        garmin = st.session_state["garmin"]
        st.success(f"Bienvenue {garmin.get_full_name()} ✅")
        st.success("✅ Connecté à Garmin Connect")

        if st.button("Se déconnecter"):
            auth.garmin_logout()

        # Sélecteurs de dates
        st.write("### 📅 Sélection des activités")
        col1, col2 = st.columns(2)
        date_debut = col1.date_input("Date début")
        date_fin = col2.date_input("Date fin")

        if st.button("Charger les activités"):
            try:
                activities = garmin.get_activities_by_date(date_debut, date_fin)
                st.session_state["activities"] = activities
                st.success(f"{len(activities)} activités récupérées ✅")
            except Exception as e:
                st.error(f"Erreur récupération activités : {e}")

        if "activities" in st.session_state:
            df = pd.DataFrame(st.session_state["activities"])

            # Normalisation colonnes
            df["distance_km"] = df["distance"] / 1000
            df["duration_minutes"] = df["duration"] / 60
            df["elev_gain"] = df["elevationGain"]
            df["Sélectionner"] = False  # checkbox colonne

            # Filtres
            st.write("### 🎛️ Filtres")
            min_dist = st.number_input("Distance min (km)", 0.0, 500.0, 0.0, step=1.0)
            max_dist = st.number_input("Distance max (km)", 0.0, 500.0, 500.0, step=1.0)
            min_dplus = st.number_input("Dénivelé + min (m)", 0, 10000, 0, step=100)
            min_duree = st.number_input("Durée min (min)", 0, 10000, 0, step=10)

            df_filtered = df[
                (df["distance_km"] >= min_dist)
                & (df["distance_km"] <= max_dist)
                & (df["elev_gain"] >= min_dplus)
                & (df["duration_minutes"] >= min_duree)
            ]

            df_filtered_display = format_garmin_df(df_filtered)

            # Tableau interactif
            edited_df = st.data_editor(
                df_filtered_display,
                hide_index=True,
                use_container_width=True,
                num_rows="dynamic",
            )

            # Bouton pour analyser la sélection
            selected = edited_df[edited_df["Sélectionner"] == True]
            if not selected.empty and st.button("📥 Analyser la sélection"):
                try:
                    # On prend le premier sélectionné (tu peux adapter pour multi)
                    act_id = selected.iloc[0]["activityId"]
                    raw_bytes = garmin.download_activity(act_id, garmin.ActivityDownloadFormat.GPX)
                    gpx = gpxpy.parse(io.BytesIO(raw_bytes))
                    route = open_gpx(io.BytesIO(raw_bytes))
                    st.session_state["route"] = route
                    st.success(f"✅ Activité {act_id} envoyée dans l’onglet Analyse GPX !")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur lors du téléchargement/analyse GPX : {e}")
