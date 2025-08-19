import email
import pandas as pd
import numpy as np
import climb_detector as climb_detector
import get_locations as locator
import parser as parser
import plotter as plotter
import util as util

import streamlit as st
from streamlit_folium import st_folium

import garminconnect


st.set_page_config(layout="wide", page_title="GPX Analyzer 📍")

# ---- Paramètres par défaut ----
DEFAULT_PARAMS = {
    "max_pause_length_m": 200,
    "max_pause_descent_m": 10,
    "start_threshold_slope": 2.0
}
localisation_name_precision = 10000
window_m = 100

st.title("GPX Climb Analyzer")

tab_params, tab_analysis = st.tabs(["⚙️ Paramètres avancés", "📈 Analyse"])
with tab_params:
    st.subheader("Réglage des paramètres")

    advanced_mode = st.checkbox("Activer le mode avancé", value=False, key="advanced_mode")

    if advanced_mode:

        params = {
            "max_pause_length_m": st.number_input(
                "Longueur max de pause (m)", value=DEFAULT_PARAMS["max_pause_length_m"], min_value=0
            ),
            "max_pause_descent_m": st.number_input(
                "Descente max en pause (m)", value=DEFAULT_PARAMS["max_pause_descent_m"], min_value=0
            ),
            "start_threshold_slope": st.number_input(
                "Pente seuil de départ (%)", value=DEFAULT_PARAMS["start_threshold_slope"], min_value=1.0, step=0.1
            ),
            }
        localisation_name_precision = st.number_input(
            "Précision des noms de lieux", value=localisation_name_precision, min_value=10000, step=1
        )
        window_m = st.number_input(
            "Précision du lissage pour le calcul de la pente", value=window_m, min_value=1, step=1
        )

        if st.button("💾 Sauvegarder & Valider"):
            # Sauvegarde dans la session pour l'utiliser dans l'onglet Analyse
            st.session_state["params"] = params
            st.session_state["localisation_name_precision"] = localisation_name_precision
            st.session_state["window_m"] = window_m
            st.success("Paramètres mis à jour ✅")
            st.table(pd.DataFrame(params.items(), columns=["Paramètre", "Valeur"]))

    else:
        st.markdown("Par défaut, les paramètres suivants sont utilisés :")
        st.subheader("Paramètres par défaut")
        display_param = DEFAULT_PARAMS.copy()
        display_param["localisation_name_precision"] = localisation_name_precision
        display_param["window_m"] = window_m

        df_param = pd.DataFrame(display_param.items(), columns=["Paramètre", "Valeur"])
        df_param.round(1)
        st.table(df_param)

        params = DEFAULT_PARAMS

with tab_analysis:
    st.subheader("Analyse des données")

    with st.sidebar:

        # Barre latérale avec les choix
        option = st.sidebar.radio(
            "Choisissez une source :",
            ("Import a GPX file", "Authentification with Strava", "Authentification with Garmin")
        )

        if option == "Import a GPX file":
            uploaded_file = st.file_uploader("Import a GPX file", type=["gpx"])

        elif option == "Authentification with Strava":
            st.info("🚧 Work in Progress : connexion Strava")

        elif option == "Authentification with Garmin":
            email = st.text_input("Enter login")
            password = st.text_input("Enter a password", type="password")
            garmin = garminconnect.Garmin(email, password)
            garmin.login()

            st.write(f"Bienvenue {garmin.display_name} !")
            st.info("🚧 Work in Progress : connexion Garmin")


    if option == "Import a GPX file":
        if uploaded_file is not None:
            try:
                route = parser.open_gpx(uploaded_file)
                st.success("✅ Fichier GPX chargé avec succès !")
                if len(route.gpx.tracks) > 1:
                    raise ValueError("Le fichier GPX contient plus d'un tracé. Veuillez fournir un fichier avec un seul tracé.")
                df = parser.parse_gpx(route)
                route.df = parser.apply_slope_smoothing(route.df, target_meters=100)
                stats = parser.compute_stats_gpx(route)

                climbs_df = climb_detector.detect_significant_segments(route.df, kind="climb", **params)
                descents_df = climb_detector.detect_significant_segments(route.df, kind="descent", **params)

                route.descents_df = descents_df
                route.climbs_df = climbs_df

                df_with_locations = locator.add_location(route, localisation_name_precision)

                # st.write(stats)
                st.header("📊 Statistiques du parcours")
                col1, col2, col3 = st.columns(3)
                col1.metric("Distance", f"{stats['total_distance_km']:.2f} km")
                col2.metric("Dénivelé +", f"{stats['elevation_gain']:.0f} m")
                col3.metric("Dénivelé -", f"{stats['elevation_loss']:.0f} m")

                col4, col5, col6 = st.columns(3)
                col4.metric("Altitude min", f"{stats['min_elevation']:.1f} m")
                col5.metric("Altitude max", f"{stats['max_elevation']:.1f} m")
                col6.metric("Pente moyenne", f"{stats['average_grade']:.1f} %")

                st.header("📍 Carte du parcours")
                # st.write(df_with_locations)
                m = plotter.create_route_map(route, color_by_slope=True)
                st_folium(m, width=900, height=500, returned_objects=[])

                st.header("📈 Profil d'élévation")
                fig = plotter.display_profile(route)
                st.plotly_chart(fig)    

                # Création du df d'affichage
                display_df = pd.DataFrame({
                    "🏁 Démarrage (km)" : (climbs_df['start_km']).round(2),
                    "📏 Longueur (km)": (climbs_df['length_m'] / 1000).round(2),
                    "⛰️ Dénivelé (m)": climbs_df['elev_gain'].round(0),
                    "📈 Pente moyenne (%)": climbs_df['avg_slope'].round(1),
                    "🔺 Pente max (%)": [f"{s:.1f}" if s is not None else "-" for s in climbs_df['max_slope']]
                })
                display_df.insert(0, "No", range(1, len(display_df) + 1))

                
                st.subheader("Segments de montée détectés 🚵")
                st.dataframe(display_df, hide_index=True, use_container_width=True)

                st.subheader("Profil détaillé par segment 🚵")
                bool_display_each_segment = st.checkbox("Afficher chaque segment", value=False)

                if bool_display_each_segment:
                    dico_fig = plotter.display_each_segment(route, window_m)

                    for i, fig in dico_fig.items():
                        st.subheader(f"Segment {i+1}")
                        st.plotly_chart(dico_fig[i], use_container_width=True)

            except Exception as e:
                st.error(f"Erreur lors du parsing du fichier : {e}")


