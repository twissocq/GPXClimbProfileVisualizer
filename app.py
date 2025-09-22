# app.py
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import parser as parser
import climb_detector as climb_detector
import get_locations as locator
import plotter as plotter

from pages.garmin_tab import render_garmin_tab

# -------------------------------
# ⚙️ CONFIG
# -------------------------------
st.set_page_config(layout="wide", page_title="GPX Analyzer 📍")

DEFAULT_PARAMS = {
    "max_pause_length_m": 200,
    "max_pause_descent_m": 10,
    "start_threshold_slope": 2.0
}
DEFAULT_LOCALISATION_PRECISION = 10000
DEFAULT_WINDOW_M = 100

if "params" not in st.session_state:
    st.session_state["params"] = DEFAULT_PARAMS.copy()
if "localisation_name_precision" not in st.session_state:
    st.session_state["localisation_name_precision"] = DEFAULT_LOCALISATION_PRECISION
if "window_m" not in st.session_state:
    st.session_state["window_m"] = DEFAULT_WINDOW_M


# -------------------------------
# 📊 SIDEBAR PARAMETRES
# -------------------------------
st.sidebar.header("⚙️ Paramètres avancés")

advanced_mode = st.sidebar.checkbox("Activer le mode avancé", value=False)

if advanced_mode:
    st.session_state["params"]["max_pause_length_m"] = st.sidebar.number_input(
        "Longueur max de pause (m)", value=st.session_state["params"]["max_pause_length_m"], min_value=0
    )
    st.session_state["params"]["max_pause_descent_m"] = st.sidebar.number_input(
        "Descente max en pause (m)", value=st.session_state["params"]["max_pause_descent_m"], min_value=0
    )
    st.session_state["params"]["start_threshold_slope"] = st.sidebar.number_input(
        "Pente seuil de départ (%)", value=st.session_state["params"]["start_threshold_slope"], min_value=1.0, step=0.1
    )
    st.session_state["localisation_name_precision"] = st.sidebar.number_input(
        "Précision des noms de lieux", value=st.session_state["localisation_name_precision"], min_value=10000, step=1
    )
    st.session_state["window_m"] = st.sidebar.number_input(
        "Lissage pente (m)", value=st.session_state["window_m"], min_value=1, step=1
    )
else:
    st.sidebar.write("Paramètres par défaut :")
    st.sidebar.table(pd.DataFrame(st.session_state["params"].items(), columns=["Paramètre", "Valeur"]))


# -------------------------------
# 📑 ONGLET PRINCIPAL
# -------------------------------
st.title("GPX Climb Analyzer")

tab_analyse, tab_garmin, tab_strava = st.tabs(["📈 Analyse GPX", "⌚ Garmin Connect", "🚧 Strava"])

# -------------------------------
# 📈 ANALYSE GPX
# -------------------------------
with tab_analyse:
    st.subheader("Analyse des données")

    if "route" not in st.session_state:
        uploaded_file = st.file_uploader("Importez un fichier GPX", type=["gpx"])
        if uploaded_file is not None:
            try:
                route = parser.open_gpx(uploaded_file)
                st.session_state["route"] = route
                st.success("✅ GPX chargé avec succès !")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur lors du parsing du fichier : {e}")
    else:
        route = st.session_state["route"]

        # ---- Bouton reset
        if st.button("❌ Décharger le GPX"):
            del st.session_state["route"]
            st.rerun()   

        # Parser et stats
        df = parser.parse_gpx(route)
        route.df = parser.apply_slope_smoothing(route.df, target_meters=100)
        stats = parser.compute_stats_gpx(route)
        climbs_df = climb_detector.detect_significant_segments(route.df, kind="climb", **st.session_state["params"])
        descents_df = climb_detector.detect_significant_segments(route.df, kind="descent", **st.session_state["params"])
        route.descents_df, route.climbs_df = descents_df, climbs_df

        df_with_locations = locator.add_location(route, st.session_state["localisation_name_precision"])

        # ---- Stats principales
        st.header("📊 Statistiques du parcours")
        col1, col2, col3 = st.columns(3)
        col1.metric("Distance", f"{stats['total_distance_km']:.2f} km")
        col2.metric("Dénivelé +", f"{stats['elevation_gain']:.0f} m")
        col3.metric("Dénivelé -", f"{stats['elevation_loss']:.0f} m")

        col4, col5, col6 = st.columns(3)
        col4.metric("Altitude min", f"{stats['min_elevation']:.1f} m")
        col5.metric("Altitude max", f"{stats['max_elevation']:.1f} m")
        col6.metric("Pente moyenne", f"{stats['average_grade']:.1f} %")

        # ---- Carte
        st.header("📍 Carte du parcours")
        m = plotter.create_route_map(route, color_by_slope=True)
        st_folium(m, width=900, height=500, returned_objects=[])

        # ---- Profil
        st.header("📈 Profil d'élévation")
        fig = plotter.display_profile(route)
        st.plotly_chart(fig)

        # ---- Segments
        st.subheader("Segments de montée détectés 🚵")
        if climbs_df.empty:
            st.warning("Aucun segment de montée détecté.")
        else:
            display_df = pd.DataFrame({
                "🏁 Début (km)" : (climbs_df['start_km']).round(2),
                "📏 Longueur (km)": (climbs_df['length_m'] / 1000).round(2),
                "⛰️ Dénivelé (m)": climbs_df['elev_gain'].round(0),
                "📈 Pente moyenne (%)": climbs_df['avg_slope'].round(1),
                "🔺 Pente max (%)": [f"{s:.1f}" if s is not None else "-" for s in climbs_df['max_slope']]
            })
            display_df.insert(0, "No", range(1, len(display_df) + 1))
            st.dataframe(display_df, hide_index=True, use_container_width=True)

            if st.checkbox("Afficher chaque segment"):
                dico_fig = plotter.display_each_segment(route, st.session_state["window_m"])
                for i, fig in dico_fig.items():
                    st.subheader(f"Segment {i+1}")
                    st.plotly_chart(fig, use_container_width=True)

   

# -------------------------------
# ⌚ GARMIN CONNECT
# -------------------------------
with tab_garmin:
    render_garmin_tab()


# -------------------------------
# 🚧 STRAVA
# -------------------------------
with tab_strava:
    st.subheader("Travail en cours 🚧")
    st.info("Connexion Strava et import d’activités en développement…")
