import streamlit as st
import matplotlib.pyplot as plt
import folium
from folium import Popup, PolyLine
import gpxpy
from streamlit_folium import st_folium
import altair as alt
import pandas as pd

import src.util as util
import src.gpx_utils as gpx_utils

@st.cache_data
def load_gpx_data(gpx_content):
    gpx = gpxpy.parse(gpx_content)
    return gpx_utils.parse_gpx(gpx)

st.title("GPX Climb Analyzer")

uploaded_gpx = st.file_uploader("Upload your GPX file", type=["gpx"])

if uploaded_gpx is not None:
    try:
        # Lecture et parsing en cache
        track_df, slopes_df, track_df_merge, places_df = load_gpx_data(uploaded_gpx)
        st.success(f"File `{uploaded_gpx.name}` loaded successfully!")

        # (Ici tu peux afficher des stats ou faire tes plots)
        # st.write(track_df.head())

    except Exception as e:
        st.error(f"Error while loading GPX file: {e}")
else:
    st.info("Please upload a GPX file to begin.")


# Summary stats
total_distance = track_df['km'].max()
total_climb = track_df['elevation'].diff()

total_climb_pos = total_climb[total_climb > 0].sum()
total_climb_neg = total_climb[total_climb < 0].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Distance", f"{total_distance:.2f} km")
col2.metric("Elevation Gain", f"{total_climb_pos:.0f} m")
col3.metric("Elevation Loss", f"{total_climb_neg:.0f} m")

# Climbs detection
climbs_df = gpx_utils.detect_climbs(track_df_merge, slopes_df)

# Map with slopes and climbs
m = folium.Map(location=[track_df['latitude'].mean(), track_df['longitude'].mean()], zoom_start=13)
colors = ['black', 'yellow', 'orange', 'orangered', 'maroon', 'darkred', 'purple']
slopesTable = [lambda x: x < 2, lambda x: (x >= 2) & (x < 4), lambda x: (x >= 4) & (x < 5),
               lambda x: (x >= 5) & (x < 8), lambda x: (x >= 8) & (x < 10),
               lambda x: (x >= 10) & (x < 12), lambda x: x >= 12]

# Draw GPX with slope colors
for seg_id in track_df_merge['segment'].unique():
    seg_data = track_df_merge[track_df_merge['segment'] == seg_id]
    slope_val = seg_data['slope'].iloc[0]
    for idx, func in enumerate(slopesTable):
        if func(slope_val):
            color = colors[idx]
            break
    coords = list(zip(seg_data['latitude'], seg_data['longitude']))
    folium.PolyLine(coords, color=color, weight=4).add_to(m)

# Distance markers every 1 km
km_markers = (
    track_df.groupby(track_df['km'].round().astype(int))
    .first()
    .rename_axis("km_marker")
    .reset_index()
)
for _, row in km_markers.iterrows():
    folium.Marker(
        location=(row['latitude'], row['longitude']),
        icon=folium.DivIcon(html=f"""<div style="font-size: 10pt; color: black;">{int(row['km'])} km</div>""")
    ).add_to(m)

# Add climbs slopes with annotations
    #TODO


st.subheader("📍 Map view")
st_folium(m, width=900, height=500)

# Full profile plot
fig, ax = plt.subplots(figsize=(12, 4))
fig = util.plot_climb_segment(track_df_merge, places_df, track_df['segment'].min(), track_df['segment'].max())
st.pyplot(fig)

# Climbs detection
# climbs_df = gpx_utils.detect_climbs(track_df_merge, slopes_df)
st.subheader("Detected Climbs")
for _, climb in climbs_df.iterrows():
    st.markdown(f"**Length:** {climb['total_length_km']:.2f} km — **Avg slope:** {climb['avg_slope']:.2f}%")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    fig2 = util.plot_climb_segment(track_df_merge, places_df, climb['segment_min'], climb['segment_max'], True)
    st.pyplot(fig2)