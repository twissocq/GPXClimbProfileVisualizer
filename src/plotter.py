from turtle import st
import matplotlib.pyplot as plt
import parser as parser
import util as util
import folium
import folium.plugins as plugins
import leafmap.foliumap as leafmap
from route import GPXData


def create_route_map(
    route: GPXData,
    tile_style: str = "OpenStreetMap",
    color_by_slope: bool = True,
) -> None:

    df = route.df
    climbs_df = route.climbs_df
    descents_df = route.descents_df
    coords = df[["lat", "lon"]].values.tolist()

    # Map center and bounds
    center = coords[len(coords) // 2]
    m = leafmap.Map(location=center, zoom_start=13, control_scale=True, tiles=None)

    legend_dict = {
        "-10% and below": "#4682B4",      # steel blue
        "-10% to -4%": "#87CEEB",        # sky blue
        "-4% to -2%": "#ADD8E6",         # light blue
        "-2% to 2%": "#FFFFFF",          # white
        "2% to 4%": "#75f60c",           # green
        "4% to 6%": "#00a0ff",           # blue
        "6% to 8%": "#ffd300",           # yellow
        "8% to 10%": "#ee0000",          # red
        "10% to 12%": "#800080",         # purple
        "12% and above": "#444444",      # black
    }

    m.add_legend(title="Slope Grade Legend", legend_dict=legend_dict)

    folium.TileLayer(tiles=tile_style, name=tile_style, opacity=0.8).add_to(m)

    # Segment coloring
    for i in range(1, len(coords)):
        segment = [coords[i - 1], coords[i]]
        color = util.getcolor(df["plot_grade"].iloc[i]) if color_by_slope else "#999999"
        folium.PolyLine(segment, color=color, weight=4, opacity=1).add_to(m)

    # Start and end markers
    folium.Marker(
        coords[0], icon=folium.Icon(color="green", icon="play"), popup="Start", 
    ).add_to(m)
    folium.Marker(
        coords[-1], icon=folium.Icon(color="red", icon="stop"), popup="End"
    ).add_to(m)

    # Climb markers
    if climbs_df is not None and not climbs_df.empty:
        for idx, row in climbs_df.iterrows():
            start_idx = (row["start_idx"])
            lat, lon = df.loc[start_idx, ["lat", "lon"]]
            folium.Marker(
                location=[lat, lon],
                popup=f"Climb {idx + 1}: {int(row['elev_gain'])}m ↑, L={int(row["length_m"]/1000)} km, slope={round(row["avg_slope"],1)} %",
                # icon=folium.DivIcon(
                #     html=f"<div style='font-size: 18px; color: red;'>{idx + 1}</div>"
                # ),
                icon = plugins.BeautifyIcon(
                     icon="mountain",
                     icon_shape="circle",
                     border_color='purple',
                     text_color="#007799",
                     background_color='white'
                 )
            ).add_to(m)

    folium.LayerControl().add_to(m)

    # Fit to bounds
    sw = df[["lat", "lon"]].min().values.tolist()
    ne = df[["lat", "lon"]].max().values.tolist()
    m.fit_bounds([sw, ne])

    return m


def plot_elevation_colored_by_slope(
    route:GPXData,
    simplified: bool = False,
    color_mode: str = "Detailed Slope",
) -> None:
    
    df = route.df
    climbs_df = route.climbs_df
    descents_df = route.descents_df

    annotationsAnchor = df['ele'].max() * 1.1
    style_city = dict(size=10, color='grey')

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlabel("Kilometers")
    ax.set_ylabel("Elevation (m)")
    ax.spines[['right', 'top']].set_visible(False)

    ax.plot(df["distance"] / 1000, df["ele"], color="#999999", linewidth=1.5, alpha=0.7)

    old_index = 0
    for index in df[df["location"].notnull()].index:
        #ensure that two annotations do not overlap
        
        if index > 0 and abs(df["distance"].values[index] - df["distance"].values[old_index])/1000 < 2:
            continue

        ax.annotate(
            df["location"].values[index],
            xy=(df["distance"].values[index] / 1000, df["ele"].values[index]),
            xytext=(df["distance"].values[index] / 1000, annotationsAnchor),
            arrowprops=dict(arrowstyle="-", color='lightgray'),
            horizontalalignment='center',
            rotation=90,
            **style_city
        )
        old_index = index

    if simplified:
        _draw_simplified_segments(ax, df, climbs_df, descents_df)
    else:
        _draw_detailed_colored_profile(
            ax, df, climbs_df, descents_df, color_mode
        )

    return fig, ax

def _draw_simplified_segments(ax, df, climbs_df, descents_df) -> None:
    style_slope = dict(size=10, color='black')
    style_max = dict(size=10, color='red')
    for segment_df, color in [(climbs_df, "#FFA500")]:
        if segment_df is not None:
            for _, row in segment_df.iterrows():
                segment = df[
                    (df["distance"] / 1000 >= row["start_km"])
                    & (df["distance"] / 1000 <= row["end_km"])
                ]
                ax.fill_between(
                    segment["distance"] / 1000, segment["ele"], color=color, alpha=0.4
                )
                
                ax.annotate(
                    f"{round(segment['ele'].values[-1])}m",
                    xy=(segment["distance"].values[-1] / 1000, segment["ele"].values[-1]),
                    xytext=(segment["distance"].values[-1] / 1000, 1.02 * segment["ele"].values[-1]),
                    horizontalalignment='center',
                    rotation=45,
                    **style_max
                )

                mean_slope = round(row["avg_slope"],1)
                start = row["start_km"]
                end = row["end_km"] 

                x_align = (start+end)/2
                if x_align - old_x_align < 5:
                    y_align = y_align + 100
                    print(y_align)
        
                ax.annotate(
                    f"{mean_slope}%",
                    xy=( x_align, y_align),
                    xytext=( x_align, y_align),
                    horizontalalignment='center',
                    rotation=0,
                    **style_slope
                )
                y_align = 0.2
                old_x_align = x_align


# 3. --- Detailed profile by slope ---
def _draw_detailed_colored_profile(
    ax, df, climbs_df, descents_df, color_mode
) -> None:
    style_slope = dict(size=10, color='black')
    style_max = dict(size=10, color='red')

    for segment_df in [climbs_df]:
        if segment_df is not None:
            for _, row in segment_df.iterrows():
                segment = df[
                    (df["distance"] / 1000 >= row["start_km"])
                    & (df["distance"] / 1000 <= row["end_km"])
                ]
                
                ax.annotate(
                    f"{round(segment['ele'].values[-1])}m",
                    xy=(segment["distance"].values[-1] / 1000, segment["ele"].values[-1]),
                    xytext=(segment["distance"].values[-1] / 1000, 1.02 * segment["ele"].values[-1]),
                    horizontalalignment='center',
                    rotation=45,
                    **style_max
                )

                mean_slope = round(row["avg_slope"],1)
                start = row["start_km"]
                end = row["end_km"] 

                x_align = (start+end)/2
                if x_align - old_x_align < 5:
                    y_align = y_align + 100
                    print(y_align)
        
                ax.annotate(
                    f"{mean_slope}%",
                    xy=( x_align, y_align),
                    xytext=( x_align, y_align),
                    horizontalalignment='center',
                    rotation=0,
                    **style_slope
                )
                y_align = 0.2
                old_x_align = x_align

    if color_mode == "Detailed Slope":
        # Couleur pour chaque pente calculée
        for i in range(1, len(df)):
            x = df["distance"].iloc[i - 1 : i + 1] / 1000
            y = df["ele"].iloc[i - 1 : i + 1]
            color = util.getcolor(df["plot_grade"].iloc[i])
            ax.fill_between(x, 0, y, color=color, alpha=0.8)



    else:  # Modo "Average per Segment"
        # Primero, dibujamos todo el perfil con un color neutro de base
        ax.fill_between(df["distance"] / 1000, 0, df["ele"], color="#E0E0E0", alpha=0.6)

        # Luego, "repasamos" cada segmento detectado con su color de pendiente media
        for segment_df in [climbs_df, descents_df]:
            if segment_df is not None and not segment_df.empty:
                for _, row in segment_df.iterrows():
                    # Obtenemos el color a partir de la pendiente MEDIA del segmento
                    avg_slope_color = util.getcolor(row["avg_slope"])

                    # Seleccionamos los datos de este segmento específico del dataframe completo
                    segment_data = df.iloc[row["start_idx"] : row["end_idx"] + 1]

                    # Dibujamos solo este tramo con su color
                    ax.fill_between(
                        segment_data["distance"] / 1000,
                        0,
                        segment_data["ele"],
                        color=avg_slope_color,
                        alpha=0.9,
                    )

    # La lógica para los marcadores se mantiene igual y se aplica a ambos modos
    if show_markers:
        for segment_df, color, label in [
            (climbs_df, "black", "Climbs"),
            (descents_df, "blue", "Descents"),
        ]:
            if segment_df is not None and not segment_df.empty:
                row = segment_df.iloc[0]
                style = "--" if color == "black" else ":"
                ax.axvline(
                    x=row["start_km"],
                    color=color,
                    linestyle=style,
                    alpha=0.6,
                    label=label,
                )
                ax.axvline(x=row["end_km"], color=color, linestyle=style, alpha=0.6)

                for _, row in segment_df.iloc[1:].iterrows():
                    ax.axvline(
                        x=row["start_km"], color=color, linestyle=style, alpha=0.6
                    )
                    ax.axvline(x=row["end_km"], color=color, linestyle=style, alpha=0.6)

        if (climbs_df is not None and not climbs_df.empty) or (
            descents_df is not None and not descents_df.empty
        ):
            ax.legend()