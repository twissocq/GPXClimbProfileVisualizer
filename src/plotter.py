import matplotlib.pyplot as plt
import parser as parser
import util as util
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import uniform_filter1d

import folium
import folium.plugins as plugins
import leafmap.foliumap as leafmap
from route import GPXData


def create_route_map(
    route: GPXData,
    tile_style: str = "CyclOSM",
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
        "-10% and below": "#4D4D4A",      # steel blue
        "-10% to -4%": "#6E6E6B",        # sky blue
        "-4% to -2%": "#80807C",         # light blue
        "-2% to 2%": "#999999",          # grey
        "2% to 4%": "#75f60c",           # green
        "4% to 6%": "#00a0ff",           # blue
        "6% to 8%": "#ffd300",           # yellow
        "8% to 10%": "#ee0000",          # red
        "10% to 12%": "#800080",         # purple
        "12% and above": "#000000",      # black
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


def display_profile(route:GPXData, window_m:int =20) -> go.Figure:

    # -----------------------------
    # Example DataFrames
    # df: distance in meters, ele in meters, grade in %
    # climbs_df: type, start_km, end_km, elev_gain, length_m, avg_slope, start_idx, end_idx
    # -----------------------------
    # Parameters
    # window_m = 20
    df = route.df
    climbs_df = route.climbs_df
    def compute_avg_slope(df, window_m=100):
        median_dx = df['distance'].diff().median()
        if np.isnan(median_dx) or median_dx <= 0:
            median_dx = 20  # fallback
        window_points = max(int(window_m / median_dx), 1)
        
        df = df.copy()
        df['avg_slope'] = df['grade'].rolling(window=window_points, min_periods=1).mean()
        df['smoothed_ele'] = uniform_filter1d(df['ele'], size=window_points)
        df['color'] = df['avg_slope'].apply(util.getcolor)
        return df

    df = compute_avg_slope(df, window_m)

    # -----------------------------
    # Create figure
    # -----------------------------
    fig = go.Figure()

    # Plot background elevation as a thin grey line
    fig.add_trace(go.Scatter(
        x=df['distance']/1000,
        y=df['smoothed_ele'],
        mode='lines',
        line=dict(color='grey', width=2),
        name='Elevation',
        hoverinfo='skip',
        showlegend=False
    ))

    # Plot filled slope segments
    start_idx = 0
    for i in range(1, len(df)):
        if df['color'].iloc[i] != df['color'].iloc[start_idx] or i == len(df)-1:
            color = df['color'].iloc[start_idx]
            if color:  # only plot colored segments
                seg_x = df['distance'].iloc[start_idx:i+1]/1000
                seg_y = df['smoothed_ele'].iloc[start_idx:i+1]
                slope_length = (seg_x.iloc[-1] - seg_x.iloc[0])
                fig.add_trace(go.Scatter(
                    x=seg_x,
                    y=seg_y,
                    fill='tozeroy',
                    mode='none',
                    fillcolor=color,
                    opacity=0.8,
                    hovertemplate=(
                        "Distance: %{x:.2f} km<br>"
                        "Elevation: %{y:.0f} m<br>"
                        f"Length: {slope_length:.2f} km<br>"
                        f"Slope: {df['avg_slope'].iloc[start_idx]:.1f}%"
                    ),
                    showlegend=False
                ))
            start_idx = i



    # -----------------------------
    # Annotate climbs
    # -----------------------------
    last_annot_x = -10
    min_spacing = 8.5
    y_base = 0#df['smoothed_ele'].min() - 0.02 * (df['smoothed_ele'].max() - df['smoothed_ele'].min())

    for i, row in climbs_df.iterrows():
        x = (row['start_km'] + row["end_km"])/2

        if row['length_m']/1000 >= 4:
            fig.add_annotation(
                x=x,
                y=y_base ,
                text=f"{row['length_m']/1000:.2f} km<br>{round(row['avg_slope'],1)}%",  # <br> for line break in Plotly
                showarrow=False,
                yanchor="bottom",
                xanchor="center",
                font=dict(size=12, color="black"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=2
        )
    # annotationsAnchor = df['ele'].max() * 1.05
    annotationsAnchor = df['ele'].max() * 1.02  # place inside plot, near top

    old_index = None

    for index in df[df["location"].notnull()].index:
        loc_name = df["location"].iloc[index]
        # Skip if too close to previous annotation
        if old_index is not None and abs(df["distance"].iloc[index] - df["distance"].iloc[old_index]) / 1000 < 2:
            continue

        x_pos = df["distance"].iloc[index] / 1000
        y_elev = df["ele"].iloc[index]

        # Add dashed vertical line from elevation to text
        fig.add_shape(
            type="line",
            x0=x_pos, y0=y_elev,
            x1=x_pos, y1=annotationsAnchor,
            line=dict(color="gray", width=1, dash="dot"),
            layer="below"
        )

        # Add vertical text label, aligned at annotationsAnchor
        fig.add_annotation(
            x=x_pos,
            y=annotationsAnchor,
            text=loc_name,
            showarrow=False,
            textangle=-45,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=12, color='black')
        )
        old_index = index

    # -----------------------------
    # Legend for slope colors
    # -----------------------------
    slope_legend = [
        ("-10% <", util.getcolor(-11)),
        ("-10% – -4%", util.getcolor(-6)),
        ("-4% – -2%", util.getcolor(-3)),
        ("-2% – 2%", util.getcolor(0)),
        ("2% – 4%", util.getcolor(3)),
        ("4% – 6%", util.getcolor(5)),
        ("6% – 8%", util.getcolor(7)),
        ("8% – 10%", util.getcolor(9)),
        ("10% – 12%", util.getcolor(11)),
        (">= 12%", util.getcolor(13)),
    ]
    for label, color in slope_legend:
        fig.add_trace(go.Scatter(
            x=[0, 0], y=[None],
            mode='lines',
            line=dict(color=color, width=8),
            name=label,
            showlegend=True

        ))

    # -----------------------------
    # Layout
    # -----------------------------
    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True),
        plot_bgcolor='rgba(200,200,200,0.2)',
        hovermode="closest",
        showlegend=True,
        legend_title="Slope (%)",
        legend=dict(orientation="h", y=-0.05)
    )

    fig.update_yaxes(
        ticksuffix=" m",      # adds " m" after each tick
        showgrid=False
    )

    fig.update_xaxes(
        ticksuffix=" km",     # adds " km" after each tick
        showgrid=False
    )

    fig.update_yaxes(range=[0, 1.1*df["ele"].max()])

    fig.update_layout(
        height=600  # default is ~450
    )

    # fig.update_layout(
    # margin=dict(t=200, b=60, l=60, r=60)  # top margin 100px
    # )

    # fig.show()
    return fig


def display_each_segment(route:GPXData, window_m=20) -> dict:
    # Compute global scale
    climbs_df = route.climbs_df
    df = route.df
    median_dx = df['distance'].diff().median()
    if np.isnan(median_dx) or median_dx <= 0:
        median_dx = 20  # fallback
    window_points = max(int(window_m / median_dx), 1)

    df['avg_slope'] = df['grade'].rolling(window=window_points, min_periods=1).mean()

    global_min_x = df['distance'].min()
    global_max_x = df['distance'].max()
    global_min_y = 0
    global_max_y = df['ele'].max() * 1.15  # 15% headroom

    # Stats container
    stats_list = []
    dico_fig = dict() 

    # Loop over climbs
    for i, climb in climbs_df.iterrows():
        
        seg_df = df.iloc[climb.start_idx:climb.end_idx+1].copy()
        colors = [util.getcolor(g) for g in seg_df['avg_slope']]
        start = seg_df['distance'].iloc[0]
        seg_df['distance'] = seg_df['distance'] - start
        # Compute stats
        length_km = climb.length_m / 1000
        gain_m = climb.elev_gain
        avg_slope = climb.avg_slope
        max_slope_100m = seg_df['avg_slope'].max()
        stats_list.append({
            "Climb": i+1,
            "Length (km)": round(length_km, 2),
            "Gain (m)": round(gain_m, 1),
            "Avg slope (%)": round(avg_slope, 2),
            "Max slope 100 m (%)": round(max_slope_100m, 1)
        })

        # Build figure
        fig = go.Figure()

        start_idx = 0
        for j in range(1, len(seg_df)):
            if colors[j] != colors[j-1] or j == len(seg_df)-1:
                seg_y = seg_df['ele'].iloc[start_idx:j+1]
                seg_x = seg_df['distance'].iloc[start_idx:j+1]/1000
                # seg_y = seg_df['ele'].iloc[start_idx:j+1]
                avg_slope_val = seg_df['avg_slope'].iloc[start_idx]
                length_val = (seg_x.iloc[-1] - seg_x.iloc[0]) * 1000
                fig.add_trace(go.Scatter(
                    x=seg_x,
                    y=seg_y,
                    fill='tozeroy',
                    mode='none',
                    fillcolor=colors[start_idx],
                    hovertemplate=(
                        f"Avg slope (100m): {avg_slope_val:.1f}%<br>"
                        f"Length: {length_val/1000:.2f} m"
                    ),
                    name=f"Slope {avg_slope_val:.1f}%",
                    showlegend=False
                ))
                start_idx = j

        # -----------------------------
        # Legend for slope colors
        # -----------------------------
        slope_legend = [
            ("-10% <", util.getcolor(-11)),
            ("-10% – -4%", util.getcolor(-6)),
            ("-4% – -2%", util.getcolor(-3)),
            ("-2% – 2%", util.getcolor(0)),
            ("2% – 4%", util.getcolor(3)),
            ("4% – 6%", util.getcolor(5)),
            ("6% – 8%", util.getcolor(7)),
            ("8% – 10%", util.getcolor(9)),
            ("10% – 12%", util.getcolor(11)),
            (">= 12%", util.getcolor(13)),
        ]
        for label, color in slope_legend:
            fig.add_trace(go.Scatter(
                x=[0, 0], y=[None],
                mode='lines',
                line=dict(color=color, width=8),
                name=label,
                showlegend=True

            ))
        
        fig.update_layout(
            height=650,
            plot_bgcolor="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            title=f"Climb {i+1} — Start: {start/1000:.2f} km | {length_km:.2f} km | {gain_m:.0f} m gain | {avg_slope:.1f}% avg",
            margin=dict(t=80, b=50, l=50, r=50),
            showlegend=True,
            legend_title="Slope (%)",
            legend=dict(orientation="h", y=-0.05)
        )
        fig.update_layout(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True),
            plot_bgcolor='rgba(200,200,200,0.2)',
            hovermode="closest",
            showlegend=True,
            legend_title="Slope (%)",
            legend=dict(orientation="h", y=-0.05)
        )

        fig.update_yaxes(
            ticksuffix=" m",      # adds " m" after each tick
            showgrid=False
        )

        fig.update_xaxes(
            ticksuffix=" km",     # adds " km" after each tick
            showgrid=False
        )
        
        dico_fig[i] = fig

    return dico_fig
        # fig.show()