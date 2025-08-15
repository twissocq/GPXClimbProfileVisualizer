import requests
from math import sin, cos, sqrt, atan2, radians
import numpy as np

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

PHOTON_REVERSE_URL = "https://photon.komoot.io/reverse"
PHOTON_SEARCH_URL = "https://photon.komoot.io/api"

_cache_reverse = {}
_cache_forward = {}

def getLocationFromCoords(latitude, longitude):
    """
    Reverse geocoding: GPS → location dict
    """
    coord = (round(latitude, 5), round(longitude, 5))
    if coord in _cache_reverse:
        return _cache_reverse[coord]
    
    params = {"lat": coord[0], "lon": coord[1]}
    r = requests.get(PHOTON_REVERSE_URL, params=params)
    if r.status_code == 200:
        data = r.json()
        if data.get("features"):
            _cache_reverse[coord] = data["features"][0]
        else:
            _cache_reverse[coord] = None
    else:
        _cache_reverse[coord] = None
    return _cache_reverse[coord]


def getCoordsFromLocation(cityName):
    """
    Forward geocoding: location name → coords
    """
    if cityName in _cache_forward:
        return _cache_forward[cityName]
    
    params = {"q": cityName}
    r = requests.get(PHOTON_SEARCH_URL, params=params)
    if r.status_code == 200:
        data = r.json()
        if data.get("features"):
            coords = data["features"][0]["geometry"]["coordinates"]  # [lon, lat]
            _cache_forward[cityName] = {"latitude": coords[1], "longitude": coords[0]}
        else:
            _cache_forward[cityName] = None
    else:
        _cache_forward[cityName] = None
    return _cache_forward[cityName]


def getDistanceBetweenCoords(lat1, long1, lat2, long2):
    """
    Distance between two points in km
    """
    earthRadius = 6371
    c = 0
    if all((lat1, long1, lat2, long2)):
        latFrom = radians(lat1)          
        lonFrom = radians(long1)
        latTo = radians(lat2)
        lonTo = radians(long2)
        dlon = lonTo - lonFrom
        dlat = latTo - latFrom
        a = sin(dlat / 2)**2 + cos(latFrom) * cos(latTo) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return earthRadius * c


def getPlaceName(properties):
    """    Extracts a human-readable place name from the properties dictionary.
    """
    return properties["locality"] if "locality" in properties else \
           properties["name"] if "name" in properties else ""

def smoothProfile(signal,L=10):
    res = np.copy(signal) 
    for i in range (1,len(signal)-1): 
        L_g = min(i,L) 
        L_d = min(len(signal)-i-1,L) 
        Li=min(L_g,L_d)
        res[i]=np.sum(signal[i-Li:i+Li+1])/(2*Li+1)
    return res

def plot_climb_segment(trackDf_merge, placesDf, segment_start, segment_end, annotate_slopes=False):
    slopesTable = [
        lambda x: x < 2, 
        lambda x: (x >= 2) & (x < 4),
        lambda x: (x >= 4) & (x < 5),
        lambda x: (x >= 5) & (x < 8),
        lambda x: (x >= 8) & (x < 10),
        lambda x: (x >= 10) & (x < 12),
        lambda x: x >= 12,
    ]

    slopesColor = ['white', 'yellow', 'orange', 'orangered', 'maroon', 'darkred', 'black']
    slopesDescr = ['< 2%', '2–4%', '4–5%', '5–8%', '8–10%', '10–12%', '> 12%']

    legend = []
    style = dict(size=10, color='gray')

    climb_df = trackDf_merge[
        (trackDf_merge['segment'] >= segment_start) &
        (trackDf_merge['segment'] <= segment_end)
    ]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlabel("Kilometers")
    ax.set_ylabel("Elevation (m)")
    ax.spines[['right', 'top']].set_visible(False)

    # Background profile
    smoothed_elev = smoothProfile(climb_df['elevation'])
    ax.plot(climb_df['km'], smoothed_elev, color='black', alpha=0.3)
    ax.fill_between(climb_df['km'], smoothed_elev, color='gray', alpha=0.3, zorder=0)

    # Colored slope fills
    for i, mask_fn in enumerate(slopesTable):
        mask = mask_fn(climb_df['slope'])
        ax.fill_between(climb_df['km'], smoothed_elev, where=mask, 
                        color=slopesColor[i], zorder=1)
        legend.append(mpatches.Patch(color=slopesColor[i], label=slopesDescr[i]))
    
    annotationsAnchor = climb_df['elevation'].max() * 1.1

    # Place name annotations for this climb
    lastSegment = 0
    for idx, row in placesDf.iterrows():
        if segment_start <= row['segment'] <= segment_end:
            if row['segment'] > lastSegment + 1 or row['segment'] == segment_start or row['segment'] == segment_end:
                ax.annotate(row['place'], 
                            xy=(row['segment'], row['elevation']), 
                            xytext=(row['segment'], annotationsAnchor), 
                            arrowprops=dict(arrowstyle="-", color='lightgray'),
                            horizontalalignment='center',
                            rotation=90, **style)
                lastSegment = row['segment']

    # Annotate slopes at each full km if requested
    if annotate_slopes:
        # Définir les xticks à chaque kilomètre entier
        km_start = int(climb_df['km'].min())
        km_end = int(climb_df['km'].max()) + 1
        ax.set_xticks(range(km_start, km_end))

        # Annoter chaque kilomètre avec la pente moyenne
        for km in range(km_start, km_end):
            km_mask = (climb_df['km'] >= km) & (climb_df['km'] < km + 1)
            km_segment = climb_df[km_mask]
            if not km_segment.empty:
                slope_mean = km_segment['slope'].mean()
                ax.text(km + 0.5,                       # milieu horizontal du km
                        ax.get_ylim()[0] + 5,           # un peu au-dessus de l’axe X
                        f"{slope_mean:.1f}%", 
                        ha='center', va='bottom', fontsize=8, color='black')


    # Legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=legend, loc='center left', bbox_to_anchor=(1, 0.5))

    # Stats
    total_distance = climb_df['km'].max() - climb_df['km'].min()
    elevation_diff = climb_df['elevation'].diff()
    total_climb = elevation_diff[elevation_diff > 0].sum()

    start_place = placesDf.loc[placesDf['segment'] == segment_start, 'place'].iloc[0]
    end_place = placesDf.loc[placesDf['segment'] == segment_end, 'place'].iloc[0]

    plt.figtext(0.5, -0.10, 
                f"Climb — {total_distance:.2f} km, {total_climb:.0f} m D+ \n Segment {start_place} to {end_place}", 
                ha='center', fontsize=12, color='black')

    return fig


def getcolor(grade:int) -> str:
    """Return a color based on the slope grade in # format."""
    match grade:
        case _ if -2 <= grade < 2:
            return "#FFFFFF"  # white
        case _ if 2 <= grade < 4:
            return "#75f60c"  # green
        case _ if 4 <= grade < 6:
            return "#00a0ff"  # blue
        case _ if 6 <= grade < 8:
            return "#ffd300"  # yellow
        case _ if 8 <= grade < 10:
            return "#ee0000"  # red
        case _ if 10 <= grade < 12:
            return "#800080"  # purple
        case _ if grade >= 12:
            return "#444444"  # black
        # case _ if -4 <= grade < -2:
        #     return "#ADD8E6"  # light blue
        # case _ if -10 <= grade < -4:
        #     return "#87CEEB"  # sky blue
        # case _ if grade < -10:
        #     return "#4682B4"  # steel blue
        case _ if -4 <= grade < -2:
            return "#F5F5DC"  # beige
        case _ if -10 <= grade < -4:
            return "#FFE4C4"  # bisque
        case _ if grade < -10:
            return "#CAA473"  # burlywood

        