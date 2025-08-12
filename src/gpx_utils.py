import pandas as pd
import gpxpy
import util as util

def open_gpx(file):
    """Parse GPX and return track_df, slopes_df, places_df."""
    try:
        gpx = gpxpy.parse(file)
    except Exception as e:
        raise ValueError(f"Error parsing GPX: {e}")
    return gpx

def parse_gpx(gpx):
    track_data, km_cum, last_lat, last_lon = [], 0, None, None
    seg_unit = 1

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                dist = util.getDistanceBetweenCoords(last_lat, last_lon, point.latitude, point.longitude) if last_lat else 0
                km_cum += dist
                seg = km_cum // seg_unit
                track_data.append([seg, km_cum, point.latitude, point.longitude, point.elevation])
                last_lat, last_lon = point.latitude, point.longitude

    if not track_data:
        raise ValueError("No track points found in GPX file.")

    track_df = pd.DataFrame(track_data, columns=['segment', 'km', 'latitude', 'longitude', 'elevation'])

    # Slopes per segment
    slopes_data = []
    for segment in track_df['segment'].unique():
        seg_rows = track_df[track_df['segment'] == segment]
        length_km = seg_rows['km'].max() - seg_rows['km'].min()
        if length_km == 0:
            slope = 0
        else:
            slope = (seg_rows.iloc[-1]['elevation'] - seg_rows.iloc[0]['elevation']) / (length_km * 1000) * 100
        slopes_data.append([segment, slope])

    slopes_df = pd.DataFrame(slopes_data, columns=['segment', 'slope'])
    track_df_merge = pd.merge(track_df, slopes_df, on='segment')

    places_data, last_place, place_group = [], '', 0
    for segment in track_df['segment'].unique():
        seg_rows = track_df[track_df['segment'] == segment]
        lat, lon = seg_rows.iloc[0]['latitude'], seg_rows.iloc[0]['longitude']
        location = util.getLocationFromCoords(lat, lon)
        props = location['properties']
        name = f"{props["city"]}"
        places_data.append([segment, name, seg_rows.iloc[0]['elevation'], place_group])

    places_df = pd.DataFrame(places_data, columns=['segment', 'place', 'elevation', 'group'])
    return track_df, slopes_df, track_df_merge, places_df


def detect_climbs(track_df_merge, slopes_df, min_slope=3.0, min_climb_length=1.5):
    """Return dataframe of detected climbs."""
    segment_lengths = track_df_merge.groupby('segment')['km'].agg(['min', 'max'])
    segment_lengths['length'] = segment_lengths['max'] - segment_lengths['min']

    df = slopes_df.merge(segment_lengths, left_on='segment', right_index=True)
    df['is_climb'] = df['slope'] > 0
    df['climb_group'] = (df['is_climb'] != df['is_climb'].shift()).cumsum()

    climb_groups = df[df['is_climb']].groupby('climb_group').agg({
        'segment': ['min', 'max', 'count'],
        'length': 'sum',
        'slope': 'mean'
    })
    climb_groups.columns = ['segment_min', 'segment_max', 'segment_count', 'total_length_km', 'avg_slope']

    return climb_groups[(climb_groups['total_length_km'] >= min_climb_length) &
                        (climb_groups['avg_slope'] > min_slope)]
