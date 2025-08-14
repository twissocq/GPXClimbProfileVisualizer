from dataclasses import dataclass
import pandas as pd
import gpxpy
import numpy as np
from haversine import Unit, haversine_vector

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
 

@dataclass
class GPXData:
    gpx: gpxpy.gpx.GPX
    df: pd.DataFrame
    stats: dict = None
    locations: pd.DataFrame = None
    mean_slopes : pd.DataFrame = None

def open_gpx(file: str) -> GPXData:
    """Parse GPX and return gpx object."""
    try:
        gpx_file = open(file, "r")
        gpx = gpxpy.parse(gpx_file)
    except Exception as e:
        raise ValueError(f"Error parsing GPX: {e}")
    
    route = GPXData(gpx=gpx, df=pd.DataFrame(), stats=dict())

    return route

def parse_gpx(route:GPXData, max_points_per_km: int = 20):
    gpx = route.gpx
    data = []

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                data.append(
                    {
                        "lat": point.latitude,
                        "lon": point.longitude,
                        "ele": point.elevation,
                    }
                )

    df = pd.DataFrame(data)
    if len(df) < 2:
        raise ValueError("GPX file too short")

    df = _add_distance_and_grade(df)

    df = reduce_points_by_density(df, max_points_per_km)

    df = _add_distance_and_grade(df)

    route.df = df   
    return df

def _add_distance_and_grade(df):
    """
    Calculates distance, cumulative distance, and grade in a vectorized manner.
    Modifies the input DataFrame by adding these columns.
    """
    coords = df[["lat", "lon"]].to_numpy()

    distances_segment = np.zeros(len(df))
    distances_segment[1:] = haversine_vector(coords[:-1], coords[1:], unit=Unit.METERS)

    df["distance"] = np.cumsum(distances_segment)

    elev_diff = np.diff(df["ele"], prepend=df["ele"].iloc[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        df["grade"] = np.where(
            distances_segment > 0, (elev_diff / distances_segment) * 100, 0
        )

    return df

def reduce_points_by_density(df: pd.DataFrame, max_points_per_km) -> pd.DataFrame:
    """
    Reduces the number of points in a DataFrame based on a specified maximum density of points per kilometer.
    Parameters:
        df (pandas.DataFrame): A DataFrame containing at least a "distance" column, where "distance" is in meters
                               and represents the cumulative distance traveled.
        max_points_per_km (int): The maximum number of points allowed per kilometer.
    Returns:
        pandas.DataFrame: A reduced DataFrame with fewer points, maintaining the overall structure and resetting the index.
    Notes:
        - If the total distance is 0, the original DataFrame is returned.
        - If the number of points in the DataFrame is already less than or equal to the maximum allowed points,
          or if the calculated maximum points is 0, the original DataFrame is returned.
        - Points are reduced by selecting every nth point, where n is determined by the ratio of the total points
          to the maximum allowed points.
    """
    total_km = df["distance"].iloc[-1] / 1000
    if total_km == 0:
        return df

    max_points = int(total_km * max_points_per_km)
    if len(df) <= max_points or max_points == 0:
        return df

    step = max(1, len(df) // max_points)
    return df.iloc[::step].reset_index(drop=True)

def compute_stats_gpx(route:GPXData) -> dict:
    df= route.df
    if df.empty:
        return {}
    total_distance = df["distance"].iloc[-1]
    elev_diff = df["ele"].diff()

    gain = elev_diff[elev_diff > 0].sum()
    loss = -elev_diff[elev_diff < 0].sum()

    num_points = len(df)
    density_per_km = num_points / (total_distance / 1000)
    density_per_100m = num_points / (total_distance / 100)

    stats = {
        "total_distance_km": total_distance / 1000,
        "elevation_gain": gain,
        "elevation_loss": loss,
        "min_elevation": df["ele"].min(),
        "max_elevation": df["ele"].max(),
        "average_grade": df["grade"].mean(),
        "max_grade": df["grade"].max(),
        "num_points": num_points,
        "point_density_km": density_per_km,
        "point_density_100m": density_per_100m
    }
    route.stats = stats
    return stats


def get_location(route: GPXData, precision: int = 10000) -> pd.DataFrame:
    """
    Extracts city names from GPS coordinates of the GPXData.
    For every 10km in the route, plus the start and end points,
    creates a new DataFrame with the same id, GPS coordinates, and city names.
    """

    geolocator = Nominatim(user_agent="gpx_parser")
    df = route.df

    if df.empty:
        raise ValueError("The route DataFrame is empty.")

    # Select points at every 10km, plus the start and end points
    distances = df["distance"]
    key_points = [0] + list(range(precision, int(distances.iloc[-1]), precision)) + [distances.iloc[-1]]
    key_indices = [distances.sub(km).abs().idxmin() for km in key_points]

    locations = []
    for idx in key_indices:
        lat, lon = df.iloc[idx][["lat", "lon"]]
        try:
            location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
            city = location.raw['address']["town"] if "town" in location.raw['address'] else location.raw['address']["village"] if "village" in location.raw['address'] else location.raw['address']["city"]
        except GeocoderTimedOut:
            city = "Unknown"
        locations.append({"id": idx, "lat": lat, "lon": lon, "city": city, "distance": distances.iloc[idx]/1000})

    route.locations = pd.DataFrame(locations)
    return route.locations

def get_mean_grade(route: GPXData, precision: int = 1000) -> pd.DataFrame:
    """
    Creates a DataFrame with the mean grade for each segment of a specified distance (default 1km).
    Parameters:
        route (GPXData): The GPXData object containing the parsed route data.
        precision (int): The segment length in meters for which to calculate the mean grade.
    Returns:
        pd.DataFrame: A DataFrame with columns 'start_distance_km', 'end_distance_km', and 'mean_grade'.
    """
    df = route.df

    if df.empty:
        raise ValueError("The route DataFrame is empty.")

    distances = df["distance"]
    grades = df["grade"]

    segments = []
    start_distance = 0

    while start_distance < distances.iloc[-1]:
        end_distance = start_distance + precision
        segment_mask = (distances >= start_distance) & (distances < end_distance)
        mean_grade = grades[segment_mask].mean()

        segments.append({
            "start_distance_km": start_distance / 1000,
            "end_distance_km": end_distance / 1000,
            "mean_grade": mean_grade
        })

        start_distance = end_distance
    route.mean_slopes = pd.DataFrame(segments)
    return route.mean_slopes

def detect_climbs_from_gpx(route: GPXData, min_slope=2.0, min_climb_length=1000,
                            precision=100, plateau_slope=0, plateau_max_length=1000):
    """
    Detect climbs from parsed GPX data with plateau tolerance.
    
    plateau_slope: slope (%) below which section is considered a plateau
    plateau_max_length: max length in meters to still consider plateau as part of climb
    """
    if route.df.empty:
        raise ValueError("Route dataframe is empty.")
    
    slopes_df = get_mean_grade(route, precision=precision)

    # Mark climbing segments
    slopes_df["is_climb"] = slopes_df["mean_grade"] > min_slope

    # Handle plateaus
    segment_length_m = precision
    for i in range(1, len(slopes_df) - 1):
        if not slopes_df.loc[i, "is_climb"]:
            if abs(slopes_df.loc[i, "mean_grade"]) <= plateau_slope:
                # Check if previous and next are climbs
                if slopes_df.loc[i - 1, "is_climb"] and slopes_df.loc[i + 1, "is_climb"]:
                    # Check total plateau stretch length
                    plateau_len = 0
                    j = i
                    while j < len(slopes_df) and not slopes_df.loc[j, "is_climb"] \
                          and abs(slopes_df.loc[j, "mean_grade"]) <= plateau_slope:
                        plateau_len += segment_length_m
                        j += 1
                    if plateau_len <= plateau_max_length:
                        slopes_df.loc[i:j, "is_climb"] = True

    # Group consecutive climb segments
    slopes_df["climb_group"] = (slopes_df["is_climb"] != slopes_df["is_climb"].shift()).cumsum()

    # Extract climbs
    climbs = []
    for _, group in slopes_df.groupby("climb_group"):
        if group["is_climb"].iloc[0]:
            length_m = (group["end_distance_km"].iloc[-1] - group["start_distance_km"].iloc[0]) * 1000
            if length_m >= min_climb_length:
                climbs.append({
                    "start_km": group["start_distance_km"].iloc[0],
                    "end_km": group["end_distance_km"].iloc[-1],
                    "length_km": length_m / 1000,
                    "avg_slope": group["mean_grade"].mean()
                })

    return pd.DataFrame(climbs)



if __name__ == "__main__":

    route = open_gpx(r"C:\Users\Thibaut\Documents\Python Scripts\Cycling\traces\Étape_du_tour_femmes.gpx")
    df = parse_gpx(route)

    stats = compute_stats_gpx(route)


    locations = get_location(route, precision=10000)
    slopes = get_mean_grade(route, precision=1000)
    
    route.locations = locations
    route.mean_slopes = slopes

    climbs = detect_climbs_from_gpx(route, 1.5)
    print("ok")