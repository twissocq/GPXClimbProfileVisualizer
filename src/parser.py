import pandas as pd
import gpxpy
import numpy as np
from haversine import Unit, haversine_vector

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
 
from route import GPXData

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

def apply_slope_smoothing(df, target_meters: int = 100):
    """
    Smooths the slope (grade) data in a DataFrame by applying a rolling mean.
    This function calculates a rolling average for the "grade" column in the 
    provided DataFrame to smooth out variations in slope data. The size of the 
    rolling window is determined based on the target distance in meters and 
    the distance per data point.
    Args:
        df (pandas.DataFrame): The input DataFrame containing at least the 
            columns "distance" (cumulative distance) and "grade" (slope values).
        target_meters (int, optional): The target distance in meters over which 
            to smooth the slope data. Defaults to 300 meters.
    Returns:
        pandas.DataFrame: The input DataFrame with an additional column 
        "plot_grade" containing the smoothed slope values.
    Notes:
        - If the calculated distance per point is zero, the function returns 
          the original DataFrame without modifications.
        - The rolling window size is adjusted to ensure it is at least 3 and 
          always an odd number for proper centering.
    """
    meters_per_point = df["distance"].iloc[-1] / len(df)
    # Avoid a window that is too large or too small
    if meters_per_point == 0:
        return df  # Cannot calculate
    window = max(3, int(target_meters / meters_per_point))
    # Ensure the window is an odd number for clear centering
    if window % 2 == 0:
        window += 1

    df["plot_grade"] = (
        df["grade"].rolling(window=window, center=True, min_periods=1).mean()
    )
    return df

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

# def find_climbs(df: pd.DataFrame, min_grade: float = 3.0, min_length: float = 500.0) -> pd.DataFrame:
#     """
#     Identify climbs as segments where grade > min_grade for at least min_length meters.
#     Returns a DataFrame with start/end indices, distance, elevation gain, and average grade.
#     """
#     df["is_climb"] = (df["grade"] > min_grade) & (df["distance"].diff() > 0)
#     df["climb_id"] = (df["is_climb"].diff() != 0).cumsum()
#     climbs = df[df["is_climb"]].groupby("climb_id").agg(
#         start_idx=("distance", "first"),
#         end_idx=("distance", "last"),
#         length=("distance", "max"),
#         gain=("ele", lambda x: x.iloc[-1] - x.iloc[0]),
#         avg_grade=("grade", "mean"),
#     )
#     climbs = climbs[(climbs["length"] >= min_length) & (climbs["gain"] > 0)]
#     return climbs.reset_index(drop=True)


if __name__ == "__main__":

    route = open_gpx(r"C:\Users\Thibaut\Documents\Python Scripts\Cycling\traces\Étape_du_tour_femmes.gpx")
    df = parse_gpx(route)

    stats = compute_stats_gpx(route)


    locations = get_location(route, precision=10000)
    slopes = get_mean_grade(route, precision=1000)
    
    route.locations = locations
    # route.mean_slopes = slopes

    # climbs = find_climbs(route.df)

    print("ok")