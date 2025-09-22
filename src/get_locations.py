import requests
import pandas as pd
from route import GPXData

PHOTON_REVERSE_URL = "https://photon.komoot.io/reverse"
PHOTON_SEARCH_URL = "https://photon.komoot.io/api"

_cache_reverse = {}
_cache_forward = {}

def getLocationFromCoords(latitude, longitude) -> dict:
    """
    Reverse geocoding: GPS → location dict
    """
    return None
    coord = (round(latitude, 10), round(longitude, 10))
    if coord in _cache_reverse:
        return getNamefromProperties(_cache_reverse[coord])
    
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

    return getNamefromProperties(_cache_reverse[coord])

def getNamefromProperties(location:dict) -> str:
    """
    Extract the name from the properties dictionary.
    """
    if location is None:
        return ""
    
    name = location["properties"]["district"] if "district" in location["properties"] else location["properties"]["city"] if "city" in location["properties"] else None
    city = location["properties"]["city"] if "city" in location["properties"] else None

    if name != city:
        full_name = f"{city}, {name}"
    else:
        full_name = name

    return city #full_name

def get_indexes_at_intervals(df, interval=1000):
    """
    Generate a list of indexes from a DataFrame at specified distance intervals.

    This function iterates through the 'distance' column of the given DataFrame
    and finds the index where the distance is closest to the current interval.
    It continues this process until the maximum distance in the DataFrame is reached.

    Args:
        df (pandas.DataFrame): A DataFrame containing a 'distance' column with numeric values.
        interval (int, optional): The distance interval at which to select indexes. Defaults to 1000.

    Returns:
        list: A list of indexes corresponding to the rows in the DataFrame where the distance
              is closest to the specified intervals.
    """
    indexes = []
    current_distance = 0
    while current_distance <= df['distance'].max():
        # Trouver l'index où la distance est la plus proche de current_distance
        idx = (df['distance'] - current_distance).abs().idxmin()
        indexes.append(idx)
        current_distance += interval
    return indexes

def add_location(route:GPXData, interval = 10000) -> pd.DataFrame:
    """
    Adds location information to the DataFrame of a given route.
    This function updates the `route.df` DataFrame by adding a "location" column
    that contains location names derived from latitude and longitude coordinates.
    It also ensures that specific indexes, including those at regular intervals
    and the end indexes of climbs, are populated with location data.
    Args:
        route: An object containing the following attributes:
            - df (pd.DataFrame): The main DataFrame representing the route, which must
              include "lat" and "lon" columns for latitude and longitude, respectively.
            - climbs_df (pd.DataFrame): A DataFrame containing information about climbs,
              which must include an "end_idx" column indicating the end index of each climb.
    Returns:
        pd.DataFrame: The updated DataFrame with a "location" column added.
    Raises:
        ValueError: If the `route.df` DataFrame is empty.
    Notes:
        - The function uses `get_indexes_at_intervals` to determine indexes at regular
          intervals (e.g., every 10,000 units).
        - The `getLocationFromCoords` function is used to fetch location names based
          on latitude and longitude coordinates.
    """
    df = route.df
    climbs_df = route.climbs_df
    if df.empty:
        raise ValueError("The route DataFrame is empty.")

    indexes = get_indexes_at_intervals(df, interval=interval)
    df["location"] = None
    for i in indexes:
        df.at[i, "location"] = getLocationFromCoords(df.at[i, "lat"], df.at[i, "lon"])
    for i in climbs_df.index:
        start_index = climbs_df.at[i, "start_idx"]
        end_index = climbs_df.at[i, "end_idx"]
        df.at[start_index, "location"] = getLocationFromCoords(df.at[start_index, "lat"], df.at[start_index, "lon"])
        df.at[end_index, "location"] = getLocationFromCoords(df.at[end_index, "lat"], df.at[end_index, "lon"])

    route.df = df
    return df