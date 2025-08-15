from dataclasses import dataclass
import pandas as pd
import gpxpy

@dataclass
class GPXData:
    gpx: gpxpy.gpx.GPX
    df: pd.DataFrame
    stats: dict = None
    climbs_df: pd.DataFrame = None
    descents_df: pd.DataFrame = None
