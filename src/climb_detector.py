import pandas as pd


def detect_significant_segments(
    df,
    kind: str = "climb",
    # --- New parameters for user control ---
    start_threshold_slope: float = 2.0,  # Minimum slope to start considering a climb
    end_threshold_slope: float = 1.0,  # If the slope drops below this, we enter "pause"
    max_pause_length_m: int = 200,  # If the pause lasts longer than this, the climb ends
    max_pause_descent_m: int = 10,  # If we lose more elevation than this, the climb ends
):
    segments = []
    state = "SEARCHING"

    # Sign to differentiate between climbs (1) and descents (-1)
    slope_sign = 1 if kind == "climb" else -1

    start_idx = 0
    current_segment_points = []

    for i in range(1, len(df)):
        point_data = df.iloc[i]
        slope = point_data["plot_grade"] * slope_sign
        elev_diff = (df["ele"].iloc[i] - df["ele"].iloc[i - 1]) * slope_sign
        dist_diff = point_data["distance"] - df["distance"].iloc[i - 1]

        # ---- STATE MACHINE ----
        if state == "SEARCHING":
            if slope >= start_threshold_slope:
                state = "IN_CLIMB"
                start_idx = i - 1
                current_segment_points = [
                    df.iloc[i - 1].to_dict(),
                    point_data.to_dict(),
                ]  # Start with 2 points

        elif state == "IN_CLIMB":
            if slope >= end_threshold_slope:
                # The climb continues, add the point
                current_segment_points.append(point_data.to_dict())
            else:
                # The slope has dropped, start a possible pause
                state = "EVALUATING_PAUSE"
                pause_start_idx = i - 1
                pause_length = 0
                pause_descent = 0
                current_segment_points.append(point_data.to_dict())

        elif state == "EVALUATING_PAUSE":
            current_segment_points.append(point_data.to_dict())
            pause_length += dist_diff
            if elev_diff < 0:  # Only count negative elevation change
                pause_descent += abs(elev_diff)

            # Criterion 1: The slope rises again, the pause ends, and the climb continues
            if slope >= end_threshold_slope:
                state = "IN_CLIMB"

            # Criterion 2 and 3: The pause is too long or we have descended too much
            elif (
                pause_length > max_pause_length_m or pause_descent > max_pause_descent_m
            ):
                # End of the climb. Save the segment BEFORE the pause
                final_segment_df = pd.DataFrame(
                    current_segment_points[: -(i - pause_start_idx)]
                )
                # Here we would validate and save the segment if it meets the requirements (length, elevation gain...)
                _validate_and_append_segment(
                    segments, final_segment_df, kind, start_idx
                )

                # Go back to searching for a new climb
                state = "SEARCHING"
                current_segment_points = []

    # At the end of the loop, check if we were in the middle of a climb
    if state in ["IN_CLIMB", "EVALUATING_PAUSE"] and current_segment_points:
        _validate_and_append_segment(
            segments, pd.DataFrame(current_segment_points), kind, start_idx
        )

    return pd.DataFrame(segments)


def _validate_and_append_segment(
    segments_list,
    segment_df,
    kind,
    start_idx,
    min_gain: int = 20,
    min_length: int = 300,
) -> None:
    """Helper function to avoid repeating validation code."""
    if segment_df.empty or len(segment_df) < 2:
        return

    length = segment_df["distance"].iloc[-1] - segment_df["distance"].iloc[0]

    if kind == "climb":
        gain = segment_df[segment_df["ele"].diff() > 0]["ele"].diff().sum()
    else:  # descent
        gain = abs(segment_df[segment_df["ele"].diff() < 0]["ele"].diff().sum())

    if length > min_length and gain > min_gain:
        avg_slope = (gain / length) * 100 if length > 0 else 0
        end_idx = start_idx + len(segment_df) - 1

        segments_list.append(
            {
                "type": kind,
                "start_km": segment_df["distance"].iloc[0] / 1000,
                "end_km": segment_df["distance"].iloc[-1] / 1000,
                "elev_gain" if kind == "climb" else "elev_loss": gain,
                "length_m": length,
                "avg_slope": avg_slope if kind == "climb" else -avg_slope,
                "start_idx": start_idx,
                "end_idx": end_idx,
            }
        )