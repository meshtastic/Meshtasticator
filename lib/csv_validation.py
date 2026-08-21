"""Small validation helpers for simulator CSV inputs."""

import math

from lib.geo import valid_lat_lon


def finite_float(row, column, csv_name, row_number):
    """Parse a required finite float from a CSV row with a useful error."""
    try:
        value = float(row[column])
    except KeyError as err:
        raise ValueError(f"{csv_name} CSV row {row_number} needs {column} column") from err
    except (TypeError, ValueError) as err:
        raise ValueError(f"{csv_name} CSV row {row_number} has invalid {column}: {row.get(column)!r}") from err

    if not math.isfinite(value):
        raise ValueError(f"{csv_name} CSV row {row_number} has non-finite {column}: {row.get(column)!r}")
    return value


def finite_lat_lon(row, csv_name, row_number):
    """Parse and range-check lat/lon columns from a CSV row."""
    lat = finite_float(row, "lat", csv_name, row_number)
    lon = finite_float(row, "lon", csv_name, row_number)
    if not valid_lat_lon(lat, lon):
        raise ValueError(f"{csv_name} CSV row {row_number} has invalid latitude/longitude degrees")
    return lat, lon
