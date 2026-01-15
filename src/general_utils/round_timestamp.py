from datetime import datetime, timedelta

def round_to_nearest_hour(dt: datetime) -> datetime:
    seconds = (dt - dt.replace(minute=0, second=0, microsecond=0)).total_seconds()
    if seconds >= 1800:
        dt = dt + timedelta(hours=1)
    return dt.replace(minute=0, second=0, microsecond=0)