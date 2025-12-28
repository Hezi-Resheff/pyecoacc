
import pandas as pd 


def long_to_wide_segments(
    df: pd.DataFrame, 
    segment_duration: str = "1s",
    xcol: str = "accX",
    ycol: str = "accY",
    zcol: str = "accZ",
    timestamp_col: str = "Timestamp",
    sort_by_time: bool = True,
) -> pd.DataFrame:
    """ Make a wide dataframe: one row per non-overlapping segment. Appropriate to use for a single animal with continuous data.    
    
    The input format uses the original long-shape columns: X, Y, Z, timestamp. Assumes (approximately) constant sampling interval; drops the last partial segment.
    Assumes no gaps. 
    
    We thank an anonymous reviewer for contributing this function.
    

    Args:
        df (pd.DataFrame): input dataframe
        segment_duration (str, optional): encodes the duration of each segment. Defaults to "1s".
        xcol (str, optional): the name of the column for the X acceleration axis. Defaults to "accX".
        ycol (str, optional): the name of the column for the Y acceleration axis. Defaults to "accY".
        zcol (str, optional): the name of the column for the Z acceleration axis. Defaults to "accZ".
        timestamp_col (str, optional): the name of the column for the timestamp. Defaults to "Timestamp".
        sort_by_time (bool, optional): if True, sort the data by timestamp. Defaults to True.


    Returns:
        segment_table (pd.DataFrame): a wide dataframe with one row per non-overlapping segment.
    """
    
    d = df[[timestamp_col, xcol, ycol, zcol]].copy()
    d[timestamp_col] = pd.to_datetime(d[timestamp_col], errors="coerce")
    d = d.dropna(subset=[timestamp_col])

    if sort_by_time:
        d = d.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)

    if len(d) < 2:
        raise ValueError("Need at least 2 rows to infer sampling interval")

    dt = (d[timestamp_col].iloc[1] - d[timestamp_col].iloc[0]).total_seconds()
    if dt <= 0:
        raise ValueError("Non-positive sampling interval (check timestamp ordering/duplicates)")

    seg_s = pd.to_timedelta(segment_duration).total_seconds()
    samples_per_seg = int(round(seg_s / dt))
    if samples_per_seg <= 0:
        raise ValueError("segment_duration too small for inferred sampling interval")

    acc = d[[xcol, ycol, zcol]].to_numpy()  # (N, 3)

    nseg = acc.shape[0] // samples_per_seg
    acc = acc[: nseg * samples_per_seg]

    wide = acc.reshape(nseg, samples_per_seg * 3)  # (nseg, 3*T)

    # column names: x,y,z,x.1,y.1,z.1,...
    base = ["x", "y", "z"]
    cols = []
    for i in range(samples_per_seg):
        suffix = "" if i == 0 else f".{i}"
        cols.extend([f"{b}{suffix}" for b in base])

    out = pd.DataFrame(wide, 
                       columns=cols,
                       index=d[timestamp_col].iloc[::samples_per_seg].iloc[:nseg].to_numpy())
    return out


def long_to_wide_multi_animal(
    df: pd.DataFrame, 
    id_col: str = "AnimalID",
    segment_duration: str = "1s",
    xcol: str = "accX",
    ycol: str = "accY",
    zcol: str = "accZ",
    timestamp_col: str = "Timestamp",
    sort_by_time: bool = True,
) -> pd.DataFrame:
    """Make a wide dataframe: one row per non-overlapping segment. Appropriate to use for multiple animals with continuous data for each one.     


    Assumes (approximately) constant sampling interval; drops the last partial segment.
    Assumes no gaps in the data of each animal. 

    Args:
        df (pd.DataFrame): input dataframe
        segment_duration (str, optional): encodes the duration of each segment. Defaults to "1s".
        xcol (str, optional): the name of the column for the X acceleration axis. Defaults to "accX".
        ycol (str, optional): the name of the column for the Y acceleration axis. Defaults to "accY".
        zcol (str, optional): the name of the column for the Z acceleration axis. Defaults to "accZ".
        timestamp_col (str, optional): the name of the column for the timestamp. Defaults to "Timestamp".
        sort_by_time (bool, optional): if True, sort the data by timestamp. Defaults to True.
        id_col (str, optional): _description_. Defaults to "AnimalID".
       
    Returns:
        segment_table (pd.DataFrame): a wide dataframe with one row per non-overlapping segment.
    """

    all_animals = [] 
    
    for animal_id, animal in df.groupby(id_col):
        segments = long_to_wide_segments(animal, 
                                         segment_duration=segment_duration,
                                         xcol=xcol,
                                         ycol=ycol,
                                         zcol=zcol,
                                         timestamp_col=timestamp_col,
                                         sort_by_time=sort_by_time)
        segments.index = pd.MultiIndex.from_product([animal_id, segments.index])
        all_animals.append(segments)    
    
    out = pd.concat(all_animals, axis=0)
    return out



    
    
    
    