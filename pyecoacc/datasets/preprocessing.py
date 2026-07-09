import pandas as pd 
import os 

from reader import RAW_DIR


BEAR_PATH = os.path.join(RAW_DIR, "pagano_bears")
BEAR_ACC_FILE = "PolarBear_archival_logger_data_southernBeaufortSea_2014_2016_revised.csv"
BEAR_ANNOTATIONS = "PolarBear_video-derived_behaviors_southernBeaufortSea_2014_2016_revised.csv"
BEAR_OUTPUT_FILE = "bears_with_behav_10percent.csv"
BEAR_OUTPUT_FILE_FULL = "bears_with_behav.csv"



def make_bears(save_full=True):
    """
    Don't literally make bears in the office. They are too dangerous.
    
    This converts the raw data from the Pagano et al. polar bear dataset into a single CSV file that contains both the accelerometer data and the behavior annotations.
    """
    # ACC data file
    acc = pd.read_csv(os.path.join(BEAR_PATH, BEAR_ACC_FILE))
    acc.columns = ["Animal", "dt", "x", "y", "z", "_"]
    acc.drop("_", axis=1, inplace=True)
    acc.dt = pd.DatetimeIndex(acc.dt)

    # annotations file
    labels = pd.read_csv(os.path.join(BEAR_PATH, BEAR_ANNOTATIONS))
    labels.columns = ["Animal", "dt_begin", "dt_end", "Behavior", "_"]
    labels.drop("_", axis=1, inplace=True)
    labels["dt_begin"] = pd.DatetimeIndex(labels.dt_begin)
    labels["dt_end"] = pd.DatetimeIndex(labels.dt_end)

    valid_animals = acc.Animal.unique()
    labels.drop(labels.loc[~labels.Animal.isin(valid_animals)].index, axis=0, inplace=True)

    # combine & save
    acc["Behavior"] = pd.NA

    for i, row in labels.iterrows():
        if i % 100 < 10 or save_full:  # consecutive 10% of segments unless save full
            ix = acc[(acc.Animal == row.Animal) & acc.dt.between(row.dt_begin, row.dt_end)].index
            acc.loc[ix, "Behavior"] = row.Behavior

        if i % 100 == 0:
            print(f"Processing segment... {i}")

    acc.dropna(subset="Behavior", inplace=True)

    acc.to_csv(os.path.join(BEAR_PATH, BEAR_OUTPUT_FILE_FULL if save_full else BEAR_OUTPUT_FILE))


if __name__ == "__main__":
    make_bears(save_full=True)