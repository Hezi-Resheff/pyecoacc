import os
import pandas as pd
from reader import data_registry, DATA_DIR


SEGMENTS_DIR = os.path.join(DATA_DIR, "segments")


# Supported formats for segmented datasets
FORMAT_XYZXYZ = 1
FORMAT_LONG = 2
FORMAT_TENSOR = 3


reg = data_registry()


def load_raw_dataset(dataset_name):
    load_func = eval(reg.loc[dataset_name, "loader"])
    return load_func()


def load_segmented_dataset(dataset_name, format=FORMAT_XYZXYZ):
    segments_file = reg.loc[dataset_name, "segmented"] 
    data = pd.read_csv(os.path.join(SEGMENTS_DIR, segments_file), index_col=0)
    # TODO: convert to the requested format
    return data


def load_all_segmented_datasets(format=FORMAT_XYZXYZ):
    """
    Provides a convenient way to iterate over all segmented datasets in the data registry.
    
    Yields:
        tuple: A tuple containing the dataset name and the corresponding segmented dataset.
    """
    for dataset_name in reg.index:
        yield dataset_name, load_segmented_dataset(dataset_name, format=format)
    




