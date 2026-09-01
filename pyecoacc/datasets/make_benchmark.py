""" A utility for making the "benchmark" datasets with programmable parameters 
"""
    
import numpy as np 
import pandas as pd 

from segment import *
from loader import *
from reader import *


# Keep only behaviors that have at least this many segments in the dataset when possible 
MINIMUM_BEHAVIOR_SEGMENTS = 200

# Keep only at most this many segments per behavior in the dataset
MAXIMUM_BEHAVIOR_SEGMENTS = 1000

# These are names of non-behavior segments that should not be included
DROP_CODES = ["NotScored", "Out of sight", "other", "unknown"]

# Save benchmark to here 
BENCHMARK_TARGET_DIR = os.path.join(DATA_DIR, "segments")

# Use only these behavs per animal
BEHAV_FILTER = {
    "Pagano-Bears": ["laying", "walking", "eating", "swimming", "digging", "grooming"]
}


def make_min_max_segments(min_seg=MINIMUM_BEHAVIOR_SEGMENTS, 
                          max_seg=MAXIMUM_BEHAVIOR_SEGMENTS, 
                          path=BENCHMARK_TARGET_DIR, 
                          behav_filter_per_animal=BEHAV_FILTER):
    
    registry = data_registry()
    
    for animal in registry.index:
        print("Starting with animal: ", animal)
        
        # Just prints 
        target_csv = registry.loc[animal, "segmented"]
        target_path = os.path.join(path, target_csv)
        if os.path.exists(target_path):
            print(f"  --> File {target_path} already exists, skipping")
            continue
        else:
            print("  --> Working... ")
        
        # Make the segments 
        raw_data = eval(registry.loc[animal, "load-func"])()
        
        seg_length = 1.0 if animal in ("Ladds-Seals", "Harel-Baboons") else 2.0 if animal in ("Rotics-Molerats", "Rotics-Meerkats") else 3.0
        print("  -> segment length: ", seg_length)
        
        if animal not in ("Spiegel-Vultures", "Efrat-Vultures", "Rotics-Storks", "Sasha-Cranes"): # These already come as segments 
            segments = segment(raw_data, 
                            id_column=ANIMAL_ID_COL_NAME, 
                            time_column=TIMESTAMP_COL_NAME, 
                            behav_column=BEHAVIOR_COL_NAME, 
                            x_column=ACC_X_COL_NAME, 
                            y_column=ACC_Y_COL_NAME, 
                            z_column=ACC_Z_COL_NAME, 
                            allow_behav_switches=False, 
                            segment_length=seg_length)
            
            segments_df = make_segments_csv(segments)
        else:
            segments_df = raw_data
        
        # Process segments 
        # --> per animal filters 
        if animal in behav_filter_per_animal.keys():
            use = behav_filter_per_animal[animal]
            segments_df = segments_df[segments_df.behavior.isin(use)].copy()
        
        # --> Keep only behaviors that have at least MINIMUM_BEHAVIOR_SEGMENTS segments in the dataset, and drop any behaviors that are in DROP_CODES
        min_segments_keep = 100 if animal == "Harel-Baboons" else 50 if animal in ("Rotics-Storks", "Sasha-Cranes", "Efrat-Vultures") else min_seg
        keep_behavs = segments_df.behavior.value_counts()[segments_df.behavior.value_counts() >= min_segments_keep].index 
        keep_behavs = [b for b in keep_behavs if b not in DROP_CODES]
        segments_df = segments_df[segments_df.behavior.isin(keep_behavs)].copy()

        # --> Drop the start_time and end_time columns since they are not needed for the dataset
        if "start_time" in segments_df.columns:
            segments_df.drop(columns=["start_time", "end_time"], inplace=True)

        # Sample at most MAXIMUM_BEHAVIOR_SEGMENTS segments per behavior in the dataset
        segments_df = segments_df.groupby("behavior").apply(lambda x: x.sample(n=max_seg) if len(x) > max_seg else x).reset_index(drop=True)

        # save 
        segments_df.to_csv(target_path, index=False)

        print(segments_df.behavior.value_counts())
        

if __name__ == "__main__":
    make_min_max_segments(min_seg=MINIMUM_BEHAVIOR_SEGMENTS, max_seg=MAXIMUM_BEHAVIOR_SEGMENTS, path=BENCHMARK_TARGET_DIR)
    make_min_max_segments(min_seg=MINIMUM_BEHAVIOR_SEGMENTS, max_seg=1e6, path=BENCHMARK_TARGET_DIR+"_all") # ./segments_all 