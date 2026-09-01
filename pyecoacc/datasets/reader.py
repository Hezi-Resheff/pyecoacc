import pandas as pd 
import numpy as np 
import os 

# Change this to the location the data will be downloaded to 
DATA_DIR = os.path.expanduser("~/Desktop/acc-data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

# Standardize names of columns 
ANIMAL_ID_COL_NAME = "ID"
TIMESTAMP_COL_NAME = "ts"
BEHAVIOR_COL_NAME = "behavior"
ACC_X_COL_NAME = "X"
ACC_Y_COL_NAME = "Y"
ACC_Z_COL_NAME = "Z"


def data_registry():
    """ Load the data registry
    """
    here = os.path.dirname(__file__)
    registry = pd.read_csv(os.path.join(here, "registry.csv"), index_col=0) 
    return registry

reg = data_registry()


def read_rotics_molerats():
    raw_folder = reg.loc["Rotics-Molerats", "raw-folder"]
    df = pd.read_csv(os.path.join(RAW_DIR, raw_folder, "obs_ACC.csv"), index_col=0)
    df["dt"] = pd.to_datetime(df.date + " " + df.time, format="%d/%m/%Y %H:%M:%S.%f")
    df.drop_duplicates(subset=["Animal", "dt"], inplace=True)
    
    df.rename(columns={
        "Animal": ANIMAL_ID_COL_NAME,
        "Behavior": BEHAVIOR_COL_NAME,
        "dt": TIMESTAMP_COL_NAME,
        "x": ACC_X_COL_NAME,
        "y": ACC_Y_COL_NAME, 
        "z": ACC_Z_COL_NAME 
    }, inplace=True)
    
    return df 

  
def read_rotics_meerkats():
    raw_folder = reg.loc["Rotics-Meerkats", "raw-folder"]
    path = os.path.join(RAW_DIR, raw_folder, "DataAcc_label_2")
    
    all_files = [f for f in os.listdir(path) if f.endswith(".csv")]

    frames = []

    for f in all_files:
        data = pd.read_csv(os.path.join(path, f), index_col=0)
        ts = pd.to_datetime(data["Date"] + " " + data["Time.hh.mm.ss.ddd"], format="%Y-%m-%d %H:%M:%S.%f")
        data.insert(0, "ts", ts)
        data.rename({"Acc_x": "x", "Acc_y": "y", "Acc_z": "z", "Behaviour": "Behavior", "ID": "Animal"},
                    axis=1, inplace=True)
        data.drop_duplicates(subset=["Animal", "ts"], inplace=True)
        frames.append(data)

    data = pd.concat(frames, axis=0)

    data.rename(columns={
        "Animal": ANIMAL_ID_COL_NAME,
        "Behavior": BEHAVIOR_COL_NAME,
        "ts": TIMESTAMP_COL_NAME,
        "x": ACC_X_COL_NAME,
        "y": ACC_Y_COL_NAME, 
        "z": ACC_Z_COL_NAME 
    }, inplace=True)
    
    return data 


def read_rotics_storks():
    raw_folder = reg.loc["Rotics-Storks", "raw-folder"]
    segments_df = pd.read_csv(os.path.join(RAW_DIR, raw_folder, "storks_obs.csv"), header=None, index_col=None)
    
    segments_df.insert(0, ANIMAL_ID_COL_NAME, pd.NA)
    segments_df.insert(0, "end_time", pd.NA)
    segments_df.insert(0, "start_time", pd.NA)

    segments_df.columns = [ANIMAL_ID_COL_NAME, "start_time", "end_time"] + \
    [ACC_X_COL_NAME, ACC_Y_COL_NAME, ACC_Z_COL_NAME] * 40 + [BEHAVIOR_COL_NAME]

    # Move behav to 4-th position
    behav = segments_df.pop(BEHAVIOR_COL_NAME)
    segments_df.insert(3, BEHAVIOR_COL_NAME, behav)
    
    return segments_df 
    

def read_sasha_cranes():
    raw_folder = reg.loc["Sasha-Cranes", "raw-folder"]
    segments_df = pd.read_csv(os.path.join(RAW_DIR, raw_folder, "AcceleRaterToUSE.csv"), header=None, index_col=None)
    
    segments_df.insert(0, ANIMAL_ID_COL_NAME, pd.NA)
    segments_df.insert(0, "end_time", pd.NA)
    segments_df.insert(0, "start_time", pd.NA)

    segments_df.columns = [ANIMAL_ID_COL_NAME, "start_time", "end_time"] + \
    [ACC_X_COL_NAME, ACC_Y_COL_NAME, ACC_Z_COL_NAME] * 40 + [BEHAVIOR_COL_NAME]

    # Move behav to 4-th position
    behav = segments_df.pop(BEHAVIOR_COL_NAME)
    segments_df.insert(3, BEHAVIOR_COL_NAME, behav)
    
    return segments_df 
    

def read_harel_baboons():
    raw_folder = reg.loc["Harel-Baboons", "raw-folder"]
    path = os.path.join(RAW_DIR, raw_folder) 
    data = pd.read_csv(os.path.join(path, "baboons_acc_raw_behav_2019_cleaned_ver2.csv"), parse_dates=["timestamp"])
    
    # bab.rename({"timestamp": "ts", "behav": "behavior", "tag": "Animal"}, axis=1, inplace=True)
    data.dropna(how="any", inplace=True)
    
    data.rename(columns={
        "tag": ANIMAL_ID_COL_NAME,
        "behav": BEHAVIOR_COL_NAME,
        "timestamp": TIMESTAMP_COL_NAME,
        "x": ACC_X_COL_NAME,
        "y": ACC_Y_COL_NAME, 
        "z": ACC_Z_COL_NAME 
    }, inplace=True)
    
    return data 
    

def read_pagano_bears():
   raw_folder = reg.loc["Pagano-Bears", "raw-folder"]
   path = os.path.join(RAW_DIR, raw_folder)
   f_name = "bears_with_behav.csv"
   data = pd.read_csv(os.path.join(path, f_name), parse_dates=["dt"], index_col=0)
   
   data.rename(columns={
        "Animal": ANIMAL_ID_COL_NAME,
        "Behavior": BEHAVIOR_COL_NAME,
        "dt": TIMESTAMP_COL_NAME,
        "x": ACC_X_COL_NAME,
        "y": ACC_Y_COL_NAME, 
        "z": ACC_Z_COL_NAME 
    }, inplace=True)
   
   return data 
   
   
def read_efrat_vultures():
    raw_folder = reg.loc["Efrat-Vultures", "raw-folder"]
    path = os.path.join(RAW_DIR, raw_folder)
    data = pd.read_csv(os.path.join(path, "segments.csv"), header=None, index_col=None)
    
    l = (data.shape[1] - 2) // 3
    
    data.columns = [ANIMAL_ID_COL_NAME] + [ACC_X_COL_NAME, ACC_Y_COL_NAME, ACC_Z_COL_NAME] * l + [BEHAVIOR_COL_NAME]
        
    return data
    
    
def read_spiegel_vultures():
    raw_folder = reg.loc["Spiegel-Vultures", "raw-folder"]
    path = os.path.join(RAW_DIR, raw_folder)
    data = pd.read_csv(os.path.join(path, "training_dataset.csv"))
    
    data = data.loc[:, :"acc_z_100"]
    data.drop(columns=["bout_id"], inplace=True)
   
    col_order = ["device_id", "observed_beh"] + [f"acc_{axis}_{i}" for i in range(1, 101) for axis in ["x", "y", "z"]]
    data = data[col_order]
    
    data.rename(columns={
        "device_id": ANIMAL_ID_COL_NAME,
        "observed_beh": BEHAVIOR_COL_NAME,
        "X": ACC_X_COL_NAME,
        "Y": ACC_Y_COL_NAME, 
        "Z": ACC_Z_COL_NAME 
    }, inplace=True)
    
    return data 
    
    
    
def read_agarwal_african_wild_dogs():
    raw_folder = reg.loc["Agarwal-Dogs", "raw-folder"]
    path = os.path.join(RAW_DIR, raw_folder)
    
    # Load segments frame 
    segments = pd.read_csv(os.path.join(path, "matched_acceleration_data_out.csv"), parse_dates=["behavior_start", "behavior_end"])
   
    # Load annotations 
    annot = pd.read_csv(os.path.join(path, "matched_acceleration_metadata_out.csv"))
    
    # Combine 
    df = pd.concat([segments, annot], axis=1)        
    
    # Make format 
    sampling_rate = 16 # from paper 
    all_data = []

    for i, row in df.iterrows():
        
        xyz = np.stack(row["acc_x acc_y acc_z".split()].apply(eval).values).T
        time_index = pd.date_range(start=row["behavior_start"], periods=xyz.shape[0], freq='62.5ms', name="ts")
        
        seg = pd.DataFrame(xyz, index=time_index, columns="X Y Z".split()).reset_index() 
        seg["ID"] = row["individual ID"]
        seg["behav"] = row["behavior"]
        
        all_data.append(seg)
        
    data = pd.concat(all_data)
    
    data.rename(columns={
         "ID": ANIMAL_ID_COL_NAME,
         "ts": TIMESTAMP_COL_NAME,
         "behav": BEHAVIOR_COL_NAME,
         "X": ACC_X_COL_NAME,
         "Y": ACC_Y_COL_NAME, 
         "Z": ACC_Z_COL_NAME 
    }, inplace=True)
    
    return data 
    
    
def read_ladds_seals():
    raw_folder = reg.loc["Ladds-Seals", "raw-folder"]
    path = os.path.join(RAW_DIR, raw_folder, "raw_data")
    
    # structure is animal/data...
    all_data = [] 
    animals = os.listdir(path)
    animals = filter(lambda f: not f.startswith("."), animals)
    
    for animal in animals:
        data_files = os.listdir(os.path.join(path, animal))
        data_files = filter(lambda f: f.endswith(".csv"), data_files)
        
        for file in data_files:
            frame = pd.read_csv(os.path.join(path, animal, file), 
                                usecols="x y z behaviour date".split())
            frame["date"] = pd.to_datetime(frame['date'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
            frame["ID"] = animal 
            frame.dropna(subset=['date'], inplace=True)
            all_data.append(frame)
    
    data = pd.concat(all_data)
    
    data.rename(columns={
         "ID": ANIMAL_ID_COL_NAME,
         "date": TIMESTAMP_COL_NAME,
         "behaviour": BEHAVIOR_COL_NAME,
         "x": ACC_X_COL_NAME,
         "y": ACC_Y_COL_NAME, 
         "z": ACC_Z_COL_NAME 
    }, inplace=True)
    
    return data 

    
def read_maekawa_gulls():
    raw_folder = reg.loc["Maekawa-Gulls", "raw-folder"]
    path = os.path.join(RAW_DIR, raw_folder)

    # ACC data 
    data = pd.read_csv(os.path.join(path, "raw_data.csv"))
    
    # behav 
    behav = pd.read_csv(os.path.join(path, "labels.csv"))
    
    # combine 
    data["behav"] = pd.NA
    for i, row in behav.iterrows():
        obs_rows = (data.animal_tag == row.animal_tag) & (data.timestamp >= row.stt_timestamp) & (data.timestamp <= row.stp_timestamp)
        data.loc[obs_rows, "behav"] = row.activity

    data['timestamp'] = pd.to_datetime(data['timestamp'],  format='%Y-%m-%dT%H:%M:%S.%fZ',  errors='coerce')
    data.dropna(subset=['timestamp', 'behav'], inplace=True)
    
    data.rename(columns={
         "animal_tag": ANIMAL_ID_COL_NAME,
         "timestamp": TIMESTAMP_COL_NAME,
         "behav": BEHAVIOR_COL_NAME,
         "acc_x": ACC_X_COL_NAME,
         "acc_y": ACC_Y_COL_NAME, 
         "acc_z": ACC_Z_COL_NAME 
    }, inplace=True)
    
    return data 


if __name__ == "__main__":
    df = read_efrat_vultures()
    print(df.head())
    print(df.groupby("behavior").size())
    