import pandas as pd 
import numpy as np 
import os 

from loader import data_registry, reg, DATA_DIR, RAW_DIR


# Standardize names of columns 
ANIMAL_ID_COL_NAME = "ID"
TIMESTAMP_COL_NAME = "ts"
BEHAVIOR_COL_NAME = "behavior"
ACC_X_COL_NAME = "X"
ACC_Y_COL_NAME = "Y"
ACC_Z_COL_NAME = "Z"

G_EARTH = 9.8
E_OBS_OFFSET = 2048
E_OBS_SLOPE = 0.0027


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
    
    df[[ACC_X_COL_NAME, ACC_Y_COL_NAME, ACC_Z_COL_NAME]] *= G_EARTH # G-> ms^-2
    
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
    
    data[[ACC_X_COL_NAME, ACC_Y_COL_NAME, ACC_Z_COL_NAME]] *= G_EARTH # G-> ms^-2
    
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
    
    data[[ACC_X_COL_NAME, ACC_Y_COL_NAME, ACC_Z_COL_NAME]] -= E_OBS_OFFSET
    data[[ACC_X_COL_NAME, ACC_Y_COL_NAME, ACC_Z_COL_NAME]] *= E_OBS_SLOPE
    data[[ACC_X_COL_NAME, ACC_Y_COL_NAME, ACC_Z_COL_NAME]] *= G_EARTH 
    
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
    }, inplace=True)
    
    acc_cols = [col for col in data.columns if col.startswith("acc_")]
    data[acc_cols] *= G_EARTH
    
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
    
    data[[ACC_X_COL_NAME, ACC_Y_COL_NAME, ACC_Z_COL_NAME]] *= G_EARTH
    
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
    
    data[[ACC_X_COL_NAME, ACC_Y_COL_NAME, ACC_Z_COL_NAME]] *= G_EARTH
    
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
    
    data[[ACC_X_COL_NAME, ACC_Y_COL_NAME, ACC_Z_COL_NAME]] *= G_EARTH
    
    return data 


def read_weibke_hares():
    raw_folder = reg.loc["Weibke-Hares", "raw-folder"]
    path = os.path.join(RAW_DIR, raw_folder)

    all_segments = []
        
    for i, f in enumerate(os.listdir(path)):
        if f.endswith(".txt"):
            print(f)
            frame = pd.read_csv(os.path.join(path, f), delimiter=" ", parse_dates=["time"], date_format="%H:%M:%S")
            
            frame["behaviour"].replace({"Sitting_upright": "Sitting", "Running_zigzag": "Running"}, inplace=True)
            
            # Make the segments per 1s 
            frame["hour"] = frame["time"].dt.hour
            frame["minute"] = frame["time"].dt.minute
            frame["second"] = frame["time"].dt.second
            
            segments = frame.groupby(["hour", "minute", "second"]).apply(
                lambda seg: [f"animal_{i}", seg.name, seg.name] + seg["x_ms y_ms z_ms".split()].values.flatten().tolist() + [seg["behaviour"].mode()[0]] 
            )
            segments = pd.DataFrame(segments.tolist(), index=segments.index).dropna(how="any").reset_index(drop=True)
            
            segments.columns =[ANIMAL_ID_COL_NAME, "start_time", "end_time"] + [ACC_X_COL_NAME, ACC_Y_COL_NAME, ACC_Z_COL_NAME] * 18 + [BEHAVIOR_COL_NAME]
            all_segments.append(segments)        
            
    all_segments = pd.concat(all_segments, axis=0)
    return all_segments


def read_annett_glider():
    raw_folder = reg.loc["Annett-Gliders", "raw-folder"]
    df = pd.read_csv(os.path.join(RAW_DIR, raw_folder, "Annett_Glider_labelled.csv"), index_col=None)
    
    activity_codes = pd.read_csv(os.path.join(RAW_DIR, raw_folder, "Mahog_Glider_Behaviour_Act_Number.csv"), index_col="number")
    df["Activity"] = df["Activity"].map(activity_codes["activity"])
    
    df.rename(columns={
        "ID": ANIMAL_ID_COL_NAME,
        "Time": TIMESTAMP_COL_NAME,
        "Activity": BEHAVIOR_COL_NAME,
        "X": ACC_X_COL_NAME,
        "Y": ACC_Y_COL_NAME, 
        "Z": ACC_Z_COL_NAME 
        }, inplace=True)
     
    df[TIMESTAMP_COL_NAME] = pd.to_datetime(df[TIMESTAMP_COL_NAME], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')
    df[[ACC_X_COL_NAME, ACC_Y_COL_NAME, ACC_Z_COL_NAME]] *= G_EARTH    
    
    return df 
    
    
def read_clemente_echidna():
    raw_folder = reg.loc["Clemente-Echidna", "raw-folder"]
    df = pd.read_csv(os.path.join(RAW_DIR, raw_folder, "Clemente_Echidna_labelled.csv"), index_col=None)
    
    # Choose an arbitrary start date
    start_date = pd.Timestamp('2000-01-01')

    # Create the new datetime column by adding the seconds
    df['Time'] = start_date + pd.to_timedelta(df['Time'], unit='s')   
    
    activity_codes = {0: "Unknown",
                      1: "Inactivity",
                      2: "Digging",
                      3: "Walking",
                      4: "Climbing"} 
    
    df["Activity"] = df["Activity"].map(activity_codes)
    
    df.rename(columns={
            "ID": ANIMAL_ID_COL_NAME,
            "Time": TIMESTAMP_COL_NAME,
            "Activity": BEHAVIOR_COL_NAME,
            "X": ACC_X_COL_NAME,
            "Y": ACC_Y_COL_NAME, 
            "Z": ACC_Z_COL_NAME 
            }, inplace=True)
         
    df[[ACC_X_COL_NAME, ACC_Y_COL_NAME, ACC_Z_COL_NAME]] *= G_EARTH    
    return df 


def read_clemente_impala():
    raw_folder = reg.loc["Clemente-Impala", "raw-folder"]
    df = pd.read_csv(os.path.join(RAW_DIR, raw_folder, "Clemente_Imapla_labelled.csv"), index_col=None)
    
    df.rename(columns={
                "ID": ANIMAL_ID_COL_NAME,
                "utc_datetime": TIMESTAMP_COL_NAME,
                "Activity": BEHAVIOR_COL_NAME,
                "RawAX.cl": ACC_X_COL_NAME,
                "RawAY.cl": ACC_Y_COL_NAME, 
                "RawAZ.cl": ACC_Z_COL_NAME 
                }, inplace=True)
    
    df[TIMESTAMP_COL_NAME] = pd.to_datetime(df[TIMESTAMP_COL_NAME], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')
    df[[ACC_X_COL_NAME, ACC_Y_COL_NAME, ACC_Z_COL_NAME]] *= G_EARTH    
    return df 
        

def read_gaschk_quoll():
    raw_folder = reg.loc["Gaschk-Quoll", "raw-folder"]
    df = pd.read_csv(os.path.join(RAW_DIR, raw_folder, "Gashk_Quoll_formatted.csv"), index_col=None, parse_dates=["Time"])
    
    activity_codes = pd.read_csv(os.path.join(RAW_DIR, raw_folder, "Quoll_Behaviour_ActNo.csv"), index_col="Number")
    df["Activity"] = df["Activity"].map(activity_codes["Activity"])
        
    df.rename(columns={
        "ID": ANIMAL_ID_COL_NAME,
        "Time": TIMESTAMP_COL_NAME,
        "Activity": BEHAVIOR_COL_NAME,
        "X": ACC_X_COL_NAME,
        "Y": ACC_Y_COL_NAME, 
        "Z": ACC_Z_COL_NAME 
        }, inplace=True)
    
    df = df.dropna(how='any').reset_index(drop=True)
    df[TIMESTAMP_COL_NAME] = pd.to_datetime(df[TIMESTAMP_COL_NAME], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')
    df[[ACC_X_COL_NAME, ACC_Y_COL_NAME, ACC_Z_COL_NAME]] *= G_EARTH    
    return df 

            

def read_galea_cat():
    raw_folder = reg.loc["Galea-Cat", "raw-folder"]
    df = pd.read_csv(os.path.join(RAW_DIR, raw_folder, "Galea_Cat_formatted.csv"), index_col=None)
    
    df.rename(columns={
        "ID": ANIMAL_ID_COL_NAME,
        "Time": TIMESTAMP_COL_NAME,
        "Activity": BEHAVIOR_COL_NAME,
        "X": ACC_X_COL_NAME,
        "Y": ACC_Y_COL_NAME, 
        "Z": ACC_Z_COL_NAME 
        }, inplace=True)
            
    
    df[[ACC_X_COL_NAME, ACC_Y_COL_NAME, ACC_Z_COL_NAME]] *= G_EARTH    
    return df 

        



if __name__ == "__main__":
    df = read_gaschk_quoll()
    print(df.head())
    print(df.info())
    print(df.shape)
    print(df.groupby(BEHAVIOR_COL_NAME).size())
    print(type(df.loc[0, "ts"]))
    print(df.loc[0, "ts"])
    
    frame = df 
    sample_gap = frame["ts"].diff().median()
    print(f"Median sample gap: {sample_gap}")
    
    # sample_hz = 1/sample_gap.total_seconds()
    