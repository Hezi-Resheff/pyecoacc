import pandas as pd
from .reader import * 


def segment(frame, id_column, time_column, behav_column, x_column="x", y_column="y", z_column="z", allow_behav_switches=True, segment_length=2.):
    
    sample_gap = frame[time_column].diff().median()
    sample_hz = 1/sample_gap.total_seconds()
    rows_per_segment = int(segment_length * sample_hz)
    print(f"Sampling rate is {sample_hz:.2f} Hz, so each segment will have {rows_per_segment} rows.")
    
    
    all_segments = []
    
    for animal_id, animal_data in frame.groupby(id_column):
        
        n = len(animal_data)
        i_seg_start = 0
        
        for i_current_row in range(1, n-1):
            gap = abs(frame[time_column].iloc[i_current_row] - frame[time_column].iloc[i_current_row - 1])
            switch = frame[behav_column].iloc[i_current_row] != frame[behav_column].iloc[i_current_row - 1]
            
            if gap > 2 * sample_gap:
                # Drop current and start a new segment
                i_seg_start = i_current_row
                continue
            
            if switch and not allow_behav_switches:
                # Drop current and start a new segment
                i_seg_start = i_current_row
                continue
            
            this_segment_row_count = i_current_row - i_seg_start + 1
            
            if this_segment_row_count == rows_per_segment:
                # This is the last row of the segment
                segment_data = animal_data.iloc[i_seg_start:i_current_row + 1]
                i_seg_start = i_current_row + 1
                all_segments.append({
                    ANIMAL_ID_COL_NAME: animal_id,
                    "start_time": segment_data[time_column].iloc[0],
                    "end_time": segment_data[time_column].iloc[-1],
                    BEHAVIOR_COL_NAME: segment_data[behav_column].mode()[0],
                    "segment_data": segment_data[[x_column, y_column, z_column]].values 
                })
            
    return all_segments


def make_segments_csv(segments):
    """
    Converts a list of segments into a DataFrame so it can be saved as a single CSV file.
    """
    segments_rows = [[segment[ANIMAL_ID_COL_NAME], segment["start_time"], segment["end_time"], segment[BEHAVIOR_COL_NAME]] + list(segment["segment_data"].flatten())
                     for segment in segments]
    segment_column_names = [ANIMAL_ID_COL_NAME, "start_time", "end_time", BEHAVIOR_COL_NAME] + [ACC_X_COL_NAME, ACC_Y_COL_NAME, ACC_Z_COL_NAME] * (segments[0]["segment_data"].shape[0])
    return pd.DataFrame(segments_rows, columns=segment_column_names)
            
    


    