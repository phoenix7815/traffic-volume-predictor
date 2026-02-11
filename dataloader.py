import os
import random
import numpy as np
import pandas as pd

class DataLoader:
    def __init__(self, path_to_csv, sensor_split=0.9, time_split=0.8, index: np.ndarray = None):
        if not os.path.exists(path_to_csv):
            raise FileNotFoundError(f"File {path_to_csv} does not exist.")

        if index is None:
            # Load with first column as index (time steps)
            self.data = pd.read_csv(path_to_csv, index_col=0)
        else:
            # Load without index and set custom index
            self.data = pd.read_csv(path_to_csv)
            if len(index) != len(self.data):
                raise ValueError(f'{len(index)} timestamps provided, but CSV has {len(self.data)} rows.')
            self.data.set_index(pd.Index(index), inplace=True)
        
        rows, cols = self.data.shape
        if rows == 0 or cols == 0:
            raise ValueError("The CSV file is empty.")
        else:
            print(f"Loaded data with {cols} sensors and {rows} time steps.")
        
        # Split sensors into seen and unseen
        sensor_names = self.data.columns.tolist()
        random.shuffle(sensor_names)

        split_index = int(len(sensor_names) * sensor_split)
        self.seen_sensors = sensor_names[:split_index]
        self.unseen_sensors = sensor_names[split_index:]

        self.train_data = self.data[self.seen_sensors].iloc[:int(len(self.data) * time_split)]

        self.test_seen_sensors_data = self.data[self.seen_sensors].iloc[int(len(self.data) * time_split):]
        self.test_unseen_sensors_data = self.data[self.unseen_sensors]

        print(f"Seen sensors: {len(self.seen_sensors)}, Unseen sensors: {len(self.unseen_sensors)}")
        print(f"Training data shape: {self.train_data.shape}")
        print(f"Test data (seen sensors) shape: {self.test_seen_sensors_data.shape}")
        print(f"Test data (unseen sensors) shape: {self.test_unseen_sensors_data.shape}")

    def get_train_data(self):
        return self.train_data

    def get_test_seen_sensors_data(self):
        return self.test_seen_sensors_data

    def get_test_unseen_sensors_data(self):
        return self.test_unseen_sensors_data