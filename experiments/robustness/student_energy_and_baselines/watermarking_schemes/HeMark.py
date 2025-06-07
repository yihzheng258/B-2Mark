import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import hashlib
import math

class HeMarkWatermarkEmbedding:
    def __init__(self, watermarked_column, dataset="housing", seed=10000, m=10, gamma=1/2):
        self.dataset = dataset
        self.seed = seed
        self.m = m
        self.watermarked_column = watermarked_column
        self.origin = None
        self.green_domain_values = None
        self.green_mid_values = None
        np.random.seed(self.seed)

    def load_data(self, file_path):
        self.origin = pd.read_csv(file_path)


    def generate_segments(self):
        intervals = np.linspace(0, 1, self.m + 1)
        segments = [(intervals[i], intervals[i + 1]) for i in range(self.m)]
        np.random.shuffle(segments)
        half_m = self.m // 2
        green_domains = segments[:half_m]
        self.green_domain_values = [(low, np.nextafter(high, low)) for low, high in green_domains]
        self.green_mid_values = [(seg[0] + seg[1]) / 2 for seg in self.green_domain_values]

    def process_data(self):
        for idx in range(len(self.origin)):
            original_medv = self.origin.loc[idx, self.watermarked_column]
            
            int_part = int(np.floor(original_medv))
            dec_part = original_medv - int_part
            
            closest_mid = min(self.green_mid_values, key=lambda x: abs(x - dec_part))
            closest_idx = self.green_mid_values.index(closest_mid)

            if dec_part >= self.green_domain_values[closest_idx][0] and \
               dec_part < self.green_domain_values[closest_idx][1]:
                new_frac = dec_part
            else:
                new_frac = np.random.uniform(self.green_domain_values[closest_idx][0],
                                                  self.green_domain_values[closest_idx][1])
            self.origin.loc[idx, self.watermarked_column] = int_part + new_frac

    def save_results(self, output_path):
        results = {'watermarked_data': self.origin}
        np.save(output_path, results)




class HeMarkWatermarkDetection:
    def __init__(self, watermarked_column, dataset="housing", seed=10000, m=10, gamma=1/2):
        self.dataset = dataset
        self.seed = seed
        self.m = m
        self.watermarked_column = watermarked_column
        self.origin = None
        self.green_domain_values = None
        self.green_mid_values = None
        np.random.seed(self.seed)

    def load_data(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        if(file_extension == '.csv'):
            data = pd.read_csv(file_path)
        elif(file_extension == '.npy'):
            loaded_results = np.load(file_path, allow_pickle=True).item()
            data = loaded_results['watermarked_data']
            
        return data

    def generate_segments(self):
        intervals = np.linspace(0, 1, self.m + 1)
        segments = [(intervals[i], intervals[i + 1]) for i in range(self.m)]
        np.random.shuffle(segments)
        
        half_m = self.m // 2
        self.green_domains = segments[:half_m]
        self.red_domains = segments[half_m:]

    def detect_watermark(self, watermarked_data):
        green_cell = 0
        n_cell = 0

        for idx in range(len(watermarked_data)):
            n_cell += 1
            
            int_part = int(np.floor(watermarked_data.loc[idx, self.watermarked_column]))
            dec_part = watermarked_data.loc[idx, self.watermarked_column] - int_part
            
            for low, high in self.green_domains:
                if low <= dec_part < high:
                    green_cell += 1
                    break

        return green_cell, n_cell

    def compute_z_score(self, green_cell, n_cell):
        return (green_cell - n_cell / 2) / math.sqrt(n_cell / 4)

    def run_detection(self, file_path):
        watermarked_data = self.load_data(file_path)
        
        np.random.seed(self.seed)
        self.generate_segments()

        green_cell, n_cell = self.detect_watermark(watermarked_data)
        z_score = self.compute_z_score(green_cell, n_cell)
        
        return z_score
   