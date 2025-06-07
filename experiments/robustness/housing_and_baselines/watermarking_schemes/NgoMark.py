import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import hashlib
import math

class NgoMarkWatermarkEmbedding:
    def __init__(self, dataset="housing", seed=10000, b=10):
        self.dataset = dataset
        self.seed = seed
        self.b = b
        self.green_domain_values = None
        self.green_mid_values = None
        np.random.seed(self.seed)

    def load_data(self, file_path):
        self.origin = pd.read_csv(file_path)
        self.medv_max = self.origin['MEDV'].max()
        self.medv_min = self.origin['MEDV'].min()

    def generate_segments(self):
        intervals = np.linspace(self.medv_min, self.medv_max, self.b + 1)
        segments = [(intervals[i], intervals[i + 1]) for i in range(self.b)]
        np.random.shuffle(segments)

        half_b = self.b // 2
        green_domains = segments[:half_b]
        self.green_domain_values = [(low, np.nextafter(high, low)) for low, high in green_domains]
        self.green_mid_values = [(seg[0] + seg[1]) / 2 for seg in self.green_domain_values]

    def process_data(self):
        for idx in range(len(self.origin)):
            original_medv = self.origin.loc[idx, 'MEDV']
            closest_mid = min(self.green_mid_values, key=lambda x: abs(x - original_medv))
            closest_idx = self.green_mid_values.index(closest_mid)

            if self.origin.loc[idx, 'MEDV'] >= self.green_domain_values[closest_idx][0] and \
               self.origin.loc[idx, 'MEDV'] < self.green_domain_values[closest_idx][1]:
                closest_value = original_medv
            else:
                closest_value = np.random.uniform(self.green_domain_values[closest_idx][0],
                                                  self.green_domain_values[closest_idx][1])

            self.origin.loc[idx, 'MEDV'] = closest_value

    def save_results(self, output_path):
        results = {'watermarked_data': self.origin}
        np.save(output_path, results)


class NgoMarkWatermarkDetection:
    def __init__(self, dataset="housing", seed=10000, b=10):
        self.dataset = dataset
        self.seed = seed
        self.b = b
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

    def generate_segments(self, medv_min, medv_max):
        intervals = np.linspace(medv_min, medv_max, self.b + 1)
        segments = [(intervals[i], intervals[i + 1]) for i in range(self.b)]
        np.random.shuffle(segments)
        
        half_b = self.b // 2
        self.green_domains = segments[:half_b]
        self.red_domains = segments[half_b:]

    def detect_watermark(self, watermarked_data):
        green_cell = 0
        n_cell = 0

        for idx in range(len(watermarked_data)):
            n_cell += 1

            # Check if MEDV is in any of the green domains
            for low, high in self.green_domains:
                if low <= watermarked_data.loc[idx, 'MEDV'] < high:
                    green_cell += 1
                    break
        return green_cell, n_cell

    def compute_z_score(self, green_cell, n_cell):
        return (green_cell - n_cell / 2) / math.sqrt(n_cell / 4)

    def run_detection(self, file_path, medv_min, medv_max):
        watermarked_data = self.load_data(file_path)
        
        np.random.seed(self.seed)
        self.generate_segments(medv_min, medv_max)

        green_cell, n_cell = self.detect_watermark(watermarked_data)
        z_score = self.compute_z_score(green_cell, n_cell)
        
        return z_score
