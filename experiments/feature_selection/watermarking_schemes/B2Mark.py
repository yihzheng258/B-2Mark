import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import hashlib

class B2MarkWatermarkEmbedding:
    def __init__(self, dataset="wine", seed=10000, k=10, g=3, gamma=1/2, secret_key="123", columns_of_interest = ['RM', 'AGE'], watermarked_colomn='MEDV'):
        self.dataset = dataset
        self.seed = seed
        self.k = k
        self.g = g
        self.secret_key = secret_key
        self.columns_of_interest = columns_of_interest
        self.origin = None
        self.watermarked_colomn_max = None
        self.watermarked_colomn_min = None
        self.green_domain_values = None
        self.green_mid_values = None
        self.watermarked_colomn = watermarked_colomn
        

    def load_data(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        if file_extension == '.npy':
            loaded_results = np.load(file_path, allow_pickle=True).item()
            self.origin = loaded_results['watermarked_data']
        else:
            self.origin = pd.read_csv(file_path, sep=';', quotechar='"')
        self.origin[self.columns_of_interest] = self.origin[self.columns_of_interest].fillna(0)
        self.watermarked_colomn_max = self.origin[self.watermarked_colomn].max()
        self.watermarked_colomn_min = self.origin[self.watermarked_colomn].min()

    def hash_mod(self, key, mod_value):
        combined = f"{self.secret_key}{key}"
        hash_value = int(hashlib.sha256(combined.encode()).hexdigest(), 16)
        return hash_value % mod_value

    def first_n_digits(self, x, n=2):
        if x == 0:
            return "0" * n  
        digits = str(x).replace('.', '')  
        if len(digits) < n:
            return digits + "0" * (n - len(digits)) 
        return digits[:n]

    def generate_segments(self):
        intervals = np.linspace(self.watermarked_colomn_min, self.watermarked_colomn_max, self.k + 1)
        segments = [(intervals[i], intervals[i + 1]) for i in range(self.k)]
        np.random.seed(self.seed)
        np.random.shuffle(segments)
        half_k = self.k // 2
        green_domains = segments[:half_k]
        self.green_domain_values = [(low, np.nextafter(high, low)) for low, high in green_domains]
        self.green_mid_values = [(seg[0] + seg[1]) / 2 for seg in self.green_domain_values]

    def process_data(self):
        for idx in range(len(self.origin)):
            selected_data = self.origin.loc[idx, self.columns_of_interest]
            first_n_digits_data = selected_data.apply(self.first_n_digits)
            composite_numbers = ''.join(first_n_digits_data.values)

            if self.hash_mod(composite_numbers, self.g) != 0:
                continue

            original_value = self.origin.loc[idx, self.watermarked_colomn]
            closest_mid = min(self.green_mid_values, key=lambda x: abs(x - original_value))
            closest_idx = self.green_mid_values.index(closest_mid)

            if self.origin.loc[idx, self.watermarked_colomn] >= self.green_domain_values[closest_idx][0] and \
               self.origin.loc[idx, self.watermarked_colomn] <= self.green_domain_values[closest_idx][1]:
                closest_value = original_value
            else:
                closest_value = np.random.uniform(self.green_domain_values[closest_idx][0],
                                                  self.green_domain_values[closest_idx][1])

            self.origin.loc[idx, self.watermarked_colomn] = closest_value

    def save_results(self, output_path):
        results = {'watermarked_data': self.origin}
        np.save(output_path, results)




import os
import numpy as np
import pandas as pd
import hashlib
import math

class B2MarkWatermarkDetection:
    def __init__(self, dataset, secret_key, k=10, g=3, seed=10000, columns_of_interest=['RM', 'AGE'], watermarked_colomn='MEDV'):
        self.dataset = dataset
        self.secret_key = secret_key
        self.k = k
        self.g = g
        self.seed = seed
        self.columns_of_interest = columns_of_interest
        self.watermarked_colomn = watermarked_colomn
        self.green_domains = None
        self.red_domains = None

    def load_data(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        if(file_extension == '.csv'):
            data = pd.read_csv(file_path)
        elif(file_extension == '.npy'):
            loaded_results = np.load(file_path, allow_pickle=True).item()
            data = loaded_results['watermarked_data']
            
        data[self.columns_of_interest] = data[self.columns_of_interest].fillna(0)
        return data

    def hash_mod(self, key, mod_value):
        combined = f"{self.secret_key}{key}"
        hash_value = int(hashlib.sha256(combined.encode()).hexdigest(), 16)
        return hash_value % mod_value

    def first_n_digits(self, x, n=2):
        if x == 0:
            return "0" * n  
        digits = str(x).replace('.', '')  
        if len(digits) < n:
            return digits + "0" * (n - len(digits)) 
        return digits[:n]

    def generate_segments(self, watermarked_colomn_min, watermarked_colomn_max):
        intervals = np.linspace(watermarked_colomn_min, watermarked_colomn_max, self.k + 1)
        segments = [(intervals[i], intervals[i + 1]) for i in range(self.k)]
        np.random.seed(self.seed)
        np.random.shuffle(segments)
        
        half_k = self.k // 2
        self.green_domains = segments[:half_k]
        self.red_domains = segments[half_k:]

    def detect_watermark(self, watermarked_data):
        green_cell = 0
        n_cell = 0

        for idx in range(len(watermarked_data)):
            selected_data = watermarked_data.loc[idx, self.columns_of_interest]
            first_n_digits_data = selected_data.apply(self.first_n_digits)
            composite_numbers = ''.join(first_n_digits_data.values)

            if self.hash_mod(composite_numbers, self.g) != 0:
                continue
            n_cell += 1

            # Check if MEDV is in any of the green domains
            for low, high in self.green_domains:
                if low <= watermarked_data.loc[idx, self.watermarked_colomn] < high:
                    green_cell += 1
                    break

        return green_cell, n_cell

    def compute_z_score(self, green_cell, n_cell):
        return (green_cell - n_cell / 2) / math.sqrt(n_cell / 4)

    def run_detection(self, file_path, watermarked_colomn_min, watermarked_colomn_max):
        watermarked_data = self.load_data(file_path)
        
        
        self.generate_segments(watermarked_colomn_min, watermarked_colomn_max)

        green_cell, n_cell = self.detect_watermark(watermarked_data)
        z_score = self.compute_z_score(green_cell, n_cell)
        
        return z_score
