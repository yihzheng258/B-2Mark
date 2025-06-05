import os
import numpy as np
import math
import pandas as pd
import hashlib

class WatermarkEmbedding:
    def __init__(self, dataset, seed=10000, k=10, g=3, gamma=1/2, secret_key="123", columns_of_interest = ['RM', 'AGE']):
        self.dataset = dataset
        self.seed = seed
        self.k = k
        self.g = g
        self.secret_key = secret_key
        self.columns_of_interest = columns_of_interest
        self.origin = None
        self.medv_max = None
        self.medv_min = None
        self.green_domain_values = None
        self.green_mid_values = None
        np.random.seed(self.seed)

    def load_data(self, file_path):
        self.origin = pd.read_csv(file_path)
        self.origin[self.columns_of_interest] = self.origin[self.columns_of_interest].fillna(0)
        self.medv_max = self.origin['MEDV'].max()
        self.medv_min = self.origin['MEDV'].min()

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
        intervals = np.linspace(self.medv_min, self.medv_max, self.k + 1)
        segments = [(intervals[i], intervals[i + 1]) for i in range(self.k)]
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

            original_medv = self.origin.loc[idx, 'MEDV']
            closest_mid = min(self.green_mid_values, key=lambda x: abs(x - original_medv))
            closest_idx = self.green_mid_values.index(closest_mid)

            if self.origin.loc[idx, 'MEDV'] >= self.green_domain_values[closest_idx][0] and \
               self.origin.loc[idx, 'MEDV'] <= self.green_domain_values[closest_idx][1]:
                closest_value = original_medv
            else:
                closest_value = np.random.uniform(self.green_domain_values[closest_idx][0],
                                                  self.green_domain_values[closest_idx][1])

            self.origin.loc[idx, 'MEDV'] = closest_value

    def save_results(self, output_path):
        results = {'watermarked_data': self.origin}
        np.save(output_path, results)


class WatermarkDetection:
    def __init__(self, dataset, secret_key, k=10, g=3, seed_range=range(10000,10100), columns_of_interest=['RM', 'AGE']):
        self.dataset = dataset
        self.secret_key = secret_key
        self.k = k
        self.g = g
        self.seed_range = seed_range
        self.columns_of_interest = columns_of_interest
        self.medv_max = None
        self.medv_min = None
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

    def generate_segments(self, medv_min, medv_max):
        intervals = np.linspace(medv_min, medv_max, self.k + 1)
        segments = [(intervals[i], intervals[i + 1]) for i in range(self.k)]
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
                if low <= watermarked_data.loc[idx, 'MEDV'] < high:
                    green_cell += 1
                    break

        return green_cell, n_cell

    def compute_z_score(self, green_cell, n_cell):
        return (green_cell - n_cell / 2) / math.sqrt(n_cell / 4)

    def run_detection(self):
        z_scores = []

        for seed in self.seed_range:
            # file_path = f"../../datasets/watermark/{self.dataset}/{self.dataset}-{seed}.npy"
            file_path = "/home/zhengyihao/BlindTabularMark-v2/datasets/HousingData.csv"
            watermarked_data = self.load_data(file_path)

            self.medv_max = watermarked_data['MEDV'].max()
            self.medv_min = watermarked_data['MEDV'].min()
            
            np.random.seed(seed)
            self.generate_segments(self.medv_min, self.medv_max)

            green_cell, n_cell = self.detect_watermark(watermarked_data)
            z_score = self.compute_z_score(green_cell, n_cell)
            
            print(z_score)
            z_scores.append(z_score)

        print("The average z-score is", np.mean(z_scores))
