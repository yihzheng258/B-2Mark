import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import hashlib
import math

class HeMarkWatermarkEmbedding:
    def __init__(self, dataset="housing", seed=10000, m=10, gamma=1/2):
        self.dataset = dataset
        self.seed = seed
        self.m = m
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
            original_medv = self.origin.loc[idx, 'MEDV']
            
            int_part = int(np.floor(abs(original_medv)))  # 取绝对值后取整
            sign = np.sign(original_medv)  # 保留符号 (+1 or -1)

            # 随机从 green domain 中选一个小数部分
            chosen_idx = np.random.randint(len(self.green_domain_values))
            low, high = self.green_domain_values[chosen_idx]
            new_frac = np.random.uniform(low, high)

            # 构造新的值（保留正负号）
            new_value = sign * (int_part + new_frac)
            self.origin.loc[idx, 'MEDV'] = new_value


    def save_results(self, output_path):
        results = {'watermarked_data': self.origin}
        np.save(output_path, results)




class HeMarkWatermarkDetection:
    def __init__(self, dataset="housing", seed=10000, m=10, gamma=1/2):
        self.dataset = dataset
        self.seed = seed
        self.m = m
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
        # print("segments", segments)
        np.random.shuffle(segments)
        
        half_m = self.m // 2
        self.green_domains = segments[:half_m]
        self.red_domains = segments[half_m:]

    def detect_watermark(self, watermarked_data):
        green_cell = 0
        n_cell = 0

        for idx in range(len(watermarked_data)):
            n_cell += 1
            
            int_part = int(np.floor(watermarked_data.loc[idx, 'MEDV']))
            dec_part = watermarked_data.loc[idx, 'MEDV'] - int_part
            
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
   