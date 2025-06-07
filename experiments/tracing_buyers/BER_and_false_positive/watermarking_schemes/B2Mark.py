import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import hashlib


class WatermarkEmbedding:
    def __init__(self, dataset='covertype', watermark_information="10101100", 
                 g=10, seed=10000, secret_key_1="123", secret_key_2="456", 
                 columns_of_interest=['Elevation', 'Aspect'], original_file='/home/zhengyihao/BlindTabularMark-v2/revision/compare/dataset/covtype_with_key.subset.data'):
        self.watermark_information = watermark_information
        self.g = g
        self.seed = seed
        self.secret_key_1 = secret_key_1
        self.secret_key_2 = secret_key_2
        self.columns_of_interest = columns_of_interest
        self.original_file = original_file
        self.dataset = dataset
        self.origin = None
        self.covertype = None
        self.shuffled_covertype = None
        self.green_domain = None
        self.red_domain = None
        np.random.seed(self.seed)

    def load_dataset(self):
        _, file_extension = os.path.splitext(self.original_file)
        if file_extension == '.npy':
            loaded_results = np.load(self.original_file, allow_pickle=True).item()
            self.origin = loaded_results['watermarked_data']
        else:
            self.origin = pd.read_csv(self.original_file)
        
        self.origin[self.columns_of_interest] = self.origin[self.columns_of_interest].fillna(0)  # Fill NA with 0
        self.covertype = self.origin['Cover_Type'].unique()
        self.covertype.sort()
        self.shuffled_covertype = list(self.covertype)
        np.random.shuffle(self.shuffled_covertype)
        
    def hash_mod(self, key, mod_value, secret_key):
        combined = f"{secret_key}{key}"
        hash_value = int(hashlib.sha256(combined.encode()).hexdigest(), 16)
        return hash_value % mod_value

    def first_n_digits(self, x, n=2):
        if x == 0:
            return "0" * n  
        digits = str(x).replace('.', '')  
        if len(digits) < n:
            return digits + "0" * (n - len(digits)) 
        return digits[:n]

    def get_quality_domains(self):
        half_size = len(self.shuffled_covertype) // 2
        self.green_domain = self.shuffled_covertype[:half_size]
        self.red_domain = self.shuffled_covertype[half_size:]
        # print("Green Domain:", self.green_domain)
        # print("Red Domain:", self.red_domain)

    def apply_watermark(self):
        """Apply the watermarking technique to the dataset."""
        for idx in range(len(self.origin)):
            selected_data = self.origin.loc[idx, self.columns_of_interest]
            first_n_digits = selected_data.apply(self.first_n_digits)
            composite_numbers = ''.join(first_n_digits.values)
            
            if self.watermark_information[self.hash_mod(composite_numbers, len(self.watermark_information), self.secret_key_1)] == '1':
                if self.hash_mod(composite_numbers, self.g, self.secret_key_2) == 0:
                    if self.origin.loc[idx, 'Cover_Type'] in self.red_domain:
                        self.origin.loc[idx, 'Cover_Type'] = np.random.choice(self.green_domain)
                        
    def save_results(self, output_path):
        """Save the watermarked data to a file."""
        decimal = int(self.watermark_information, 2)
        results = {'watermarked_data': self.origin}
        np.save(output_path, results)

    def run(self):
        """Run the watermark embedding process."""
        self.load_dataset()  # Load the dataset
        self.get_quality_domains()  # Get the green and red domain
        self.apply_watermark()  # Apply watermarking
        self.save_results()  # Save the results



class WatermarkDetection:
    def __init__(self,
                 dataset='winequality',
                 seed=10000,
                 secret_key_1='123',
                 secret_key_2='456',
                 watermark_information='10101100',
                 threshold=2.15,
                 columns_of_interest=['Elevation', 'Aspect'],
                 g=10,
                 gamma = 1/2):
        self.dataset = dataset
        self.seed = seed
        self.secret_key_1 = secret_key_1
        self.secret_key_2 = secret_key_2
        self.watermark_information = watermark_information
        self.threshold = threshold
        self.columns_of_interest = columns_of_interest
        self.g = g
        self.gamma = gamma
        # 下列属性在检测过程中会用到
        self.detected_watermark_information = ""
        self.z_scores = None
        self.green_domain = None
        self.red_domain = None

    def hash_mod(self, key, mod_value, secret_key):
        combined = f"{secret_key}{key}"
        hash_value = int(hashlib.sha256(combined.encode()).hexdigest(), 16)
        return hash_value % mod_value

    def first_n_digits(self, x, n=2):
        if x == 0:
            return "0" * n  
        digits = str(x).replace('.', '')  
        if len(digits) < n:
            return digits + "0" * (n - len(digits)) 
        return digits[:n]

    def load_data(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        if file_extension == '.npy':
            loaded_results = np.load(file_path, allow_pickle=True).item()
            data = loaded_results['watermarked_data']
            data['Cover_Type'] = data['Cover_Type'].round() 
        else:
            data = pd.read_csv(file_path)
            data['Cover_Type'] = data['Cover_Type'].round() 

        data[self.columns_of_interest] = data[self.columns_of_interest].fillna(0)
        return data

    def generate_domains(self, watermarked_data):
        covertype = watermarked_data['Cover_Type'].unique()
        covertype.sort()  
        np.random.seed(self.seed)

        shuffled_covertype = sorted(list(covertype))
        np.random.shuffle(shuffled_covertype)
        half_size = len(shuffled_covertype) // 2

        self.green_domain = shuffled_covertype[:half_size]
        self.red_domain = shuffled_covertype[half_size:]
        self.gamma = len(self.green_domain) / (len(self.red_domain) + len(self.green_domain))
        

    def run_detection(self, file_path):
        watermarked_data = self.load_data(file_path)

        self.generate_domains(watermarked_data)

        watermark_length = len(self.watermark_information)
        green_cells = np.zeros(watermark_length, dtype=np.float64)
        n_cells = np.zeros(watermark_length, dtype=np.float64)
        self.z_scores = np.zeros(watermark_length, dtype=np.float64)

        for idx in range(len(watermarked_data)):
            selected_data = watermarked_data.loc[idx, self.columns_of_interest]
            first_n_digits_data = selected_data.apply(self.first_n_digits)
            composite_numbers = ''.join(first_n_digits_data.values)

            # 找到在水印比特串中的索引位置
            w_index = self.hash_mod(composite_numbers, watermark_length, self.secret_key_1)

            if self.hash_mod(composite_numbers, self.g, self.secret_key_2) == 0: 
                n_cells[w_index] += 1    
                if watermarked_data.loc[idx, 'Cover_Type'] in self.green_domain:
                    green_cells[w_index] += 1

        for i in range(watermark_length):
            if n_cells[i] != 0:
                self.z_scores[i] = (green_cells[i] - n_cells[i]*self.gamma) / math.sqrt(n_cells[i]*self.gamma*(1-self.gamma))
            else:
                self.z_scores[i] = 0

        self.detected_watermark_information = ""
        for score in self.z_scores:
            if score > self.threshold:
                self.detected_watermark_information += '1'
            else:
                self.detected_watermark_information += '0'

        return self.detected_watermark_information
    
    def get_z_scores(self):
        # print(self.z_scores)
        return self.z_scores

    def print_detection_result(self):
       
        if self.z_scores is None or not self.detected_watermark_information:
            print("请先调用 run_detection() 方法进行检测。")
            return
        
        print("=== Z-scores for Each Bit ===")
        for i, score in enumerate(self.z_scores):
            print(f"Bit {i}: z-score = {score}")

        print("\n=== Detected Watermark Information ===")
        print(self.detected_watermark_information)
