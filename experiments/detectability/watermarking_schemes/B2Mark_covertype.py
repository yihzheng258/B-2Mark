import os
import numpy as np
import pandas as pd
import hashlib

class WatermarkEmbedding:
    def __init__(self, dataset, seed=10000, g=10, secret_key="123", columns_of_interest=['Elevation', 'Aspect']):
        self.dataset = dataset
        self.seed = seed
        self.g = g
        self.secret_key = secret_key
        self.columns_of_interest = columns_of_interest
        self.origin = None
        
        np.random.seed(self.seed)

    def load_data(self, file_path):
        self.origin = pd.read_csv(file_path)
        self.origin[self.columns_of_interest] = self.origin[self.columns_of_interest].fillna(0)
        self.cover_types = self.origin['Cover_Type'].unique()
        self.cover_types.sort()
        self.shuffled_cover_types = list(self.cover_types)
        np.random.shuffle(self.shuffled_cover_types)

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

    def process_data(self):
        half_size = len(self.shuffled_cover_types) // 2
        green_domain = self.shuffled_cover_types[:half_size]
        red_domain = self.shuffled_cover_types[half_size:]

        for idx in range(len(self.origin)):
            selected_data = self.origin.loc[idx, self.columns_of_interest]
            first_n_digits_data = selected_data.apply(self.first_n_digits)
            composite_numbers = ''.join(first_n_digits_data.values)

            if self.hash_mod(composite_numbers, self.g) != 0:
                continue
            
            if self.origin.loc[idx, 'Cover_Type'] in red_domain:
                perturb_value = np.random.choice(green_domain)
                self.origin.loc[idx, 'Cover_Type'] = perturb_value

    def save_results(self, output_path):
        results = {'watermarked_data': self.origin}
        np.save(output_path, results)




class WatermarkDetection:
    def __init__(self, dataset, secret_key, seed_range=range(10000,10050), g=10, columns_of_interest=['Elevation', 'Aspect']):
        self.dataset = dataset
        self.secret_key = secret_key
        self.seed_range = seed_range
        self.g = g
        self.columns_of_interest = columns_of_interest
        self.green_domains = None
        self.red_domains = None

    def load_data(self, file_path):
        _, file_extension = os.path.splitext(file_path)
       
        if(file_extension == '.npy'):
            loaded_results = np.load(file_path, allow_pickle=True).item()
            data = loaded_results['watermarked_data']
        else:
            data = pd.read_csv(file_path)
            
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

    def generate_segments(self, cover_types):
        shuffled_cover_types = list(cover_types)
        np.random.shuffle(shuffled_cover_types)
        half_size = len(shuffled_cover_types) // 2
        self.green_domains = shuffled_cover_types[:half_size]
        self.red_domains = shuffled_cover_types[half_size:]

    def detect_watermark(self, watermarked_data):
        green_cell = 0
        n_cell = 0
       
        # Iterate over the dataset to detect watermark
        for idx in range(len(watermarked_data)):
            selected_data = watermarked_data.loc[idx, self.columns_of_interest]
            first_n_digits_data = selected_data.apply(self.first_n_digits)
            composite_numbers = ''.join(first_n_digits_data.values)

            if self.hash_mod(composite_numbers, self.g) != 0:
                continue
            
            n_cell += 1
                        
            if watermarked_data.loc[idx, 'Cover_Type'] in self.green_domains:
                green_cell += 1

        return green_cell, n_cell
    
    def compute_z_score(self, green_cell, n_cell):
        return (green_cell - n_cell / 2) / math.sqrt(n_cell / 4)   

    def run_detection(self):
        z_scores = []

        for seed in self.seed_range:
            file_path = f"../../datasets/watermark/{self.dataset}/{self.dataset}-{seed}.npy"
            # file_path = '../../datasets/covtype_with_key.subset.data'
            watermarked_data = self.load_data(file_path)
            
            cover_types = watermarked_data['Cover_Type'].unique()
            cover_types.sort()  
            np.random.seed(seed)
            self.generate_segments(cover_types)
            
            green_cell, n_cell = self.detect_watermark(watermarked_data)
            z_score = self.compute_z_score(green_cell, n_cell)
            
            # print(z_score)
            z_scores.append(z_score)
            
        print("The average z-score is", np.mean(z_scores))
