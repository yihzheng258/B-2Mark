import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import hashlib
from collections import Counter

class WatermarkEmbedding:
    def __init__(self, dataset='covertype', watermark_information="10101100", seed=10000, Ks="123", N_g = 4,
                 columns_of_interest=['Elevation', 'Aspect'], original_file='/home/zhengyihao/BlindTabularMark-v2/revision/compare/dataset/covtype_with_key.subset.data'):
        self.watermark_information = watermark_information
        self.seed = seed
        self.Ks = Ks
        self.N_g = len(watermark_information)
        self.columns_of_interest = columns_of_interest
        self.original_file = original_file
        self.dataset = dataset
        self.origin = None
        np.random.seed(self.seed)
        
    def load_dataset(self):
        _, file_extension = os.path.splitext(self.original_file)
        if file_extension == '.npy':
            loaded_results = np.load(self.original_file, allow_pickle=True).item()
            self.origin = loaded_results['watermarked_data']
        else:
            self.origin = pd.read_csv(self.original_file)
        
        self.origin[self.columns_of_interest] = self.origin[self.columns_of_interest].fillna(0)  # Fill NA with 0

    def first_n_digits(self, x, n=2):
        if x == 0:
            return "0" * n  
        digits = str(x).replace('.', '')  
        if len(digits) < n:
            return digits + "0" * (n - len(digits)) 
        return digits[:n]
    
    def generate_primary_key(self, row):
        first_digits = [self.first_n_digits(row[col]) for col in self.columns_of_interest]
        return ''.join(first_digits)

    def hash_function(self, composite_str):
        hash_obj = hashlib.sha256()
        inner_value = (str(self.Ks) + composite_str).encode('utf-8')
        hash_obj.update(inner_value)
        digest = hash_obj.hexdigest()
        group_number = int(digest, 16) % self.N_g
        return group_number
    
    def apply_watermark(self):
        self.origin['primary_key'] = self.origin.apply(self.generate_primary_key, axis=1)
        
        # self.origin['primary_key'] = self.origin.index.astype(str)
        self.origin['group_number'] = self.origin['primary_key'].apply(self.hash_function)
        
        self.temp = self.origin.copy()
                
        # 初始化 group_number
        group_number = 0
        _max = self.origin['Cover_Type'].max()
        _min = self.origin['Cover_Type'].min()

        # 创建 y_hat 列
        self.origin['y_hat'] = (_max + _min) / 2
        self.pa = {}
        self.mp = {}
        
        np.random.seed(self.seed)
        selected_indices = np.random.choice(len(self.origin), int(len(self.origin)* 0.2), replace=False)
        selected_index_labels = self.origin.iloc[selected_indices].index
        
        for bit in self.watermark_information:
            # 选取当前 group 的数据
            mask = self.origin['group_number'] == group_number
            group_data = self.origin[mask]
            group_data = group_data.loc[group_data.index.intersection(selected_index_labels)]

            # 计算 p_e (忽略 min 和 max)
            mask2 = (group_data['Cover_Type'] != _max) & (group_data['Cover_Type'] != _min)
            p_e = group_data.loc[mask2, 'Cover_Type'] - group_data.loc[mask2, 'y_hat']
            
            # 找出出现频数最高的 p_e 并赋值给 p
            if not p_e.empty:
                counter = Counter(np.abs(p_e))
                p = counter.most_common(1)[0][0]
            else:
                p = 0  # 或者 None，视需求而定

            
            # 把当前 group 中, quality 为 min 或 max 的 primary_key 存入 mp
            mask_min_or_max = (group_data['Cover_Type'] == _max) | (group_data['Cover_Type'] == _min)
        
            # self.mp.extend(group_data.loc[mask_min_or_max, 'primary_key'].values.tolist())
            min_max_keys = group_data.loc[mask_min_or_max, 'primary_key'].values.tolist()
            self.mp[group_number] = min_max_keys
            
            # 对于那些不等于 min 或 max 的 quality，更新原始数据集数据
            mask &= mask2

            group_quality = self.origin.loc[mask, 'Cover_Type']
            y_hat = (_max + _min) / 2
            p_e = group_quality - y_hat
            
            # 更新 p_e 的值
            p_e = np.where((p_e == p) & (bit == '0'), p_e,
                        np.where((p_e == p) & (bit == '1'), p_e + 1,
                                    np.where((p_e == -p) & (bit == '0'), p_e,
                                            np.where((p_e == -p) & (bit == '1'), p_e - 1,
                                                    np.where(p_e >= p + 1, p_e + 1,
                                                            np.where(p_e <= -(p + 1), p_e - 1, p_e))))))
            # 计算 y_prime，并更新 'quality'
            self.origin.loc[mask, 'Cover_Type'] = p_e + y_hat
            self.pa[group_number] = p

            # 更新 group_number 到下一组
            group_number += 1
            
    def save_results(self, output_path):
        results = {
            'watermarked_data': self.origin,
            # 'original_data': self.temp,
            'pa': self.pa,
            'mp': self.mp
        }
        np.save(output_path, results)
        
        
class WatermarkDetection:
    def __init__(self, dataset='covertype', watermark_information="10101100", seed=10000, Ks="123", 
                 columns_of_interest=['Elevation', 'Aspect'], original_file='/home/zhengyihao/BlindTabularMark-v2/revision/compare/dataset/covtype_with_key.subset.data'):
        self.watermark_information = watermark_information
        self.seed = seed
        self.Ks = Ks
        self.columns_of_interest = columns_of_interest
        self.original_file = original_file
        self.dataset = dataset
        self.origin = None
        self.N_g = len(watermark_information)
        np.random.seed(self.seed)
        
    def load_data(self, file_path):
        loaded_results = np.load(file_path, allow_pickle=True).item()
        data = loaded_results['watermarked_data']
        data['Cover_Type'] = data['Cover_Type'].round()
        pa = loaded_results['pa']
        mp = loaded_results['mp']
        
        data[self.columns_of_interest] = data[self.columns_of_interest].fillna(0)
        return data, pa, mp
    
    def first_n_digits(self, x, n=2):
        if x == 0:
            return "0" * n  
        digits = str(x).replace('.', '')  
        if len(digits) < n:
            return digits + "0" * (n - len(digits)) 
        return digits[:n]
    
    def generate_primary_key(self, row):
        first_digits = [self.first_n_digits(row[col]) for col in self.columns_of_interest]
        return ''.join(first_digits)

    def hash_function(self, composite_str):
        hash_obj = hashlib.sha256()
        inner_value = (str(self.Ks) + composite_str).encode('utf-8')
        hash_obj.update(inner_value)
        digest = hash_obj.hexdigest()
        group_number = int(digest, 16) % self.N_g
        return group_number
    
    def run_detection(self, file_path, detected_data):
        watermarked_data, pa, mp = self.load_data(file_path)
        watermarked_data = pd.read_csv(detected_data)
        
        watermarked_data['primary_key'] = watermarked_data.apply(self.generate_primary_key, axis=1)
        
        # self.origin['primary_key'] = self.origin.index.astype(str)
        watermarked_data['group_number'] = watermarked_data['primary_key'].apply(self.hash_function)
        
        # 计算 y_hat
        _max = watermarked_data['Cover_Type'].max()
        _min = watermarked_data['Cover_Type'].min()

        y_hat = (_max + _min) / 2
        
        watermarked_data['pe'] = watermarked_data['Cover_Type'] - y_hat
        
        # 将原有的list类型转化为集合数据类型，提高在其中查找项的速度
        W_det = ""
        
        np.random.seed(self.seed)
        selected_indices = np.random.choice(len(watermarked_data), int(len(watermarked_data)* 0.2), replace=False)
        selected_index_labels = watermarked_data.iloc[selected_indices].index
        
        for group_number, bit in enumerate(self.watermark_information):
            mp_set = set(mp[group_number])
            # 对当前组进行操作
            group_data = watermarked_data[watermarked_data['group_number'] == group_number]
            group_data = group_data.loc[group_data.index.intersection(selected_index_labels)]
            
            p = pa[group_number]

            a = 0 # count bit = 0
            b = 0 # count bit = 1

            # 通过将一组条件（每行是否满足要求）应用于数据框并进行求和，避免了逐行运算
            mask = ~group_data['primary_key'].isin(mp_set) & ((group_data['pe'] == p+1) | (group_data['pe'] == -p-1))
            b = mask.sum()
            mask = ~group_data['primary_key'].isin(mp_set) & ((group_data['pe'] == p) | (group_data['pe'] == -p))
            a = mask.sum()

            W_det += '0' if a > b else '1'
 
        return W_det

