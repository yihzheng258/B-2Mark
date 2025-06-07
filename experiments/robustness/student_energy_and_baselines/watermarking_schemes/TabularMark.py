import os
import numpy as np
import pandas as pd
import argparse
import math

import os
import numpy as np
import pandas as pd
import argparse

class TabualrMarkWatermarkEmbedding:
    def __init__(self, watermarked_column, original_file, n=int(500/3), gamma=1/2, seed=10000, dataset='housing', p = 25, k = 10):
        self.original_file = original_file
        self.n = n
        self.p = p
        self.k = k
        self.gamma = gamma
        self.seed = seed
        self.dataset = dataset
        self.watermarked_column = watermarked_column
        
        self.origin = pd.read_csv(self.original_file)
        
        if len(self.origin) < self.n:
            raise ValueError("数据中的记录数小于所请求的n个记录")
        np.random.seed(self.seed)
        self.divide_seeds = np.random.randint(0, 2**32 - 1, size=self.n)
        self.indices = np.random.choice(len(self.origin), size=self.n, replace=False)
    
    
    def apply_watermark(self):
        
        # 验证索引列表和种子列表的长度是否一致
        if len(self.indices) != len(self.divide_seeds):
            raise ValueError("索引文件和种子文件的长度不一致")

        for idx, divide_seed in zip(self.indices, self.divide_seeds):
            np.random.seed(divide_seed)
            
            # 生成等分点
            intervals = np.linspace(-self.p, self.p, self.k + 1)
            # 将 [-p, p] 等分为 k 份
            segments = [(intervals[i], intervals[i + 1]) for i in range(self.k)]
            np.random.shuffle(segments)

            half_k = self.k // 2
            green_domains = segments[:half_k]
            red_domains = segments[half_k:]

            # 选择绿区中的一个值作为扰动值
            green_domain_values = [np.random.uniform(low, np.nextafter(high, low)) for low, high in green_domains]
            perturb_value = np.random.choice(green_domain_values)

            self.origin.loc[idx, self.watermarked_column] += perturb_value
        
    def save_results(self, save_path):
        """
        保存水印数据和相关信息到指定路径
        """
        results = {
            'watermarked_data': self.origin,
            'divide_seeds': self.divide_seeds,
            'indices': self.indices
        }
        np.save(save_path, results)
    

class TabularMarkWatermarkDetection:
    def __init__(self, origin_file, watermarked_column, results_file,watermarked_data=None, n=int(500/3), gamma=1/2, seed=10000, p = 25, k = 10, dataset='housing', primary_key_cols=None):
        self.origin_data = self.load_data(origin_file)
        self.results = np.load(results_file, allow_pickle=True).item()
        self.primary_key_cols = primary_key_cols or ['CRIM','ZN']
        self.n = n
        self.p = p
        self.k = k
        self.gamma = gamma
        self.seed = seed
        self.dataset = dataset
        self.watermarked_column = watermarked_column
        
        # 获取水印的结果
        self.divide_seeds = self.results['divide_seeds']
        self.indices = self.results['indices']
        
        if watermarked_data is None:
            self.watermarked_data = self.results['watermarked_data']
        else:
            self.watermarked_data = self.load_data(watermarked_data)
        
        

        
    def load_data(self, file_path):
        _, file_extension = os.path.splitext(file_path)
       
        if(file_extension == '.npy'):
            loaded_results = np.load(file_path, allow_pickle=True).item()
            data = loaded_results['watermarked_data']
        else:
            data = pd.read_csv(file_path)
        
        return data
    
    def binary_search(self, arr, key):
       
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == key:
                return mid
            elif arr[mid] < key:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    def match_tuples(self, origin_data, watermarked_data, indices):
  
        match_indices = []
        watermarked_keys = [tuple(row) for row in watermarked_data[self.primary_key_cols].values]
        for idx in indices:
            key_do = tuple(origin_data.loc[idx, self.primary_key_cols])
            match_idx = self.binary_search(watermarked_keys, key_do)
            if match_idx != -1:
                match_indices.append(watermarked_data.index[match_idx])
            else:
                match_indices.append(-1)
        return match_indices

    def detect_watermark(self):

        green_cell = 0
        self.watermarked_data = self.watermarked_data.sort_values(by=self.primary_key_cols).reset_index(drop=True)
        
        # 获取原始数据和水印数据中的匹配索引
        match_indices = self.match_tuples(self.origin_data, self.watermarked_data, self.indices)
        
        n_cell = 0
        
        for idx, match_idx, divide_seed in zip(self.indices, match_indices, self.divide_seeds):
            if match_idx == -1:
                continue
            n_cell += 1
            np.random.seed(divide_seed)
            intervals = np.linspace(-self.p, self.p, self.k + 1)
            # 将 [-p, p] 等分为 k 份
            segments = [(intervals[i], intervals[i + 1]) for i in range(self.k)]
            np.random.shuffle(segments)
            # 将 segments 分为 green domains 和 red domains
            half_k = self.k // 2
            green_domains = segments[:half_k]
            red_domains = segments[half_k:]

            difference = self.watermarked_data.loc[match_idx, self.watermarked_column] - self.origin_data.loc[idx, self.watermarked_column]
            for low, high in green_domains:
                if low <= difference < high:
                    green_cell += 1
                    break
        z_score = (green_cell - n_cell / 2) / math.sqrt(n_cell / 4)
        return z_score

