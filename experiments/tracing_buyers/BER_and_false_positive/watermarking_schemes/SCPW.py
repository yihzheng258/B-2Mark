import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random

class WatermarkEmbedding:
    def __init__(self, dataset='covertype', watermark_information="10101100", seed=10000, Ks="123", N_g=2,
                 k=1, sigma_r=2, columns_of_interest=['Elevation', 'Aspect'],
                 original_file='/path/to/your.csv'):
        self.watermark_information = watermark_information
        self.seed = seed
        self.Ks = Ks
        self.N_g = len(watermark_information) // k
        self.columns_of_interest = columns_of_interest
        self.original_file = original_file
        self.dataset = dataset
        self.origin = None
        self.k = k
        self.sigma_r = sigma_r
        np.random.seed(self.seed)

    def load_dataset(self):
        _, ext = os.path.splitext(self.original_file)
        if ext == '.npy':
            data = np.load(self.original_file, allow_pickle=True).item()
            self.origin = data['watermarked_data']
        else:
            self.origin = pd.read_csv(self.original_file)
        self.origin[self.columns_of_interest] = self.origin[self.columns_of_interest].fillna(0)

    def first_n_digits(self, x, n=2):
        if x == 0:
            return "0" * n
        digits = str(x).replace('.', '')
        return (digits + "0" * n)[:n]

    def generate_primary_key(self, row):
        return ''.join([self.first_n_digits(row[col]) for col in self.columns_of_interest])

    def hash_sha256(self, s):
        return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)

    def get_potential_watermark(self, xi, Pki, k):
        binary = self.float_to_binary(xi, 1)
        # print("binary", binary)
        Lx = len(binary)
        bits = ''
        for j in range(k):
            pkj = str(Pki) + '/' + str(j + 1)
            sj = self.hash_sha256(self.Ks + pkj) % Lx
            bits += binary[sj]
        return bits

    def float_to_binary(self, x, precision=16):
        int_part = bin(int(x))[2:]
        frac_part = x - int(x)
        frac_bin = ''
        while len(frac_bin) < precision:
            frac_part *= 2
            bit = int(frac_part)
            frac_bin += str(bit)
            frac_part -= bit
        return int_part + frac_bin

    def group_tuples(self, p=1):
        self.origin['primary_key'] = self.origin.apply(self.generate_primary_key, axis=1)
        # self.origin['primary_key'] = self.origin.index.astype(str)
        self.origin['group_number'] = -1
        self.temp = self.origin.copy()
        for i in range(len(self.origin)):
            pk = self.origin.at[i, 'primary_key']
            if self.hash_sha256(self.Ks + self.hash_sha256(self.Ks + pk).__str__()) % p == 0:
                g = self.hash_sha256(self.Ks + self.hash_sha256(self.Ks + pk).__str__()) % self.N_g
                self.origin.at[i, 'group_number'] = g

    def apply_watermark(self):
        self.group_tuples()
        LW = len(self.watermark_information)
        NG = self.N_g
        grouped = self.origin[self.origin['group_number'] >= 0].groupby('group_number')

        # 预计算一次统计参数
        stats = self.get_ri_params()
        
        max_iter = 10000
        
        self.valid_set = {}
        for g, group in grouped:
            print(f"Processing group {g + 1}/{NG}...")
            Wg = self.watermark_information[g * self.k:(g + 1) * self.k]
            # print(f"Watermark for group {g}: {Wg}") 
            self.valid_set[g] = []
            for idx, row in group.iterrows():
                xi = row["Cover_Type"] 
                yi = row[self.columns_of_interest[0]] 
                Pki = row['primary_key']
                print("Pki", Pki)
                w_hat = self.get_potential_watermark(xi, Pki, self.k)
                # print(f"Watermark for row {idx}: {w_hat}")
                print("xi: ", xi)
                if w_hat != Wg:
                    # rejection sampling until w_hat == Wg
                    while True:
                        alpha = np.random.normal(0, 1) 
                        ri = self.construct_ri(xi, yi, alpha, stats)
                        x_new = round(xi + ri, 0)
                        print("x_new", x_new)
                        Wg_new = self.get_potential_watermark(x_new, Pki, self.k)
                        print("Wg", Wg)
                        print("Wg_new", Wg_new)
                        # if Wg_new == Wg or max_iter <= 0:
                        if Wg_new == Wg and x_new in self.origin["Cover_Type"].unique():
                            self.origin.at[idx, "Cover_Type"] = x_new  
                            self.valid_set[g].append(Pki)
                            break
                        if max_iter <= 0:
                            self.origin.at[idx, "Cover_Type"] = xi
                            break
                        max_iter -= 1
                        
                    # print(f"new x for row {idx} after embedding: {x_new}")

    def get_ri_params(self):
        X = self.origin[self.columns_of_interest[0]].values
        Y = self.origin[self.columns_of_interest[1]].values

        mu_X = np.mean(X)
        mu_Y = np.mean(Y)
        sigma_X = np.std(X)
        sigma_Y = np.std(Y)
        sigma_XY = np.cov(X, Y)[0, 1]
        sigma_XY = np.corrcoef(X, Y)[0, 1]

        sigma_R = self.sigma_r

        denominator = 2 * (1 - sigma_XY**2)
        kx = -sigma_R**2 / (denominator * sigma_X**2)
        ky = sigma_R**2 * sigma_XY / (denominator * sigma_X * sigma_Y)
        k1_squared = sigma_R**2 - (kx**2 * sigma_X**2) - (ky**2 * sigma_Y**2) - 2 * kx * ky * sigma_XY
        k1 = np.sqrt(max(k1_squared, 0))  # 防止数值误差导致负数开根号
        k2 = -kx * mu_X - ky * mu_Y

        return {'kx': kx, 'ky': ky, 'k1': k1, 'k2': k2}

    def construct_ri(self, xi, yi, alpha, stats):
        ri = stats['kx'] * xi + stats['ky'] * yi + stats['k1'] * alpha + stats['k2']
        return ri


    def save_results(self, path):
        np.save(path, {
            'watermarked_data': self.origin,
            'original_data': self.temp,
            'valid_set': self.valid_set
        })

class WatermarkDetection:
    def __init__(self, dataset='covertype', watermark_information="10101100", seed=10000, Ks="123",
                 k=1, columns_of_interest=['Elevation', 'Aspect'],
                 original_file='/path/to/your.csv'):
        self.watermark_information = watermark_information
        self.seed = seed
        self.Ks = Ks
        self.k = k
        self.columns_of_interest = columns_of_interest
        self.original_file = original_file
        self.dataset = dataset
        self.origin = None
        np.random.seed(self.seed)

    def load_data(self, file_path):
        loaded_results = np.load(file_path, allow_pickle=True).item()
        # data = loaded_results['watermarked_data']
        data = loaded_results['original_data']
        data[self.columns_of_interest] = data[self.columns_of_interest].fillna(0)
        self.valid_set = loaded_results['valid_set']
        # print(len(self.valid_set[0]))
        # data['quality'] = data['quality'].round() 
        
        return data

    def first_n_digits(self, x, n=2):
        if x == 0:
            return "0" * n
        digits = str(x).replace('.', '')
        return (digits + "0" * n)[:n]

    def generate_primary_key(self, row):
        return ''.join([self.first_n_digits(row[col]) for col in self.columns_of_interest])

    def hash_sha256(self, s):
        return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)

    def get_potential_watermark(self, xi, Pki, k):
        binary = self.float_to_binary(xi, 1)
        Lx = len(binary)
        bits = ''
        for j in range(k):
            pkj = str(Pki) + '/' + str(j + 1)
            sj = self.hash_sha256(self.Ks + pkj) % Lx
            bits += binary[sj]
        return bits

    def float_to_binary(self, x, precision=16):
        int_part = bin(int(x))[2:]
        frac_part = x - int(x)
        frac_bin = ''
        while len(frac_bin) < precision:
            frac_part *= 2
            bit = int(frac_part)
            frac_bin += str(bit)
            frac_part -= bit
        return int_part + frac_bin

    def group_tuples(self, df, p, NG):
        df['primary_key'] = df.apply(self.generate_primary_key, axis=1)
        # df['primary_key'] = df.index.astype(str)
        df['group_number'] = -1
        for i in range(len(df)):
            pk = df.at[i, 'primary_key']
            if self.hash_sha256(self.Ks + str(self.hash_sha256(self.Ks + pk))) % p == 0:
                g = self.hash_sha256(self.Ks + str(self.hash_sha256(self.Ks + pk))) % NG
                df.at[i, 'group_number'] = g
        return df

    def get_group_length(self, LW, k, g, NG):
        return LW - k * (NG - 1) if g == NG - 1 else k

    def vote(self, V):
        Wg = ''
        for V0, V1 in V:
            if V0 == V1:
                Wg += str(np.random.randint(0, 2))
            elif V0 < V1:
                Wg += '1'
            else:
                Wg += '0'
        return Wg

    def run_detection(self, file_path, detected_data):
        df = self.load_data(file_path)
        df = pd.read_csv(detected_data)
        LW = len(self.watermark_information)
        NG = int(np.ceil(LW / self.k))
        df = self.group_tuples(df, p=1, NG=NG)

        final_watermark = ""

        for g in range(NG):
            l = self.get_group_length(LW, self.k, g, NG)
            V = [[0, 0] for _ in range(l)]

            group_data = df[df['group_number'] == g]
            for _, row in group_data.iterrows():
                if row['primary_key'] in self.valid_set[g]:
                    xi = row["Cover_Type"]
                    Pki = row["primary_key"]
                    # print("Pki", Pki)
                    wx_hat = self.get_potential_watermark(xi, Pki, l)
                    # print("wx_hat", wx_hat)
                    for j in range(l):
                        if wx_hat[j] == '0':
                            V[j][0] += 1
                        else:
                            V[j][1] += 1

            Wg = self.vote(V)
            # print(V)
            final_watermark += Wg
        
        return final_watermark
