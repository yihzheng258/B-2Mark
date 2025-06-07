import numpy as np
import pandas as pd
import hashlib
import os

class B2MarkWatermarkEmbedding:
    def __init__(self, dataset="iris", seed=10000, g=6, gamma=1/2,
                 secret_key="123", columns_of_interest=None, label_column='class'):
        self.dataset = dataset
        self.seed = seed
        self.g = g
        self.gamma = gamma
        self.secret_key = secret_key
        self.columns_of_interest = columns_of_interest or ['sepal_length', 'sepal_width']
        self.label_column = label_column
        self.origin = None
        self.classes = None
        self.shuffled_classes = None
        np.random.seed(self.seed)

    def load_data(self, file_path):
        self.origin = pd.read_csv(file_path)
        self.origin[self.columns_of_interest] = self.origin[self.columns_of_interest].fillna(0)
        self.classes = self.origin[self.label_column].unique()
        self.classes.sort()
        self.shuffled_classes = list(self.classes)
        np.random.shuffle(self.shuffled_classes)

    def hash_mod(self, key, mod_value):
        combined = f"{self.secret_key}{key}"
        hash_value = int(hashlib.sha256(combined.encode()).hexdigest(), 16)
        return hash_value % mod_value

    def first_two_digits(self, x):
        if x == 0:
            return "00"
        digits = str(x).lstrip('0.').replace('.', '')
        return (digits + "0")[:2]

    def process_data(self):
        for idx in range(len(self.origin)):
            selected_data = self.origin.loc[idx, self.columns_of_interest]
            first_two_digits_data = selected_data.apply(self.first_two_digits)
            composite_numbers = ''.join(first_two_digits_data.values)

            if self.hash_mod(composite_numbers, self.g) != 0:
                continue

            half_size = len(self.shuffled_classes) // 2
            green_domain = self.shuffled_classes[:half_size]
            red_domain = self.shuffled_classes[half_size:]

            current_class = self.origin.loc[idx, self.label_column]
            if current_class in red_domain:
                new_class = np.random.choice(green_domain)
                self.origin.loc[idx, self.label_column] = new_class

    def save_results(self, output_path=None):
        if output_path is None:
            output_path = f"{self.dataset}-{self.g}.csv"
        results = self.origin
        results.to_csv(output_path, index=False)
