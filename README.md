# B^2Mark
This repository implements experiments for the paper "B^2Mark: A Blind and Buyer-Traceable Watermarking Scheme for Tabular Datasets".

### Prerequisites
- Python, NumPy, Scikit-learn, Pandas, Matplotlib, Hashlib, Torch, Be-great

### Datasets
- Forest Cover Type
- Boston House Prices
- Energy Efficiency
- Student Performance
- Iris
- Wine
- Generated Synthetic Datasets

The generation method for the synthetic dataset is mentioned in Section 5.1 of the original paper.

## Experiment
```
.  
└── experiment   # Experiments for Sections 5 and 6 
    ├── dataeval       # Algorithms for B^2Mark, baselines, and other required utils
    ├── detectability       # Watermark detectability experiments for Section 5.1     
    ├── non_intrusiveness   # Watermark non_intrusiveness experiments for Section 5.2 
    ├── robustness          # Watermark robustness experiments for Section 5.3 
    ├── tracing_buyers      # Tracing buyers experiments for Section 5.4
    ├── model_robustness    # Model robustness experiments for Section 6.2
    └── feature_selection   # Feature selection experiments for Section 6.3
```

## Usage
All the experimental code is located in the /experiments directory. Each experiment is contained in its respective folder. To run an experiment, simply navigate to the appropriate directory and execute the scripts.


