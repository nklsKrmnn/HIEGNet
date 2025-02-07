import numpy as np
import pandas as pd
import os

from src.wsi_preprocessing.function.path_io import get_path_up_to, extract_index

ROOT_DIR = get_path_up_to(os.path.abspath(__file__), "repos")
TARGET_DIR = f"{ROOT_DIR}/data/3_extracted_features/EXC/"

f = np.load(ROOT_DIR + "/data/3_extracted_features/EXC/predictions_using_standardization.npz")

d = dict(zip(("data1{}".format(k) for k in f), (f[k] for k in f)))

arr = d['data1predictions']
features = np.mean(arr, (1,2))

ids = pd.DataFrame(d['data1filenames'])
ids = ids[0].apply(extract_index)

df_features = pd.DataFrame(features, columns=[f'byol_feature_{i}' for i in range(features.shape[1])])
df_features['glom_index'] = ids

# Save the features
df_features.to_csv(f"{TARGET_DIR}/byol_features.csv", index=False)

print(f'Features saved to {TARGET_DIR}/byol_features.csv')