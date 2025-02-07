import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.filters import sobel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance

RANDDOM_SEED = 69
NUM_INITIALIZATIONS = 20

# Load features
df_features = pd.read_csv('/home/dascim/data/3_extracted_features/EXC/glom_features_uniform.csv')

# Load annotations
annotation_dir = '/home/dascim/data/1_cytomine_downloads/EXC/annotations/25/'
df_annotations = pd.concat([pd.read_csv(annotation_dir + f'annotations_{patient}_25.csv') for patient in ['001', '002', '003']])
df_annotations_004 = pd.read_csv(annotation_dir + f'annotations_004_25.csv')

df = df_features.merge(df_annotations, left_on='glom_index', right_on='ID')
df_004 = df_features.merge(df_annotations_004, left_on='glom_index', right_on='ID')

feature_list = ['aspect_ratio', 'eccentricity', 'circularity' ] + ['Area (microns²)', 'Perimeter (mm)']
feature_list += [f'lbp_uniform_{_}' for _ in range(10) if _ != 8]

# load test indices
file_path = '/data/input/set_indices/test15_val15_test.txt'
test_indices = np.loadtxt(file_path, dtype=int)

df_train_within = df[~df['glom_index'].isin(test_indices)]
df_test_within = df[df['glom_index'].isin(test_indices)]

X_train = df_train_within[feature_list]
y_train = df_train_within['Term']
X_test_within = df_test_within[feature_list]
y_test_within = df_test_within['Term']
X_test_between = df_004[feature_list]
y_test_between = df_004['Term']

# One hot encode targets
y_train = pd.get_dummies(y_train)
y_test_within = pd.get_dummies(y_test_within)
y_test_between = pd.get_dummies(y_test_between)


# Normalize features (min max scaling)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test_within = scaler.transform(X_test_within)
X_test_between = scaler.transform(X_test_between)

# Defined grid for random forrest grid search
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 50, 100, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search once
rf = RandomForestRegressor(random_state=RANDDOM_SEED)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=4, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Store scores for 20 initializations
within_scores_acc = []
between_scores_acc = []
within_scores_f1 = []
between_scores_f1 = []

# Loop for 20 random initializations
for seed in range(NUM_INITIALIZATIONS):
    print(f"Random Seed {seed + 1}/{NUM_INITIALIZATIONS}")
    model = RandomForestRegressor(random_state=seed, **best_params)

    # Train and test on within patients
    model.fit(X_train, y_train)
    y_pred_within = model.predict(X_test_within)
    y_pred_within = np.argmax(y_pred_within, axis=1)
    y_test_within_labels = np.argmax(y_test_within.values, axis=1)
    within_score_acc = accuracy_score(y_test_within_labels, y_pred_within)
    within_scores_acc.append(within_score_acc)
    within_score_f1 = f1_score(y_test_within_labels, y_pred_within, average='macro')
    within_scores_f1.append(within_score_f1)

    # Train and test on between patients
    X_train_between = np.concatenate([X_train, X_test_within])
    y_train_between = pd.concat([y_train, y_test_within])
    model.fit(X_train_between, y_train_between)
    y_pred_between = model.predict(X_test_between)
    y_pred_between = np.argmax(y_pred_between, axis=1)
    y_test_between_labels = np.argmax(y_test_between.values, axis=1)
    between_score_acc = accuracy_score(y_test_between_labels, y_pred_between)
    between_scores_acc.append(between_score_acc)
    between_score_f1 = f1_score(y_test_between_labels, y_pred_between, average='macro')
    between_scores_f1.append(between_score_f1)


# Compute mean and standard deviation of scores
within_mean = np.mean(within_scores_acc)
within_std = np.std(within_scores_acc)
between_mean = np.mean(between_scores_acc)
between_std = np.std(between_scores_acc)

# Report results
print("\nWithin Patients:")
print(f"Accuracy: {within_mean:.4f} +/- {within_std:.4f}")

print("\nBetween Patients:")
print(f"Accuracy: {between_mean:.4f} +/- {between_std:.4f}")

# Computer f1 scores
within_mean_f1 = np.mean(within_scores_f1)
within_std_f1 = np.std(within_scores_f1)
between_mean_f1 = np.mean(between_scores_f1)
between_std_f1 = np.std(between_scores_f1)

# Report results
print("\nWithin Patients:")
print(f"F1: Mean = {within_mean_f1:.4f}, Std = {within_std_f1:.4f}")

print("\nBetween Patients:")
print(f"F1: Mean = {between_mean_f1:.4f}, Std = {between_std_f1:.4f}")

# Save scores
np.save('/data/output/random_forest_within_acc.npy', within_scores_acc)





