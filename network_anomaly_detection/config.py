# --- Data Configuration ---

# Path to the dataset file, relative to the project root (../data)
DATA_PATH = 'kdd_train.csv' 

# Name of the target column in the dataset
TARGET_COL = 'label'

# The value in TARGET_COL that represents a "Normal" connection
NORMAL_LABEL = 'normal'

# Categorical columns in the KDD dataset (used for one-hot encoding)
# KDD columns are typically 'protocol_type', 'service', and 'flag'
CATEGORICAL_COLS = [
    'feature_2', # protocol_type
    'feature_3', # service
    'feature_4'  # flag
]

# --- Model Configuration ---

# Keras MLP Hyperparameters
MLP_EPOCHS = 10
MLP_BATCH_SIZE = 512

# Isolation Forest Hyperparameter: Expected proportion of anomalies in the dataset
# Based on your data (58630 / (67343 + 58630)) ~ 0.465
IFOR_CONTAMINATION = 0.465