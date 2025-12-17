Network Anomaly Detection Pipeline

An end-to-end machine learning project for detecting network intrusions (anomalies) using the KDD (Knowledge Discovery and Data Mining) dataset. The pipeline compares the effectiveness of supervised tree-based, deep learning, and unsupervised models for binary classification (Normal vs. Anomaly).

🚀 Key Features

⦁	Data Preprocessing: Handles data loading, categorical feature encoding (One-Hot), and target binarization (Normal = 0, Anomaly = 1).

⦁	Model Comparison: Implements and evaluates three different models:

⦁	Random Forest Classifier (Supervised)

⦁	Multi-Layer Perceptron (MLP) using Keras (Supervised)

⦁	Isolation Forest (Unsupervised Outlier Detection)

⦁	Performance Metrics: Generates detailed Classification Reports, AUC scores, Confusion Matrices, and Feature Importance plots.

⦁	Configuration: All model and data parameters are easily configurable via config.py.



🛠️ Project Setup

Prerequisites

Ensure you have Python 3.8+ and Git installed on your system.

1. Clone the Repository

git clone [https://github.com/sayuj5/Anomaly-Detection-Project.git](https://github.com/sayuj5/Anomaly-Detection-Project.git)

cd Anomaly-Detection-Project


2. Set up the Environment

It is highly recommended to use a virtual environment:

# Create and activate environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Or activate environment (Linux/macOS)
# source venv/bin/activate


3. Install Dependencies

Install all necessary libraries using the provided requirements.txt:

pip install -r network_anomaly_detection/requirements.txt


4. Data Preparation

This repository does not include the large dataset (kdd_train.csv).

⦁	Create a folder named data/ in the project root.

⦁	Place your KDD training dataset file inside this folder.

⦁	Rename the file to kdd_train.csv to match the path defined in config.py.



▶️ Execution

Run the main script from the root of the project folder:

python network_anomaly_detection/main.py



📊 Results 

OverviewThe supervised models demonstrated high performance on the KDD test set, confirming the separability of the features after preprocessing.





