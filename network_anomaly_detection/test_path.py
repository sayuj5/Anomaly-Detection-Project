import os

# --- Configuration copied from config.py for testing ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: For this quick test, we assume test_path.py is in the root: AnomalyDetectionProject/
# Therefore, we need to adjust BASE_DIR to point to the network_anomaly_detection folder.
# We will use the explicit path that the main script tried to access:

data_path_components = ['network_anomaly_detection', '..', 'data', 'kdd_train.csv']
# We are running this test_path.py from the root. The components below mimic the path used
# when running from the network_anomaly_detection folder.

# Let's verify the actual, simplified path:
ACTUAL_DATA_PATH = os.path.join(os.getcwd(), 'data', 'kdd_train.csv')

print(f"Checking Path: {ACTUAL_DATA_PATH}")

try:
    with open(ACTUAL_DATA_PATH, 'r') as f:
        print("\n✅ SUCCESS: File found and opened! Data pipeline will work.")
        # Read the first line to confirm content
        first_line = f.readline().strip()
        print(f"First line of data (Expected: No header, data values): {first_line[:80]}...")
except FileNotFoundError:
    print("\n❌ FAILURE: File NOT found at the checked path.")
    print("ACTION REQUIRED: Go to your 'data' folder and ensure the file is named exactly 'kdd_train.csv'.")
except Exception as e:
    print(f"\n⚠️ ERROR: Found file, but could not read it. Error: {e}")