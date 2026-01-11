import requests
import json

BASE_URL = "http://localhost:5001/api"

# 1. Create a dummy CSV with Tabs (simulating user data)
content = """M	69	1	2	2	1	1	2	1	2	2	2	2	2	2	YES
M	74	2	1	1	1	2	2	2	1	1	1	2	2	2	YES
F	59	1	1	1	2	1	2	1	2	1	2	2	1	2	NO"""

with open("test_data.csv", "w") as f:
    f.write(content)

# 2. Upload File (Node requires upload first usually)
print("Uploading file...")
try:
    files = {'file': ('test_data.csv', open('test_data.csv', 'rb'), 'text/csv')}
    res = requests.post(f"{BASE_URL}/upload", files=files)
    if res.status_code == 200:
        file_path = res.json()['filePath']
        print(f"Upload: Success ({file_path})")
    else:
        print("Upload: Failed", res.text)
        exit(1)
except Exception as e:
    print(f"Upload Connection Failed: {e}")
    exit(1)

# 3. Load Data
print("Loading data...")
try:
    # Node endpoint expects 'filePath' not 'file_path'
    res = requests.post(f"{BASE_URL}/data/load", json={"filePath": file_path, "fileType": "csv"})
    if res.status_code == 200:
        print("Load: Success")
    else:
        print("Load: Failed", res.text)
        exit(1)
except Exception as e:
    print(f"Load Connection Failed: {e}")
    exit(1)

# 4. Clean Data
print("\nCleaning data (dropna)...")
try:
    res = requests.post(f"{BASE_URL}/data/clean", json={"operation": "dropna"})
    if res.status_code == 200:
        print("Clean: Success")
    else:
        print("Clean: Failed", res.text)
except Exception as e:
    print(f"Clean Connection Failed: {e}")
