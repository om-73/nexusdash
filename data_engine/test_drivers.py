import requests
import json
import pandas as pd
import io

import os
BASE_URL = "http://localhost:8000"

# 1. Create Dummy Data
# y = 2*xA + random noise
# xB is random noise
df = pd.DataFrame({
    'xA': range(100),
    'xB': [1] * 100,
    'y': [x * 2 for x in range(100)]
})

# Save locally
file_path = os.path.abspath("drivers_data.csv")
df.to_csv(file_path, index=False)
print(f"Created local file: {file_path}")

# 2. Skip Upload (Node requires auth). Python just needs path.

# 3. Load
print("Loading...")
res = requests.post(f"{BASE_URL}/load", json={"file_path": file_path, "file_type": "csv"})
if res.status_code != 200:
    print("Load Failed:", res.text)
    exit(1)

# 4. Analyze Drivers
print("Analyzing Drivers (Target: y)...")
try:
    res = requests.post(f"http://localhost:8000/analyze/drivers", json={"target_column": "y"})
    if res.status_code == 200:
        data = res.json()
        print("Success!")
        print(json.dumps(data, indent=2))
        
        # Validation
        drivers = data['drivers']
        top_driver = drivers[0]['feature']
        if top_driver == 'xA':
            print("PASS: xA is the top driver.")
        else:
            print(f"FAIL: xA should be top driver, but got {top_driver}")
            
    else:
        print("Analysis Failed:", res.text)
except Exception as e:
    print(f"Connection Failed: {e}")
