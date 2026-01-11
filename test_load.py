import requests
import json

url = "http://localhost:8000/load"
file_path = "/Users/omprakashsingh/web_ditector/data visulizer/server/uploads/file-1768119529263-511141150.csv"

payload = {
    "file_path": file_path,
    "file_type": "csv"
}

try:
    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
