#!/bin/bash
# Upload file
echo "Uploading file..."
RESPONSE=$(curl -s -X POST -H "Content-Type: multipart/form-data" -F "file=@test_upload.csv" http://localhost:5001/api/upload)
echo "Upload Response: $RESPONSE"

# Extract file path (simple grep/awk hack or just copy manually if needed)
FILE_PATH=$(echo $RESPONSE | grep -o '"filePath":"[^"]*"' | cut -d'"' -f4)
echo "Extracted FilePath: $FILE_PATH"

# Load Data
echo "Loading Data..."
curl -v -X POST -H "Content-Type: application/json" -d "{\"filePath\": \"$FILE_PATH\", \"fileType\": \"csv\"}" http://localhost:5001/api/data/load
