import requests
import os
import json

SERVER = "http://localhost:8000"
TEST_IMAGE = r"C:\Users\induj\OneDrive\Desktop\20180104_091723.jpg"

def run_tests():
    # Test 1: Check server connection
    print("\n=== Testing API Connection ===")
    try:
        response = requests.get(f"{SERVER}/test-connection")
        print(f"Connection Test: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Connection failed: {str(e)}")
        return

    # Test 2: File upload
    print("\n=== Testing File Upload ===")
    if not os.path.exists(TEST_IMAGE):
        print(f"Error: Test image not found at {TEST_IMAGE}")
        return

    try:
        with open(TEST_IMAGE, 'rb') as f:
            response = requests.post(
                f"{SERVER}/upload",
                files={'file': f},
                timeout=10
            )
        print(f"Upload Test: {response.status_code}")
        print("Response:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Upload failed: {str(e)}")

if __name__ == "__main__":
    run_tests()