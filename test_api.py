import requests
import os

# Test image path (replace with yours)
TEST_IMAGE = r"C:\Users\induj\OneDrive\Desktop\20180104_091723.jpg"

def test_upload():
    url = "http://localhost:8000/upload"
    
    if not os.path.exists(TEST_IMAGE):
        print(f"Error: Test image not found at {TEST_IMAGE}")
        return

    with open(TEST_IMAGE, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
        
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")

if __name__ == "__main__":
    test_upload()