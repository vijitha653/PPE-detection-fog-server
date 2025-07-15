import requests
import os

def test_upload():
    url = 'http://localhost:5000/upload'
    test_image = 'test_image.jpg'  # Replace with a test image path
    
    with open(test_image, 'rb') as f:
        files = {'file': (os.path.basename(test_image), f)}
        response = requests.post(url, files=files)
    
    print(response.status_code)
    print(response.json())

if __name__ == '__main__':
    test_upload()