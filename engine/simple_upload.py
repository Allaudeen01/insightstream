import requests
import json

BASE_URL = "http://localhost:8000"
FILE_PATH = "test_chart_data.csv"

def test():
    print(f"Uploading {FILE_PATH}...")
    with open(FILE_PATH, "rb") as f:
        files = {"file": f}
        try:
            response = requests.post(f"{BASE_URL}/upload", files=files)
        except Exception as e:
            print(f"Request Failed: {e}")
            return

    print(f"Status: {response.status_code}")
    print(f"Content Repr: {repr(response.content)}")
    
    try:
        data = response.json()
        print("JSON Decode Success!")
        print(data)
    except Exception as e:
        print(f"JSON Decode Failed: {e}")
        # Try manual match
        import re
        m = re.search(b'session_id":"([^"]+)"', response.content)
        if m:
            print(f"Regex found session_id: {m.group(1)}")
        else:
            print("Regex found nothing")

if __name__ == "__main__":
    test()
