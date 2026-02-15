import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"
FILE_PATH = "test_chart_data.csv"

def verify_filtering():
    # 1. Upload
    print(f"Uploading {FILE_PATH}...")
    with open(FILE_PATH, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{BASE_URL}/upload", files=files)
    
    if response.status_code != 200:
        print("Upload failed:", response.text)
        return
        
    print(f"Upload Status: {response.status_code}")
    print(f"Upload Response Content: {repr(response.content)}")
    import json
    print(f"Response Encoding: {response.encoding}")
    try:
        data = json.loads(response.content)
        session_id = data["session_id"]
    except Exception as e:
        print(f"Manual JSON Load Error: {e}")
        return
    
    # 2. Test Default (no filter)
    print("\nTesting Default Viz...")
    resp = requests.get(f"{BASE_URL}/generate-viz/{session_id}")
    if resp.status_code != 200:
        print(f"Generate Viz Failed: {resp.status_code}")
        print(f"Response: {resp.text}")
        return
    data = resp.json()
    print(f"Generated {len(data['charts'])} charts.")
    
    # 3. Test GroupBy
    print("\nTesting GroupBy='Region'...")
    resp = requests.get(f"{BASE_URL}/generate-viz/{session_id}?groupby=Region")
    data = resp.json()
    print(f"Generated {len(data['charts'])} charts.")
    # Check if histogram color is Region
    hist_chart = next((c for c in data['charts'] if c['chart_type'] == 'histogram'), None)
    if hist_chart:
        print(f"Histogram used columns: {hist_chart['columns_used']}")
        if 'Region' in hist_chart['columns_used']:
            print("SUCCESS: Histogram used Region for grouping.")
        else:
            print("FAILURE: Histogram did not use Region.")
    
    # 4. Test Chart Type Filter
    print("\nTesting ChartType=['pie']...")
    resp = requests.get(f"{BASE_URL}/generate-viz/{session_id}?chart_types=pie")
    data = resp.json()
    types = set(c['chart_type'] for c in data['charts'])
    print(f"Chart types returned: {types}")
    if types == {'pie'} or types == set():
        print("SUCCESS: Only pie charts returned.")
    else:
        print(f"FAILURE: Returned {types}")

if __name__ == "__main__":
    verify_filtering()
