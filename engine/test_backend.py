import requests
import sys

try:
    print("Checking backend health...")
    r = requests.get("http://localhost:8000/docs", timeout=5)
    print(f"Backend status: {r.status_code}")
except Exception as e:
    print(f"Backend not reachable: {e}")
    sys.exit(1)

# Run the filtering test
import verify_filtering
verify_filtering.verify_filtering()
