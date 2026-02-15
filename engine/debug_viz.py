import sys
import os
import shutil
from pathlib import Path

# Add current directory to path so we can import main
sys.path.append(os.getcwd())

# Mock FastAPI dependencies
class MockFile:
    def __init__(self, filename):
        self.filename = filename

# Import main
try:
    import main
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Clean sessions
SESSION_DIR = Path("/tmp/sessions")
if SESSION_DIR.exists():
    try:
        shutil.rmtree(SESSION_DIR)
        print("Sessions cleared.")
    except Exception as e:
        print(f"Failed to clear sessions: {e}")
else:
    # Windows fallback
    SESSION_DIR = Path("C:/tmp/sessions")
    if SESSION_DIR.exists():
         try:
            shutil.rmtree(SESSION_DIR)
            print("Sessions (Win) cleared.")
         except Exception as e:
            print(f"Failed to clear sessions: {e}")

# Run debug
def debug():
    print("Starting debug...")
    
    # 1. Upload
    print("Simulating upload...")
    with open("test_chart_data.csv", "rb") as f:
        content = f.read()
    
    # Manually call upload_dataset logic
    # We can't easily call FastAPI endpoint directly without TestClient, but we can import logic.
    # Actually, main.upload_dataset expects UploadFile.
    
    # Let's use TestClient
    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    
    files = {"file": ("test_chart_data.csv", content, "text/csv")}
    response = client.post("/upload", files=files)
    
    if response.status_code != 200:
        print(f"Upload failed: {response.text}")
        return
    
    session_id = response.json()["session_id"]
    print(f"Session ID: {session_id}")
    
    # 2. Generate Viz
    print("Calling generate-viz...")
    response = client.get(f"/generate-viz/{session_id}?max_charts=8")
    
    if response.status_code != 200:
        print(f"Generate Viz Failed: {response.status_code}")
        print(f"Response: {response.text}")
    else:
        print("Generate Viz Success!")
        data = response.json()
        print(f"Generated {len(data['charts'])} charts.")

if __name__ == "__main__":
    debug()
