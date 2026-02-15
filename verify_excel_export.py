import urllib.request
import urllib.parse
import json
import os
import mimetypes

API_URL = "http://localhost:8000"
FILE_PATH = "test_multisheet.xlsx"

def encode_multipart_formdata(files, fields=None):
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    body = []
    
    if fields:
        for key, value in fields.items():
            body.append(f'--{boundary}'.encode())
            body.append(f'Content-Disposition: form-data; name="{key}"'.encode())
            body.append(b'')
            body.append(str(value).encode())
            
    for key, filepath in files.items():
        filename = os.path.basename(filepath)
        mime_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        with open(filepath, 'rb') as f:
            file_content = f.read()
            
        body.append(f'--{boundary}'.encode())
        body.append(f'Content-Disposition: form-data; name="{key}"; filename="{filename}"'.encode())
        body.append(f'Content-Type: {mime_type}'.encode())
        body.append(b'')
        body.append(file_content)
        
    body.append(f'--{boundary}--'.encode())
    body.append(b'')
    
    content_type = f'multipart/form-data; boundary={boundary}'
    return b'\r\n'.join(body), content_type

def test_export_excel():
    if not os.path.exists(FILE_PATH):
        print(f"File {FILE_PATH} not found. Run create_excel.py first.")
        # Re-create if missing (using simplified logic or assuming it exists)
        return

    print("1. Uploading file to get Session ID...")
    session_id = None
    try:
        # Upload specific sheet to get session
        data, content_type = encode_multipart_formdata({"file": FILE_PATH}, {"sheet_name": "Detailed_Sales"})
        req = urllib.request.Request(f"{API_URL}/upload", data=data, headers={"Content-Type": content_type})
        with urllib.request.urlopen(req) as response:
            result = json.load(response)
            session_id = result.get("session_id")
            print(f"Session ID: {session_id}")
    except Exception as e:
        print(f"Upload failed: {e}")
        return

    if not session_id:
        print("Failed to get session ID.")
        return

    print("2. Requesting Excel Export...")
    try:
        export_url = f"{API_URL}/export-excel/{session_id}"
        print(f"URL: {export_url}")
        with urllib.request.urlopen(export_url) as response:
            print(f"Status: {response.status}")
            headers = response.info()
            print(f"Content-Type: {headers.get_content_type()}")
            content = response.read()
            print(f"Content-Length: {len(content)} bytes")
            
            if len(content) > 1000 and "spreadsheetml.sheet" in headers.get_content_type():
                print("PASS: Excel file received.")
                with open("exported_report.xlsx", "wb") as f:
                    f.write(content)
                print("Saved to exported_report.xlsx")
            else:
                print("FAIL: Invalid response content.")
                
    except urllib.error.HTTPError as e:
        print(f"FAIL: HTTP Error {e.code}: {e.read().decode()}")
    except Exception as e:
        print(f"FAIL: Error: {e}")

if __name__ == "__main__":
    test_export_excel()
