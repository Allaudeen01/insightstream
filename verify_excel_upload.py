import urllib.request
import urllib.parse
import json
import os
import mimetypes

API_URL = "http://localhost:8000/upload"
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

def test_upload_excel_multisheet():
    if not os.path.exists(FILE_PATH):
        print(f"File {FILE_PATH} not found. Run create_excel.py first.")
        return

    print(f"Testing upload with {FILE_PATH}...")
    
    # 1. Test without sheet name
    try:
        data, content_type = encode_multipart_formdata({"file": FILE_PATH})
        req = urllib.request.Request(API_URL, data=data, headers={"Content-Type": content_type})
        with urllib.request.urlopen(req) as response:
            result = json.load(response)
            print("Response 1 received.")
            if result.get("requires_selection") is True:
                print("PASS: Backend correctly identified multi-sheet file and requested selection.")
                print(f"Sheets found: {result.get('sheets')}")
            else:
                print("FAIL: Backend did not request selection for multi-sheet file.")
                print(result)
    except urllib.error.HTTPError as e:
        print(f"FAIL: HTTP Error {e.code}: {e.read().decode()}")
    except Exception as e:
        print(f"FAIL: Error: {e}")

    # 2. Test with sheet name
    print("\nTesting upload with specific sheet 'Detailed_Sales'...")
    try:
        data, content_type = encode_multipart_formdata({"file": FILE_PATH}, {"sheet_name": "Detailed_Sales"})
        req = urllib.request.Request(API_URL, data=data, headers={"Content-Type": content_type})
        with urllib.request.urlopen(req) as response:
            result = json.load(response)
            print("Response 2 received.")
            if result.get("requires_selection") is False:
                print("PASS: Backend accepted specific sheet and processed it.")
                print(f"Row count: {result.get('row_count')}")
            else:
                print("FAIL: Backend requested selection even when sheet_name was provided.")
    except urllib.error.HTTPError as e:
        print(f"FAIL: HTTP Error {e.code}: {e.read().decode()}")
    except Exception as e:
        print(f"FAIL: Error: {e}")

if __name__ == "__main__":
    test_upload_excel_multisheet()
