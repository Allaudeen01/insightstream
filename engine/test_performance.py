"""Benchmark: test background analysis + cache performance."""
import time
import requests
import io
import random
import polars as pl

API = "http://127.0.0.1:8001"

# Create a synthetic e-commerce CSV
print("=== Creating 1000-row test dataset ===")
random.seed(42)
n = 1000
data = {
    "Order_ID":         [f"ORD-{i:05d}" for i in range(n)],
    "Customer_ID":      [f"CUST-{random.randint(1,50):04d}" for _ in range(n)],
    "Product_Category": [random.choice(["Electronics","Clothing","Books","Home","Sports"]) for _ in range(n)],
    "City":             [random.choice(["Mumbai","Delhi","Bangalore","Chennai","Kolkata"]) for _ in range(n)],
    "Payment_Method":   [random.choice(["Credit Card","Credit Card","Credit Card","UPI","Cash"]) for _ in range(n)],
    "Price":            [round(random.uniform(10, 500), 2) for _ in range(n)],
    "Quantity":         [random.randint(1, 5) for _ in range(n)],
    "Delivery_Days":    [random.randint(1, 15) for _ in range(n)],
    "Returned":         [random.choice(["Yes","Yes","No","No","No"]) for _ in range(n)],
}
df = pl.DataFrame(data)
csv_bytes = df.write_csv()

# 1. Upload
print("\n--- Upload ---")
t0 = time.time()
r = requests.post(f"{API}/upload", files={"file": ("test.csv", io.BytesIO(csv_bytes.encode()), "text/csv")})
upload_time = time.time() - t0
assert r.status_code == 200, f"Upload failed: {r.text}"
sid = r.json()["session_id"]
print(f"  Session: {sid[:12]}... ({upload_time:.2f}s)")

# 2. POST /analyze (background)
print("\n--- POST /analyze (background kick-off) ---")
t0 = time.time()
r = requests.post(f"{API}/analyze/{sid}")
print(f"  Status: {r.status_code} ({time.time()-t0:.3f}s)")
print(f"  Response: {r.json()}")

# 3. Poll /analyze-status until done
print("\n--- Polling /analyze-status ---")
for i in range(30):
    time.sleep(0.5)
    r = requests.get(f"{API}/analyze-status/{sid}")
    status = r.json()
    print(f"  Poll {i+1}: progress={status['progress']}%, status={status['status']}")
    if status["status"] in ("done", "error"):
        break

# 4. Benchmark: GET /insights (should be cache hit)
print("\n--- GET /insights (first call = cache hit from background job) ---")
t0 = time.time()
r = requests.get(f"{API}/insights/{sid}")
first_time = time.time() - t0
assert r.status_code == 200
data = r.json()
print(f"  {len(data['insights'])} insights, {first_time*1000:.0f}ms")

# 5. Benchmark: GET /insights (second call = cache hit)
print("\n--- GET /insights (second call = pure cache hit) ---")
t0 = time.time()
r = requests.get(f"{API}/insights/{sid}")
cached_time = time.time() - t0
print(f"  {cached_time*1000:.0f}ms (should be <50ms)")

# 6. GET /kpis (should use cached profile)
print("\n--- GET /kpis ---")
t0 = time.time()
r = requests.get(f"{API}/kpis/{sid}")
kpi_time = time.time() - t0
print(f"  {len(r.json()['kpis'])} KPIs, {kpi_time*1000:.0f}ms")

# 7. GET /eda (should use cached profile)
print("\n--- GET /eda ---")
t0 = time.time()
r = requests.get(f"{API}/eda/{sid}")
eda_time = time.time() - t0
print(f"  {eda_time*1000:.0f}ms")

# 8. GET /eda (second call = full cache hit)
print("\n--- GET /eda (cached) ---")
t0 = time.time()
r = requests.get(f"{API}/eda/{sid}")
eda_cached = time.time() - t0
print(f"  {eda_cached*1000:.0f}ms")

# 9. Check cache status
print("\n--- Cache Status ---")
r = requests.get(f"{API}/cache-status/{sid}")
print(f"  {r.json()}")

# 10. Check GZip
print("\n--- GZip check ---")
r = requests.get(f"{API}/insights/{sid}", headers={"Accept-Encoding": "gzip"})
print(f"  Content-Encoding: {r.headers.get('Content-Encoding', 'none')}")
print(f"  Content-Length: {r.headers.get('Content-Length', 'unknown')}")

# Summary
print("\n" + "="*50)
print("PERFORMANCE SUMMARY")
print("="*50)
print(f"  Upload:           {upload_time*1000:.0f}ms")
print(f"  Insights (first): {first_time*1000:.0f}ms")
print(f"  Insights (cache): {cached_time*1000:.0f}ms")
print(f"  KPIs:             {kpi_time*1000:.0f}ms")
print(f"  EDA (first):      {eda_time*1000:.0f}ms")
print(f"  EDA (cache):      {eda_cached*1000:.0f}ms")
print(f"\n  Cache hit speedup: {first_time/max(cached_time,0.001):.0f}x")
print("  [PASS]" if cached_time < 0.1 else "  [WARN] Cache may not be working")
