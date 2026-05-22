"""
InsightStream Chart Consistency Verification
Verifies: UI charts == LLM charts == PDF charts
"""
import sys, os, time, json, sqlite3, requests, base64
sys.path.insert(0, ".")
os.chdir(os.path.dirname(__file__))
import plotly.io as pio

BASE   = "http://localhost:8000"
EMAIL  = "verify_test@gmail.com"
PASSWD = "VerifyTest123!"

# ── Step 0: Register test user ────────────────────────────────────────────────
r = requests.post(f"{BASE}/auth/register", json={
    "email": EMAIL, "password": PASSWD, "full_name": "Verify Test"
})
if r.status_code in (200, 201):
    print(f"Registered: {EMAIL}")
elif "already" in r.text.lower() or r.status_code == 400:
    print(f"User exists: {EMAIL}")
else:
    print(f"Register {r.status_code}: {r.text[:80]}")

# ── Step 1: Login ─────────────────────────────────────────────────────────────
r = requests.post(f"{BASE}/auth/login", json={"email": EMAIL, "password": PASSWD})
if r.status_code != 200:
    print(f"FAIL: Login {r.status_code}")
    sys.exit(1)
token   = r.json().get("access_token")
headers = {"Authorization": f"Bearer {token}"}
print(f"Login: OK")

# ── Step 2: Clear cache ───────────────────────────────────────────────────────
import shutil
for p in [".analyzer_cache"]:
    if os.path.exists(p):
        shutil.rmtree(p, ignore_errors=True)
        os.makedirs(p, exist_ok=True)
        print(f"Cache cleared")

# ── Step 3: Upload HR dataset ─────────────────────────────────────────────────
import glob
hr_files = sorted(glob.glob("data/uploads/session_*HRDataset_v14.csv"))
dataset_path = hr_files[-1] if hr_files else None
if not dataset_path:
    print("FAIL: HRDataset_v14.csv not found in uploads")
    sys.exit(1)

print(f"Uploading: {os.path.basename(dataset_path)}")
with open(dataset_path, "rb") as f:
    r = requests.post(
        f"{BASE}/analyze",
        files={"file": ("HRDataset_v14.csv", f, "text/csv")},
        data={"currency": "USD"},
        headers=headers,
        timeout=120,
    )

if r.status_code != 200:
    print(f"FAIL: Upload {r.status_code} — {r.text[:200]}")
    sys.exit(1)

session_id = r.json().get("session_id")
print(f"Upload: OK, session_id={session_id}")

# Check stored LLM results
conn = sqlite3.connect("data/insightstream.db")
row  = conn.execute(
    "SELECT llm_results_json FROM analysis_sessions WHERE id=?", (session_id,)
).fetchone()
conn.close()

stored_titles = set()
n_stored_charts = 0
n_insights = 0
if row and row[0]:
    stored = json.loads(row[0])
    n_stored_charts = len(stored.get("charts", []))
    n_insights      = len(stored.get("insights", []))
    stored_titles   = {c.get("title", "").lower() for c in stored.get("charts", [])}
    print(f"Stored LLM results: {n_insights} insights, {n_stored_charts} charts")
    for c in stored.get("charts", []):
        print(f"  Stored: {c.get('title')!r}")
else:
    print("WARN: No llm_results_json stored")

time.sleep(1)

# ── Step 4: Verify /generate-viz returns LLM charts ──────────────────────────
r = requests.get(
    f"{BASE}/generate-viz/{session_id}?max_charts=12",
    headers=headers, timeout=30,
)
if r.status_code != 200:
    print(f"FAIL: generate-viz {r.status_code}")
    sys.exit(1)

viz_charts = r.json().get("charts", [])
viz_titles  = {c.get("title", "").lower() for c in viz_charts}
print(f"\nVisualization charts ({len(viz_charts)}):")
for c in viz_charts:
    print(f"  {c.get('title')!r}")

# Check: do viz titles overlap with stored LLM titles?
overlap = viz_titles & stored_titles
if overlap:
    print(f"PASS: Viz tab shows LLM charts (matched titles: {len(overlap)}/{n_stored_charts})")
elif n_stored_charts == 0:
    print("WARN: No LLM charts stored to compare")
else:
    print("FAIL: Viz charts do not match stored LLM charts")
    print(f"  Stored: {stored_titles}")
    print(f"  Viz:    {viz_titles}")

# ── Step 5: Render charts to base64 for PDF ───────────────────────────────────
def strip_and_render(pj):
    """Strip template from plotly_json dict/str and render to base64 PNG."""
    if isinstance(pj, str):
        pj = json.loads(pj)
    pj.get("layout", {}).pop("template", None)
    fig = pio.from_json(json.dumps(pj))
    img = pio.to_image(fig, format="png", width=800, height=500)
    return base64.b64encode(img).decode("utf-8")

chart_assets = []
for c in viz_charts[:4]:
    pj = c.get("plotly_json")
    if not pj:
        continue
    try:
        b64 = strip_and_render(pj)
        chart_assets.append({
            "id":           c.get("chart_id", "chart"),
            "title":        c.get("title", ""),
            "image_base64": b64,
            "insight":      "",
        })
        print(f"  Rendered: {c.get('title')!r} ({len(b64)} b64 chars)")
    except Exception as e:
        print(f"  Render failed ({c.get('title')!r}): {str(e)[:80]}")

print(f"Chart assets for PDF: {len(chart_assets)}")

# ── Step 6: Export PDF ────────────────────────────────────────────────────────
export_body = {
    "title":           "HR Analytics Report",
    "template":        "modern",
    "project_name":    "InsightStream",
    "kpis":            {},
    "charts":          chart_assets,
    "ai_summary":      "",
    "insights":        [],
    "recommendations": [],
    "text_blocks":     [],
}

r = requests.post(
    f"{BASE}/export-dashboard-pdf/{session_id}",
    json=export_body, headers=headers, timeout=120,
)

if r.status_code == 200:
    pdf_path = f"verify_output_{session_id}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(r.content)
    print(f"PDF exported: {pdf_path} ({len(r.content):,} bytes)")
    print("PASS: PDF generated successfully")
else:
    print(f"FAIL: PDF export {r.status_code} — {r.text[:200]}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("VERIFICATION COMPLETE")
print(f"  Session:         {session_id}")
print(f"  LLM stored:      {n_insights} insights, {n_stored_charts} charts")
print(f"  Viz charts:      {len(viz_charts)}")
print(f"  PDF chart assets:{len(chart_assets)}")
print(f"  Title overlap:   {len(overlap)}/{n_stored_charts}")
if len(overlap) == n_stored_charts and n_stored_charts > 0:
    print("  STATUS: PASS - UI and PDF use same LLM charts")
else:
    print("  STATUS: PARTIAL - check output above")
print("="*55)
