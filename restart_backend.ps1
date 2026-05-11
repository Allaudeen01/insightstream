# Restart Backend with V2 Engine
# Run this script to load the new code with all fixes

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Restarting Backend with V2 Engine" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Stop backend
Write-Host "[1/4] Stopping backend process..." -ForegroundColor Yellow
try {
    Stop-Process -Id 15296 -Force -ErrorAction SilentlyContinue
    Write-Host "✅ Backend stopped" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Backend may already be stopped" -ForegroundColor Yellow
}
Start-Sleep -Seconds 2

# Step 2: Clear Python cache
Write-Host ""
Write-Host "[2/4] Clearing Python cache..." -ForegroundColor Yellow
$cacheDir = "c:\Users\ALI\Downloads\insightstream_-ai-data-analyst\engine\__pycache__"
if (Test-Path $cacheDir) {
    Remove-Item -Path $cacheDir -Recurse -Force
    Write-Host "✅ Cache cleared" -ForegroundColor Green
} else {
    Write-Host "✅ No cache found (already clean)" -ForegroundColor Green
}

# Step 3: Clear .pyc files
Write-Host ""
Write-Host "[3/4] Clearing .pyc files..." -ForegroundColor Yellow
Get-ChildItem -Path "c:\Users\ALI\Downloads\insightstream_-ai-data-analyst\engine" -Recurse -Filter "*.pyc" | Remove-Item -Force
Write-Host "✅ .pyc files cleared" -ForegroundColor Green

# Step 4: Instructions to start
Write-Host ""
Write-Host "[4/4] Ready to start backend" -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  NOW RUN THESE COMMANDS:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "cd c:\Users\ALI\Downloads\insightstream_-ai-data-analyst" -ForegroundColor White
Write-Host ".\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "python engine/main.py" -ForegroundColor White
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  LOOK FOR THIS IN CONSOLE:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ V2 ENGINE ACTIVE — 2026-05-09 BUILD" -ForegroundColor Green
Write-Host ""
Write-Host "If you see this, the new code is loaded!" -ForegroundColor Green
Write-Host ""
