Set-Location D:\Quant_trading
$profiles = "regime_omni_1k,flexible_regime_1k,hybrid_trend_conservative_1k,baseline_1k"

# 0. Kiểm tra điều kiện
if (-not (Test-Path flatten_testnet2.py)) { Write-Error "Thiếu flatten_testnet2.py"; exit 1 }
git check-ignore .env | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error ".env CHƯA nằm trong .gitignore, dừng lại!"; exit 1 }

# 1. Commit Sprint 28 (không kèm snapshot/log/state)
git add -A
git reset -q -- .agy logs data/state 2>$null
git status --short | Select-Object -First 20
git commit -m "Sprint 28: orthogonal regime optimization (regime_omni_1k, 842 passed)"

# 2. Chống Windows tự Sleep
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

# 3. Flatten Perp trên Testnet
python -X utf8 flatten_testnet2.py
if ($LASTEXITCODE -ne 0) { Write-Error "Chưa flatten xong, dừng để tránh đặt chồng lệnh!"; exit 1 }

# 4. Cycle 1: tạo baseline
python scripts/run_parallel_live_demo.py --profiles $profiles --live-testnet-profile regime_omni_1k --reset-demo-state --cycles 1
if ($LASTEXITCODE -ne 0) { Write-Error "Cycle 1 lỗi, dừng lại!"; exit 1 }

# 5. Kiểm tra SYNCED trước khi vào vòng lặp
$log = "logs\sprint25_parallel_demo_scoreboard.jsonl"
$rows = Get-Content $log -Tail 4 | ForEach-Object { $_ | ConvertFrom-Json }
$rows | Format-Table cycle, profile, mode, reconcile_status, recon_drift_usd -AutoSize
$live = $rows | Where-Object { $_.profile -eq "regime_omni_1k" } | Select-Object -Last 1
if ($live.reconcile_status -ne "SYNCED") { Write-Error "regime_omni_1k chưa SYNCED, không vào vòng lặp. Gửi bảng trên cho tôi."; exit 1 }

# 6. Vòng lặp 24/7
while ($true) {
    Write-Host "`n[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] === 4-PROFILE TOURNAMENT (regime_omni_1k = TESTNET_REAL) ===" -ForegroundColor Cyan
    python scripts/run_parallel_live_demo.py --profiles $profiles --live-testnet-profile regime_omni_1k --interval-seconds 3600
    Write-Warning "Tiến trình dừng hoặc mạng chập chờn. Nối lại sau 30 giây..."
    Start-Sleep -Seconds 30
}
