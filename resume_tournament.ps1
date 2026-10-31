# Resume 24/7 Quantitative Tournament from existing state with Active Heartbeat Watchdog
# (No reset, No flatten - Preserves testnet positions & profile states)
Set-Location D:\Quant_trading
$profiles = "regime_omni_1k,flexible_regime_1k,carry_max_eta_1k,baseline_1k"
$liveProfile = "regime_omni_1k"
$intervalSeconds = 3600
$heartbeatPath = "data\state\tournament_heartbeat.json"
$maxStaleSeconds = 300  # 5 minutes without heartbeat = zombie hang

# 1. Anti-Sleep system settings
try {
    powercfg /change standby-timeout-ac 0
    powercfg /change hibernate-timeout-ac 0
} catch {}

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting 24/7 background tournament daemon with Active Heartbeat Watchdog..." -ForegroundColor Green
Write-Host "Profiles: $profiles"
Write-Host "Live Profile: $liveProfile (TESTNET_REAL)"
Write-Host "Interval: $intervalSeconds seconds"
Write-Host "Heartbeat Check: every 60s (Stale Threshold: ${maxStaleSeconds}s)"

while ($true) {
    Write-Host "`n[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] === STARTING TOURNAMENT RUNNER PROCESS ===" -ForegroundColor Cyan
    
    $pythonArgs = @(
        "-u",
        "scripts/run_parallel_live_demo.py",
        "--profiles", $profiles,
        "--live-testnet-profile", $liveProfile,
        "--interval-seconds", $intervalSeconds
    )
    
    $proc = Start-Process -FilePath "python.exe" -ArgumentList $pythonArgs -PassThru -NoNewWindow
    $procId = $proc.Id
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Python runner launched with PID: $procId" -ForegroundColor Yellow
    
    # Active Watchdog Monitoring Loop
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 60
        
        # Check if process exited during sleep
        if ($proc.HasExited) {
            break
        }
        
        # Check Heartbeat File
        if (Test-Path $heartbeatPath) {
            try {
                $content = Get-Content -Raw -Path $heartbeatPath | ConvertFrom-Json
                $nowEpoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
                $hbEpoch = [double]$content.epoch_sec
                $ageSeconds = $nowEpoch - $hbEpoch
                
                # Fallback to LastWriteTime if epoch_sec is missing/invalid
                if ($hbEpoch -le 0) {
                    $item = Get-Item $heartbeatPath
                    $ageSeconds = ((Get-Date) - $item.LastWriteTime).TotalSeconds
                }
                
                if ($ageSeconds -gt $maxStaleSeconds) {
                    Write-Host "`n[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] [WATCHDOG ALERT] ZOMBIE_HANG_DETECTED!" -ForegroundColor Red
                    Write-Host "Heartbeat is stale: age=$([math]::Round($ageSeconds, 1))s > threshold=${maxStaleSeconds}s. PID=$procId" -ForegroundColor Red
                    Write-Host "Terminating frozen Python process PID $procId..." -ForegroundColor Red
                    
                    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                    Start-Sleep -Seconds 5
                    break
                } else {
                    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] [WATCHDOG OK] PID=$procId alive | Cycle=$($content.cycle) Phase=$($content.phase) Heartbeat age: $([math]::Round($ageSeconds, 1))s" -ForegroundColor DarkGray
                }
            } catch {
                Write-Warning "[WATCHDOG] Warning reading heartbeat file: $_"
            }
        } else {
            # File doesn't exist yet, check process running time
            $runtime = ((Get-Date) - $proc.StartTime).TotalSeconds
            if ($runtime -gt $maxStaleSeconds) {
                Write-Host "`n[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] [WATCHDOG ALERT] Heartbeat file missing after $([math]::Round($runtime, 1))s! Killing PID $procId..." -ForegroundColor Red
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                Start-Sleep -Seconds 5
                break
            }
        }
    }
    
    $exitCode = $proc.ExitCode
    Write-Warning "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Process PID $procId stopped with exit code $exitCode. Resuming in 15 seconds..."
    Start-Sleep -Seconds 15
}
