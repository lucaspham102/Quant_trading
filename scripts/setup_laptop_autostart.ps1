<#
.SYNOPSIS
    Đăng ký Scheduled Task tự động khởi chạy giải đấu Quant Trading 24/7 (resume_tournament.ps1)
    khi Windows khởi động lại / đăng nhập sau sự cố sập nguồn hoặc mất điện trên Laptop.

.DESCRIPTION
    - Tắt chế độ Sleep / Hibernate khi cắm sạc (AC Power) trên Laptop.
    - Đăng ký Windows Scheduled Task "QuantTrading_24_7_Tournament" kích hoạt khi AtLogOn
      (và AtStartup nếu chạy với quyền Administrator).
    - Bật cờ AllowStartIfOnBatteries & DontStopIfGoingOnBatteries để Laptop không tự ngắt bot
      khi mất điện lưới và chuyển sang chạy bằng Pin.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_laptop_autostart.ps1
    powershell -ExecutionPolicy Bypass -File scripts\setup_laptop_autostart.ps1 -Unregister
#>
param(
    [string]$TaskName = "QuantTrading_24_7_Tournament",
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ResumeScript = Join-Path $RepoRoot "resume_tournament.ps1"

if ($Unregister) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "[OK] Đã gỡ bỏ Scheduled Task '$TaskName'." -ForegroundColor Green
    } else {
        Write-Host "[INFO] Scheduled Task '$TaskName' không tồn tại." -ForegroundColor Yellow
    }
    exit 0
}

if (-not (Test-Path $ResumeScript)) {
    Write-Error "Không tìm thấy file '$ResumeScript'!"
    exit 1
}

Write-Host "=== CẤU HÌNH CHỐNG SLEEP & TỰ KHỞI ĐỘNG 24/7 TRÊN LAPTOP ===" -ForegroundColor Cyan
Write-Host "Thư mục dự án : $RepoRoot"
Write-Host "Script khởi chạy: $ResumeScript"

# 1. Ngăn Laptop tự Sleep / Hibernate khi cắm sạc (AC) và trên Pin (DC)
try {
    powercfg /change standby-timeout-ac 0
    powercfg /change hibernate-timeout-ac 0
    powercfg /change standby-timeout-dc 0
    Write-Host "[1/2] Đã tắt chế độ tự động Sleep/Hibernate trên Windows PowerCfg." -ForegroundColor Green
} catch {
    Write-Warning "Không thể chỉnh powercfg tự động: $_"
}

# 2. Đăng ký Windows Scheduled Task
$ActionArgs = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Normal -File `"$ResumeScript`""
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $ActionArgs -WorkingDirectory $RepoRoot
$TriggerLogon = New-ScheduledTaskTrigger -AtLogOn
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $TriggerLogon `
    -Settings $Settings `
    -Description "Auto-resume 24/7 Quant Trading Tournament from latest cycle state after power/reboot." `
    -Force | Out-Null

Write-Host "[2/2] Đã đăng ký Scheduled Task '$TaskName' thành công!" -ForegroundColor Green
Write-Host "      -> Tự động chạy lại '$ResumeScript' khi Windows khởi động/đăng nhập." -ForegroundColor Gray
Write-Host "      -> Giữ tiến trình chạy liên tục ngay cả khi Laptop chuyển sang dùng Pin." -ForegroundColor Gray
Write-Host "      -> Để gỡ bỏ: powershell -ExecutionPolicy Bypass -File scripts\setup_laptop_autostart.ps1 -Unregister" -ForegroundColor DarkGray
