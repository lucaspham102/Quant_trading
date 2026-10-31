# Quant Trading System — Multi-Sleeve Barbell & 24/7 Live Tournament Engine

Hệ thống giao dịch định lượng đa chiến lược (**Multi-Sleeve Quantitative Portfolio**) trên sàn **Binance (Spot & USDⓈ-M Futures)** kết hợp dữ liệu liên sàn (**Binance, Bybit, OKX**). Hệ thống được thiết kế để vận hành tự động 24/7 trên PC hoặc Laptop với cơ chế chống đứt gãy toàn diện (chống Windows Sleep, tự động khôi phục file trạng thái `.json.bak` khi sập nguồn, tự chờ mạng Wi-Fi/Internet phục hồi, đối soát tài khoản cô lập và báo cáo Telegram HTML thông minh).

- **Trạng thái kiểm thử**: `859 / 859 passed (100%)`
- **Trạng thái giải đấu hiện tại**: **Chu kỳ 105 (`SYNCED`)** — sẵn sàng chạy tiếp ngay lập tức.
- **Tài liệu kiến trúc chuyên sâu**:
  - [Báo cáo Phân tích Chuyên sâu Codebase (`CODEBASE_DEEP_DIVE.md`)](file:///d:/Quant_trading/CODEBASE_DEEP_DIVE.md)
  - [Bản Kiến trúc Toàn diện & Ngữ cảnh Hệ thống (`ARCHITECTURE_CONTEXT_GEMINI.md`)](file:///d:/Quant_trading/ARCHITECTURE_CONTEXT_GEMINI.md)

---

## MỤC LỤC
1. [Tổng Quan Kiến Trúc Chiến Lược & Giải Đấu 4 Profile](#1-tổng-quan-kiến-trúc-chiến-lược--giải-đấu-4-profile)
2. [Yêu Cầu Hệ Thống & Cài Đặt Môi Trường](#2-yêu-cầu-hệ-thống--cài-đặt-môi-trường)
3. [Cấu Hình Khóa API & Telegram (`.env`)](#3-cấu-hình-khóa-api--telegram-env)
4. [Hướng Dẫn Chuyển Sang Laptop & Chạy Tiếp Từ Chu Kỳ 105 (3 Bước)](#4-hướng-dẫn-chuyển-sang-laptop--chạy-tiếp-từ-chu-kỳ-105-3-bước)
5. [Hướng Dẫn Vận Hành Giải Đấu 24/7 & Các Tùy Chọn CLI](#5-hướng-dẫn-vận-hành-giải-đấu-247--các-tùy-chọn-cli)
6. [Cơ Chế Chống Đứt Gãy 24/7 & Báo Cáo Telegram Thông Minh](#6-cơ-chế-chống-đứt-gãy-247--báo-cáo-telegram-thông-minh)
7. [Hướng Dẫn Chạy Backtest, Đánh Giá Holdout & Bot Đơn Lẻ](#7-hướng-dẫn-chạy-backtest-đánh-giá-holdout--bot-đơn-lẻ)
8. [Kiểm Thử Tự Động & Xử Lý Sự Cố Thường Gặp (Troubleshooting)](#8-kiểm-thử-tự-động--xử-lý-sự-cố-thường-gặp-troubleshooting)

---

## 1. Tổng Quan Kiến Trúc Chiến Lược & Giải Đấu 4 Profile

Hệ thống phân bổ gói vốn chuẩn **`$1,000` USDT** (tách biệt `$700` ví Spot + `$300` ví Futures) vào 4 nguồn sinh lời độc lập trên rổ 12 tài sản thanh khoản lớn nhất (`BTC`, `ETH`, `SOL`, `BNB`, `XRP`, `ADA`, `DOGE`, `AVAX`, `DOT`, `LINK`, `MATIC`/`POL`, `LTC`):

1. **Sleeve 1 — Core Delta-Neutral Cash & Carry (`src/dynamic_carry_basis.py`, `src/cross_exchange_carry.py`)**:
   - Mua Spot + Bán khống Perp tỉ lệ 1:1 trên Top 3 đồng coin có điểm Funding Rate & Basis cao nhất (`carry_top_k = 3`, `carry_buffer_rank = 4`, giữ tối thiểu `96h`).
2. **Sleeve 2 — Cross-Sectional Momentum Overlay (`src/barbell_portfolio.py`, `src/cross_sectional.py`)**:
   - Mua các đồng mạnh nhất và bán khống các đồng yếu nhất theo thiết diện ngang (tái cân bằng mỗi `72h`), được bảo vệ bởi cổng phân tán **Dispersion SNR Gate** và ép trung hòa Đô-la sau khi làm tròn lô (`|Long - Short| <= $10`).
3. **Sleeve 3 — Long-Only Triple-Confirmed Trend (`src/directional_trend_sleeve.py`)**:
   - Chỉ giải ngân Long khi xu hướng đồng thuận trên cả 3 khung thời gian (`14d / 21d / 30d`) với `trend_z_entry = 1.25`, đi kèm cầu chì ngắt mạch 2 tầng (**2-Tier Drawdown Circuit Breaker**).
4. **Sleeve 4 — Passive Idle Cash Yield (`src/strategy_profiles.py`)**:
   - Tích lũy lãi suất tiết kiệm linh hoạt trên phần vốn nhàn rỗi (`4.5% APR`).

### Bảng 4 Profile Đang Thi Đấu Trực Tiếp (`scripts/run_parallel_live_demo.py`)

| Profile | Chế độ chạy | Cấu trúc Sleeve | Trạng thái `Trend CB` |
| :--- | :--- | :--- | :--- |
| **`regime_omni_1k`** | **`TESTNET_REAL`** (Đặt lệnh thật trên Binance Testnet) | Carry + XS Overlay + Long-Only Trend | `ACTIVE (Tier 0)` |
| **`flexible_regime_1k`** | `SHADOW_LIVE` (Khớp lệnh giả lập theo sổ lệnh thực) | Carry + XS Overlay + Long-Only Trend (Linh hoạt) | `ACTIVE (Tier 0)` |
| **`carry_max_eta_1k`** | `SHADOW_LIVE` (Khớp lệnh giả lập theo sổ lệnh thực) | Thuần Carry hiệu suất cao + XS Overlay | `N/A (No-Trend)` |
| **`baseline_1k`** | `SHADOW_LIVE` (Khớp lệnh giả lập theo sổ lệnh thực) | Chuẩn đối chiếu 85% Carry / 15% XS Overlay | `N/A (No-Trend)` |

---

## 2. Yêu Cầu Hệ Thống & Cài Đặt Môi Trường

- **Hệ điều hành**: Windows 10 / 11 (khuyến nghị PowerShell 5.1+ hoặc PowerShell 7) hoặc Linux/macOS.
- **Python**: Python `3.11+` (đã kiểm thử toàn diện trên Python `3.14`).

### Cài đặt thư viện:
```powershell
# 1. Di chuyển vào thư mục dự án
cd d:\Quant_trading

# 2. (Tùy chọn) Tạo và kích hoạt môi trường ảo
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Cài đặt các thư viện cần thiết
pip install numpy pandas pyarrow requests ccxt scikit-learn scipy pytest hypothesis python-dotenv
```

---

## 3. Cấu Hình Khóa API & Telegram (`.env`)

Tạo file `.env` tại thư mục gốc của dự án (file `.env` đã được đưa vào `.gitignore` để bảo mật tuyệt đối, không bao giờ bị đẩy lên Git):

```ini
# 1. Khóa API Binance Spot Testnet (https://testnet.binance.vision/)
BINANCE_SPOT_TESTNET_API_KEY=your_64_char_spot_testnet_api_key
BINANCE_SPOT_TESTNET_API_SECRET=your_64_char_spot_testnet_api_secret

# 2. Khóa API Binance USD(S)-M Futures Testnet (https://testnet.binancefuture.com/)
BINANCE_FUTURES_TESTNET_API_KEY=your_64_char_futures_testnet_api_key
BINANCE_FUTURES_TESTNET_API_SECRET=your_64_char_futures_testnet_api_secret

# 3. Cấu hình Bot Telegram nhận báo cáo (Tùy chọn nhưng rất khuyến khích khi treo 24/7)
TELEGRAM_BOT_TOKEN=1234567890:AAH_your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

> **Mẹo kiểm tra nhanh kết nối ví Testnet (Read-Only, không đặt lệnh):**
> ```powershell
> python scripts/check_testnet_equity_breakdown.py
> ```

---

## 4. Hướng Dẫn Chuyển Sang Laptop & Chạy Tiếp Từ Chu Kỳ 105 (3 Bước)

Toàn bộ trạng thái sổ cái, vị thế và đối soát của **Chu kỳ 105 (`SYNCED`)** đã được lưu sẵn trong `data/state/`. Khi chuyển sang Laptop mới để treo 24/7, bạn chỉ cần thực hiện 3 bước:

### Bước 1: Clone / Pull dự án và chép file `.env`
```powershell
git clone https://github.com/lucaspham102/Quant_trading.git
cd Quant_trading
# Nếu đã có thư mục dự án từ trước:
git pull origin master
```
*(Sau đó tạo hoặc copy file `.env` chứa API Key Testnet & Telegram vào thư mục gốc `Quant_trading`).*

### Bước 2: (Tùy chọn) Đăng ký tự động bật lại Giải đấu khi Windows khởi động
Nếu Laptop bị sập nguồn hoặc tự khởi động lại sau khi cập nhật Windows, Scheduled Task này sẽ tự động bật lại giải đấu và tiến trình canh gác Watchdog:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_laptop_autostart.ps1
```
*(Để gỡ bỏ tính năng tự khởi động: `powershell -ExecutionPolicy Bypass -File .\scripts\setup_laptop_autostart.ps1 -Remove`).*

### Bước 3: Khởi chạy tiếp Giải đấu từ Chu kỳ 105
```powershell
powershell -ExecutionPolicy Bypass -File .\resume_tournament.ps1
```
Lệnh trên sẽ:
- Kiểm tra và dọn dẹp các tiến trình giải đấu cũ bị treo (nếu có).
- Mở cửa sổ chạy tiếp giải đấu từ **Chu kỳ 106** (nối tiếp hoàn hảo trạng thái Chu kỳ 105).
- Bật tiến trình canh gác ngầm [`scripts/watchdog_tournament.ps1`](file:///d:/Quant_trading/scripts/watchdog_tournament.ps1) để tự động cứu bot nếu có sự cố treo.

---

## 5. Hướng Dẫn Vận Hành Giải Đấu 24/7 & Các Tùy Chọn CLI

Nếu muốn chạy trực tiếp giải đấu trên cửa sổ Terminal hiện tại với các tham số tùy chỉnh, sử dụng [`scripts/run_parallel_live_demo.py`](file:///d:/Quant_trading/scripts/run_parallel_live_demo.py):

```powershell
# Chạy giải đấu 24/7 (chu kỳ 1 giờ = 3600s, gửi báo cáo định kỳ mỗi 4 chu kỳ + gửi ngay khi có sự kiện)
python scripts/run_parallel_live_demo.py --interval-sec 3600 --tg-report-every 4
```

### Các tham số CLI quan trọng của `scripts/run_parallel_live_demo.py`:
| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `--interval-sec` | `3600` | Khoảng cách giữa 2 chu kỳ tính bằng giây (`3600` = 1 giờ). |
| `--tg-report-every` | `4` | Chu kỳ gửi báo cáo Telegram định kỳ (ví dụ `4` = 4 giờ gửi 1 lần; đặt `1` nếu muốn nhận mỗi giờ, đặt `0` nếu chỉ muốn nhận khi có sự kiện giao dịch/cảnh báo). |
| `--cycles` | `0` | Số chu kỳ muốn chạy (`0` = chạy vô hạn 24/7; đặt `1` nếu chỉ muốn chạy thử 1 chu kỳ rồi dừng). |
| `--capital` | `1000.0` | Quy mô vốn chuẩn của mỗi profile (`$1,000` USDT). |
| `--state-dir` | `data/state` | Thư mục lưu sổ cái `.json`, bản sao `.bak` và `tournament_heartbeat.json`. |
| `--log-dir` | `logs` | Thư mục lưu nhật ký kiểm toán `demo_Scoreboard.jsonl` và báo cáo Markdown. |
| `--no-send-orders` | `False` | Nếu thêm cờ này, cả 4 profile đều chạy ở chế độ `SHADOW_LIVE` (không bắn lệnh lên Testnet). |

---

## 6. Cơ Chế Chống Đứt Gãy 24/7 & Báo Cáo Telegram Thông Minh

1. **Ngăn Windows tự Sleep (`prevent_windows_sleep`)**:
   - Tự động gọi `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` khi khởi chạy và đầu mỗi chu kỳ để ngăn Laptop rơi vào chế độ Sleep khi không có thao tác chuột/bàn phím.
2. **Tự động khôi phục khi sập nguồn (`fsync` + `.json.bak` Auto-Recovery)**:
   - Trước khi ghi đè bất kỳ file trạng thái `.json` nào trong `data/state/`, hệ thống đều ép ghi xuống đĩa vật lý (`os.fsync`) và lưu lại bản sao `.json.bak`. Nếu mất điện đột ngột làm hỏng file `.json` chính, lần khởi động kế tiếp sẽ tự động khôi phục từ `.json.bak` và ghi log `WARNING: RECOVERED_STATE_FROM_BACKUP`.
3. **Tự động chờ mạng Wi-Fi/Internet phục hồi (`wait_for_network_connectivity`)**:
   - Khi mất mạng đầu chu kỳ, bot không đánh dấu lỗi ngay mà chuyển Heartbeat sang `WAITING_FOR_NETWORK` và thử kết nối lại mỗi 15 giây (tối đa 5 phút). Khi có mạng lại, bot tự động chạy tiếp chu kỳ và báo `[NETWORK RECOVERED]` qua Telegram.
4. **5 Mốc Gửi Báo Cáo Telegram Thông Minh (`src/notifier.py`)**:
   - **Startup / Resume**: Gửi ngay ở chu kỳ đầu tiên khi vừa bật bot hoặc sau khi Watchdog khởi động lại.
   - **Trade Executed / Position Changed**: Gửi ngay khi `regime_omni_1k` có phát sinh lệnh khớp mới hoặc thay đổi số lượng vị thế.
   - **Network Recovered**: Gửi ngay khi vừa tự phục hồi sau sự cố mất mạng.
   - **Risk / Reconcile Alert**: Gửi ngay khi trạng thái đối soát khác `SYNCED`, hoặc Circuit Breaker kích hoạt, hoặc có lỗi thực thi.
   - **Periodic / Daily Milestone**: Gửi định kỳ mỗi `N` chu kỳ (`--tg-report-every`) và mốc chuyển ngày mới (`00:00 UTC`).

---

## 7. Hướng Dẫn Chạy Backtest, Đánh Giá Holdout & Bot Đơn Lẻ

### 7.1. Chạy Backtest Giải đấu các Profile `$1,000` (2022–2026)
```powershell
python scripts/run_1k_profile_tournament_backtest.py
```

### 7.2. Kiểm định Out-of-Sample trên dữ liệu gần nhất (90 & 180 ngày)
```powershell
python scripts/evaluate_recent_holdout.py
```

### 7.3. Chạy Kiểm tra Tiền trạm 6 Cổng An Toàn (Preflight Check)
```powershell
python scripts/run_testnet_preflight.py
```

### 7.4. Chạy Bot Đơn Lẻ (`run_live.py`) hoặc Backtest Cơ Sở (`run_backtest.py`)
```powershell
# Chạy thử 1 chu kỳ ở chế độ Shadow-Live (chỉ đọc dữ liệu thực, không đặt lệnh)
python run_live.py --mode shadow-live --strategy barbell --capital 3000 --once

# Chạy backtest đơn chiến lược
python run_backtest.py
```

---

## 8. Kiểm Thử Tự Động & Xử Lý Sự Cố Thường Gặp (Troubleshooting)

### Chạy toàn bộ 859 bài kiểm thử tự động:
```powershell
# Chạy toàn bộ test suite (859 passed)
pytest -q

# Chạy riêng bộ kiểm thử chống đứt gãy, tự phục hồi .bak, UI Terminal & Telegram
pytest tests/test_anti_hang_resilience.py -v
```

### Xử lý sự cố thường gặp:
- **Lỗi thiếu API Key hoặc báo `MISSING_CREDENTIALS`**:
  - Kiểm tra lại file `.env` ở thư mục gốc xem đã điền đủ 4 biến `BINANCE_SPOT_TESTNET_API_KEY`, `BINANCE_SPOT_TESTNET_API_SECRET`, `BINANCE_FUTURES_TESTNET_API_KEY`, `BINANCE_FUTURES_TESTNET_API_SECRET` (đúng 64 ký tự hex, không dư khoảng trắng hoặc dấu ngoặc kép) hay chưa.
- **Muốn kiểm tra xem giải đấu có đang chạy ngầm hay không**:
  - Xem file `data/state/tournament_heartbeat.json` hoặc kiểm tra dòng cuối cùng của `logs/demo_Scoreboard.jsonl`:
    ```powershell
    Get-Content data\state\tournament_heartbeat.json
    Get-Content logs\demo_Scoreboard.jsonl -Tail 4
    ```
- **Muốn dừng hẳn giải đấu để bảo trì hoặc tắt máy**:
  - Đóng cửa sổ PowerShell của giải đấu và Watchdog, hoặc chạy lệnh dừng tiến trình an toàn:
    ```powershell
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_parallel_live_demo|watchdog_tournament' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    ```
