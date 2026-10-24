# HỆ THỐNG GIAO DỊCH ĐỊNH LƯỢNG BARBELL DUAL-ALPHA (TIER 1 CARRY + TIER 2 OVERLAY)
## BÁO CÁO PHÂN TÍCH TOÀN DIỆN VÀ HƯỚNG DẪN KIẾN TRÚC CODEBASE TỪNG CHÂN TƠ KẼ TÓC

> **Phiên bản:** Sprint 22 Production Ready  
> **Thị trường mục tiêu:** Binance Spot & USDⓈ-M Futures (Hỗ trợ mở rộng Bybit & OKX)  
> **Kiến trúc cốt lõi:** Barbell Capital Allocation (Tier 1 Core Cash & Carry 85–90% + Tier 2 Tactical Cross-Sectional Momentum Alpha Overlay 10–15%)  
> **Nguyên tắc kỹ thuật:** Fail-Closed Security, Zero-Compounding Leak Invariance, UTC Epoch Rebalance Phase Anchoring, Inside-Spread Smart Maker, Multi-Venue Signed Reconciler.

---

## MỤC LỤC
1. [Tổng Quan Hệ Thống & Lịch Sử Tiến Hóa Định Lượng](#1-tổng-quan-hệ-thống--lịch-sử-tiến-hóa-định-lượng)
2. [Cây Thư Mục Toàn Diện & Vai Trò Từng Thư Mục](#2-cây-thư-mục-toàn-diện--vai-trò-từng-thư-mục)
3. [5 Định Luật Bất Biến Định Lượng (The 5 Golden Quant Invariants)](#3-5-định-luật-bất-biến-định-lượng-the-5-golden-quant-invariants)
4. [Bản Đồ Chi Tiết Từng Module Trong `src/` (Làm Gì & Làm Như Thế Nào)](#4-bản-đồ-chi-tiết-từng-module-trong-src-làm-gì--làm-như-thế-nào)
   - [4.1. `src/config.py` — Tham Số & Thiết Lập Vũ Trụ Giao Dịch](#41-srcconfigpy--tham-số--thiết-lập-vũ-trụ-giao-dịch)
   - [4.2. `src/quant_invariants.py` — Kiểm Định Bất Biến Định Lượng](#42-srcquant_invariantspy--kiểm-định-bất-biến-định-lượng)
   - [4.3. `src/data_loader.py` — Pipeline Tải & Chuẩn Hóa OHLCV Futures](#43-srcdata_loaderpy--pipeline-tải--chuẩn-hóa-ohlcv-futures)
   - [4.4. `src/funding_loader.py` — Funding Rate & Khớp Spot-Perp Basis](#44-srcfunding_loaderpy--funding-rate--khớp-spot-perp-basis)
   - [4.5. `src/multi_exchange_loader.py` — Funding Rate Đa Sàn (Binance, Bybit, OKX)](#45-srcmulti_exchange_loaderpy--funding-rate-đa-sàn-binance-bybit-okx)
   - [4.6. `src/features.py` — Feature Engineering Vector Hóa Không Look-Ahead](#46-srcfeaturespy--feature-engineering-vector-hóa-không-look-ahead)
   - [4.7. `src/strategy.py` — Bộ Động Cơ Chiến Lược Định Hướng Directional](#47-srcstrategypy--bộ-động-cơ-chiến-lược-định-hướng-directional)
   - [4.8. `src/carry_strategy.py` — Funding Carry Cơ Bản & MTM Bar-by-Bar](#48-srccarry_strategypy--funding-carry-cơ-bản--mtm-bar-by-bar)
   - [4.9. `src/cross_sectional.py` — Cross-Sectional Momentum Market-Neutral](#49-srccross_sectionalpy--cross-sectional-momentum-market-neutral)
   - [4.10. `src/dynamic_carry_basis.py` — Dynamic Carry & Hội Tụ Basis](#410-srcdynamic_carry_basispy--dynamic-carry--hội-tụ-basis)
   - [4.11. `src/cross_exchange_carry.py` — Carry Đa Sàn & Lượng Hóa Khối Lượng](#411-srccross_exchange_carrypy--carry-đa-sàn--lượng-hóa-khối-lượng)
   - [4.12. `src/barbell_portfolio.py` — Trái Tim Danh Mục Barbell Portfolio](#412-srcbarbell_portfoliopy--trái-tim-danh-mục-barbell-portfolio)
   - [4.13. `src/risk_manager.py` — Quản Trị Vị Thế & Circuit Breaker](#413-srcrisk_managerpy--quản-trị-vị-thế--circuit-breaker)
   - [4.14. `src/ml_filter.py` — Meta-Labeling & Purged Walk-Forward](#414-srcml_filterpy--meta-labeling--purged-walk-forward)
   - [4.15. `src/backtester.py` — Engine Khớp Lệnh & Mô Phỏng Lịch Sử](#415-srcbacktesterpy--engine-khớp-lệnh--mô-phỏng-lịch-sử)
   - [4.16. `src/state_store.py` — Quản Lý Trạng Thái Nguyên Tử (Atomic State)](#416-srcstate_storepy--quản-lý-trạng-thái-nguyên-tử-atomic-state)
   - [4.17. `src/binance_public_feed.py` — Public Feed Cho Chế Độ Shadow-Live](#417-srcbinance_public_feedpy--public-feed-cho-chế-độ-shadow-live)
   - [4.18. `src/binance_signed_client.py` — HMAC-SHA256 Client & Smart Maker](#418-srcbinance_signed_clientpy--hmac-sha256-client--smart-maker)
   - [4.19. `src/exchange_reconciler.py` — Đối Soát Vị Thế & Sức Khỏe Ký Quỹ](#419-srcexchange_reconcilerpy--đối-soát-vị-thế--sức-khỏe-ký-quỹ)
   - [4.20. `src/carry_live_executor.py` — Live Daemon Engine Vận Hành 24/7](#420-srccarry_live_executorpy--live-daemon-engine-vận-hành-247)
   - [4.21. `src/notifier.py` — Hệ Thống Cảnh Báo Telegram Chịu Lỗi](#421-srcnotifierpy--hệ-thống-cảnh-báo-telegram-chịu-lỗi)
5. [CLI Entrypoint `run_live.py` & Các Scripts Nghiên Cứu](#5-cli-entrypoint-run_livepy--các-scripts-nghiên-cứu)
6. [Dòng Chảy Dữ Liệu & Quy Trình Khớp Lệnh Đa Tầng (End-to-End Pipeline)](#6-dòng-chảy-dữ-liệu--quy-trình-khớp-lệnh-đa-tầng-end-to-end-pipeline)
7. [Các Cạm Bẫy Đã Được Khắc Phục & Bài Học Thực Chiến (Battle-Tested Fixes)](#7-các-cạm-bẫy-đã-được-khắc-phục--bài-học-thực-chiến-battle-tested-fixes)
8. [Hướng Dẫn Vận Hành Hệ Thống 24/7 (Runbook)](#8-hướng-dẫn-vận-hành-hệ-thống-247-runbook)

---

## 1. TỔNG QUAN HỆ THỐNG & LỊCH SỬ TIẾN HÓA ĐỊNH LƯỢNG

Hệ thống bắt đầu từ mục tiêu xây dựng một trading bot tự động cho crypto, nhưng qua quá trình nghiên cứu thực nghiệm nghiêm ngặt, codebase đã trải qua **22 Sprint phát triển** để chuyển mình từ một bot giao dịch kỹ thuật đơn giản thành một **nền tảng định lượng cấp tổ chức (institutional-grade quantitative platform)**.

```mermaid
flowchart LR
    A["Sprint 1 - 5:<br/>Directional Alpha<br/>(Donchian/ATR/ML)"] -->|Phí giao dịch & false breakout ăn mòn PnL| B["Sprint 6 - 8:<br/>Delta-Neutral Pivot<br/>(Cash & Carry + Dynamic Basis)"]
    B -->|Mở rộng đa sàn & quản trị rủi ro| C["Sprint 9 - 11:<br/>Barbell Architecture<br/>(Core Yield + XS Momentum Overlay)"]
    C -->|Thực thi an toàn & đối soát sàn thật| D["Sprint 12 - 16:<br/>Production Hardening<br/>(Dollar Neutrality + Audit JSONL)"]
    D -->|Khắc phục lệch pha đồng hồ & Smart Maker| E["Sprint 17 - 22:<br/>Live Testnet/Mainnet Guard<br/>(Inside-Spread + Epoch Clock)"]
```

### 1.1. Tại sao từ bỏ Directional Breakout thuần túy?
Trong Sprint 1 đến Sprint 5, hệ thống thử nghiệm các chiến lược phá vỡ kênh Donchian kết hợp lọc biến động ATR, Regime EMA và mô hình Machine Learning Meta-Labeling. Mặc dù trên dữ liệu thô chưa trừ phí chiến lược mang lại lợi nhuận, nhưng khi áp dụng mô hình chi phí thực tế (Taker fee $0.05\%$ + Slippage $0.02\%$ mỗi chiều):
- **Phí giao dịch ăn mòn toàn bộ alpha**: Chi phí khứ hồi $14\text{ bps}$ trên mỗi lệnh lặp đi lặp lại khiến kỳ vọng toán học chuyển sang âm.
- **Rủi ro đuôi (Fat tails) và biến động bất đối xứng**: Các đợt quét thanh khoản (liquidity sweeps) kích hoạt SL trước khi thị trường chạy theo xu hướng.
- **Chẩn đoán Newey-West HAC**: Kết quả phân tích trong `diagnostics.py` và `deep_analysis.py` chỉ ra rằng lợi nhuận của Directional phụ thuộc gần như $100\%$ vào Beta thị trường, $\alpha \approx 0$.

### 1.2. Bước ngoặt Barbell Portfolio
Từ Sprint 6 trở đi, hệ thống chuyển hướng sang kiến trúc **Barbell Portfolio**:
1. **Tier 1 — Core Yield Sleeve (Chiếm 85% – 90% NAV)**:
   - Bản chất: **Delta-Neutral Cash & Carry** (Long Spot + Short Perp hoặc Short Coin-M Perp).
   - Nguồn lợi nhuận: Thu phí **Funding Rate** được trả định kỳ mỗi 8 giờ từ các nhà giao dịch đòn bẩy quá mức, cộng thêm lợi nhuận từ sự **hội tụ chênh lệch giá (Basis Convergence)** giữa Spot và Perp.
   - Đặc tính: Lợi nhuận ổn định, biến động danh mục cực thấp, Sharpe Ratio lịch sử $> 3.5$, Drawdown $< 1.5\%$.
2. **Tier 2 — Tactical Alpha Overlay (Chiếm 10% – 15% NAV tận dụng Unified Margin)**:
   - Bản chất: **Cross-Sectional Market-Neutral Momentum** (Long đồng coin mạnh nhất, Short đồng coin yếu nhất trong rổ thanh khoản cao).
   - Tận dụng vốn: Không cần ký quỹ thêm bằng tiền mặt mới; chạy trực tiếp trên hạn mức ký quỹ dư thừa của tài khoản ký quỹ thống nhất (Unified Margin).
   - Bộ lọc rủi ro: Chỉ mở vị thế khi độ phân tán tín hiệu trên nhiễu ($SNR\text{ Dispersion}$) vượt ngưỡng, tự động co cụm đòn bẩy bằng **Pair Volatility Targeting**, và tự ngắt qua **Drawdown Cushion**.

---

## 2. CÂY THƯ MỤC TOÀN DIỆN & VAI TRÒ TỪNG THƯ MỤC

```
D:\Quant_trading
├── .agy/                     # Nhật ký thiết kế, kiến trúc sư & audit locks của AGY
├── data/
│   ├── raw/                  # Kho lưu trữ Parquet OHLCV Spot, Perp & Funding rates
│   ├── state/                # File trạng thái JSON nguyên tử của bot (bot_state.json, baseline)
│   ├── bridge_*/             # Dữ liệu bridge kiểm định qua các sprint
│   └── forensics_*/          # Bằng chứng phân tích sai lệch và đối soát
├── docs/                     # 38 tài liệu báo cáo kỹ thuật từ Sprint 3 đến Sprint 21
├── logs/                     # File log vận hành daemon và audit trails JSONL
├── scripts/                  # 11 scripts tiện ích, diễn tập sự cố (fire drill) & burn-in
├── src/                      # MÃ NGUỒN CỐT LÕI (21 modules Python)
│   ├── config.py             # Cấu hình tham số, vũ trụ coin, fail-fast validations
│   ├── quant_invariants.py   # 5 định luật bất biến định lượng bảo vệ hệ thống
│   ├── data_loader.py        # Pipeline nến OHLCV Binance USDⓈ-M Futures
│   ├── funding_loader.py     # Pipeline funding rate và khớp Spot-Perp Basis
│   ├── multi_exchange_loader.py # Pipeline funding rates đa sàn (Binance, Bybit, OKX)
│   ├── features.py           # Feature engineering vector hóa chuẩn không look-ahead
│   ├── strategy.py           # Chiến lược kỹ thuật Directional (Breakout/Pullback/MR)
│   ├── carry_strategy.py     # Chiến lược Cash & Carry cơ bản + MTM Bar-by-bar
│   ├── cross_sectional.py    # Chiến lược Cross-Sectional Momentum Market-Neutral
│   ├── dynamic_carry_basis.py# Dynamic Carry + Khai thác hội tụ Basis Spot-Perp
│   ├── cross_exchange_carry.py # Carry liên sàn (Perp-Perp, Best-Venue) & Lượng hóa Lot
│   ├── barbell_portfolio.py  # Trái tim danh mục Barbell (Sleeve + Overlay + SNR Gate)
│   ├── risk_manager.py       # Quản trị rủi ro, Sizing, Clamping, Circuit Breaker
│   ├── ml_filter.py          # Machine Learning Meta-Labeling & Purged Walk-Forward
│   ├── backtester.py         # Backtest engine chi tiết đến từng nến, khớp next-bar open
│   ├── state_store.py        # Lưu trữ trạng thái bot nguyên tử (Atomic StateStore)
│   ├── binance_public_feed.py# Public feed REST read-only phục vụ shadow mode
│   ├── binance_signed_client.py # Client ký HMAC-SHA256, Dual-venue Auth, Smart Maker
│   ├── exchange_reconciler.py# Động cơ đối soát sàn vs trạng thái local, đo Margin Health
│   ├── carry_live_executor.py# Daemon Live Executor 24/7, xử lý Maker-Taker pair
│   └── notifier.py           # Cảnh báo Telegram chịu lỗi cao kèm fallback
├── tests/                    # 63 file test với gần 700 ca kiểm thử tự động
├── run_live.py               # CLI Daemon Runner chính cho Live/Paper/Testnet/Shadow
├── run_backtest.py           # CLI chạy backtest chiến lược directional
├── run_deep_analysis.py      # CLI phân tích OLS Newey-West, Bootstrap, Exposure
├── run_diagnostics.py        # CLI chẩn đoán alpha vs beta, breakeven fees
├── run_walkforward.py        # CLI chạy kiểm định Walk-Forward tối ưu tham số
├── run_sprint*.py            # Các runner thực nghiệm cho từng sprint riêng biệt
└── .env.testnet              # File cấu hình khóa API Testnet (Spot & Futures độc lập)
```

---

## 3. 5 ĐỊNH LUẬT BẤT BIẾN ĐỊNH LƯỢNG (THE 5 GOLDEN QUANT INVARIANTS)

Được định nghĩa tập trung trong [`src/quant_invariants.py`](file:///d:/Quant_trading/src/quant_invariants.py), 5 định luật bất biến này là chốt chặn toán học ngăn ngừa toàn bộ các lỗi giả lập (quant artifacts) và rò rỉ dữ liệu:

```mermaid
graph TD
    Inv1["1. Warmup Saturation Invariant<br/>(Jaccard >= 0.98, L1 diff <= 0.02)"]
    Inv2["2. Zero Compounding Leak Invariant<br/>(Max Abs Diff <= 1e-6)"]
    Inv3["3. Rebalance Clock Phase Shift Invariant<br/>(Jaccard = 1.0000)"]
    Inv4["4. Accounting Identity Invariant<br/>(Net == Gross - Fees, Tol <= 1e-9)"]
    Inv5["5. Lot Neutrality & Delta Invariant<br/>(Imbalance <= 10 USD, Zero Naked Legs)"]
```

### 3.1. Warmup Saturation Invariant (`verify_warmup_saturation_invariant`)
- **Vấn đề**: Khi tính toán EMA, ATR, hoặc các chỉ báo rolling, nếu cửa sổ khởi động (warmup) quá ngắn hoặc bị cắt lát dữ liệu khác nhau, các giá trị số học ở thời điểm bắt đầu đánh giá $t_0$ sẽ bị lệch (undershoot), dẫn đến việc lựa chọn tài sản bị sai lệch hoàn toàn.
- **Quy tắc**: So sánh trọng số danh mục sinh ra tại $t_0$ giữa 2 cửa sổ warmup chuẩn ($720\text{h}$) và gấp đôi ($1440\text{h}$).
  $$\text{Jaccard}(\text{Active\_Signals}_{720}, \text{Active\_Signals}_{1440}) \ge 0.98$$
  $$\sum_{i} |w_{i, 720} - w_{i, 1440}| \le 0.02$$
- Nếu vi phạm: Ném lỗi `BLOCKER: QUANT_ARTIFACT: WARMUP_UNDERSHOOT`.

### 3.2. Zero Compounding Leak Invariant (`verify_zero_compounding_leak_invariant`)
- **Vấn đề**: Trong nghiên cứu backtest khi đặt `reinvest_profits=False`, nhiều hệ thống vô tình tính tỷ lệ vị thế theo `current_equity` tăng dần, khiến lợi nhuận sinh ra bị thổi phồng bởi lãi kép ngầm.
- **Quy tắc**: Chuỗi PnL tính bằng USD mỗi kỳ của hai lần chạy với số vốn thặng dư khác nhau nhưng cùng `initial_capital` phải khớp tuyệt đối:
  $$\max |PnL_{\text{low\_eq}, t} - PnL_{\text{high\_eq}, t}| \le 10^{-6}$$
- Nếu vi phạm: Ném lỗi `BLOCKER: QUANT_ARTIFACT: COMPOUNDING_CAPITAL_LEAK`.

### 3.3. Rebalance Clock Phase Shift Invariant (`verify_clock_shift_invariance`)
- **Vấn đề**: Chiến lược tái cân bằng mỗi $48\text{h}$. Nếu dùng phép chia lấy dư theo index cục bộ `local_idx % 48 == 0`, khi dữ liệu bị cắt lát (ví dụ cắt ngắn để test holdout hoặc live bot khởi động lại), nhịp đồng hồ bị trôi pha khiến bot mua bán vào những ngày hoàn toàn khác với backtest.
- **Quy tắc**: Lịch tái cân bằng phải được neo vào mốc tuyệt đối (**UTC Epoch Timestamp**). Tập hợp các timestamp tái cân bằng giữa 2 lần chạy có chiều dài lịch sử khác nhau phải trùng khớp $100\%$:
  $$\text{Jaccard}(\text{Rebal\_Timestamps}_A, \text{Rebal\_Timestamps}_B) = 1.0000$$
- Nếu vi phạm: Ném lỗi `BLOCKER: QUANT_ARTIFACT: REBALANCE_CLOCK_PHASE_SHIFT`.

### 3.4. Accounting Identity Invariant (`verify_accounting_identity_invariant`)
- **Vấn đề**: Sai lệch kế toán, tính trùng phí (double-fee) hoặc bỏ sót phí giao dịch/funding giữa các nhánh.
- **Quy tắc**: Đẳng thức kế toán đóng tuyệt đối:
  $$\left| \text{Net\_PnL} - \left( \sum \text{Sleeve\_Gross\_PnL} - \sum \text{Fee\_Components} \right) \right| \le 10^{-9}$$
- Nếu vi phạm: Ném lỗi `BLOCKER: QUANT_ARTIFACT: ACCOUNTING_IDENTITY_VIOLATION`.

### 3.5. Lot Neutrality & Delta Leak Invariant (`verify_lot_neutrality_invariant`)
- **Vấn đề**: Trên Binance, mỗi cặp coin có `step_size` khác nhau (ví dụ: $0.001$ BTC $\approx \$96$, trong khi $0.1$ SOL $\approx \$18$). Nếu làm tròn độc lập từng chân, chân Short có thể lớn hơn chân Long tới hàng chục USD, hoặc tệ hơn là một chân bị làm tròn về $0$ tạo thành vị thế trần trụi (**Naked Leg**).
- **Quy tắc**:
  1. Tuyệt đối không có chân nào bằng $0$ khi chân kia $> 0$ (Chống Naked Leg).
  2. Độ lệch danh nghĩa sau khi làm tròn phải nằm trong giới hạn cho phép:
  $$|\text{Leg}_A - \text{Leg}_B| \le \$10.0$$
- Nếu vi phạm: Ném lỗi `BLOCKER: QUANT_ARTIFACT: LOT_QUANTIZATION_DELTA_LEAK`.

---

## 4. BẢN ĐỒ CHI TIẾT TỪNG MODULE TRONG `src/` (LÀM GÌ & LÀM NHƯ THẾ NÀO)

### 4.1. `src/config.py` — Tham Số & Thiết Lập Vũ Trụ Giao Dịch
- **Làm gì**: Định nghĩa lớp dữ liệu bất biến `@dataclass(frozen=True) TradingConfig`, danh sách các bộ vũ trụ coin và các hàm tính toán chu kỳ nến theo thời gian.
- **Làm như thế nào**:
  - `CORE_4_SYMBOLS`: `["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"]`.
  - `EXTRA_8_SYMBOLS`: Rổ 8 đồng thanh khoản cao bổ sung gồm `XRP`, `DOGE`, `ADA`, `AVAX`, `LINK`, `DOT`, `LTC`, `TRX`.
  - `UNIVERSE_12_SYMBOLS`: Hợp của Core 4 và Extra 8.
  - Lớp `TradingConfig` thiết lập toàn bộ tham số mặc định: vốn ban đầu ($400$ USDT cho directional hoặc $3000$ USDT cho barbell), phí taker $0.05\%$, maker $0.02\%$, trượt giá $0.02\%$, đòn bẩy tối đa $2.0\text{x}$.
  - Cơ chế **Fail-Fast `__post_init__`**: Ngăn chặn cấu hình vô hiệu, ví dụ nếu bật Trailing Stop mà khoảng cách kích hoạt $\ge$ khoảng cách chốt lời ($D_{\text{act}} \ge TP$) sẽ ném `ValueError` ngay lập tức.
  - Hàm `candles_per_year_for_timeframe(timeframe)`: Tính số nến chuẩn trong năm ($365 \times 24 \times 4 = 35,040$ nến $15\text{m}$; $8,760$ nến $1\text{h}$).

### 4.2. `src/quant_invariants.py` — Kiểm Định Bất Biến Định Lượng
- **Làm gì**: Chứa mã nguồn thực thi 5 hàm kiểm định bất biến đã mô tả tại [Mục 3](#3-5-định-luật-bất-biến-định-lượng-the-5-golden-quant-invariants).
- **Làm như thế nào**: Sử dụng thư viện NumPy, Pandas để so sánh khoảng cách ma trận L1, hệ số tương đồng Jaccard trên nhãn tín hiệu, và kiểm tra sai số dấu phẩy động với dung sai khắt khe ($10^{-6}$ đến $10^{-9}$).

### 4.3. `src/data_loader.py` — Pipeline Tải & Chuẩn Hóa OHLCV Futures
- **Làm gì**: Tải dữ liệu lịch sử nến từ Binance Futures qua CCXT hoặc FakeExchange (dùng cho test), chuẩn hóa cấu trúc dữ liệu và lưu trữ cache cục bộ dạng Parquet.
- **Làm như thế nào**:
  - `raw_ohlcv_to_dataframe()`: Chuyển dữ liệu mảng thô sang DataFrame, ép kiểu sang `float64`, đặt `DatetimeIndex` với múi giờ chuẩn hóa UTC nanosecond (`tz='UTC', as_unit('ns')`), đổi tên cột index thành `timestamp`.
  - `drop_incomplete_last_candle()`: Loại bỏ nến cuối cùng nếu thời gian mở nến cộng khung thời gian lớn hơn thời gian hiện tại của server sàn (chống rò rỉ dữ liệu nến chưa đóng hoàn toàn).
  - `merge_and_deduplicate_ohlcv()`: Hợp nhất dữ liệu mới vào dữ liệu cũ, deduplicate theo timestamp với cơ chế `keep='last'`, bảo toàn thuộc tính metadata `attrs['symbol']` và `attrs['timeframe']`.
  - `DataLoader.update_incremental()`: Kiểm tra cache Parquet, chỉ tải phần nến còn thiếu tính từ timestamp cuối cùng của file cache, sau đó ghi đè nguyên tử.

### 4.4. `src/funding_loader.py` — Funding Rate & Khớp Spot-Perp Basis
- **Làm gì**: Thu thập lịch sử Funding Rate của Binance (mỗi 8h: 00:00, 08:00, 16:00 UTC), tải lịch sử nến Spot tương ứng và ghép nối với nến Perp để tính toán Basis.
- **Làm như thế nào**:
  - `fetch_funding_history()`: Tải phân trang funding rate (tối đa 1000 bản ghi/lần gọi) và lưu vào `data/raw/<symbol>_funding.parquet`.
  - `align_ohlcv_with_funding()`: Khớp nến OHLCV $1\text{h}$ với Funding Rate mà **hoàn toàn không có look-ahead bias**:
    * `funding_rate_event`: Chỉ nhận giá trị funding rate tại đúng cây nến đóng thanh toán ($00, 08, 16\text{ UTC}$), các nến khác bằng $0.0$.
    * `last_known_funding_rate`: Sử dụng `pd.merge_asof(..., direction='backward')` để tại nến $t$, chiến lược chỉ biết tỷ lệ funding gần nhất đã chốt tại thời điểm $\le t$.
  - `align_spot_perp_funding()`: Đồng bộ hóa đồng thời nến Spot Close và Perp Close tại cùng timestamp, tính toán phần trăm chênh lệch Basis:
    $$\text{basis\_pct}_t = \frac{\text{perp\_close}_t - \text{spot\_close}_t}{\text{spot\_close}_t}$$

### 4.5. `src/multi_exchange_loader.py` — Funding Rate Đa Sàn (Binance, Bybit, OKX)
- **Làm gì**: Mở rộng khả năng thu thập dữ liệu funding rate từ 3 sàn giao dịch lớn nhất thế giới để phục vụ chiến lược chênh lệch lãi suất liên sàn (Sprint 9).
- **Làm như thế nào**:
  - `clean_symbol_name()`: Chuẩn hóa mọi ký hiệu cặp tiền (ví dụ `BTC/USDT:USDT`, `BTC_USDT_USDT`) về dạng chuẩn liền mạch `BTCUSDT`.
  - `fetch_exchange_funding_history()`: Xử lý sự khác biệt API giữa các sàn:
    * Binance: `limit=1000`.
    * Bybit: `limit=200`, yêu cầu truyền cửa sổ thời gian `startTime` và `endTime`.
    * OKX: `limit=100`.
  - `align_multi_exchange_funding()`: Đồng bộ funding của cả 3 sàn vào một DataFrame thống nhất trên trục nến $1\text{h}$. Xác định sàn có funding cao nhất (`best_short_ex`) và thấp nhất (`best_long_ex`) để tính `cross_ex_spread`.

### 4.6. `src/features.py` — Feature Engineering Vector Hóa Không Look-Ahead
- **Làm gì**: Tính toán toàn bộ các chỉ báo kỹ thuật và đặc trưng vi cấu trúc (microstructure) thị trường bằng mã thuần Pandas/NumPy vector hóa cao tốc.
- **Làm như thế nào**:
  - `compute_ema()`: Sử dụng `ewm(span=period, adjust=False, min_periods=period)`.
  - `compute_atr()`: ATR chuẩn Wilder (RMA) tính trên True Range bằng hàm `ewm(alpha=1/period, adjust=False)`.
  - `compute_rsi()`: RSI chuẩn Wilder. Xử lý an toàn mẫu số $0$: nếu `avg_loss == 0`, RSI ép về $100.0$ (nếu cả gain và loss $= 0$ thì về $50.0$).
  - `compute_donchian()`: Dùng `high.shift(1).rolling(window).max()` và `low.shift(1).rolling(window).min()`. **Bắt buộc `.shift(1)` trước khi rolling** để loại trừ đỉnh/đáy của chính cây nến hiện tại.
  - `compute_candle_microstructure()`: Tính tỷ lệ thân nến `body_to_range_ratio`, râu nến trên `upper_wick_ratio`, râu nến dưới `lower_wick_ratio` trên biên độ $[low, high]$. Khi $high == low$, gán an toàn $= 0.0$.
  - `compute_bb_width_squeeze()`: Đo độ rộng dải Bollinger: $(upper - lower) / mid$.
  - `compute_time_cyclical()`: Mã hóa chu kỳ 24 giờ UTC thành 2 đặc trưng tuần hoàn: $\sin(2\pi \cdot hour / 24)$ và $\cos(2\pi \cdot hour / 24)$.
  - `compute_adx()`: Tính ADX, +DI, -DI chuẩn Wilder.

### 4.7. `src/strategy.py` — Bộ Động Cơ Chiến Lược Định Hướng Directional
- **Làm gì**: Sinh tín hiệu giao dịch mua/bán ($1, -1, 0$) và mức Stop Loss, Take Profit ban đầu.
- **Làm như thế nào**:
  - Hỗ trợ 4 chế độ: `breakout` (Donchian + Volume Z-Score + Regime), `trend_pullback` (hồi quy về EMA20/Bollinger Mid trong xu hướng ADX $> 25$), `mean_reversion` (đảo chiều khi chạm biên Bollinger ngoài và RSI quá mua/quá bán trong thị trường đi ngang ADX $< 25$), và `regime_adaptive` (tự động chuyển đổi giữa pullback và mean reversion).
  - Cơ chế **Ratchet Trailing Stop** (`compute_trailing_stop`): Chỉ dời Stop Loss theo hướng có lợi cho vị thế, tuyệt đối không bao giờ nới lỏng rủi ro.
  - Cơ chế **Breakeven Stop** (`compute_breakeven_price`): Dời SL về điểm hòa vốn đã cộng bù phí Taker 2 chiều và trượt giá 2 chiều ngay khi giá đạt lợi nhuận $1.0\text{R}$.

### 4.8. `src/carry_strategy.py` — Funding Carry Cơ Bản & MTM Bar-by-Bar
- **Làm gì**: Động cơ mô phỏng danh mục Cash & Carry trên rổ coin với tính toán Mark-to-Market (MTM) từng nến, theo dõi ký quỹ và tái cân bằng đệm rủi ro.
- **Làm như thế nào**:
  - Hỗ trợ 2 cấu trúc vị thế:
    * `coinm_1leg`: Dùng chính coin làm tài sản thế chấp và Short Perp Coin-M (hiệu suất vốn $\eta = 1.0$).
    * `spot_perp_2leg`: Mua Spot và Short Perp USDT-M (hiệu suất vốn $\eta = 2/3$).
  - 2 chế độ vào lệnh: `hysteresis` (vào khi funding EWMA $> 1\text{ bps}$, thoát khi $< 0$, giữ tối thiểu 3 kỳ) và `passive_long_term` (vào khi $> 0.5\text{ bps}$, thoát khi $< -0.2\text{ bps}$, giữ tối thiểu 21 kỳ).
  - Tự động kích hoạt **Margin Rebalance** khi tỷ lệ ký quỹ duy trì chạm ngưỡng $5\%$, hoặc **Price Rebalance** khi giá biến động $\pm 40\%$.

### 4.9. `src/cross_sectional.py` — Cross-Sectional Momentum Market-Neutral
- **Làm gì**: Động cơ chiến lược Cross-Sectional Momentum hoàn toàn trung tính với thị trường (Dollar-Neutral).
- **Làm như thế nào**:
  - Tại mỗi chu kỳ tái cân bằng (mặc định $24\text{h}$ hoặc $48\text{h}$):
    1. Tính điểm động lượng chuẩn hóa (Cross-sectional z-score) cho từng coin:
       $$M_{i, t} = \frac{\frac{Close_{i, t-1}}{Close_{i, t-1-L}} - 1}{\sigma_{\text{rolling}}(r_{i, t-1}, L) + 10^{-8}}$$
       *(Sử dụng triệt để `.shift(1)` để loại trừ hoàn toàn look-ahead bias)*.
    2. Nếu có cấu hình `carry_tilt_weight` ($\lambda > 0$), kết hợp thêm điểm Funding Rate kỳ vọng $F_{i, t}$ để ưu tiên Long những đồng có funding dương và Short đồng có funding âm.
    3. Chọn Top-$K$ coin cao nhất để Long ($+w$) và Bottom-$K$ coin thấp nhất để Short ($-w$).
    4. Áp dụng **Buffer Rank** (ngưỡng trễ xếp hạng): Một coin đang nắm giữ chỉ bị loại khỏi danh mục nếu thứ hạng của nó rơi xuống dưới $Top\_K + Buffer\_Rank$, giúp giảm thiểu $60\%$ chi phí đảo lệnh (churn turnover).
    5. **Pair Volatility Targeting**: Tự động co giãn tổng đòn bẩy dựa trên độ biến động lịch sử của cặp vị thế so với mức biến động mục tiêu hàng năm.

### 4.10. `src/dynamic_carry_basis.py` — Dynamic Carry & Hội Tụ Basis
- **Làm gì**: Xoay vòng vốn giữa các cặp Spot-Perp để vừa ăn Funding Rate, vừa bắt trọn phần chênh lệch Basis khi giá Perp và Spot hội tụ về nhau.
- **Làm như thế nào**:
  - Tính điểm tổng hợp:
    $$\text{Score}_{i, t} = F_{\text{ewma}, i, t} + w_{\text{basis}} \cdot \frac{\text{Basis\_Dev}_{i, t}}{\text{Expected\_Hold\_Periods}}$$
    Trong đó $\text{Basis\_Dev} = \text{basis\_pct}_{t-1} - \text{EMA}(\text{basis\_pct}, 168\text{h})_{t-1}$.
  - Cơ chế **Switch Advantage Hurdle**: Vị thế hiện tại chỉ bị đảo sang một đồng coin mới nếu chênh lệch điểm số vượt qua chi phí khứ hồi:
    $$\text{Score}_{\text{new}} - \text{Score}_{\text{current}} \ge \frac{2 \times \text{Cost}_{\text{round\_trip}}}{\text{Expected\_Hold\_Periods}}$$

### 4.11. `src/cross_exchange_carry.py` — Carry Đa Sàn & Lượng Hóa Khối Lượng
- **Làm gì**: Khai thác chênh lệch Funding Rate giữa các sàn giao dịch (ví dụ: Short trên Binance ăn $0.03\%$ và Long trên Bybit trả $0.01\%$, bỏ túi chênh lệch ròng $0.02\%$/8h mà không chịu rủi ro giá crypto).
- **Làm như thế nào**:
  - Bảng quy chuẩn lot sàn `EXCHANGE_LOT_SPECS`: Khóa cứng `step_size` và `min_notional` cho từng coin.
  - Hàm `quantize_order_qty()`: Làm tròn khối lượng lệnh xuống theo đúng bước nhảy của sàn (`math.floor(q / step) * step`) và kiểm tra giá trị danh nghĩa tối thiểu ($10$ USD).
  - Hỗ trợ 2 chế độ:
    * Chế độ A (`perp_perp_spread`): Bắt chênh lệch liên sàn giữa 2 hợp đồng Perp.
    * Chế độ B (`best_venue_carry`): Chọn sàn có funding tốt nhất để đánh Cash & Carry Spot-Perp.
  - Mô hình khớp lệnh Maker Post-Only tất định (`maker_taker_hedge`): Chân 1 đặt Maker tiết kiệm phí ($0.02\%$), sau khi khớp lập tức bắn Taker chân 2 ($0.05\%$) để khóa Delta ngay lập tức.

### 4.12. `src/barbell_portfolio.py` — Trái Tim Danh Mục Barbell Portfolio
- **Làm gì**: Module quan trọng nhất của hệ thống, tích hợp toàn bộ Tier 1 (Core Yield) và Tier 2 (Tactical Alpha Overlay) trên cùng một tài khoản ký quỹ thống nhất (Unified Margin).
- **Làm như thế nào**:
  - `compute_snr_dispersion()`: Đo lường tỷ số Tín hiệu trên Nhiễu ($SNR$) của xu hướng 14 ngày trên toàn rổ coin:
    $$SNR_i = \frac{Return_{14d, i}}{\sigma_{1h, i} \cdot \sqrt{336}}$$
    $$\text{SNR\_Dispersion} = \max(SNR) - \min(SNR)$$
  - **SNR Regime Gate**:
    * Chế độ `hard_binary`: Nếu $\text{SNR\_Dispersion} < 1.35$, tắt hoàn toàn nhánh Overlay ($L_t = 0$), đưa $100\%$ vốn về trú ẩn tại Core Carry an toàn. Chỉ kích hoạt lại khi thị trường xuất hiện sự phân hóa xu hướng rõ rệt.
    * Chế độ `smooth_conviction`: Điều chỉnh mượt mà hệ số niềm tin:
      $$c_t = \text{clip}\left(\frac{\text{SNR\_Dispersion}}{1.0}, 0.50, 1.00\right)$$
  - **Enforce Overlay Dollar Neutrality** (`enforce_overlay_dollar_neutrality`): Thuật toán độc quyền giải quyết triệt để lỗi lệch lot BTC: Sau khi làm tròn theo lot-size sàn, nếu $|Long - Short| > \$10$, thuật toán tự động co chân lớn lại hoặc tinh chỉnh từng bước lot trên đồng coin có bước giá nhỏ nhất cho đến khi độ lệch $< \$10$ USD.
  - **UTC Epoch Boundary-Crossing Schedule** (`build_rebalance_mask`): Neo lịch tái cân bằng $48\text{h}$ vào gốc thời gian tuyệt đối `1970-01-01T00:00:00Z`. Khi nến tại mốc giao thời bị khuyết dữ liệu, hệ thống tự động nhận diện bước nhảy bucket và kích hoạt tái cân bằng ngay ở nến kế tiếp, **triệt tiêu vĩnh viễn hiện tượng trôi pha đồng hồ**.

### 4.13. `src/risk_manager.py` — Quản Trị Vị Thế & Circuit Breaker
- **Làm gì**: Tính toán kích thước vị thế vào lệnh, giới hạn đòn bẩy danh mục, quản lý Stop Loss tối thiểu và cầu dao ngắt rủi ro trong ngày.
- **Làm như thế nào**:
  - `calculate_position_size()`: Tính khối lượng theo công thức rủi ro cố định ($1\%$ NAV / khoảng cách SL). Hỗ trợ chế độ `use_fee_aware_sizing` (cộng gộp phí giao dịch và trượt giá vào khoảng cách rủi ro hiệu dụng để không bao giờ bị lỗ vượt quá $1\%$).
  - Kẹp chặt đòn bẩy danh mục không vượt quá `max_effective_leverage` ($2.0\text{x}$).
  - `Circuit Breaker`: Theo dõi lỗ thực tế trong ngày UTC (`daily_realized_pnl`). Nếu khoản lỗ chạm ngưỡng `-3%` NAV khởi đầu ngày (`max_daily_drawdown_pct`), cầu dao kích hoạt ngay lập tức: từ chối toàn bộ tín hiệu mở lệnh mới cho đến khi bước sang $00:00\text{ UTC}$ ngày hôm sau.

### 4.14. `src/ml_filter.py` — Meta-Labeling & Purged Walk-Forward
- **Làm gì**: Sử dụng học máy (Machine Learning) để lọc nhiễu tín hiệu theo phương pháp Triple-Barrier của Marcos López de Prado.
- **Làm như thế nào**:
  - `simulate_single_trade_outcome()`: Mô phỏng độc lập từng lệnh phát sinh từ tín hiệu kỹ thuật. Gán nhãn nhị phân: $y = 1$ nếu lệnh chốt lời dương (sau khi trừ phí/trượt giá), $y = 0$ nếu chạm SL hoặc hết thời gian nắm giữ mà lỗ.
  - `PurgedWalkForwardSplit`: Chia tập dữ liệu dạng Walk-Forward, thực hiện **Purging** (loại bỏ các quan sát trùng lặp thời gian nắm giữ giữa tập Train và Test) và **Embargo** (khoảng đệm sau tập Test) để loại trừ triệt để hiện tượng rò rỉ thông tin tương lai (data leakage).
  - `HistGradientBoostingClassifier`: Huấn luyện mô hình phân loại xác suất thắng dựa trên 13 đặc trưng kỹ thuật & vi cấu trúc. Chỉ cho phép mở lệnh nếu xác suất dự báo $P(\text{Win}) \ge 55\%$.

### 4.15. `src/backtester.py` — Engine Khớp Lệnh & Mô Phỏng Lịch Sử
- **Làm gì**: Động cơ mô phỏng backtest khớp lệnh chính xác đến từng nến, phản ánh đầy đủ trượt giá và chi phí sàn.
- **Làm như thế nào**:
  - **Khớp tại Open nến kế tiếp**: Tín hiệu tính toán tại nến $t$ chỉ được thực thi tại giá `open[t+1]` (cộng trượt giá).
  - **Thứ tự ưu tiên trong nến**: Nếu trong cùng một nến cả Stop Loss và Take Profit đều bị vi phạm, Backtester luôn mặc định **ưu tiên xử lý Stop Loss trước** (nguyên tắc thận trọng tối đa, chống ảo tưởng kết quả).
  - Hỗ trợ lệnh chờ `PendingLimitOrder` (Limit Pullback): Theo dõi thời gian sống TTL của lệnh limit, tự động hủy nếu quá hạn không khớp.

### 4.16. `src/state_store.py` — Quản Lý Trạng Thái Nguyên Tử (Atomic State)
- **Làm gì**: Lưu trữ và khôi phục trạng thái hoạt động của bot ra đĩa cứng dưới dạng JSON phục vụ vận hành $24/7$.
- **Làm như thế nào**:
  - Cơ chế **Ghi nguyên tử (Atomic Write)**: Trạng thái được tuần tự hóa và ghi vào file tạm `.tmp` trong cùng thư mục, gọi `f.flush()` và `os.fsync(f.fileno())` để ép hệ điều hành đẩy dữ liệu từ RAM xuống đĩa cứng, sau đó dùng `os.replace(tmp_path, final_path)` để tráo đổi file tức thì. Ngăn ngừa $100\%$ nguy cơ file state bị corrupt hoặc rỗng nếu máy chủ mất điện đột ngột.
  - Lưu trữ chi tiết: Vị thế mở (`open_position`), vị thế carry 2 chân (`carry_positions`), phân bổ barbell (`carry_sleeve`, `momentum_overlay`), cờ `kill_switch_triggered`, và nhật ký các đồng bị drift không thể tự sửa (`drift_unrepairable_symbols`).

### 4.17. `src/binance_public_feed.py` — Public Feed Cho Chế Độ Shadow-Live
- **Làm gì**: Thu thập nến giá và sổ lệnh từ các endpoint công khai (Public REST API) của Binance mà **hoàn toàn không cần API Key**.
- **Làm như thế nào**: Thiết lập cơ chế timeout nghiêm ngặt ($5.0\text{s}$), tự động retry khi gặp lỗi mạng, cung cấp dữ liệu an toàn tuyệt đối cho chế độ chạy ngầm giám sát (`shadow-live`).

### 4.18. `src/binance_signed_client.py` — HMAC-SHA256 Client & Smart Maker
- **Làm gì**: Module giao tiếp mạng có ký khóa mật mã HMAC-SHA256 với Binance Spot và USDⓈ-M Futures Testnet/Mainnet.
- **Làm như thế nào**:
  - **Kiến trúc Khóa Kép Độc Lập (Dual-Venue Auth)**: Quản lý riêng rẽ `spot_api_key/secret` và `futures_api_key/secret`. Tự động vệ sinh khóa (`_clean_credential`: loại bỏ khoảng trắng, dấu nháy kép, newline do copy/paste). Kiểm định độ dài đúng chuẩn 64 ký tự hex (`validate_credential_format`).
  - **Triệt tiêu lỗi lệch giờ -1021**: Hàm `sync_server_time()` gọi `/api/v3/time` và `/fapi/v1/time`, tính độ lệch đồng hồ máy chủ `time_offset_ms` và tự động bù vào trường `timestamp` của mọi payload ký HMAC.
  - **Smart Inside-Spread Maker** (`compute_maker_limit_price`): Khi đặt lệnh Maker, nếu spread $\ge 2\text{ ticks}$, bot đặt giá nằm sâu bên trong spread (`best_bid + 1 tick` cho BUY) để tăng tối đa xác suất khớp; nếu spread $\le 1\text{ tick}$, đặt tại `best_bid`. Nếu sổ lệnh bị khóa/chéo, tự động fallback về MARKET.
  - **Cơ chế Rollback Khẩn Cấp Chân 1** (`_rollback_filled_spot`): Trong trường hợp Chân 1 (Spot) đã khớp nhưng Chân 2 (Futures) bị sàn từ chối hoặc lỗi mạng, bot lập tức bán tháo lượng Spot vừa mua bằng lệnh Market để đưa tài khoản về trạng thái Flat, triệt tiêu hoàn toàn rủi ro lộ vị thế một chân (Naked Leg).

### 4.19. `src/exchange_reconciler.py` — Đối Soát Vị Thế & Sức Khỏe Ký Quỹ
- **Làm gì**: Động cơ đối soát độc lập so sánh giữa trạng thái lưu trữ cục bộ (`bot_state.json`) và ảnh chụp thực tế trên sàn (`exchange_snapshot`).
- **Làm như thế nào**:
  - Phân loại trạng thái từng vị thế: `SYNCED` (khớp hoàn hảo), `MISSING_ON_EXCHANGE` (sổ sách có nhưng sàn mất), `ORPHAN_ON_EXCHANGE` (sàn có nhưng sổ sách không theo dõi), `QTY_DRIFT` (lệch khối lượng).
  - Phân tích sức khỏe ký quỹ (`analyze_equity_and_margin_health`): Kiểm tra tỷ lệ ký quỹ duy trì ($Maint\_Margin\_Ratio \le 50\%$), đo khoảng cách từ giá hiện tại tới giá thanh lý ($Liquidation\_Buffer \ge 25\%$).
  - **Cơ chế Baseline Delta Phân Lập Faucet**: Trên Testnet, tài khoản có hàng trăm đồng coin rác do faucet cấp sẵn. Hàm `compute_testnet_delta_snapshot()` trừ đi số dư ban đầu tại `testnet_baseline.json`, chỉ đối soát đúng những biến động tài sản thực sự do bot tạo ra.

### 4.20. `src/carry_live_executor.py` — Live Daemon Engine Vận Hành 24/7
- **Làm gì**: Lớp điều phối cấp cao nhất thực thi toàn bộ chu trình sống của bot giao dịch Live/Testnet.
- **Làm như thế nào**:
  - Quản lý vòng lặp daemon `run_loop()`: Ngủ theo chu kỳ nến hoặc tham số `--interval-sec`, tải dữ liệu thị trường mới nhất, kiểm tra điều kiện khởi động (Warmup gate $720\text{h}$).
  - Điều phối lệnh Maker-Taker: Đặt Spot Maker $\rightarrow$ Lắng nghe khớp $\rightarrow$ Bắn Futures Taker $\rightarrow$ Kiểm định Lot Neutrality $\rightarrow$ Ghi log đối soát.
  - Tự động phát hiện và sửa lệch Delta (`audit_delta_drift` & `repair_delta_drift`).
  - Cầu dao tổng `check_kill_switch()`: Nếu NAV sụt giảm quá ngưỡng cho phép, tự động đóng toàn bộ vị thế khẩn cấp (`_close_carry_position_emergency`) và dừng bot.

### 4.21. `src/notifier.py` — Hệ Thống Cảnh Báo Telegram Chịu Lỗi
- **Làm gì**: Gửi thông báo tức thời về việc mở lệnh, đóng lệnh, PnL và sự cố hệ thống về kênh Telegram của người quản trị.
- **Làm như thế nào**:
  - Sử dụng REST API Telegram (`sendMessage`) kèm cơ chế retry tự động (mặc định 2 lần) và timeout an toàn ($5\text{s}$).
  - Thiết kế **Chịu lỗi tuyệt đối (Fault-Tolerant)**: Toàn bộ các lệnh gọi mạng đều được bọc trong `try-except`. Nếu mất mạng, sai token hoặc Telegram bị chặn, bot chỉ ghi log cảnh báo ra console, **tuyệt đối không bao giờ làm crash vòng lặp giao dịch chính**.

---

## 5. CLI ENTRYPOINT `run_live.py` & CÁC SCRIPTS NGHIÊN CỨU

### 5.1. Bộ Điều Khiển Chính `run_live.py`
Tệp [`run_live.py`](file:///d:/Quant_trading/run_live.py) là entrypoint duy nhất để vận hành bot ngoài đời thực. Hỗ trợ các cờ dòng lệnh quan trọng:

| Cờ Dòng Lệnh | Kiểu Dữ Liệu | Mặc Định | Ý Nghĩa Kỹ Thuật |
|---|---|---|---|
| `--mode` | `choice` | `paper` | Chế độ chạy: `paper` (giả lập), `live` (tiền thật), `testnet` (Binance Testnet), `shadow-live` (chỉ đọc public feed). |
| `--strategy` | `choice` | `barbell` | Chiến lược vận hành: `barbell` (Barbell Portfolio), `dynamic_carry`, hoặc `directional`. |
| `--capital` | `float` | `3000.0` | Hạn mức vốn phân bổ cho danh mục (USDT). |
| `--interval-sec` | `float` | `60.0` | Chu kỳ thời gian giữa các lần quét nến và đối soát (giây). |
| `--once` | `flag` | `False` | Chỉ chạy đúng 1 chu kỳ kiểm tra rồi thoát (dùng cho cronjob hoặc CI/CD). |
| `--cycles` | `int` | `None` | Giới hạn số chu kỳ chạy trước khi thoát tự động (dùng cho burn-in test). |
| `--reset-state` | `flag` | `False` | Xóa bỏ trạng thái cũ trong `bot_state.json`, khởi tạo lại từ đầu. |
| `--dry-run-sign` | `flag` | `False` | Đóng gói và tạo chữ ký HMAC nhưng không gửi lệnh lên sàn (kiểm thử an toàn). |
| `--testnet-send-orders` | `flag` | `False` | Cho phép gửi lệnh thật lên Binance Testnet (phải bật cờ này lệnh mới được bắn đi). |
| `--live-snapshot` | `flag` | `False` | Bật tính năng kéo snapshot tài khoản thật có chữ ký từ sàn để đối soát trước khi vào lệnh. |
| `--reset-testnet-baseline` | `flag` | `False` | (Sprint 22) Xóa và chụp lại baseline testnet mới từ snapshot sàn hiện tại. |
| `--skip-testnet-baseline` | `flag` | `False` | (Sprint 22) Bỏ qua đối soát baseline-delta trên testnet, đối soát trực tiếp raw snapshot (chỉ dùng cho chẩn đoán). |
| `--testnet-baseline-file` | `path` | `None` | (Sprint 22) Đường dẫn tùy chọn tới file testnet baseline (mặc định: `data/state/testnet_baseline.json`). |
| `--audit-log` | `path` | `None` | Đường dẫn ghi nhật ký đối soát chi tiết dạng JSONL (ví dụ `logs/audit.jsonl`). |

> [!IMPORTANT]
> **Đồng Bộ Kiểm Thử `test_parse_args_strategy_default_and_choices` (Sprint 22 Phase 10)**:  
> Trước đây, bài test `tests/test_carry_live_executor.py::test_parse_args_strategy_default_and_choices` bị fail do test cũ kỳ vọng `--strategy` mặc định là `directional`. Tại commit `07dbfac`, kiến trúc sư hệ thống đã chính thức nâng cấp giá trị mặc định của hệ thống lên `"barbell"` để phản ánh đúng năng lực cốt lõi hiện tại của bot. Ở Phase 10 này, bài test trên đã được đồng bộ chính thức sang assert `args.strategy == "barbell"`, đưa toàn bộ test suite toàn repo (699/699 tests) đạt trạng thái PASS 100% với 0 failed.

### 5.2. Các Scripts Nghiên Cứu & Vận Hành Trong `scripts/`
- [`scripts/run_testnet_multi_cycle_burnin.py`](file:///d:/Quant_trading/scripts/run_testnet_multi_cycle_burnin.py): Script kiểm định khép kín 3 chu kỳ thực tế trên Binance Testnet:
  * **Chu kỳ 1**: Bootstrap & Reconcile (Kiểm tra format key 64 ký tự, cổng warmup $720\text{h}$, đối soát 8/8 cặp SYNCED, chụp baseline).
  * **Chu kỳ 2**: Khớp lệnh thực tế Dual-Leg (Spot Maker + Perp Taker, kiểm tra Lot Neutrality).
  * **Chu kỳ 3**: Đóng vị thế sạch sẽ (Clean Flatten), xác nhận độ lệch vốn residual $\approx 0$.
- [`scripts/run_testnet_preflight.py`](file:///d:/Quant_trading/scripts/run_testnet_preflight.py): Kiểm tra 6 cổng an toàn trước khi bấm nút chạy bot.
- [`scripts/run_e2e_fire_drill.py`](file:///d:/Quant_trading/scripts/run_e2e_fire_drill.py): Diễn tập 20 tình huống sự cố khẩn cấp (mất kết nối, trượt giá cực đoan, sàn từ chối lệnh, chạm ngưỡng kill-switch).
- [`scripts/check_testnet_equity_breakdown.py`](file:///d:/Quant_trading/scripts/check_testnet_equity_breakdown.py): Script an toàn (chỉ đọc) in ra chi tiết tài sản ví Spot và ví Futures, bóc tách tiền mặt thực tế và coin rác faucet.

---

## 6. DÒNG CHẢY DỮ LIỆU & QUY TRÌNH KHỚP LỆNH ĐA TẦNG (END-TO-END PIPELINE)

Biểu đồ tuần tự dưới đây mô tả chính xác những gì diễn ra trong một chu kỳ quét của hệ thống:

```mermaid
sequenceDiagram
    autonumber
    participant D as Daemon Runner (run_live)
    participant C as BinanceSignedClient
    participant F as Data / Feed Engine
    participant B as Barbell Portfolio Engine
    participant R as Exchange Reconciler
    participant E as Binance Exchange (Spot & Perp)
    participant S as StateStore (bot_state.json)
    participant N as Telegram Notifier

    D->>C: 1. sync_server_time("both")
    C->>E: GET /api/v3/time & /fapi/v1/time
    E-->>C: server_time_ms
    C-->>D: time_offset_ms synchronized (-1021 error eliminated)

    D->>F: 2. fetch_latest_aligned_universe()
    F-->>D: Data 12 coins (Spot + Perp + Funding 3 sàn)
    
    D->>D: 3. compute_warmup_readiness() (Require >= 720h)
    
    D->>C: 4. fetch_exchange_reconcile_snapshot()
    C->>E: Signed GET /account & /positionRisk
    E-->>C: Raw Balances & Positions
    C-->>D: Exchange Snapshot

    D->>R: 5. reconcile_portfolio_state(local_state, snapshot)
    alt Reconcile Failed / Health Compromised
        R-->>D: Status != SYNCED (MISSING / DRIFT / ORPHAN)
        D->>D: Trigger Delta Repair or Emergency Flatten
        D->>N: Alert Reconcile Mismatch
    else Reconcile Passed (SYNCED)
        R-->>D: Status == SYNCED, Margin Health OK

        D->>B: 6. build_barbell_allocations(NAV, Universe)
        Note over B: Tier 1: Cash & Carry Allocation (85-90%)<br/>Tier 2: SNR Dispersion Gate -> Vol Target -> XS Momentum (10-15%)<br/>Enforce Dollar Neutrality (|Long - Short| <= $10)
        B-->>D: Barbell Plan (Target Quantized Lots)

        alt Có thay đổi vị thế cần thực thi
            D->>C: 7. place_maker_with_fallback(Leg 1: Spot Buy Inside-Spread)
            C->>E: POST /api/v3/order (LIMIT_MAKER)
            alt Maker khớp thành công
                E-->>C: Leg 1 FILLED
                C->>E: 8. Leg 2: POST /fapi/v1/order (Perp Short MARKET)
                E-->>C: Leg 2 FILLED
                C-->>D: Dual-Leg Successfully Hedged
            else Leg 1 xịt hoặc Leg 2 lỗi
                C->>E: 9. Rollback emergency: SELL Spot MARKET immediately
                E-->>C: Rollback Complete
                C-->>D: Abort Cycle (No Naked Exposure)
            end
        end

        D->>S: 10. save() Atomic Write (.tmp -> fsync -> replace)
        D->>D: 11. append_audit_record() to JSONL
        D->>N: 12. Send Cycle Summary Message
    end
```

---

## 7. CÁC CẠM BẪY ĐÃ ĐƯỢC KHẮC PHỤC & BÀI HỌC THỰC CHIẾN (BATTLE-TESTED FIXES)

Trong suốt 22 Sprint phát triển, hệ thống đã phát hiện và tiêu diệt nhiều lỗi kỹ thuật tinh vi mà hầu hết các dự án bot thương mại thông thường đều mắc phải:

### 7.1. Cạm bẫy Trôi Pha Đồng Hồ Tái Cân Bằng (Clock Phase Shift)
- **Hiện tượng**: Khi phân tích chuỗi dữ liệu bị đứt đoạn 1 nến do sự cố mạng, lịch tái cân bằng $48\text{h}$ tính theo modulo nến cục bộ (`bar_idx % 48`) bị trôi đi 1 giờ mãi mãi.
- **Giải pháp**: Xây dựng hàm `build_rebalance_mask` với cơ chế **UTC Epoch Boundary Crossing**. Quy đổi toàn bộ timestamp sang số giờ tính từ mốc `1970-01-01`. Dù mạng có mất bao nhiêu nến, khi bước sang mốc giờ chia hết cho 48 tiếp theo của thế giới thực, bot vẫn kích hoạt tái cân bằng chuẩn xác từng giây.

### 7.2. Cạm bẫy Lệch Delta Hàng Chục Đô Do Khối Lượng Lot BTC
- **Hiện tượng**: Bước nhảy tối thiểu của BTC trên Binance Futures là $0.001$ BTC ($\approx \$96$). Nếu một danh mục $3000$ USDT dành $10\%$ cho Overlay ($\$300$), chia đều Long 1 coin và Short 1 coin ($\$150$ mỗi bên). Nếu chân Short chọn BTC, việc làm tròn có thể đẩy giá trị thực tế lên $\$192$ hoặc tụt về $\$96$, trong khi chân Long (ví dụ SOL) khớp chuẩn xác $\$150$. Danh mục bị lệch Delta tới $\$46$ – $\$54$ USD, hoàn toàn mất đi tính chất Market-Neutral.
- **Giải pháp**: Hàm `enforce_overlay_dollar_neutrality` áp dụng thuật toán co chân và tinh chỉnh tối ưu hóa tổ hợp (combinatorial optimization) trên từng bước lot của đồng có bước giá mịn hơn. Nếu độ lệch không thể giảm xuống dưới $\$10$, bot thà từ chối mở vị thế (`qty = 0`) chứ không bao giờ chấp nhận rò rỉ Delta.

### 7.3. Cạm bẫy Lệnh Post-Only Bị Từ Chối Do Cắt Qua Spread (-2010 / -2021)
- **Hiện tượng**: Lệnh Maker (`LIMIT_MAKER` trên Spot hoặc `GTX` trên Futures) đặt tại mức giá chạm vào sổ lệnh đối ứng bị Binance từ chối lập tức với mã lỗi `-2010` hoặc `-2021`, dẫn đến việc bot phải liên tục dùng lệnh Market chịu phí cao.
- **Giải pháp**: Hàm `compute_maker_limit_price` kiểm tra độ rộng bid/ask trước khi phát lệnh: Nếu spread $\ge 2$ ticks, đặt sâu bên trong spread (`best_bid + 1 tick` đối với lệnh Mua). Lệnh vừa được ưu tiên khớp trước hàng đợi tại best_bid, vừa đảm bảo tính chất Maker $100\%$.

### 7.4. Cạm bẫy Mất Đồng Bộ Giờ Server (-1021 Outside of recvWindow)
- **Hiện tượng**: Đồng hồ hệ điều hành Windows bị trôi vài giây so với máy chủ Binance ở Tokyo khiến chữ ký HMAC bị từ chối với lỗi `-1021`.
- **Giải pháp**: Hàm `sync_server_time()` liên tục đo lường `time_offset_ms = server_time - local_time` ở đầu mỗi chu kỳ và tự động cộng dồn vào tham số `timestamp` khi ký lệnh.

### 7.5. Cạm bẫy Nhiễm Bẩn Faucet Testnet Làm Hỏng Reconciler
- **Hiện tượng**: Tài khoản Binance Testnet tự động được cấp hàng trăm đồng token lạ (faucet coins). Khi reconciler quét toàn bộ tài khoản, nó phát hiện hàng trăm vị thế mồ côi (`ORPHAN_ON_EXCHANGE`) và kích hoạt cơ chế khóa an toàn từ chối giao dịch.
- **Giải pháp**: Hàm `reconcile_portfolio_state` phân tách ranh giới rõ ràng: chỉ đối soát trên rổ tài sản định sẵn (`CANONICAL_UNIVERSE_SYMBOLS`). Các tài sản faucet lạ được gom vào danh sách theo dõi thụ động `untracked_spot_assets` và không bao giờ làm gián đoạn luồng giao dịch của các cặp chính.

---

## 8. HƯỚNG DẪN VẬN HÀNH HỆ THỐNG 24/7 (RUNBOOK)

### 8.1. Thiết Lập Khóa Môi Trường `.env.testnet`
Tạo file `.env.testnet` tại thư mục gốc với định dạng sau (đảm bảo độ dài chính xác 64 ký tự hex):

```ini
# Binance Spot Testnet (https://testnet.binance.vision)
BINANCE_SPOT_TESTNET_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
BINANCE_SPOT_TESTNET_API_SECRET=yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy

# Binance USD(S)-M Futures Testnet (https://testnet.binancefuture.com)
BINANCE_FUTURES_TESTNET_API_KEY=zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
BINANCE_FUTURES_TESTNET_API_SECRET=wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

# Cảnh báo Telegram (Tùy chọn)
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
TELEGRAM_CHAT_ID=-1001234567890
```

### 8.2. Các Lệnh Vận Hành Tiêu Biểu

#### 1. Kiểm tra số dư ví và cấu thành tài sản (Read-Only, an toàn tuyệt đối):
```powershell
python scripts/check_testnet_equity_breakdown.py
```

#### 2. Chạy kiểm tra tiền trạm 6 cổng an toàn (Preflight):
```powershell
python scripts/run_testnet_preflight.py
```

#### 3. Chạy kiểm thử tự động toàn bộ codebase:
```powershell
pytest -q
```

#### 4. Khởi động Bot ở chế độ Giả Lập Giám Sát Thị Trường Thực (Shadow-Live):
```powershell
python run_live.py --mode shadow-live --strategy barbell --capital 3000
```

#### 5. Khởi động Bot ở chế độ Ký Thử Nghiệm Không Bắn Lệnh (Dry-Run Signing):
```powershell
python run_live.py --mode testnet --strategy barbell --capital 3000 --dry-run-sign --audit-log logs/audit.jsonl
```

#### 6. VẬN HÀNH BOT CHÍNH THỨC TRÊN TESTNET ($24/7$):
```powershell
python run_live.py --mode testnet --strategy barbell --capital 3000 --testnet-send-orders --live-snapshot --interval-sec 60 --audit-log logs/live_audit.jsonl
```

#### 7. Diễn tập sự cố khẩn cấp (Fire Drill):
```powershell
python scripts/run_e2e_fire_drill.py
```

---

## LỜI KẾT & ĐÁNH GIÁ CHẤT LƯỢNG CODEBASE

Codebase này đại diện cho một sản phẩm kỹ thuật tài chính có mức độ hoàn thiện cao, tuân thủ chặt chẽ các nguyên tắc:
1. **Toán học vững chắc**: Không đưa ra quyết định dựa trên cảm tính; mọi tham số đều có nguồn gốc từ kiểm định Newey-West HAC, Monte Carlo Bootstrap và 5 Định luật Bất biến.
2. **Kỹ thuật phần mềm bền bỉ**: Sử dụng triệt để Atomic Write, cơ chế Rollback hai chân, bọc lót lỗi mạng, và hệ thống đối soát liên tục loại trừ khả năng thất thoát tài sản.
3. **Sẵn sàng triển khai**: Kiến trúc tách rời hoàn toàn giữa tầng dữ liệu, tầng tính toán alpha, tầng quản lý danh mục và tầng kết nối sàn, cho phép hệ thống dễ dàng mở rộng sang các sàn giao dịch phái sinh khác trong tương lai.
