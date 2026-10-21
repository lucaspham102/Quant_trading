# Sprint 13 — Fire Drill Evidence Report

Kich ban dien tap 5 chu ky offline tren o dia, su dung `CarryLiveExecutor` + `StateStore` that voi `FakeExchange`.

| Chu ky | Hanh dong | Ket qua mong doi | Ket qua thuc te | PASS/FAIL |
| :--- | :--- | :--- | :--- | :--- |
| Chu ky 1 | Mo vi the Maker-Taker pair va Overlay rebalance | Vi the mo hop le, state duoc ghi len dia, qty theo lot-step | carry=1, overlay=0, file=True | PASS |
| Chu ky 2 | Delta drift audit (sub-lot -> unrepairable flag; >1 lot -> repair) | BTC lech nho bi danh dau, SOL sua thanh cong, BTC lech lon tu go co va sua | flag_a=True, sol_rep=True, flag_cleared=True | PASS |
| Chu ky 3 | Kiem tra spread guard voi wide spread va crossed book | Ca hai bi tu choi (SPREAD_TOO_WIDE, SPREAD_CROSSED_BOOK), 0 lenh moi | wide=SPREAD_TOO_WIDE, crossed=SPREAD_CROSSED_BOOK, orders=0 | PASS |
| Chu ky 4 | Kich hoat Drawdown Kill-switch tai equity=2812.5 (-6.25%) | Kill-switch kich hoat, dong toan bo vi the, chi dung lenh market close, luu co | triggered=True, carry_pos=0, new_orders=3 | PASS |
| Chu ky 5 | Kiem tra co sticky tren bot moi va go bo bang reset=True | Khoa moi van giu co kill-switch, actions=[]; reset=True go sach co ve 3000.0 | sticky=True, reset=True | PASS |

**Tong: 20/20 checks PASS**
