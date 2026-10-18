"""Test ràng buộc fail-fast của TradingConfig cho Trailing Stop (Sprint 3).

Kiểm chứng mặc định mới `trailing_activation_r=0.75` luôn bật được trailing stop
với cấu hình mặc định, và `__post_init__` chặn các cấu hình trailing vô hiệu.
"""
from dataclasses import replace

import pytest

from src.config import TradingConfig


def test_default_config_trailing_is_activatable():
    """Mặc định trailing_activation_r=0.75 phải cho trailing có thể kích hoạt trước TP."""
    assert TradingConfig().trailing_activation_r == pytest.approx(0.75)
    assert TradingConfig().trailing_activation_r * TradingConfig().sl_atr_multiple < TradingConfig().tp_atr_multiple

    # Bật trailing với cấu hình mặc định KHÔNG được raise và vẫn thỏa bất đẳng thức (0.75*2.0 = 1.5 < 3.0).
    cfg = replace(TradingConfig(), use_trailing_stop=True)
    assert cfg.use_trailing_stop is True
    assert cfg.trailing_activation_r * cfg.sl_atr_multiple < cfg.tp_atr_multiple


def test_invalid_trailing_config_is_rejected():
    """Cấu hình đúng biên (activation_dist == tp) phải bị chặn vì dùng >=."""
    with pytest.raises(ValueError, match="Trailing Stop vô hiệu"):
        replace(
            TradingConfig(),
            use_trailing_stop=True,
            trailing_activation_r=1.5,
            tp_atr_multiple=3.0,
        )


def test_trailing_exempt_when_tp_disabled():
    """TP thả nổi (tp_atr_multiple=0) bỏ qua ràng buộc; vi phạm rõ khi tp>0 vẫn bị chặn."""
    # tp_atr_multiple=0.0: không ràng buộc -> kích thước kích hoạt lớn tùy ý vẫn hợp lệ.
    cfg = replace(
        TradingConfig(),
        use_trailing_stop=True,
        tp_atr_multiple=0.0,
        trailing_activation_r=5.0,
    )
    assert cfg.tp_atr_multiple == 0.0
    assert cfg.use_trailing_stop is True

    # tp>0 với activation_dist (1.5) >= tp (1.0) -> phải bị chặn.
    with pytest.raises(ValueError):
        replace(TradingConfig(), use_trailing_stop=True, tp_atr_multiple=1.0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"trailing_activation_r": 0.0}, "trailing_activation_r phải > 0"),
        ({"trailing_activation_r": -0.5}, None),
        ({"sl_atr_multiple": 0.0}, "sl_atr_multiple phải > 0"),
        ({"trailing_atr_mult": 0.0}, "trailing_atr_mult phải > 0"),
    ],
)
def test_trailing_rejects_non_positive_params(overrides, message):
    """Khi bật trailing, các tham số không dương phải bị chặn fail-fast bằng ValueError."""
    with pytest.raises(ValueError) as excinfo:
        replace(TradingConfig(), use_trailing_stop=True, **overrides)
    if message is not None:
        assert message in str(excinfo.value)

    # Case hợp lệ với cấu hình mặc định KHÔNG được bị chặn.
    cfg = replace(TradingConfig(), use_trailing_stop=True)
    assert cfg.use_trailing_stop is True

