"""Small, fast checks of the physics and the guarantees NABDH relies on.

    python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nabdh import governance, physics, pulse  # noqa: E402
from nabdh.audit import AuditLog, verify  # noqa: E402
from nabdh.config import BIAS, DRIFT, FABRICATED, STUCK  # noqa: E402
from nabdh.injectors import inject  # noqa: E402
from nabdh.simulator import EpisodeSpec, simulate  # noqa: E402


@pytest.fixture(scope="module")
def clean():
    return simulate(EpisodeSpec("t", "summer_evening", duration_s=900, seed=7))


def test_energy_balance_closes_on_clean_data(clean):
    gap = physics.balance_gap(clean.tx_p, clean.meter_p, clean.meter_i)
    assert abs(np.median(gap)) < 0.1          # kW, on a ~35 kW feeder


def test_voltage_twin_matches_clean_meters(clean):
    v_hat = physics.voltage_twin(clean.tx_v, clean.meter_i)
    assert np.abs(np.median(clean.meter_v - v_hat, axis=0)).max() < 0.2


def test_genuine_meter_carries_grid_pulse(clean):
    w = slice(300, 360)
    c = pulse.grid_pulse(clean.meter_f[w], clean.tx_f[w])
    assert c.min() > 0.6


def test_replay_lag_is_recovered(clean):
    ep = simulate(EpisodeSpec("r", "summer_evening", duration_s=900, seed=7))
    rng = np.random.default_rng(0)
    inj = inject(ep, 4, "REPLAY", 700, 800, rng)
    lag, corr = pulse.find_replay_lag(ep.meter_f[700:760, 4], ep.tx_f, 700, 30, 700)
    assert lag == int(inj.param) and corr > 0.6


def test_fabricated_stream_has_no_pulse():
    ep = simulate(EpisodeSpec("f", "winter_morning", duration_s=900, seed=3))
    inject(ep, 2, FABRICATED, 400, 700, np.random.default_rng(1))
    c = pulse.grid_pulse(ep.meter_f[500:560], ep.tx_f[500:560])
    assert abs(c[2]) < 0.4 and np.delete(c, 2).min() > 0.6


def test_stuck_sensor_is_frozen():
    ep = simulate(EpisodeSpec("s", "winter_morning", duration_s=900, seed=3))
    inject(ep, 5, STUCK, 400, 700, np.random.default_rng(1))
    fr = pulse.frozen_fraction(ep.meter_v[500:560], ep.meter_i[500:560])
    assert fr[5] == 1.0 and np.delete(fr, 5).max() < 0.1


def test_kirchhoff_imputation_recovers_single_current(clean):
    bad = np.zeros_like(clean.meter_i, dtype=bool)
    bad[:, 9] = True
    est = physics.impute_untrusted(clean.meter_i, clean.tx_i, bad)
    assert np.median(np.abs(est[:, 9] - clean.meter_i[:, 9])) < 1.0   # amps


def test_high_risk_verdicts_always_need_a_human():
    for v in (BIAS, FABRICATED, "REPLAY", "EQUIPMENT_FAULT"):
        assert governance.route(v, 0.999)[0] == governance.HUMAN
    assert governance.route(DRIFT, 0.99)[0] == governance.AUTO


def test_audit_log_detects_tampering(tmp_path):
    log = AuditLog(tmp_path / "log.jsonl")
    for k in range(5):
        log.append(meter=k, verdict="STUCK", confidence=0.9)
    assert verify(tmp_path / "log.jsonl") == (True, -1)
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    lines[2] = lines[2].replace('"STUCK"', '"NORMAL"')
    (tmp_path / "log.jsonl").write_text("\n".join(lines) + "\n")
    assert verify(tmp_path / "log.jsonl") == (False, 2)
