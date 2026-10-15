"""Feeder, measurement and detection parameters.

Values are chosen to be realistic for a Tunisian low-voltage (230 V / 50 Hz)
residential feeder supplied by one MV/LV transformer. Everything lives here so
the simulator, the detector and the docs all read from one place.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeederConfig:
    n_meters: int = 24              # houses on the radial feeder
    n_pv: int = 8                   # houses with rooftop PV (PROSOL-style)
    pv_kw: float = 3.0              # rated PV power per roof
    r_segment: float = 0.005        # ohm, cable segment between two houses (loop)
    r_service: float = 0.04         # ohm, service drop from pole to meter
    r_transformer: float = 0.004    # ohm, transformer equivalent resistance
    v_nominal: float = 230.0
    v_source: float = 233.0         # MV-side voltage referred to LV, no load
    f_nominal: float = 50.0


@dataclass(frozen=True)
class GridDynamics:
    # Grid frequency: Ornstein-Uhlenbeck around 50 Hz (stationary sd ~20 mHz)
    f_theta: float = 0.02
    f_sigma: float = 0.004
    # MV-side voltage wobble seen by the whole feeder (the "feeder pulse")
    v_theta: float = 0.01
    v_sigma: float = 0.15


@dataclass(frozen=True)
class MeterNoise:
    v_sd: float = 0.03       # volts
    i_rel_sd: float = 0.002  # relative current error
    f_sd: float = 0.001      # Hz


@dataclass(frozen=True)
class DetectorConfig:
    window_s: int = 60           # analysis window
    step_s: int = 30             # hop between windows
    lookback_s: int = 600        # history used for balance / fault regression
    max_replay_lag_s: int = 2400 # how far back the replay search looks
    replay_min_lag_s: int = 30
    replay_corr: float = 0.6     # pulse match needed to call a replay
    # Thresholds below are re-calibrated from clean data (NabdhDetector.fit).
    pulse_quantile: float = 0.001
    residual_quantile: float = 0.999
    if_margin: float = 0.08      # Isolation Forest alarm = clean max score + margin
    freq_event_hz: float = 0.08  # |f - 50| above this = grid-wide frequency event
    sag_event_pct: float = 5.0   # transformer voltage drop that counts as a sag


@dataclass(frozen=True)
class Config:
    feeder: FeederConfig = field(default_factory=FeederConfig)
    grid: GridDynamics = field(default_factory=GridDynamics)
    noise: MeterNoise = field(default_factory=MeterNoise)
    detector: DetectorConfig = field(default_factory=DetectorConfig)


DEFAULT = Config()

# Verdict vocabulary shared by the injector, the detector and the evaluation.
NORMAL = "NORMAL"
GRID_EVENT = "GRID_EVENT"
STUCK = "STUCK"
DRIFT = "DRIFT"
BIAS = "BIAS"
FABRICATED = "FABRICATED"
REPLAY = "REPLAY"
EQUIPMENT_FAULT = "EQUIPMENT_FAULT"
UNEXPLAINED = "UNEXPLAINED"

DATA_PROBLEMS = (STUCK, DRIFT, BIAS, FABRICATED, REPLAY)
PHYSICAL_PROBLEMS = (EQUIPMENT_FAULT,)
ALL_LABELS = (NORMAL, GRID_EVENT, STUCK, DRIFT, BIAS, FABRICATED, REPLAY, EQUIPMENT_FAULT)
