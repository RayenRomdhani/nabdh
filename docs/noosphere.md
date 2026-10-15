# Noosphere: shared learning between NABDH nodes

Phase 1 runs on a single node. This document is the design for many nodes, as the challenge
allows. It will be prototyped in Phase 2.

## Nodes

A node is one NABDH instance watching one slice of the grid: a residential feeder, a
factory's LV board, a microgrid. It runs at the edge (substation PC or data concentrator)
and keeps its own data.

## What is shared, what stays local, and why

| Item | Shared? | Why |
|---|---|---|
| Grid pulse (1 Hz frequency trace) | yes | It is identical everywhere on the synchronous grid, so it says nothing about any household. Peers use it to check each other's reference. |
| Grid event attestations ("frequency dip of 160 mHz at 14:02:07") | yes | Lets the network confirm that an event is grid-wide. A dip seen by one node alone points at that node's reference meter. |
| Anomaly fingerprints of confirmed incidents | yes, after privacy filtering | The evidence vector (pulse matches, frozen share, residual, gap ratio, replay lag) describes how an attack or a fault *looks*. It has no meter ID, no customer and no raw readings. |
| Calibration statistics (threshold quantiles, Isolation Forest summary) | yes, aggregated | A new node can start with network thresholds instead of waiting weeks for clean data. |
| Raw per-meter V, I, P | **no** | At 1 Hz, household power reveals presence, sleep, cooking times and individual appliances (load disaggregation). It is personal data under Tunisia's Organic Law 2004-63 on personal data protection. |
| Meter IDs, addresses, billing, local topology | **no** | Identifies customers and maps the grid for attackers. |
| Operator decisions | counts only | Over-trust and under-trust rates per node help compare training needs without naming anyone. |

## Messages

All messages are JSON, signed with the node's Ed25519 key, sent over MQTT with TLS.

```json
{"type": "pulse",       "node": "N-SFX-014", "t0": "2026-10-15T14:02:00Z",
 "f_mhz_dev": [12, 9, 4, -3, ...]}

{"type": "event",       "node": "N-SFX-014", "t": "2026-10-15T14:02:07Z",
 "kind": "freq_dip", "magnitude_mhz": 162, "confidence": 0.97}

{"type": "fingerprint", "node": "N-SFX-014", "verdict": "REPLAY", "count": 6,
 "features": {"f_corr": 0.08, "v_corr": 0.11, "replay_corr": 0.91,
              "replay_lag_bucket": "15-30 min"}, "noise": "laplace(eps=1.0)"}

{"type": "calibration", "node": "N-SFX-014", "windows": 28800,
 "quantiles": {"f_corr_p001": 0.79, "v_corr_p001": 0.81, "res_abs_p999": 0.09}}
```

## Privacy guards

- A fingerprint is only published when at least k = 5 incidents of that verdict exist on
  the node, and its features get Laplace noise (differential privacy, ε about 1 per week).
- Timestamps in fingerprints are bucketed to the hour; replay lags to ranges.
- The pulse message carries frequency only. Voltage stays local because it depends on local
  load.

## Collective defense

1. **Cross-checking references.** Each node compares its transformer frequency with its
   peers'. A node whose reference loses the shared pulse is itself under attack or broken.
   Its verdicts are frozen and peers flag it.
2. **Event confirmation.** A GRID_EVENT gets "confirmed by 7 of 7 nodes" in its explanation.
   An event seen by one node is re-labelled as local.
3. **Campaign detection.** The same fingerprint (for example replays with a 15 to 30 minute
   lag) appearing on several nodes within a day raises one security incident for the
   network operator instead of scattered local tickets.
4. **Shared thresholds.** Network thresholds are the median of node statistics, so one
   poisoned node cannot drag them. Nodes whose statistics sit far from the median lose
   weight (a simple reputation score).

## Learning loop

Operator decisions on HUMAN incidents become labels. Confirmed fingerprints update the
shared library weekly. Each node retrains its Isolation Forest on its own recent clean
windows and adopts network thresholds only when its own data is too thin.
