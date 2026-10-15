# Example verdicts produced by NABDH

## NORMAL  (confidence 0.99, route NONE)

Meter M05, window 19:30-20:30: healthy. Its readings carry the grid pulse (match 0.89) and the feeder pulse (match 0.95), and its voltage agrees with the physics model within 0.00 V.

*Action:* accept data

## GRID_EVENT  (confidence 0.98, route AUTO)

Meter M02, window 42:30-43:30: genuine grid event. The grid frequency moved 157 mHz away from 50 Hz, and this meter saw it in step with the reference (pulse match 0.96). The data is real: act on the event, not on the meter.

*Action:* accept data, alert grid operations

## STUCK  (confidence 0.99, route AUTO)

Meter M11, window 31:30-32:30: sensor frozen. Voltage and current did not change at all for 100% of the window. Real mains voltage always wobbles, so these values are a repeated last reading, not a measurement.

*Action:* quarantine readings, open maintenance ticket

## DRIFT  (confidence 0.99, route AUTO)

Meter M05, window 30:00-31:00: calibration drift. The meter still carries the pulse (match 0.91), so it is real and in place, but it reads +4.25 V away from what Ohm's law and its neighbours predict, and that offset does not follow its own current. The sensor moved, not the grid.

*Action:* apply offset correction, schedule recalibration

## BIAS  (confidence 0.99, route HUMAN)

Meter M21, window 23:00-24:00: under-reporting. The transformer delivers 1.60 kW that no meter accounts for. That gap opened at the same second this meter's reading dropped by the same amount. A real load switching off would have lowered the transformer reading too. Possible CT miscalibration or tampering: needs a human decision.

*Action:* quarantine billing data, request field inspection

## FABRICATED  (confidence 0.99, route HUMAN)

Meter M05, window 26:00-27:00: readings not measured here. Values look plausible, but their wobble does not match the grid pulse (match 0.05) or the feeder pulse (match -0.15), and no earlier moment matches either (best 0.47). Someone or something is generating these numbers.

*Action:* quarantine readings, open security incident

## REPLAY  (confidence 0.90, route HUMAN)

Meter M07, window 33:00-34:00: replayed data. The wobble is genuine but belongs to 1130 s ago (match 0.90 at that delay, 0.07 now). Old real readings are being re-sent as new ones.

*Action:* quarantine readings, open security incident

## EQUIPMENT_FAULT  (confidence 0.99, route HUMAN)

Meter M03, window 23:00-24:00: likely bad connection at the customer side. Grid pulse is intact (match 0.87), so the meter is real, but the voltage sits 1.71 V below the physics model and its voltage flickers out of step with the feeder (feeder pulse match 0.11), as arcing does. That is the signature of a loose or burnt joint: a real safety issue, not a data issue.

*Action:* dispatch crew to customer connection
