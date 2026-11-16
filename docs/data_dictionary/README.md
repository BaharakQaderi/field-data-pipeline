# TK10 field data dictionary

How to read the ground-station InfluxDB data (`tk` measurement, `source` tag, fields prefixed by source)
and the archives made by `export_influx_day.py`. Built 2026-10-23 from tk-hmi-monorepo, flight-pod-automation-v2,
tk10-plc-sim/tk10-sim, the PDFs in Kite-projects/docs, and checked against the 2026-10-22 flights.

| File | Content |
|---|---|
| `01_concepts_power_chain_kpis.md` | Energy chain, PLC state machine, constants, power definitions + sign conventions, KPIs, open questions |
| `02_ground_station_hmi_fields.md` | OPC (PLC) fields by subsystem, small sources (WIND, THROTTLE, TARANIS, BRIDGE, WS, SYSTEM), enums |
| `03_flight_segment_pod_fields.md` | FLIGHT_SEGMENT / BRIDGE / GS_UPLINK_COMMAND / FS_MAINTENANCE: pod, IPOK position, servos, links |
| `field_profile_2026-10-22_flights.csv` | Every field over the three flights: dtype, distinct values, min/max/mean (only ~160 of ~870 vary) |

## Conventions verified on data (2026-10-22)

| Item | Rule |
|---|---|
| Array index | 1-based. `DsInverters[1]` recovery motor, `[2]` generator (main), `[3]` thread guide. `ConvStruct[1..4]` = converter/battery 1..4 |
| Drum | D = 0.491 m; `outTamburo_SpeedMSec` = RPM/60·π·0.491. `Velocity_ActualValue[2]` = drum rpm (1:1) |
| Reel direction | `outTamburo_SpeedMSec` **< 0 = reel-out** (generation), > 0 = reel-in |
| Tether | `outTamburo_WindedLength` = length still **wound on the drum** (≈214 m at launch). Kite radius ≈ 313.6 − WindedLength (r = −0.95) |
| Tether force | `DsLoadCells.MeasureFloat_SUM` in kg (backend sum of 4 cells under the drum) → N = 9.81·(SUM − tare) |
| Battery power | `ConvStruct.CONV_READ.CONV_MEAS_FB_LS_PWR_SCALED_CALC` kW, **> 0 = charging** (backend Σ V_LS·I_LS/1000) |
| Mechanical power | P_mech = F·v_t with v_t = −SpeedMSec (positive on reel-out) |
| Pumping phases | **Not** from the PLC state: flights stay in state 4 (GENERATION) throughout; segment by drum speed sign |
| Do not use | `DsInverters.Power[i]` (unsigned raw V×I), `CONV_POWER_SCALED_HS[i]` (unsigned), `Torque_ActualValue[2]` for absolute power (scale ≈4× off, unconfirmed), `DsEncoder.outSpeed` (unsigned wrap) |
| Pod | No autopilot (status hard-coded 0) — manual flying via Taranis. `pitch` = depower axis, `roll` = steering, both %. Position from IPOK: azimuth 0 = N clockwise, `radius` = straight-line distance, 6553.5 = invalid |
| Time | All `_time` are backend receive times (not PLC/pod clocks). OPC 10 Hz snapshot; FLIGHT_SEGMENT ~20 Hz sparse; BRIDGE/GS_UPLINK 2 Hz |

## Open questions for the team
1. `Torque_ActualValue[2]` scale/unit (Metronix rated torque, CiA-402 per-mille?) and gearbox vs direct drive.
2. Load-cell tare and geometry: does SUM·g equal tether tension?
3. Is 0.491 m the bare-drum or effective diameter (layer build-up)? Total tether length?
4. Intended generation control mode (spec: torque; data: velocity via VelocityN/VelocityP throttle).
5. Wind sensor height; XSens yaw convention; pod pitch % ↔ mm limits used in flight.
