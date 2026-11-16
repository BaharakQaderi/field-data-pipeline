# Data dictionary: TK10 Flight Segment (pod) telemetry in InfluxDB

Scope: InfluxDB measurement `tk`, tag `source` ∈ {`FLIGHT_SEGMENT`, `BRIDGE`, `GS_UPLINK_COMMAND`, `FS_MAINTENANCE`}. Field names are `<SOURCE>_<field>`.
Sources read (read-only):
- `FPA` = `Kite-projects/flight-pod-automation-v2` (HEAD 2026-10-07, `244dd8a`)
- `HMI` = `Kite-projects/tk-hmi-monorepo`

Confidence levels: **C** = confirmed in code, **D** = stated in docs only, **I** = inferred by me (needs checking against data or with the team).

---

## 1. Overview of the flight segment (pod)

### 1.1 What the pod is

- **Hardware.** The pod is a Raspberry Pi (`controller_temp` is the RPi SoC temperature, read from `/sys/class/thermal/thermal_zone0/temp`). It drives two MD80 servo drives over CANopen at 1 Mbit/s, using CiA-402 CSP position mode (FPA `configs/motor_config.yaml:44-48,59-66`). It also carries:
  - an I²C power monitor on the battery: LTC2945-style registers, 1.5 mΩ shunt (FPA `flight_segment/peripherals/power_sensor.py:12-19`);
  - a RunCam Thumb 2 camera on UART;
  - two XBee radios: MAIN 2.4 GHz DigiMesh and BACKUP 868/915 MHz.
- **Mechanical axes.** There are two logical axes (FPA `web_gui/src/app/core/models/gui-config.ts:34-40`):
  - **pitch** is the **depower** axis (power/depower, i.e. angle of attack);
  - **roll** is the **steering** axis.
- **Pod versions:**
  - **V2** has independent motors: pitch motor = node 18, roll motor = node 1.
  - **V3** has coupled motors: left = node 11, right = node 12 (`motor_config.yaml:59-66`; an older repo note says 5/10, which is stale). The transform is `left = pitch + roll` and `right = −pitch + roll`, with inverse `pitch = (L−R)/2` and `roll = (L+R)/2` (FPA `flight_segment/control/strategies/v3_strategy.py:4-5`).
  - The deployed FS config says V3 (`configs/system_config.yaml:56`). The 2026-05 ground-fixed-point campaigns used V2 (`configs/campaign_2026-05-27.yaml:23`).
  - `FLIGHT_SEGMENT_pod_version` tells you which version was flying.
- **IPOK.** IPOK is a separate box with an XSens MTi IMU/INS and a u-blox F9P RTK receiver. It streams lat/lon/alt, ENU velocity, Euler angles and RTK status at about 100 Hz over a WebSocket on the pod's Wi-Fi (`10.42.0.1:8770`). The pod is a client of that stream (FPA `docs/architecture/FLIGHT_SEGMENT_OVERVIEW.md:24,48`; `ipok/tasks.py:346-354`).
  - All of `pitch_angle`, `roll_angle`, `yaw_angle`, `azimuth`, `elevation`, `radius` and `velocity_*` come from IPOK, not from the pod's own sensors (FPA `flight_segment/peripherals/ipok_adapter.py:72-110`).
  - **I:** IPOK is airborne next to the pod (it is on the pod's Wi-Fi hotspot, and the az/el/r it produces is "kite as seen from the GS").
- **No autopilot on the pod.** The header always sends `autopilot_status=0`, hard-coded (FPA `flight_segment/transports/downlink_frame_builder.py:129,159`; HMI `docs/GS_BRIDGE_CONTRACT.md:67`). The kite is flown by a human:
  - with the Taranis RC transmitter (sticks map to `pitch`/`roll` from −100 to +100 %);
  - or with HMI sliders in MAINTENANCE (HMI `GS_BRIDGE_CONTRACT.md:227-243`).

### 1.2 Control loop

The control path runs as follows.

1. The ground segment (GS) streams an uplink "control state" at **20 Hz**: roll %, pitch %, flags and status request (FPA `docs/architecture/DATALINK_PROTOCOL.md:12,194-236`).
2. The flight segment (FS) state machine gates the command (§1.3).
3. `UplinkCommandMapper` issues one combined `set_position` with both axes. It does this only when **both** axes carry real values, because the sentinel `0x80` means "no command" (FPA `flight_segment/transports/uplink_dispatch.py:86-98`; `DATALINK_PROTOCOL.md:206-213`).
4. The percent value is converted to mm, linear within the axis limits: `mm = center + pct/100·half_range`, where `center=(min+max)/2` and `half_range=(max−min)/2` (FPA `common/percent_converter.py:11-52`).
5. On V3, mm goes through the left/right transform. The MD80 then runs its internal position/velocity PID: default kp=25, ki=0.1, kd=0, kp_vel=1 (`motor_config.yaml:92-109`).

Further rules:
- **Arm interlock.** On every READY→OPERATIONAL arm, the pod holds its measured pose. It ignores the streamed command until that command is within **±5 %** of the actual position (`system_config.yaml:74-77`). A lag right after arming is expected.
- **FAIL_SAFE pose.** Pitch goes to −100 % (the axis minimum, i.e. the "safe" end, **I: fully depowered**) and roll goes to 0 % (centred) (FPA `flight_segment/control/motor_controller.py:128-140`).
- **Comm-loss watchdog.** `t_loss_s = 5.0` s of uplink silence fires comm-loss (`system_config.yaml:17-19`). It enters FAIL_SAFE automatically **only** when the last `gs_operating_status` is GENERATION, RECOVERY, LANDING or EMERGENCY (FPA `flight_segment/core/state_machine.py:31-33,351-354`). Recovery needs 5 consecutive good frames plus `fail_safe==0`.

### 1.3 FS state machine (`fs_operating_status`)

The states are INIT(7) → READY(0) ⇄ OPERATIONAL(1), plus:
- MAINTENANCE(3) ⇄ CONFIGURATION(6);
- FAULT(2), which is degraded flight on the surviving drive;
- FAIL_SAFE(4);
- EMERGENCY(5), which disables all drives.

Sources: FPA `common/datalink/enums.py:57-68`; `DATALINK_PROTOCOL.md:318-332`; `FLIGHT_SEGMENT_OVERVIEW.md:67-232`.

**Flying under pod control means `fs_operating_status==1`** (or 2/4 in degraded cases).

### 1.4 Flight phases

The pod itself has no notion of flight phase. Phases are the **ground-station PLC** states. They are mirrored onto the uplink as `gs_operating_status` and stored as `GS_UPLINK_COMMAND_gs_operating_status` (compact code) and `FS_MAINTENANCE_gs_operating_status` (raw PLC code, sent on change only).

| Raw PLC `ACTUAL_STATE` (HMI `stateToText.pipe.ts`) | Compact wire code (FPA `ground_segment/bridge/mapping.py:168-201`) | Physical meaning (I unless noted) |
|---|---|---|
| −1 no request | 0 NO_GS_CONNECTION | backend not connected to the PLC |
| 0 SELF TEST | 1 SELF_TEST | |
| 2 PREFLIGHT, 13 DISABLE DC POWER | 2 PREFLIGHT | on ground |
| 3 TAKE-OFF, 5 TRANSITION TO GENERATION, 9 ABORT TAKE-OFF, 14 BACK TO PREFLIGHT | 3 TAKEOFF | launch / climb-out |
| 4 GENERATION, 7 TRANSITION RECOVERY | 4 GENERATION | **traction / reel-out** (power phase, figure-8 or crosswind patterns) |
| 6 RECOVERY | 5 RECOVERY | **retraction / reel-in** (depowered) |
| 8 LANDING | 6 LANDING | |
| 10 SERVICE | 7 SERVICE | |
| 31 EMERGENCY | 8 EMERGENCY | |
| 1 ENABLE DC POWER, 15 POWERON | 9 POWERON | |
| 11 PRE-CHARGE, 12 CHARGE, other | 10 INVALID | |

**Caveat:** the compact code merges transitions. GENERATION includes TRANSITION_RECOVERY(7), and TAKEOFF includes TRANSITION_TO_GENERATION(5). For exact phase boundaries, use the raw PLC state. That is `FS_MAINTENANCE_gs_operating_status`, or the OPC source `DsStateMachine.ACTUAL_STATE` if it is in the DB.

### 1.5 What is sent down and at what rate

- **Downlink.** Each downlink frame is 55 B at **20 Hz**, sent on both radios. An arbiter dedups by counter (`DATALINK_PROTOCOL.md:12-16`). The variant cycles as `counter % 4` → A,B,C,D (FPA `flight_segment/transports/downlink_scheduler.py:3`):

| Block | Carried in | Effective rate | Fields |
|---|---|---|---|
| Header | every frame | 20 Hz | node_id, fs_id, pod_version, counter, fs_datalink_status, fs_operating_status, autopilot_status, controller_time, uplink_per |
| FlightData (IPOK) | A, B, C, D | 20 Hz (but IPOK sample is "latest", stale after 0.5 s) | pitch_angle, roll_angle, yaw_angle, velocity_az/el/r, azimuth, elevation, radius |
| Positions | A, B, C, D | 20 Hz | pitch, roll (fifth_line_length is always null) |
| Torques | A, C | 10 Hz | servo_a_torque, servo_b_torque |
| Temps + Status | B | 5 Hz | servo_a/b_temp, controller_temp, servo_a/b_status, gps/ipok/controller/wing_lights_status, force_power_depower_status, power/depower_mode_engaged (pod_temp always null) |
| Battery + Uptime | D | 5 Hz | voltage, current, charge, power, activity_time |
| Type E (config readback) | 4-frame burst on config commit or `get_servo_config` | event | goes to `BRIDGE.config_readback`. **This is a nested object and is NOT written to Influx** (HMI `python/utils/data_manager.py:344-349`) |

- **Path into InfluxDB:**
  1. The GS `HmiBridge` posts each received frame as `{"source":"FLIGHT_SEGMENT","data":…}` without throttling (`fs_min_period_s: 0.0`, FPA `system_config.yaml:117-120`). The payload contains **only that variant's fields** (sparse).
  2. The tk-hmi backend writes **one point per message**. The timestamp is the **backend's `time.time_ns()` at arrival**, not the pod clock (HMI `python/restapi/workers/managers/influxdb.py:82-105`).
  3. Null fields are omitted. That is why `pod_temp` and `fifth_line_length` never appear among the 39 fields.
- **Expected FLIGHT_SEGMENT point rate:** about 20 Hz minus downlink loss.
- **BRIDGE and GS_UPLINK_COMMAND:** posted every **0.5 s** (2 Hz), even when the link is dead (`hmi_bridge.py:55-56`).
- **FS_MAINTENANCE:** one point per HMI/backend command event, as a `label → value` single field (`data_manager.py:339-343`).

---

## 2. Field tables

### 2.1 `FLIGHT_SEGMENT` (39 fields)

Wire definitions: FPA `common/datalink/downlink_models.py` (DM), `DATALINK_PROTOCOL.md` (DP), and HMI `GS_BRIDGE_CONTRACT.md` (GBC).

**No-data sentinel:** each field's range minimum encodes "no data" and decodes to null (DM:55-72). Consequences:
- A genuine 0 for `yaw_angle`, `azimuth`, `radius` or `charge` is lost.
- A genuine value **at or below** the floor is also lost. `voltage` ≤ 37.0 V reads null, `elevation` = −90 reads null, and so on.

| Field | Meaning | Unit / wire scaling | Sign / frame / enum | Source in code | Conf. |
|---|---|---|---|---|---|
| `node_id` | Pod node id (bits 7..2 of byte 0) | int 0–63 | currently always 0 (unused) | DP:88, DM:13 | C |
| `fs_id` | Flight-segment identity | int 0–3 | 1=pod1, 2=pod2, 0=not configured (from `/etc/kite-node-id`). **Use it to tell spare pod from main pod.** | DP:88; `common/node_identity.py` | C |
| `pod_version` | Pod hardware version | 2-bit | **1=V2, 2=V3** (0,3 reserved). Not the legacy CSV codes (V2=2/V3=3) | enums.py:16-20; GBC:64 | C |
| `counter` | Downlink frame counter | uint32, +1 per TX at 20 Hz | monotonic, resets to 0 on FS restart (I: process-local). Gaps = lost frames. `counter % 4` = variant | DP:90; downlink_scheduler.py:3 | C |
| `fs_operating_status` | FS state machine state | int 0–7 | 0 READY, 1 OPERATIONAL, 2 FAULT, 3 MAINTENANCE, 4 FAIL_SAFE, 5 EMERGENCY, 6 CONFIGURATION, 7 INIT | enums.py:57-68; DP:318-332 | C |
| `fs_datalink_status` | FS view of uplink health | 0/1 | 1 = ≥5 consecutive uplink frames (sticky; drops after 5 misses) | DP:92,373; `common/transport/config.py:40` | C |
| `autopilot_status` | Autopilot mode | uint8 0–4 | 0 PILOT, 1 POWER_ZONE, 2 LOITER, 3 TAKEOFF, 4 LANDING. **Always 0: hard-coded, no autopilot** | enums.py:100-105; downlink_frame_builder.py:159 | C |
| `controller_time` | Pod RPi wall clock at frame build | float **Unix epoch s (UTC)**, µs resolution. Packed as "week + TOW" but computed from the Unix epoch, so there is no GPS epoch or leap-second offset | pod clock (NTP/GPS sync unknown). `controller_time` minus Influx `_time` gives latency plus clock skew | downlink_models.py:21-23; downlink_formatter.py:72-87; downlink_parser.py:114-126 | C |
| `uplink_per` | Uplink packet error rate seen by the FS | int % 0–100, 500 ms sliding window | link quality = 100 − per | DP:94; config.py:41 | C |
| `pitch_angle` | IPOK/XSens Euler **pitch** | ° (int16 /100), −90..+90 | XSens Euler[1]. Frame is the IPOK sensor frame in XSens default ENU convention (**I**). Mounting orientation relative to the kite is unknown | ipok_adapter.py:88-95; ipok/tasks.py:349-352 | C (source) / I (frame) |
| `roll_angle` | IPOK/XSens Euler **roll** | ° (int16 /100), −180..+180 | XSens Euler[0]; same caveats | same | C / I |
| `yaw_angle` | IPOK/XSens Euler **yaw/heading** | ° (uint16 /100), normalised to [0,360) | XSens Euler[2] mod 360. **I:** XSens ENU default means yaw 0 = sensor x-axis pointing **East**, positive **counter-clockwise** (not compass heading). Verify against `azimuth` trend. Wire 0 decodes to null | ipok_adapter.py:91; GBC:77 | C / I |
| `azimuth` | Kite azimuth seen from the GS anchor | ° (uint16 /100), [0,360) | **0° = North, +90° = East** (clockwise). Computed from IPOK LLA in GS-anchor ENU: `az = atan2(E,N)`. **Wraps at 0/360, and exactly 0 decodes to null.** Unwrap or recentre on wind direction before analysis | `common/geo/observer_enu.py:117-121`; ipok_adapter.py:97-110 | C |
| `elevation` | Kite elevation above the GS anchor horizontal | ° (int16 /100), −90..+90 | `el = asin(U/r)`, + = above the anchor. The anchor is the IPOK position surveyed at the GS (alt = IPOK antenna alt), not necessarily the tether exit point | observer_enu.py:121; `docs/ipok/IPOK_FIELD_PROCEDURE.md` | C |
| `radius` | Straight-line distance kite (IPOK) ↔ GS anchor | m (uint16 /10), 0–6553.5 | **Not tether length** (sag, anchor offset). Null when there is no fix or no anchor | observer_enu.py:119; DP:108 | C |
| `velocity_az` | Tangential speed in the azimuth direction | m/s (int16 /100) | `v_az = h·d(az)/dt`, with h = horizontal distance. **+ = azimuth increasing (moving clockwise / toward East seen from above)** | observer_enu.py:170-175 | C |
| `velocity_el` | Tangential speed in the elevation direction | m/s (int16 /100) | `v_el = r·d(el)/dt`. **+ = climbing** | observer_enu.py:172-175 | C |
| `velocity_r` | Radial range-rate dr/dt | m/s (int16 /100) | **+ = moving away from GS** (I: ≈ reel-out speed plus tether-sag effects). `v_mag² = v_az²+v_el²+v_r²` | observer_enu.py:164-169; DP:110 | C |
| `pitch` | Measured **depower-axis** position | % of axis limits (int16 /100), −100..+100 | 0 % = **centre** of [min,max], not zero mm. −100 % = min (fail-safe end). V3: composed from L/R. Defaults: V2 −2500..2000 mm (centre −250); V3 −4000..1000 mm (centre −1500). Limits can be changed by Config page 1 | motor_service.py:901-907; v3_strategy.py:264-289; motor_config.yaml:68-82; percent_converter.py:11-52 | C (scaling) / I (power vs depower direction) |
| `roll` | Measured **steering-axis** position | % (int16 /100) | 0 = centred. Uplink convention: **−100 = full left** (DP:199). Default limits ±300 mm | same; DP:199 | C / D (left) |
| `servo_a_torque` | Torque of servo A | **N·m** (int16 /100). DP says "N", but the code and GBC say N·m | **V2: pitch motor; V3: LEFT motor** (`pod_version`-dependent). Sign follows the drive convention. Motor-side vs output-side is unknown (gear ratio 0.125) | motor_service.py:141-146,909-912; motor_wrapper.py:204-210; GBC:94 | C / I |
| `servo_b_torque` | Torque of servo B | N·m | V2: roll motor; V3: RIGHT motor | same | C |
| `servo_a_temp` / `servo_b_temp` | Drive/motor temperature | °C int (int8 +73 offset), −55..200 | same A/B mapping. Drive shutdown at 80 °C (`motor_config.yaml:29`) | motor_service.py:918-921; DP:138-139 | C |
| `controller_temp` | RPi CPU temperature | °C int | | peripheral_service.py:281-286 | C |
| `servo_a_status` / `servo_b_status` | Drive state | 2-bit | 0 DISABLED, 1 ENABLED (CiA-402 OPERATION_ENABLED), 2 FAULT, 3 COMM_ERROR (unknown/stale). Busy (zeroing/calibrating) reports 0 | `flight_segment/control/axis_status.py:78-95`; enums.py:108-112 | C |
| `gps_status` | GNSS fix arriving from IPOK | 2-bit | 0 = no fix or IPOK disabled, 1 = fix valid, 3 = IPOK stream stale/lost. 2 (FAULT) is not produced | ipok_adapter.py:112-167 | C |
| `ipok_status` | RTK solution quality | 2-bit | 0 NO_DATA (or no RTK / IPOK disabled), 1 RTK_FLOAT, 2 RTK_FIX, 3 COMM_ERROR (stream stale > 0.5 s) | ipok_adapter.py:118-174; enums.py:115-119 | C |
| `controller_status` | Controller health | 2-bit | servo codes. **No producer: always 0** | downlink_frame_builder.py:167-171; GBC:105 | C |
| `wing_lights_status` | Wing lights | 2-bit | servo codes. **No producer: always 0** | same; GBC:106 | C |
| `force_power_depower_status` | Echo of the forced power/depower command | 2-bit | 0 OFF, 1 reserved, 2 POWER_ARMED, 3 DEPOWER_ARMED. **Stub: echoes the uplink command, no motor action** | motor_service.py:561-567; enums.py:131-135 | C |
| `power_mode_engaged` / `depower_mode_engaged` | Power/depower mode flags | 0/1 | **No producer: always 0** | downlink_frame_builder.py:167-171; GBC:108 | C |
| `voltage` | Battery voltage (I²C monitor VIN) | V. Wire is uint8 `(V−37)·10`, range 37.0–62.0 (0.1 V steps) | ≤ 37.0 V decodes to null. Pack is 44.4 V nominal, 3300 mAh (I: 12S) | power_sensor.py:120-123; sensor_config.yaml:3-5; DP:153 | C |
| `current` | Battery current | A (int16 /100) | Docs say "+ = supplied". In code the sense reading is unsigned, so in practice it is **always ≥ 0 = discharge** (I) | power_sensor.py:121-122; DP:154 | C / I |
| `power` | Battery power | W (int16 /10) | = pod electrical consumption (≥ 0) | power_sensor.py:123 | C |
| `charge` | Estimated state of charge | % int 0–100 | **Energy counter, not measured:** `100·(1 − E_used/146.5 Wh)`, where E_used is ∫P dt persisted across reboots and reset by `reset_battery_energy`. 0 % decodes to null | power_sensor.py:125-127,142-147; peripheral_service.py:269-279 | C |
| `activity_time` | Seconds since **FS process start** | s uint32 | Docs say "lifetime counter", but code = `monotonic() − start`. **A drop marks a pod/FS restart** | downlink_frame_builder.py:142,162; DP:157 | C (code wins) |
| `pod_temp`, `fifth_line_length` | (in the model, not in the DB) | | always null, no producer, so never written | GBC:88,100 | C |

### 2.2 `BRIDGE` (16 fields): GS-local link health, every 0.5 s

Source: FPA `ground_segment/bridge/mapping.py:87-123`; HMI GBC:120-161,245-256.

| Field | Meaning | Unit / values | Conf. |
|---|---|---|---|
| `gs_datalink_status` | GS view of downlink health (merged MAIN+BACKUP) | 0/1; 1 = ≥5 consecutive downlink frames (sticky) | C |
| `downlink_loss` | Convenience flag | bool = `gs_datalink_status != 1` | C (mapping.py:118) |
| `downlink_per` | Merged downlink PER | int % 0–100, 500 ms window; **dead or cold link = 100** | C |
| `seconds_since_last_rx` | Time since last accepted downlink frame | float s; null before the first RX | C |
| `link_source` | Radio that delivered the last selected frame | 0 = MAIN 2.4 GHz, 1 = BACKUP 868 MHz; null on the WS wire (enums.py:223-231) | C |
| `lost_packets` | **Cumulative** genuine counter gaps since GS start | int; grows monotonically, so diff it | C (`common/arbiter/counters.py:40-46`) |
| `missed_cycles` | Consecutive silent 50 ms cycles (resets on RX) | int | C (counters.py:48-59) |
| `main_link_status`, `backup_link_status` | Per-radio sticky status | 0/1; **null = radio absent** (≠ down) | C (`common/metrics/per_link_stats.py:16-41`) |
| `main_downlink_per`, `backup_downlink_per` | Per-radio PER | % (100 = present but receiving nothing); null = absent | C |
| `main_seconds_since_last_rx`, `backup_seconds_since_last_rx` | Per-radio silence | s / null | C |
| `camera_command` | Echo of the commanded camera level | 0/1 (open loop) | C |
| `upload_state` | GS config-upload state machine | string: idle, uploading, wait_type_e, done, timed_out, aborted | C (enums.py:262-271) |
| `upload_bitmap` | Config pages acknowledged | 0–7 (bit0=P1, bit1=P2, bit2=P3) | C |
| (`config_readback`) | Type-E PID/limit readback | object, **dropped by the Influx writer** | C |

There is **no RSSI** anywhere (GBC:158-161).

### 2.3 `GS_UPLINK_COMMAND` (13 fields): commanded uplink state, every 0.5 s

This is a snapshot of what the GS is actually sending at 20 Hz, whoever set it (HMI, Taranis or GS-local). Source: FPA `mapping.py:126-138`, `common/datalink/uplink_models.py:36-74`; HMI GBC:163-193.

Because the snapshot is taken at only 2 Hz, **stick dynamics above about 1 Hz are aliased**. For high-rate command analysis, use `FS_MAINTENANCE_pitch/roll` (event stream) or the TARANIS source.

| Field | Meaning | Values | Conf. |
|---|---|---|---|
| `pitch` | Commanded depower-axis setpoint | int % −100..100 ("−100 = down"); **null until the first real input** (wire sentinel 0x80) | C / D |
| `roll` | Commanded steering setpoint | int % −100..100, **−100 = full left**; null until first input | C / D |
| `autopilot_command` | Commanded autopilot mode | 0–4 (as `autopilot_status`); unused | C |
| `status_request` | Requested FS state (held/latched) | 0 NO_REQUEST, **1 READY, 2 OPERATIONAL (arm), 3 MAINTENANCE**. Legacy 0/1/2 codes are dead | C (enums.py:145-150) |
| `fail_safe` | Manual fail-safe level | 0/1 (1 → FS FAIL_SAFE in any GS state) | C |
| `emergency` | Emergency level | 0/1 (1 → FS EMERGENCY, drives off) | C |
| `zeroing_pitch_roll` | Zero both servos (rising edge, MAINTENANCE only) | 0/1 momentary | C |
| `reset_battery_energy` | Reset the charge energy counter | 0/1 momentary | C |
| `maintenance_motor_arm` | Energise drives in MAINTENANCE | 0/1 | C |
| `camera_command` | RunCam record (acts only in OPERATIONAL) | 0 stop, 1 record | C |
| `get_servo_config` | Request a Type-E readback | 0/1 momentary | C |
| `force_power_depower_command` | Forced power/depower | 0 normal, 1 reset, 2 force power, 3 force depower (FS stub only) | C (enums.py:138-142) |
| `gs_operating_status` | Ground-station PLC phase, compact code | 0–10, see §1.4 | C |

### 2.4 `FS_MAINTENANCE` (9 fields): command events (label/value)

One point per command event from the HMI, or from the backend (Taranis → `pitch`/`roll`; OPC → `gs_operating_status`). Translation: FPA `mapping.py:204-256`; HMI GBC:195-243.

| Field | Meaning | Values | Conf. |
|---|---|---|---|
| `pitch`, `roll` | Axis command event (HMI slider or **Taranis stick**, about 20 Hz while moving) | int/float % −100..100 | C / D |
| `gs_operating_status` | **Raw PLC `ACTUAL_STATE`** (not the compact code), sent **on change only** | −1, 0–15, 31 (see §1.4 left column) | C (mapping.py:223) |
| `status_request` | 1 READY, 2 OPERATIONAL, 3 MAINTENANCE | | C |
| `fail_safe`, `camera_command`, `maintenance_motor_arm` | Level commands, both edges sent | 0/1 | C |
| `reset_battery_energy`, `get_servo_config` | Momentary: 1, then 0 about 300 ms later | 0/1 | D |

Not present in this DB: `emergency`, `zeroing_pitch_roll`, `config_upload`, `config_abort`. They are possible labels.

---

## 3. Fields for identifying phases, cycles, engagement and trajectory

**Pod engaged (pilot has control authority):**
- `fs_operating_status == 1`;
- and `servo_a_status == servo_b_status == 1`;
- and `fs_datalink_status == 1`.

FAULT(2) means one drive is down. FAIL_SAFE(4) means pitch is at −100 % and roll at 0, with no steering. Arm events are `GS_UPLINK_COMMAND_status_request` 1→2, followed by `fs_operating_status` 0→1. Expect a short hold from the interlock (±5 %).

**Autopilot engagement:** cannot be identified. It is always 0 and has no producer.

**Flight phases / pumping cycles:**
1. **Primary:** `GS_UPLINK_COMMAND_gs_operating_status`. Here 4 = GENERATION (reel-out) and 5 = RECOVERY (reel-in), and each 4→5→4 alternation is one pumping cycle. Better: use raw `FS_MAINTENANCE_gs_operating_status` (4 GENERATION, 6 RECOVERY, 7 TRANSITION RECOVERY, 5 TRANSITION TO GENERATION), which gives sharper boundaries.
2. **Kinematic cross-check:**
   - `velocity_r` > 0 is sustained during reel-out and < 0 during reel-in;
   - `radius` shows a saw-tooth (rising during reel-out, falling during reel-in);
   - `pitch` (depower-axis position) shows a step change at phase transitions (I: toward −100 % = depowered in recovery);
   - `elevation` typically rises during reel-in.
3. Ground winch data (OPC: tether length, reel speed, force) should be the reference; the pod data supports it.

**Launch / landing / airborne:**
- gs code 3/6;
- `radius` and `elevation` leaving ground values;
- `ipok_status` / `gps_status` must be valid (1/2 and 1) for az/el/r to be non-null.

**Pod restarts:** `activity_time` drop, or `counter` reset.

**Trajectory and figure-8:**
- Position is spherical: (`azimuth`, `elevation`, `radius`) relative to the GS anchor. Velocity is (`velocity_az`, `velocity_el`, `velocity_r`).
- Convert to ENU: `E = r·cos(el)·sin(az)`, `N = r·cos(el)·cos(az)`, `U = r·sin(el)`.
- **Figure-8 detection:**
  - in GENERATION, `azimuth` (unwrapped about the mean wind direction) oscillates about the wind axis;
  - `velocity_az` changes sign twice per loop pair;
  - `elevation` shows two peaks per figure-8 (at the turns) and lobes on either side;
  - plot az vs el: a lemniscate means figure-8, a loop means circles;
  - correlate with `roll` (steering position) and the `GS_UPLINK_COMMAND_roll` sign changes that drive each turn.
- **Turn direction:**
  - Positive `velocity_az` means moving toward East (clockwise seen from above).
  - Roll −100 % means full left (uplink convention). The measured `roll` should follow the command after arming.
  - `yaw_angle` (I: ENU heading CCW from East) and `roll_angle` give the kite attitude. Mounting offset is unknown.
- **Pitfalls:**
  - azimuth wraps at 0/360 and exact 0 reads null;
  - FlightData comes at 20 Hz, but each value is the latest IPOK sample (about 100 Hz at source) and goes stale after 0.5 s, then null;
  - Influx `_time` is GS arrival time, so use `controller_time` or `counter` for exact spacing;
  - `radius` ≠ tether length.

**Speed metrics:** `v_mag = sqrt(v_az²+v_el²+v_r²)`. The existing post-processing uses `v_mag ≥ 1 m/s` together with force ≥ 10 kgf sustained 30 s as its flight detector (FPA `post_processing/common/flight_detector.py:46-66`).

---

## 4. Controller parameters relevant to performance

| Parameter | Value | Where | Conf. |
|---|---|---|---|
| Datalink tick (uplink + downlink) | 20 Hz | `system_config.yaml:10` | C |
| Comm-loss timeout (FS → FAIL_SAFE, gated by GS phase) | 5.0 s; recovery after 5 consecutive frames | `system_config.yaml:17-19` | C |
| Arm match interlock | ±5 % of stroke | `system_config.yaml:74-77` | C |
| Fail-safe pose | pitch −100 % (min), roll 0 % | `motor_controller.py:128-140` | C |
| Position limits V2 | pitch −2500..2000 mm, roll −300..300 mm | `motor_config.yaml:68-75` | C (defaults; overridable by Config pages, readback not stored in Influx) |
| Position limits V3 | pitch −4000..1000 mm, roll −300..300 mm (L/R motor limits derived from worst-case corners) | `motor_config.yaml:76-82`; `v3_strategy.py:76-104` | C |
| Velocity-mode limit margin | 5 mm | `motor_config.yaml:84-87` | C |
| MD80 drive PID (default) | kp 25, ki 0.1, kd 0; kp_vel 1, ki_vel 0, kd_vel 0; integral limits vel 15, pos 1 | `motor_config.yaml:92-109` | C |
| Drive limits | max_torque 2500 (drive units, I: mN·m), max_current 3043 (I: mA), rated_current 6900, max_speed 1000 (rpm, I), torque constant 0.199 N·m/A, gear ratio 0.125, 21 pole pairs | `motor_config.yaml:93-101` | C / I (units) |
| Profile velocity / acceleration | 150 / 2000 | `motor_config.yaml:32-34` | C |
| Pulley diameter, encoder | 70.03 mm, 16384 inc/rev, tolerance 50 inc | `motor_config.yaml:4-11` | C |
| Drive over-temperature shutdown | 80 °C | `motor_config.yaml:29` | C |
| Battery | 44.4 V nominal, 3300 mAh (146.5 Wh) | `configs/sensor_config.yaml:3-5` | C |
| IPOK stale timeout | 0.5 s | `system_config.yaml:155` | C |
| Velocity-mode auto-stop near limits | 20 Hz loop | `motor_service.py:935-940` | C |

No autopilot setpoints exist on the pod. Flight-path setpoints (target elevation, reel speeds, force limits) belong to the GS PLC / OPC side, not to this repo.

---

## 5. Open questions

1. **XSens Euler frame:** is the MTi in the default ENU convention (yaw 0 = East, CCW+) or NED? How is the IPOK box mounted relative to the kite or pod body axes? `pitch_angle`, `roll_angle` and `yaw_angle` cannot be read as kite attitude until this is known.
2. **IPOK location:** is it on the kite (bridle/pod) for every flight, or was it ever on the ground? Which campaigns had IPOK off? Then az/el/r/v are null and `ipok_status` = 0 or 3.
3. **Pitch direction:** does −100 % pitch (axis minimum) mean fully *depowered*? The fail-safe logic implies yes; the uplink doc says "−100 = down". Which roll sign turns the kite left in the az frame?
4. **Position limits used in flight:** limits may have been re-committed via Config pages. `BRIDGE.config_readback` is dropped by the Influx writer, so the effective mm↔% mapping must come from logs (`ground_segment` packet CSVs) or from the operator.
5. **Torque:** motor-side or output-side (gear ratio 0.125)? What is the sign convention? Units are N·m per code; the DP doc says "N".
6. **`current` sign:** the contract says "+ = supplied, − = sunk", but the sensor code is unsigned. Can it ever be negative (regeneration)?
7. **`controller_time` sync:** is the pod clock disciplined by NTP or GPS? Needed before using it to correct Influx `_time`.
8. **GS anchor:** at which point was it surveyed (tether exit, winch, or another spot)? This affects elevation/radius offsets. It is persisted on the pod (`state/gs_anchor.yaml`); confirm it per campaign day.
9. **Compact `gs_operating_status`** merges transition states. Is the raw OPC `DsStateMachine.ACTUAL_STATE` stored at a higher rate in the OPC source? It would be better for cycle segmentation.
10. **`counter` behaviour across FS restarts:** does it reset to 0, as the arbiter's "restart backjump" logic implies (`common/arbiter/counters.py:28-36`)?
11. **Protocol versions:** the protocol changed during June–July 2026 (trailer removed, servo A/B renamed, `fifth_line_length`). Data in the DB from before about 2026-07 may follow an older field layout or older enums (e.g. old `status_request` 0/1/2 or `fs_operating_status` 7 = emergency). Check the date range before applying this dictionary.
12. `autopilot_status`, `controller_status`, `wing_lights_status`, `power_mode_engaged`, `depower_mode_engaged` and `force_power_depower_status` carry no physical information today. Should they be excluded from analysis?
