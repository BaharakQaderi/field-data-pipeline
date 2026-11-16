# TK10 ground-station data dictionary (from the HMI / backend repo)

Source repo: `Kite-projects/tk-hmi-monorepo` (read-only survey, HEAD of 2026-10-23).
Paths below are relative to that repo. "R:" is shorthand for `python/restapi/workers/`.
Confidence tags: **[code]** stated in code, **[doc]** stated in docs or the PLC engineer's
annotations in `docs/OPC_variable_decisions.xlsx`, **[HMI]** inferred from how the HMI labels or draws it,
**[infer]** my inference, needs to be checked against data.

---

## 1. System overview (as the HMI sees it)

TK10 is a pumping kite system. The ground station (GS) holds a winch drum driven by **two motors**:
a **GENERATOR** motor (high torque, INV2) that brakes the drum during reel-out, and a **RECOVERY** motor
(brushless, faster/low torque, INV1) that reels in. A third drive (INV3) moves the **thread guide**
(level-wind, "guida filo"), which follows the drum through an "electronic CAM"
(`DsEncoder.RelateGuidaFiloWithEncoder`, coefficient `DsInverters.TamburoGuidaFiloCoeff`).
Sources: inverter names `['', 'RECOVERY', 'GENERATOR', 'THREAD GUIDE'][m]`
(`webapp/.../synoptic-panels/inverters-panel/inverters-panels.component.html:438`, `:300-302`); electrical
synoptic "RECOV. MOTOR / GENER. MOTOR / THREAD G. MOTOR" (`.../electrical-system/electrical-system-panels.component.html:106-128`).
note.md:623 confirms "one machine but two different motors, one for generation one for recovery" and
"4 load cells placed under the drum".

Electrical chain drawn by the HMI (top to bottom, `electrical-system-panels.component.html:88-390`):

```
 RECOV. MOTOR   GENER. MOTOR   THREAD G. MOTOR      EXTERNAL SUPPLY      BRAKE (CNT6)
      |              |                |                   |
    INV1           INV2             INV3               CNT5 (EXT_SUP_CNT_*)
      |              |                |
    CNT1           CNT2             CNT3   (INVn_CNT_EN/FBK)
 ===================== HVDC BUS (>500 V = live; max of 4 converter HS voltages) =========
      |              |                |              |            CNT4 -> BRAKE RESISTOR
    CONV1          CONV2            CONV3          CONV4    (DC/DC, HS = HVDC side, LS = battery side)
      |              |                |              |
    BAT1           BAT2             BAT3           BAT4     (~48 V packs, gauge 43.5-53.5 V)
```

* **Brake**: spring brake released when contactor CNT6 `BRAKE_CNT_EN` is TRUE ("BRAKE RELEASED"),
  engaged when FALSE (`home.component.html:102`, `electrical-system-panels.component.html:93-95`). **[HMI]**
* **Brake resistor** (dump load on HVDC): contactor CNT4 `BR_RES_CNT_EN/FBK`, controller fault `BR_RES_CTRL_FAULT_N` (active-low).
* **Load cells**: 4 CANopen cells (nodes 5-8) under the drum, `DsLoadCells.MeasureFloat[1..4]`, HMI unit **kg**,
  summed in the backend. They measure drum-frame reaction, not tether tension directly (note.md:623-625). **[doc]**
* **Drum encoder**: `DsEncoder.outTamburo_*` (tamburo = drum): rpm, m/s, wound/paid-out length, angle.
* **Wind station** on a pole: TCP, speed m/s + direction "from the north" (note.md, Q7). Source `WIND`.
* **Pilot inputs**: Thrustmaster TCA throttle quadrant (source `THROTTLE`), which drives the winch speed and torque set-points
  plus the state buttons on the PLC. FrSky Taranis RC (source `TARANIS`) flies the pod's pitch/roll axes through the uplink.
* **Bridge / datalink**: external process (`flight-pod-automation-v2`) translating the 20 Hz, 55-byte radio downlink
  (main 2.4 GHz + backup 868 MHz, arbitrated) into `FLIGHT_SEGMENT`, `BRIDGE`, `GS_UPLINK_COMMAND`
  (`docs/GS_BRIDGE_CONTRACT.md`). The backend in turn sends HMI/Taranis/PLC-state commands to the pod over
  `FS_MAINTENANCE` labels.
* **PLC**: Phoenix Contact AXC F 2152 (PLCnext), OPC UA `opc.tcp://192.168.1.10:4840/`, node prefix
  `ns=6;s=Arp.Plc.Eclr/` (`python/settings.py:30`, `R:managers/opc.py:192`). `PLC_MODEL = TK10`.

**Not available anywhere in this data**: battery SoC and battery temperature (`webapp/TODO.md:18-21` "Data missing"; no
`DsBattery` node in `nodes_types.json`). There are also no motor or inverter temperatures (only the I²t warnings
`DsAlarms.IIT_WARNING[1..3]` and converter over-temp booleans). Tether tension is only available as the drum load-cell sum.

---

## 2. Sources: how they are produced, and their rates

All points go to measurement `tk`, tag `source=<SOURCE>`, field `<SOURCE>_<key>`
(`R:managers/influxdb.py:98-104`). Timestamps are taken at **backend receive time** (`time.time_ns()`), except OPC,
which is stamped with the backend's `last_update_epoch` at snapshot publish. That is the backend clock, **not** a PLC
SourceTimestamp (`R:managers/opc.py:245-247`, `python/utils/data_manager.py:333-337`; requesting PLC SourceTimestamp is
open item R2 in `docs/PLC_CHANGE_REQUEST.md:107-109`). The backend clock may be set once from XSens, but XSens is
disabled (`data_manager.py:259-320`).

| Source | Producer | Rate | Notes |
|---|---|---|---|
| **OPC** | `R:managers/opc.py` OPC UA client to the PLC | **10 Hz snapshots** (since 2026-09-30): subscription sampled every `opc_interval=0.1 s`, queue size 1; a full snapshot of **every** listed field is published every 100 ms, whether it changed or not (`opc.py:281-352`, `settings.py:18-21`). Falls back to polling at ≥200 ms (≤5 Hz) if the subscription goes stale (`opc.py:27,146-159`) | Before 2026-09-30 (commits d8b8d40/2e16c6e) it polled everything at ~3-5 Hz and logged all ~1039 fields. Values are **sample-and-hold**: a field repeats its last value. PLC task rates: load cells, converter feedback, encoder, inverter feedback, E-stop at 20 Hz; inverter commands, contactors, limit switches at 10 Hz; state machine, converter commands, analog, brake-resistor at 5 Hz; battery at 1 Hz (`OPC_variable_decisions.xlsx` sheet PLC_Rates, confirmed by the PLC engineer). Load cells update **only in flight states**, and the drum encoder from Power ON onward. Array `X[n]` in the PLC becomes fields `X[1]..X[n]`, **1-based** (`opc.py:196-202`) |
| **OPC_CONFIG** | same OPC snapshot, split by `opc_nodes.txt` mode `config` | **on change** + once per PLC (re)connection (`R:managers/opc_node_list.py:113-134`, `opc_nodes.txt:1-25,445+`) | Sparse settings and set-points. Forward-fill to use them. Same timestamp as the OPC point |
| **WIND** | `R:managers/windstation.py` TCP client to `192.168.1.20:40737`, sends `R|102|202`, parses `W|a|b|c|d` | poll every 0.2 s, so **≤5 Hz** (`settings.py:34`) | Called **METEO** (`meteo.*` keys) before 2026-10-07 (commit c48ca6b) |
| **THROTTLE** | `R:thrustmaster_throttle.py` evdev "Thrustmaster TCA Quadrant Boeing 1&2" | **event-driven** (on axis movement) + `connected` ping 1 Hz | Each event is also **written to the PLC** as the same OPC node (`data_manager.py:137-143`) |
| **TARANIS** | `R:taranis.py` evdev "OpenTX FrSky Taranis Joystick" | **event-driven** + `connected` ping 1 Hz | Pitch/Roll are re-sent as `FS_MAINTENANCE` pitch/roll unless the pod is in MAINTENANCE (`data_manager.py:187-224`) |
| **WS** | HMI browser commands `{label, value}` over `/ws` (default source WS, `python/restapi/app.py:186-187`) | **event-driven** (operator clicks) | Each one is an **OPC write request** (`data_manager.py:145-147`); field = `WS_<OPC node>`. A log of operator commands, not PLC state |
| **FS_MAINTENANCE** | HMI maintenance panel + backend (Taranis axes, PLC state mirror) | event-driven | Uplink commands to the pod, `{label,value}` gives field `FS_MAINTENANCE_<label>` (`data_manager.py:340-343`) |
| **GS_UPLINK_COMMAND** | bridge read-back of the commanded uplink | 1-2 Hz, always complete | `GS_BRIDGE_CONTRACT.md:163-193` |
| **BRIDGE** | bridge link-health watchdog | 1-2 Hz, always complete, even when the link is dead | Called **GROUND_SEGMENT_INFO** before 2026-10-07 (commit a2ad672). `config_readback` (a dict) is dropped before Influx (`data_manager.py:345-349`) |
| **SYSTEM** | `R:timesync.py` | 1 Hz | backend clock heartbeat |
| (LOADCELL) | `R:managers/loadcell.py`, added 2026-10-20 | stream | Separate load-cell server on ws://192.168.1.60:8766, `loadcell.force_kg`, `loadcell.connected`. Not in your field lists yet |

Liveness: a source counts as "active" if it updated within 2 s; errors clear after 5 s (`python/utils/timed_monitoring.py:91-106`).

---

## 3. OPC fields by subsystem (analysis-relevant)

Index conventions:
* `ConvStruct.*[1..4]` = DC/DC converter CONV1..CONV4. Each LS side is battery BAT1..BAT4 (HMI "V BAT i"/"I BAT i", `opc.config.ts:61-69`). **[HMI]**
* `DsInverters.*[1..3]` = INV1 RECOVERY motor, INV2 GENERATOR motor, INV3 THREAD GUIDE (`inverters-panels.component.html:300-302,438`). **[HMI]**
* `DsLoadCells.*[1..4]` = load cells LC1..LC4 (CANopen nodes 5-8, `PLC_CHANGE_REQUEST.md:33`). Mounting position of each index is unknown.
* `in*` = drive/device → PLC (feedback); `out*` = PLC → drive (command), per the PLC engineer (`PLC_CHANGE_REQUEST.md:25-28`). **[doc]** Caveat: for DsEncoder, `out*` fields are clearly *outputs of the encoder block* (measurements), so treat the rule as reliable for DsInverters only.
* `*_SCALED` = engineering units; the raw twin is in the PLC's own units. The PLC engineer says use the **raw** `CONV_WRITE.*_I_MAX` instead of the scaled ones; for the CONV_READ hysteresis, use the scaled one (`PLC_CHANGE_REQUEST.md:38-48`; unresolved).

### 3.1 Power: converters, battery, HVDC

| Field (prefix `OPC_`) | Meaning | Unit | Sign / values | How derived | Conf. |
|---|---|---|---|---|---|
| `ConvStruct.CONV_READ.CONV_MEAS_FB_LS_VOLTAGE_SCALED[i]` | Battery-side (LS) voltage of converter i = battery i voltage | V | ~43.5-53.5 V (HMI gauge: <45 warn, <43.5 alarm, `opc.config.ts:5-19`) | PLC (converter CAN feedback, 20 Hz) | [HMI] unit (config.ts says "XXX" placeholder, but the electrical panel prints V) |
| `...CONV_MEAS_FB_LS_AVG_CURR_SCALED[i]` | Battery-side average current | A | ±200, zero-centred (`opc.config.ts:21-42`). **V·I>0 drawn as power flowing *into* the battery (charging)**, <0 = discharging (`electrical-system-panels.component.html:340-353`) | PLC | [HMI] sign |
| `ConvStruct.CONV_READ.CONV_MEAS_FB_LS_PWR_SCALED_CALC` | **Total battery-side power, all 4 converters** | **kW** | + = net charging (generation), − = net consumption (HMI "TOTAL PWR" gauge −10..+10 kW, negative side coloured alarm, `opc.config.ts:81-95`) | **Backend:** `Σ_{i=1..4} LS_VOLTAGE_SCALED[i] × LS_AVG_CURR_SCALED[i] / 1000` (`R:managers/opc.py:207-236`). The 4 terms come from one backend snapshot, not one PLC cycle (`PLC_CHANGE_REQUEST.md:111-118`) | [code] formula; [HMI] sign |
| `...CONV_MEAS_FB_HS_VOLTAGE_SCALED[i]` | HVDC bus voltage seen by converter i | V | bus "live" if max over i > 500 V (`electrical-system-panels.component.html:281-288`, `home.component.ts:223-230`) | PLC | [HMI] |
| `...CONV_MEAS_FB_HS_AVG_CURR_SCALED[i]` | HVDC-side average current | A | sign as HS power | PLC | [HMI] |
| `...CONV_MEAS_FB_HS_AVG_PWR_SCALED[i]` | HVDC-side average power | W | >+5 W drawn as flow from the bus *into* the converter (toward battery), <−5 W reverse (`electrical-system-panels.component.html:290-304`) | PLC | [HMI] |
| `...CONV_POWER_SCALED_HS[i]`, `...CONV_POWER_SCALED_LS[i]` | Per-converter HS/LS power ("TOTAL is the sum of the 4", the PLC engineer) | W? | probably same sign as above | PLC | [doc] meaning, unit [infer]. The `_TOTAL_HS/_LS/_DIFF` totals exist in the PLC but are not logged |
| `...CONV_STATUS_FB_HS_CURRENT_REFERENCE_CC_SCALED[i]` / `..._LS_...` | Current set-point in CC (constant-current) mode | A | | PLC feedback | [HMI] label "HS/LS CURRENT TARGET" |
| `...CONV_HSVC_FB_HS_VOLTAGE_REFERENCE_SCALED[i]` / `...CONV_LSVC_FB_LS_VOLTAGE_REFERENCE_SCALED[i]` | Voltage set-point in HS / LS voltage-control mode | V | | PLC | [HMI] |
| `...CONV_CONF_FB_MODE[i]` | Converter mode | enum | 0 AUTONOMOUS, 1 CC, 3 EXTENDED (§5) | PLC (Float type) | [HMI] |
| `...CONV_STATUS_FB_POWER_DIRECTION[i]` | Allowed current direction while in CC mode | enum | 0 BIDIRECTIONAL, 1 HS CHARGE ONLY, 2 HS DISCHARGE ONLY | PLC | [HMI]+[doc] |
| `...CONV_EXT_MODE_FB_POWER_FLOW_DIRECTION[i]` | **Configured** flow direction (not the actual flow) | enum | same 0/1/2 | PLC | [doc] ("related to the configuration … not to the actual direction") |
| `...CONV_EN_DIS_FB_ERROR_CODES[i]` / `..._ERROR_VALUE[i]` | Converter error code + value | enum / V, A, °C by code | §5 | PLC | [HMI] |
| `...CONV_EN_DIS_FB_WARNING_CODES[i]` | Warning code | int | mapping unknown | PLC | — |
| `...CONV_EN_DIS_FB_POWER_STAGE_EN[i]` | Power stage enabled ("HW ENABLE") | bool | | PLC | [HMI] |
| `...CONV_EN_DIS_FB_SAFETY_DISCONNECTION_EN[i]` | Safety disconnection ("SAFETY DISC.") | bool | logic "to be checked on converter manual" | PLC | [doc] |
| `...CONV_EN_DIS_FB_VVVF_STATUS[i]` | HVDC link status column | bool | | PLC | [HMI] |
| `...CONV_STATUS_FB_CONV_EN[i]` | Converter enabled feedback | bool | | PLC | [infer] |
| `ConvStruct.CONV_WRITE.CONV_CTRL_CONV_EN[i]` | PLC command: enable converter | bool | | PLC → converter | [infer] |
| `ConvStruct.CONV_WRITE.CONV_CTRL_LS_CURRENT_REFERENCE_CC(_SCALED)[i]` | PLC command: LS current set-point in CC mode | raw / A | | PLC → converter | [infer] |
| `ConvStruct.CONV_WRITE.CONV_CURR_{HS,LS}_{CHG,DISCH}_I_MAX[i]` | Current limits (raw units) | raw | | PLC → converter | [doc] |
| `ConvStruct.CONV_WRITE.CONV_EXT_MODE_CAN_CHECK_MODE[i]` | CAN watchdog | enum | 0 KEEP LAST, 1 100 ms, 2 1 s, 3 10 s | | [HMI] |
| `ConvStruct.CONV_WRITE.CONV_MEAS_ENABLE[i]` | Measurement messages enabled | bool | | | [infer] |
| `ConvStruct.inSend/inAutonomous/inCharge/inSync/inConv_ID/inMSG_ID` | HMI → PLC converter-messaging handshake (send config msg, autonomous mode, init charging, sync) | bool / id | | written by the HMI (`home.component.ts:166-238`) | [code] |
| `CONVn_PWR_EN`, `CONVn_SAFE_DISC_EN` (n=1..4) | Converter power enable / safe-disconnect digital outputs | bool | | PLC I/O | [infer] |
| `CONVn_STATUS_FBK` | Converter status feedback | bool | TRUE = enabled (green) | PLC I/O | [HMI] (`converters-panels.component.html:786-851`) |
| `CONVn_EARTH_FAULT_N` (n=2..4; n=1 only in DsAlarms) | Earth fault, **active-low** | bool | FALSE = fault | PLC I/O | [HMI] |
| `CONVn_OVERTEMP_FAULT` | Over-temperature, active-high | bool | TRUE = fault | PLC I/O | [HMI] |
| `EXT_SUP_CNT_EN` / `EXT_SUP_CNT_FBK` | External supply contactor CNT5 (pre-charge / charging from external source) | bool | FBK TRUE = closed | | [HMI] |
| `DsExternalData.ChargeAllReq` | Request to charge all batteries (state 12 CHARGE prompt) | bool | | HMI write | [HMI] |
| `DsExternalData.Voltage_24V_Logic` / `Voltage_24V_Power` | 24 V auxiliary rails | V | `_Power` is not really used (shorted to Logic) | PLC analog 5 Hz | [doc] |

### 3.2 Inverters / motors (INV1 recovery, INV2 generator, INV3 thread guide)

Drives follow CiA 402 (ControlWord/StatusWord, modes of operation). There are **no temperatures**.

| Field | Meaning | Unit | Sign / values | Derivation | Conf. |
|---|---|---|---|---|---|
| `DsInverters.Torque_ActualValue[i]` | **Actual motor torque** | **Nm** (pilot page `<small>Nm</small>`, `hmi-pilot.component.html:148,158`) | HMI bars 0-based: INV1 0-50 Nm (warn 26, alarm 40), INV2 0-250 Nm (warn 178, alarm 215) (`opc.config.ts:164-185`). **Sign not documented**: the bars assume ≥0 in normal use | drive feedback, 20 Hz, Float | [HMI] unit; sign open |
| `DsInverters.Torque_Target[i]` | Torque target (UInt16). Pilot page draws it as the target marker on the torque bar (`hmi-pilot.component.html:142,154`) | Nm? (unsigned) | ≥0 only | PLC → drive | **conflicting**: `opc.config.ts:97-116` labels `[1]` "BRUSHLESS" RPM 0-800 and `[2]` "TORQUE" A 0-100 (stale config); the PLC engineer says the "Torque_" prefix is misleading |
| `DsInverters.Torque_ProportionalTarget[i]` | Target shown when the drive is in torque mode (inOperationMode==4) | %? | | PLC | [HMI] usage, unit [infer] |
| `DsInverters.Velocity_ActualValue[i]` | Actual motor speed (Int32) | rpm? (drive units) | signed; direction convention unknown | drive feedback | [infer] |
| `DsInverters.Velocity_Target[i]` | Velocity target (Int32). Home page shows `[1]` as "Brushless" ±800 **RPM** (`opc.config.ts:117-139`, `home.component.html:54`) | rpm | signed | PLC → drive | [HMI] |
| `DsInverters.Power[i]` | Inverter (motor) electrical power | **W** (`electrical-system-panels.component.html:159`) | **>+5 W drawn as flow motor → inverter → HVDC (generating); <−5 W = motoring** (`:102-128,150-160`) | drive feedback | [HMI] sign |
| `DsInverters.Current[i]` | Inverter current | A (`:161`) | | drive | [HMI] |
| `DsInverters.Voltage[i]` | Inverter (DC-link) voltage | V (`:231`) | | drive | [HMI] |
| `DsInverters.ProfilePosition_PositionActual[i]` / `ProfilePosition_Target[i]` | Position (Int32, counts). Mainly relevant for INV3 thread guide in Profile Position mode | counts | | | [HMI] |
| `DsInverters.inOperationMode[i]` / `outOperationMode[i]` / `OperationMode[i]` | Mode of operation: in = actual (from drive), out = commanded (to drive). `OperationMode` is a probable duplicate | enum | -1 Unknown, 1 Profile Position, 3 Profile Velocity, 4 Torque, 6 Homing, 7 Interpolated position (`inverters-panels.component.html:340-346,451-457`) | | [HMI]+[doc] |
| `DsInverters.StatusWord[i]`, `ControlWord[i]` | CiA 402 status/control words | UInt16 bitfield | standard CiA 402 bits | | [infer] |
| `DsInverters.StateDrive_{NotReadyToSwitchON,SwitchONDisabled,ReadyToSwitchON,SwitchedON,OperationEnable,Fault}[i]` | Decoded CiA 402 state (one-hot booleans; redundant with StatusWord) | bool | `OperationEnable` = drive active | PLC decode | [doc] |
| `DsInverters.ErrorCode[i]` | Drive error code | UInt16 | vendor table not in repo | | — |
| `DsInverters.Warning[i]`, `InternalLimitActive[i]` (HMI "IIT LIMIT ACTIVE"), `TargetReached[i]`, `DriveIsMoving[i]`, `Voltage_enabled[i]`, `INVPowerStageEnabled[i]` | Drive status bits | bool | | | [HMI] |
| `DsAlarms.IIT_WARNING[i]` | I²t (thermal overload) warning | bool | | | [HMI] |
| `DsInverters.{Shutdown,SwitchOn,DisableVoltage,EnableOperation,DisableOperation,QuickStop,INVForceQuickStop,Halt,ResetFault,NewSetPoint}[i]` | CiA 402 command bits (HMI service panel) | bool | QuickStop ≈ INVForceQuickStop (the PLC engineer: "same") | HMI/PLC | [doc] |
| `DsInverters.IncreaseTarget/DecreaseTarget[i]`, `IncreaseJogTarget/DecreaseJogTarget[i]` | Jog buttons. `[2]` jog is used during take-off (`home.component.html:196-222`) | bool | | HMI | [HMI] |
| `DsInverters.InverterEnable/InverterDisable/InverterNumber`, `InvertersPDORead/Write_*`, `InActivatePDO*_Service[i]` | Service / CANopen PDO management | | | | [HMI] |
| `DsInverters.inHoming`, `Homing_ProprotionalTarget` | Thread-guide homing | | "not used" (the PLC engineer) | | [doc] |
| `DsInverters.inEnableSpeedLimit/inDisableSpeedLimit` | Speed-limit enable | bool | meaning unverified | | — |
| `DsInverters.ResetInvertersControlWord` | reset of control words | bool | | | — |
| `INV1_CNT_EN/FBK`, `INV2_…`, `INV3_…` | Inverter power contactors (HMI "INVn PWR CNT") | bool | FBK TRUE = closed | | [HMI] |
| OPC_CONFIG: `DsInverters.TorqueMaxScale_{Torque,Velocity}`, `BrushlessMaxScale_{Torque,Velocity}`, `Torque_MaxTorque[i]`, `Torque_MotorRated[i]`, `Torque_Slope[i]`, `Velocity_currentLimit`, `Velocity_Window/Threshold[i]`, `ProfilePosition_*`, `Homing_*`, `JogSpeedTarget[i]`, `*_hmiTarget[i]`, `VelocityIncrementOffset`, `TamburoGuidaFiloCoeff` | Scaling of throttle % → torque/velocity for GEN ("Torque…") and REC ("Brushless…"), plus CiA 402 limits | various | `Torque_MotorRated` gives the rated torque, needed if any torque is in ‰ of rated (CiA 402 convention) | config | [infer] |

### 3.3 Drum encoder: speed, position, tether length

| Field | Meaning | Unit | Sign / values | Conf. |
|---|---|---|---|---|
| `DsEncoder.outTamburo_SpeedRPM` | **Drum speed** | rpm | signed; HMI shows ▲ when ≥0 and ▼ when ≤0 (`home.component.html:66-74`). Whether ▲ means reel-**out** is **not stated** | [HMI] |
| `DsEncoder.outTamburo_SpeedMSec` | **Tether (line) speed at the drum** | m/s | same sign as RPM presumably | [HMI] (`home.component.html:78`) |
| `DsEncoder.outTamburo_WindedLength` | Tether length ("Winded", "WINDED LINE") | m | HMI scale 0-240 m; colour bands <55 alarm, 55-95 warn, 95-170 OK, 170-210 warn, >210 alarm (`opc.config.ts:186-209`); pilot comment: "70-170 m is the working window" (`hmi-pilot.component.ts`, readoutBand). The bands imply the **paid-out** length despite the name | [HMI]; "winded" vs "unwound" open |
| `DsEncoder.outTamburo_Angle` | Drum angle | deg? | | [infer] |
| `DsEncoder.outPosition`, `outSpeed` | Raw encoder position / speed (UInt32, so no sign; wraps?) | counts | | [doc] (raw) |
| `DsEncoder.outLimitSwitchMin/Max`, `inLimitSwitchMin/Max` | Soft drum-revolution limits | counts | | [infer] |
| `DsEncoder.inZeroRequest`, `inActivate`, `EncoderPDORead_ON` | Zeroing request, encoder activation, PDO | bool | | [infer] |
| `DsEncoder.RelateGuidaFiloWithEncoder` | Electronic CAM ON: thread guide slaved to the drum | bool | FALSE shows warning "el. cam not active" (`home.component.html:115`) | [HMI] |
| `CABLE_MOTOR_LIMIT1/2` | Thread-guide ("G.FILO") end-of-travel limit switches | bool | (`opc.config.ts:71-72`) | [HMI] |
| `DsAlarms.LIMIT_SWITCH1_FAULT/2_FAULT` | "thread limit A/B reached" (red) | bool | | [HMI] |
| `DsAlarms.DRUM_ENCODER_SPEED_FAULT` / `_MAXREV_FAULT` / `_MINREV_FAULT` / `ENCODER_COMM_FAULT` | drum over-speed / upper rev limit / lower rev limit / encoder comms | bool | | [HMI] |
| OPC_CONFIG `DsEncoder.inResolution/outResolution/inPresetValue/outPresetValue/inMode/inOperatingMode`, `DsInverters.TamburoGuidaFiloCoeff` | Encoder resolution & preset, drum↔guide ratio | | needed to convert raw counts | [infer] |

**Drum geometry constant** (the only one in the repo): the pilot page converts drum rpm to line speed as
`SpeedRPM / 60 * 0.491 * 3.14` m/s (`webapp/.../pilot/hmi-pilot.component.html:168,174`). That implies an **effective drum
diameter of 0.491 m** (radius ≈ 0.2455 m, circumference ≈ 1.542 m). The home page instead shows the PLC's own
`outTamburo_SpeedMSec`. Check that SpeedMSec ≈ SpeedRPM × 0.0257 in the data. No gear ratios between motors and drum
are given anywhere in the repo. Torque → tether force needs the ratio and radius: F ≈ T·i/r (unknown i). **[code] for
the constant, [infer] for its meaning**

### 3.4 Load cells / tether force

| Field | Meaning | Unit | Notes | Conf. |
|---|---|---|---|---|
| `DsLoadCells.MeasureFloat[1..4]` | Individual load-cell readings (4 cells under the drum) | **kg** (kgf) | PLC 20 Hz, **flight states only**. ×9.81 for N | [HMI] unit (`opc.config.ts:45-60`), [doc] rate |
| `DsLoadCells.MeasureFloat_SUM` | **Total load-cell force** | kg | **Backend:** `MeasureFloat[1]+[2]+[3]+[4]` (`R:managers/opc.py:237-243`). HMI bar −50..500 kg, warn 400, alarm 450 | [code] |
| `DsLoadCells.Status[1..4]` | Per-cell status word | Int16 | meaning unknown | — |
| `DsAlarms.LOAD_CELLS_COMM_FAULT/GENERIC_FAULT` | load-cell alarms | bool | | [HMI] |

The sum is the vertical/frame reaction of the drum support, **not** tether tension directly. It depends on tether
elevation and azimuth and on drum weight (tare). Calibrate against `FLIGHT_SEGMENT` elevation if needed. **[infer]**

### 3.5 Brakes

| Field | Meaning | Values | Conf. |
|---|---|---|---|
| `BRAKE_CNT_EN` | Mechanical drum-brake contactor CNT6 | **TRUE = brake RELEASED**, FALSE = ENGAGED ("brake engaged" warning when FALSE) | [HMI] `home.component.html:102,284-290` |
| `DsExternalData.BrakeManualReq` | HMI request to toggle the brake (also in WS) | bool | [HMI] |
| `DsExternalData.PanelBrakeCheck` / `BrakeOK` / `BrakeNOK` | Brake-test operator prompt | bool. "Not yet implemented" | [doc] |
| `DsAlarms.BRAKE_FAULT` | brake fault | condition TBD, not used | [doc] |
| `BR_RES_CNT_EN` / `BR_RES_CNT_FBK` / `DsAlarms.BR_RES_CNT_FBK` | Brake-resistor contactor CNT4 (dissipates HVDC energy) | FBK TRUE = closed | [HMI] |
| `BR_RES_CTRL_FAULT_N` / `DsAlarms.BR_RES_CTRL_FAULT_N` | Brake-resistor controller fault, **active-low** | FALSE = "braking unit inhibited" | [HMI] |
| `BR_RES_CTRL_INHIBIT` | Brake-resistor controller inhibit | bool | [infer] |

Pilot "DRUM LOCKED" flag = `BRAKE_CNT_EN && inOperationMode[2]==3 && Velocity_Target[2]==0`
(`hmi-pilot.component.ts:81-86`): brake released but the generator holds the drum at zero speed.

### 3.6 State machine, operator flow, alarms

| Field | Meaning | Conf. |
|---|---|---|
| `DsStateMachine.ACTUAL_STATE` | **GS (PLC) operating state**, Int16, 5 Hz. Enum in §5. **Main phase signal for pumping-cycle segmentation: 4 GENERATION (reel-out), 6 RECOVERY (reel-in), 5/7 transitions** | [code] |
| `DsStateMachine.REQUESTED_STATE` | State requested by HMI or throttle buttons (same enum) | [code] |
| `DsStateMachine.PENDING_STATE`, `CHANGE_STATE_ALLOWED`, `CLEAR_PENDING_STATE` | Pending-transition handshake | [infer] |
| `DsStateMachine.EMERGENCY_STATE`, `EMERGENCY`, `EMERGENCY_RED`, `DsAlarms.EMERGENCY` | Emergency flags (E-stop read at 20 Hz) | [HMI]/[doc] |
| `DsStateMachine.HMIConnected` | HMI heartbeat seen by the PLC | [infer] |
| `DsExternalData.Panel{SelfTest,PreChargingCheck,ReconnectCheck,TakeOffJog,TakeOffEnd,Landing,BrakeCheck}` | PLC asks the operator for a confirmation (prompts on the home page) | [HMI] `home.component.html:189-283` |
| `DsExternalData.{SelfTestOK,SelfTestNOK,SelfTestFailMsg,PreChargingOK/NOK,ReconnectOK,ReadyToTakeOff,WingAirborne,AbortTakeOff,LandingOK}` | Operator answers (written by HMI). Note SelfTest prompt mapping is inverted on purpose (YES → SelfTestNOK = redo) | [HMI] |
| `DsExternalData.Joystick_Enabled` | Throttle enabled ("thrtl disabled" warning when FALSE) | [HMI] |
| `DsExternalData.Joystick_Percentual_Target_{Torque,VelocityP,VelocityN}` | Throttle lever %, 0-100 (from THROTTLE). Torque = generator torque lever (ABS_RX); VelocityP / VelocityN = positive / negative velocity levers (ABS_RZ / ABS_RY) | [code] `R:thrustmaster_throttle.py:106-118` |
| `DsExternalData.Joystick_Service_Brushless` / `Joystick_Service_Torque` | Throttle drives the RECOVERY / GENERATOR motor in service mode (mutually exclusive) | [HMI] |
| `DsExternalData.Target_Locked` | ? (the HMI must not write it) | [doc] |
| `DsAlarms.*` | Fault/warning booleans. Classification used by the alarm sounder: key containing `WARNING` = warning; `RESET`/`CONFIRM_RESET` = handshake not alarm; `*_FDBK`/`*_FBK`, `BATTERY_CHARGE_ACTIVE`, `BATTERY_MINIMUM_CHARGE` = info (`webapp/src/app/core/services/alarm-sound.service.ts:24-35,186-197`). `CONV6_FAULT_N` = ignore (CONV6 removed). `CONVERTER_FDBK[i]`/`INVERTER_FDBK[i]` = probably the source of the CONV/INV GENERIC FAULT telltales | [code]/[doc] |
| `DsAlarms.HVDC_OVERVOLTAGE/UNDERVOLTAGE`, `CONT_FAULT`, `CONV/INV_COMM_FAULT`, `CONV/INV_GENERIC_FAULT`, `VOLTAGE_24V_LOG/PWR_FAULT`, `BATTERY_COMM_FAULT`, `BATTERY_OVER_CHARGE`, `EXT_SUP_CNT_FBK` | Electrical alarms, as labelled on the home annunciator (`home.component.html:97-156`) | [HMI] |

---

## 4. Non-OPC fields

### WIND (`R:managers/windstation.py:34-67`)
| Field | Meaning | Unit | Notes |
|---|---|---|---|
| `WIND_wind.speed` | wind speed | m/s (`opc.config.ts:211-216`, note.md) | float, group 4 of reply `W\|a\|b\|c\|d`. Mast height unknown |
| `WIND_wind.direction` | wind direction | ° 0-360, "from the north" (meteorological "from" convention [infer]) | int, integer part of group 2 |
| `WIND_wind.connected` | always True when a reading is parsed | bool | absence of points = disconnected |
Before 2026-10-07: `METEO_meteo.speed` / `METEO_meteo.direction` / `METEO_meteo.connected` [infer from rename commit c48ca6b].

### THROTTLE (`R:thrustmaster_throttle.py`)
| Field | Meaning | Unit |
|---|---|---|
| `THROTTLE_DsExternalData.Joystick_Percentual_Target_VelocityN` | ABS_RY lever → negative-velocity lever % (written to PLC) | int 0-100 (`raw/65535*100`, clipped) |
| `THROTTLE_DsExternalData.Joystick_Percentual_Target_VelocityP` | ABS_RZ lever → positive-velocity lever % | int 0-100 |
| `THROTTLE_connected` | 1 Hz ping while device present | bool |
Also possible (not in your list): `THROTTLE_DsExternalData.Joystick_Percentual_Target_Torque` (ABS_RX) and
`THROTTLE_DsStateMachine.REQUESTED_STATE` from buttons BASE→15 POWERON, BASE2→2 PREFLIGHT, BASE3→3 TAKE-OFF,
BASE4→6 RECOVERY, BASE5→4 GENERATION (on button release) (`:94-103`). Events only fire on change, so forward-fill.

### TARANIS (`R:taranis.py:97-119`)
| Field | Meaning | Unit |
|---|---|---|
| `TARANIS_Pitch` | stick ABS_X → pod pitch command | int −100..100 % (`-100 + raw/2048*200`) |
| `TARANIS_Roll` | stick ABS_Y → pod roll command | int −100..100 % |
| `TARANIS_connected` | 1 Hz ping | bool |
These reach the pod as `FS_MAINTENANCE_pitch/roll` except while `fs_operating_status == 3` MAINTENANCE.

### SYSTEM (`R:timesync.py`)
| `SYSTEM_timestamp` | backend epoch s (ms-rounded), 1 Hz |
|---|---|
| `SYSTEM_iso_timestamp` | same as ISO-8601 UTC string |

### WS: HMI → PLC write commands (each = one OPC write; field value = value written)
| Field | Meaning |
|---|---|
| `WS_DsStateMachine.REQUESTED_STATE` | Operator state request (2 PREFLIGHT, 3 TAKE-OFF, 4 GENERATION, 6 RECOVERY, 8 LANDING, 15 POWERON; also 2 = "Abort" during take-off) (`home.component.html:163-224`) |
| `WS_DsExternalData.ReadyToTakeOff`, `WingAirborne`, `AbortTakeOff`, `LandingOK` | Take-off / landing confirmations (TRUE pulses) |
| `WS_DsExternalData.BrakeManualReq` | Brake toggle request (value = !BRAKE_CNT_EN, i.e. TRUE = release) |
| `WS_DsExternalData.BrakeOK` | Brake check answer |
| `WS_DsInverters.IncreaseJogTarget[2]` / `DecreaseJogTarget[2]` | Take-off jog of GENERATOR (TRUE on press, FALSE on release) |
| `WS_DsInverters.JogSpeedTarget[2]` | Jog speed set-point for INV2 |
| `WS_DsInverters.Velocity_hmiTarget[1]` | Manual velocity target for INV1 (service, in velocity mode) |
| `WS_DsInverters.hmiOperationMode[1..3]` | Requested CiA 402 mode (enum §5) |
| `WS_DsInverters.InverterEnable`, `InverterNumber` | Enable inverter n (number written first) |
| `WS_DsInverters.InActivatePDORead_Service[1..3]` / `InActivatePDOWrite_Service[1..3]` | PDO service toggles |
| `WS_DsInverters.inHoming` | Thread-guide homing request |
| `WS_DsEncoder.RelateGuidaFiloWithEncoder` | Electronic CAM on/off |
| `WS_DsAlarms.RESET` | Alarm reset |

### FS_MAINTENANCE: uplink commands to the pod (`GS_BRIDGE_CONTRACT.md:195-243`)
| Field | Meaning / values |
|---|---|
| `FS_MAINTENANCE_pitch`, `_roll` | Axis command −100..100 %, from the Taranis (non-maintenance) or the HMI slider (MAINTENANCE) |
| `FS_MAINTENANCE_gs_operating_status` | **Raw PLC `ACTUAL_STATE`** mirrored by the backend **on change only** (`data_manager.py:226-257`). The PLC enum (§5), not the wire enum |
| `FS_MAINTENANCE_status_request` | 1 READY, 2 OPERATIONAL (arm), 3 MAINTENANCE. **Legacy 0 idle/1 operational/2 maintenance is dead**, so data from before the protocol change has different meanings |
| `FS_MAINTENANCE_fail_safe` | 0/1 level |
| `FS_MAINTENANCE_maintenance_motor_arm` | 0/1 arm gate in MAINTENANCE |
| `FS_MAINTENANCE_camera_command` | 0/1 camera recording |
| `FS_MAINTENANCE_reset_battery_energy`, `_get_servo_config` | momentary 1 then 0 |
| (also possible: `emergency`, `zeroing_pitch_roll`, `config_abort`; `config_upload` is a dict and dropped) | |

### GS_UPLINK_COMMAND: bridge read-back of what is actually being uplinked (1-2 Hz)
| Field | Values |
|---|---|
| `GS_UPLINK_COMMAND_pitch`, `_roll` | int −100..100 % |
| `_autopilot_command` | int, unused (no autopilot) |
| `_status_request` | 1 READY, 2 OPERATIONAL, 3 MAINTENANCE (held) |
| `_fail_safe`, `_zeroing_pitch_roll`, `_reset_battery_energy`, `_maintenance_motor_arm`, `_emergency`, `_camera_command`, `_get_servo_config` | 0/1 commanded levels |
| `_force_power_depower_command` | 0-3 (2-bit; matches `force_power_depower_status` codes 0 OFF, 2 POWER_ARMED, 3 DEPOWER_ARMED [infer]) |
| `_gs_operating_status` | **Compact wire enum 0-10** (§5), translated by the bridge from the PLC state |

### BRIDGE: datalink health (1-2 Hz, `GS_BRIDGE_CONTRACT.md:120-161,245-260`)
| Field | Meaning |
|---|---|
| `BRIDGE_gs_datalink_status` | 0/1 sticky; 1 = ≥5 consecutive downlink packets |
| `BRIDGE_downlink_per` | merged downlink packet error rate %, 500 ms window; **dead link = 100** |
| `BRIDGE_seconds_since_last_rx` | s since last downlink frame; null before first RX |
| `BRIDGE_link_source` | active link: 0 MAIN 2.4 GHz, 1 BACKUP 868 MHz |
| `BRIDGE_lost_packets` | cumulative counter gaps |
| `BRIDGE_missed_cycles` | consecutive silent 50 ms cycles |
| `BRIDGE_downlink_loss` | bool = `gs_datalink_status == 0` |
| `BRIDGE_main_link_status` / `backup_link_status` | per-radio 0/1 sticky; **null = radio absent** (≠ down) |
| `BRIDGE_main_downlink_per` / `backup_downlink_per` | per-radio PER %; null = radio absent |
| `BRIDGE_main_seconds_since_last_rx` / `backup_…` | per-radio s since last frame |
| `BRIDGE_camera_command` | echo of the commanded camera level (= GS_UPLINK_COMMAND_camera_command) |
| `BRIDGE_upload_state` | string: idle, uploading, wait_type_e, done, timed_out, aborted |
| `BRIDGE_upload_bitmap` | 0-7, bit0..2 = config pages 1..3 acked |

### FLIGHT_SEGMENT (covered elsewhere): how the HMI uses it
Merged key-wise by the HMI, because each 20 Hz frame variant carries a subset (`home.component.ts:107-112`). Null =
no data, **and wire 0 also decodes to null for yaw_angle, azimuth, radius, charge**. `fs_operating_status` decides
Taranis vs HMI axis ownership (3 = MAINTENANCE) and triggers the FAIL_SAFE alarm (4). `pod_version` 1=V2, 2=V3
(servo_a/b = pitch/roll on V2, left/right on V3). Useful GS cross-checks: `radius` (FS↔GS distance, m) vs
`outTamburo_WindedLength`, and `elevation` for converting the load-cell sum to tension.

---

## 5. Enums / state mappings

**`DsStateMachine.ACTUAL_STATE` / `REQUESTED_STATE` / `PENDING_STATE`** (PLC GS state machine; `webapp/src/app/shared/pipes/stateToText.pipe.ts`, `GS_BRIDGE_CONTRACT.md:221`)

| Value | State | | Value | State |
|---|---|---|---|---|
| −1 | no request / none ("---") | | 8 | LANDING |
| 0 | SELF TEST | | 9 | ABORT TAKE-OFF (transition) |
| 1 | ENABLE DC POWER (transition) | | 10 | SERVICE |
| 2 | PREFLIGHT | | 11 | PRE-CHARGE |
| 3 | TAKE-OFF | | 12 | CHARGE (HMI prompts "INIT CHARGING") |
| 4 | **GENERATION** (reel-out) | | 13 | DISABLE DC POWER (transition) |
| 5 | TRANSITION TO GENERATION | | 14 | BACK TO PREFLIGHT (transition) |
| 6 | **RECOVERY** (reel-in) | | 15 | POWERON |
| 7 | TRANSITION RECOVERY | | 31 | EMERGENCY |

HMI groups: "On ground" = anything not in {3,4,6,8} (`home.component.html:175`).

**`GS_UPLINK_COMMAND_gs_operating_status` (compact wire enum, bridge `map_plc_state`)**: 0 NO_GS_CONNECTION (also PLC −1),
1 SELF_TEST(0), 2 PREFLIGHT(2, 13), 3 TAKEOFF(3, 5, 9, 14), 4 GENERATION(4, 7), 5 RECOVERY(6), 6 LANDING(8), 7 SERVICE(10),
8 EMERGENCY(31), 9 POWERON(15, 1), 10 INVALID(11, 12, other). Lossy: 5→TAKEOFF and 7→GENERATION, so it cannot recover transitions.
(`GS_BRIDGE_CONTRACT.md:221`)

**`DsInverters.in/out/hmiOperationMode[i]`** (CiA 402): −1 Unknown, 1 Profile Position, 3 Profile Velocity, 4 Torque, 6 Homing, 7 Interpolated Position.

**`ConvStruct.CONV_READ.CONV_CONF_FB_MODE[i]`**: 0 AUTONOMOUS, 1 CC (constant current), 3 EXTENDED.
**`CONV_EXT_MODE_FB_EXTENDED_MODE[i]`** (config): 1 HSVC (HS voltage control), 2 LSVC (LS voltage control).
**`CONV_STATUS_FB_POWER_DIRECTION[i]`, `CONV_EXT_MODE_FB_POWER_FLOW_DIRECTION[i]`**: 0 BIDIRECTIONAL, 1 HS CHARGE ONLY, 2 HS DISCHARGE ONLY.
**`CONV_(EXT_MODE_FB_/WRITE.)CAN_CHECK_MODE[i]`**: 0 KEEP LAST, 1 100 ms, 2 1 s, 3 10 s.
(`converters-panels.component.html:115-128,281-326,550-560`)

**`CONV_EN_DIS_FB_ERROR_CODES[i]`** (unit of `ERROR_VALUE`): 0 No error; 1 HS voltage range (V); 2 LS voltage range (V);
3 HS overcurrent (A); 4 Internal error 1; 5 Overtemp (°C); 6 HS error A-mode (V); 7 HS earth fault; 8 Internal error 2;
9 HS voltage hw (V); 10 LS voltage hw (V); 11 HS overcurrent hw (A). (`converters-panels.component.html:712-756`)

**Active-low booleans**: `CONVn_EARTH_FAULT_N`, `DsAlarms.CONVn_EARTH_FAULT_N`, `BR_RES_CTRL_FAULT_N`, `DsAlarms.CONV6_FAULT_N` (FALSE = fault).
**Inverted-sense**: `BRAKE_CNT_EN` TRUE = brake released.

**FLIGHT_SEGMENT-related enums used by the HMI** (`webapp/src/app/core/models/flight-segment.model.ts:11-63`):
FsOperatingStatus 0 READY, 1 OPERATIONAL, 2 FAULT, 3 MAINTENANCE, 4 FAIL_SAFE, 5 EMERGENCY, 6 CONFIGURATION, 7 INIT;
StatusRequest 0 NO_REQUEST, 1 READY, 2 OPERATIONAL, 3 MAINTENANCE; LinkSource 0 MAIN 2.4 GHz, 1 BACKUP 868 MHz;
DeviceStatus 0 DISABLED, 1 ENABLED, 2 FAULT, 3 COMM_ERROR; ForcePowerDepower 0 OFF, 2 POWER_ARMED, 3 DEPOWER_ARMED.

---

## 6. Open questions / ambiguities

1. **Torque sign**: `Torque_ActualValue[1..2]` is displayed on 0-based bars, so the sign convention during reel-out
   vs reel-in is undocumented. Check the data: is generator torque positive in GENERATION (state 4)?
2. **Drum speed sign**: does `outTamburo_SpeedRPM > 0` (HMI ▲) mean reel-out? Check against state 4/6 and
   d(WindedLength)/dt. The same question applies to `Velocity_ActualValue/Target` of each motor (two motors may
   have opposite mounting).
3. **WindedLength semantics**: "winded" suggests wound on the drum, but the HMI's working window (70-170 m,
   alarm >210 m) suggests paid-out length. Verify vs `FLIGHT_SEGMENT.radius` and the sign of SpeedMSec.
4. **Drum diameter 0.491 m** comes only from a hard-coded HMI formula. Verify by regressing `outTamburo_SpeedMSec` on
   `outTamburo_SpeedRPM`. Gear ratios between motors and drum are absent from the repo, so tether force from torque
   needs them (ask PLC/mechanical).
5. **Load cells**: unit is kg per the HMI only. The mapping of index → corner, the tare/zero, and the geometry linking
   the sum to tether tension are undocumented. They update only in flight states (stale/held otherwise). Whether 20 Hz
   is a hardware cap is open (`PLC_CHANGE_REQUEST.md:30-36`). The new `LOADCELL` source (`loadcell.force_kg`) may be a
   direct tether load cell; check.
6. **Power sign**: + = toward the batteries (generation) is inferred from the HMI arrow drawing and the "TOTAL PWR"
   colouring, not stated. Check: LS_PWR_SCALED_CALC > 0 mostly in state 4, < 0 in state 6.
7. **`Torque_Target` units/meaning** conflict (opc.config.ts: RPM for [1], A for [2]; pilot page uses it as the torque
   target marker; UInt16 so unsigned; the PLC engineer says the name is misleading).
8. `Velocity_ActualValue` units (rpm vs drive increments) and `ProfilePosition_*` counts per rev: not documented.
9. **Timing**: OPC timestamps are backend time at snapshot (10 Hz) of values the PLC produces at 5-20 Hz. Expect
   duplicated values, jitter, and no cross-variable cycle consistency. Before 2026-09-30 the rate was ~3-5 Hz and the
   field set was larger. `LS_PWR_SCALED_CALC` and `MeasureFloat_SUM` exist only since they were added in the backend
   (always present in this backend version, absent if the PLC adds its own later).
10. **Renames**: WIND was METEO, BRIDGE was GROUND_SEGMENT_INFO (both before 2026-10-07). FS_MAINTENANCE `status_request` codes
    changed with the new datalink (old 0/1/2). `fs_operating_status` 7 used to be emergency and is now INIT.
11. `FS_MAINTENANCE_gs_operating_status` (raw PLC enum, on change) ≠ `GS_UPLINK_COMMAND_gs_operating_status` (compact wire
    enum). Do not mix them.
12. Not verified by the PLC engineer: `ConvStruct.in*` handshake semantics, `DsInverters.inEnable/DisableSpeedLimit`,
    `DsLoadCells.Status`, `DsInverters.ErrorCode` / converter `WARNING_CODES` tables, `CONV_POWER_SCALED_HS/LS` units,
    `outTamburo_Angle` units, `EMERGENCY` vs `EMERGENCY_RED` vs `DsAlarms.EMERGENCY`.
13. No battery SoC/temperature and no motor/inverter/converter temperatures exist in the logged data (only booleans/I²t),
    so thermal analysis is not possible from GS data.
