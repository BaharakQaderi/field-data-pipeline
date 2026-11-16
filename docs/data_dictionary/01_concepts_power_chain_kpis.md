# TK10 pumping-kite system: physical concepts and PLC semantics for generated-power analysis

Status: research notes, 2026-10-23. Read-only study of the sources listed below. Statements are tagged:
- **[DOC]**: a design document says so
- **[CODE]**: existing software (the HMI, backend, simulators or pipeline) does it
- **[DATA]**: checked empirically on archived field data, specifically `field-data-pipeline/data_archive/2026-10-22/flights/flight_{1,2,3}/OPC.parquet` and `data/INFLUX/OPC_data_2025-07-29.csv`
- **[ASSUMED]**: inference, not yet confirmed

Source abbreviations used below:

| Tag | File |
|---|---|
| SPEC | `Kite-projects/docs/KTE_0010_TCN_0021_B_TK10Statemachine_PLC.pdf` (Issue 1, 03/11/2023, E. Fulco) |
| ONB | `Kite-projects/docs/On_boarding_guide.pdf` (Kitenergy Technical Onboarding Manual, Issue 1) |
| OVW | `Kite-projects/docs/TK10_system_overview.pdf` (1 p.) |
| DIAG | `Kite-projects/docs/TK10-diagrams (1).pdf` (3 pp.) |
| SYSD | `TK10_System_Diagrams_A4_landscape.pdf` (8 pp., itself an earlier generated summary, so lower authority) |
| HMI | `Kite-projects/tk-hmi-monorepo/webapp/src/...` |
| BE | `Kite-projects/tk-hmi-monorepo/python/restapi/workers/managers/opc.py` |
| PCR | `Kite-projects/tk-hmi-monorepo/docs/PLC_CHANGE_REQUEST.md` |
| GSB | `Kite-projects/tk-hmi-monorepo/docs/GS_BRIDGE_CONTRACT.md` |
| PSIM | `Kite-projects/tk10-plc-sim/` (spec-faithful PLC simulator) |
| PLOG | `tk10-plc-sim/docs/PROJECT_LOG.md` |
| OSIM | `Kite-projects/tk10-sim/` (retired simulator) |
| FDP | `field-data-pipeline/` (this repo) |

Not relevant to TK10:
- `Kitepower_architecture_redrawn.pdf` and `autopilot-simulation.pdf` (identical to `auato.pdf`) describe **Kitepower's** KCU/GS software, a different company. They are context only.
- One useful item from them: the Kitepower logs use the convention *reel-out speed positive* (autopilot-simulation.pdf p.7).

---

## 1. System architecture and energy chain

### 1.1 Physical chain

```
wind -> LEI kite (12 m2 ref.) + POD (steer/depower) -> single tether
     -> drum ("tamburo", encoder ENC1, 4 load cells under the drum, thread guide)
     -> M2 "Torque" machine (generator, INV2)   [+ M1 brushless recovery motor via epicyclic gear + brake, INV1]
     -> INV1/INV2/INV3 (AC <-> 600 VDC HV bus, DC-link C1/C2, brake resistor RBRAKE)
     -> 4 x DC/DC converters CONV1..4 (Epic Power EPC5k5 648HL, CAN 2.0B)
     -> 4 x Pylontech US5000 batteries BAT1..4 (48 V class, RS485)
```

Sources for the chain:
- SPEC p.6, Fig. 1: logic block diagram.
- SPEC p.7, Fig. 2: power distribution diagram.
- ONB p.58–59: thread guide and the dual-machine drivetrain.
- OVW p.1 and DIAG p.1: kite → tether → ground → storage (ISS).

### 1.2 Ground-station components (SPEC Fig. 1 p.6, Fig. 2 p.7)

| Ref | Device | Role | Bus / node |
|---|---|---|---|
| M1 / **INV1** | Metronix ARS2320(FS) | **Recovery motor** ("Ingresso INV1 (recupero)"). Brushless, higher speed and lower torque, for rapid reel-in (ONB p.59). HMI label "BRUSHLESS" / "REC". | CANopen node 2; contactor CNT1 |
| M2 / **INV2** | Metronix ARS2320(FS) | **Generator**, "G – Torque" ("Ingresso INV2 (generatore)"). Low speed, high torque (ONB p.59). HMI label "TORQUE". | CANopen node 3; CNT2 |
| M3 / **INV3** | Metronix BL4000 | **Thread guide / level-wind** ("Guida filo"). Screw-driven; its pitch ≈ tether diameter (ONB p.58–59). Limit switches LSW1/2 = `CABLE_MOTOR_LIMIT1/2`. | CANopen node 4; CNT3 |
| INV4 | single-phase 220 VAC inverter | Auxiliary (Fig. 2) | – |
| ENC1 | rotary encoder on the drum | `DsEncoder.*` | CANopen node 1 |
| ENC3 | linear encoder (MSK210, RS-422) | Hangs off a drive in Fig. 1, probably thread-guide position [ASSUMED] | – |
| LC1–LC4 | Tedea-Huntleigh load cells under the drum | `DsLoadCells.MeasureFloat[1..4]` | CANopen nodes 5–8 |
| CONV1–4 | Epic Power EPC5k5 648HL bidirectional DC/DC, 600 V HS ↔ 48 V LS | `ConvStruct.*[1..4]`; relays SW1–SW8 select converter↔battery pairs | CAN 2.0B ids 1–4 |
| BAT1–4 | Pylontech US5000 | Storage | RS485 |
| CONV5 / CONV6 | 24 V logic (MeanWell) and 24 V power (Siemens 6SL3100) DC/DC | Auxiliary bus | – |
| BRK1 | brake (BFK458, 24 VDC) via CNT6 | On the brushless-machine side (ONB p.59). `BRAKE_CNT_EN` (HMI label "Brake released") | – |
| RBRAKE + BRKC | braking resistor + controller, CNT4 | HV-bus discharge and over-voltage dump | – |
| CNT5 + BRIDGE1 + CB9 | external supply / generator set ("gruppo elettrogeno", 400 VAC rectified) | Battery charging from external power | – |

**Drivetrain.**
- ONB p.59: the torque motor and the brushless machine are coupled to the drum through an **epicyclic gearbox**, and each can be connected or disconnected. The brake sits on the brushless side.
- [DATA] `Velocity_ActualValue[2]` / `outTamburo_SpeedRPM` has a median of 1.017 (IQR 0.90–1.09). So INV2 speed equals drum speed: the generator is effectively 1:1 with the drum, or its speed is reported in drum rpm.
- [DATA] `Velocity_ActualValue[3]` ≈ −0.43 × drum rpm. The thread guide follows the drum.
- [DATA] `Velocity_ActualValue[1]` = 0 throughout all 2026-10-22 flights, with `BRAKE_CNT_EN` = 0. The recovery motor was not used for pumping.

**Inverter index mapping (OPC arrays are 1-based):**
- `[1]` = INV1 = recovery / brushless (M1)
- `[2]` = INV2 = generator / torque machine (M2): **the main power machine**
- `[3]` = INV3 = thread guide (M3)

Supporting evidence:
- SPEC p.10 node table and Fig. 2.
- HMI `opc.config.ts`: L97–116 (`Torque_Target[1]` "BRUSHLESS", `[2]` "TORQUE"), L164–181.
- `hmi-pilot.component.html` ~L150–158 ("REC" bar = `Torque_ActualValue[1]`).
- BE `_store_value` expands arrays as `name[1]..name[n]`.
- PSIM `plcnext_map.py` ("INV1..3 sit on CNT1..3").

### 1.3 Control and data layer

- **PLC.** Phoenix PLCnext AXC F 2152 at 192.168.1.10, OPC UA `ns=6;s=Arp.Plc.Eclr/<name>` (SYSD p.1, p.8; PLOG L22–30).
- **Backend.** The tk-hmi FastAPI backend reads the PLC and adds two computed fields (BE L204–242):
  - `CONV_MEAS_FB_LS_PWR_SCALED_CALC = Σ_i V_LS[i]·I_LS[i] / 1000` (kW)
  - `MeasureFloat_SUM = Σ_i MeasureFloat[i]`
- **Logging.** The backend writes to InfluxDB. The bucket is `gs`, measurement `tk` (`FDP/analyze_battery_power_flights.py` L26, L59).
- **Rates.** PLC process tables run at 20/10/5/1 Hz on 2 cores (SPEC p.13 ff.). PCR L19 confirms load cells, encoder, inverter and converter feedback at 20 Hz.
  - The OPC client polls at 100 ms (SYSD p.8).
  - [DATA] The archived OPC data has a median sample interval of 0.100 s, i.e. 10 Hz, not 20 Hz.
- **Datalink.** POD/flight-segment data comes over XBee at 20 Hz on 2.4 GHz with 868 MHz backup (SYSD p.1, p.7; ONB p.70).
- **Wind.** The wind station (`WIND_wind.speed` m/s, `WIND_wind.direction` ° from N) is a separate TCP source (SYSD p.1).
- **Pilot throttle.** The ground-station pilot throttle (Thrustmaster) writes `DsExternalData.Joystick_Percentual_Target_*`. It is also logged as source `THROTTLE`.

---

## 2. PLC state machine

### 2.1 State codes: use the real PLC numbering, not the spec's

The state codes are carried by `OPC_DsStateMachine.ACTUAL_STATE`. Related fields are `REQUESTED_STATE`, `PENDING_STATE`, `CHANGE_STATE_ALLOWED` and `EMERGENCY_STATE`.

- The spec lists 4-bit codes, shown in its section headings (SPEC p.2, p.13–40).
- **The PLC on the machine numbers states differently.** The authoritative mapping is the HMI's `webapp/src/app/shared/pipes/stateToText.pipe.ts`, confirmed by the user (PLOG L237–246; PSIM `tk10sim/plcnext_map.py` L88–119).

| Real code (`ACTUAL_STATE`) | Name (HMI) | Spec code (SPEC) | Kind | Physical meaning |
|---|---|---|---|---|
| 0 | SELF TEST | 1111 (15) | transition | contactor, relay, CAN and battery checks after power-up (p.13–16) |
| 15 | POWERON | 0000 (0) | Mode | idle, HV bus off (p.17) |
| 11 | PRE-CHARGE | 1011 | transition | CONV1 pre-charges the bus to >590 V, then CNT5 closes (p.18–19) |
| 12 | CHARGE | 1100 | Mode | charging from external supply; per-battery trickle/bulk/CV (p.20–23) |
| 13 | DISABLE DC POWER | 1101 | transition | converters off, bus discharged through RBRAKE to <2 V (p.24–25) |
| 1 | ENABLE DC POWER | 0001 | transition | converters raise the bus to >585 V; inverter modes configured (p.26–27) |
| 2 | PREFLIGHT | 0010 | Mode | ready for take-off, kite on the ground (p.28) |
| 3 | TAKE-OFF | 0011 | Mode | launch; SoC ≥25 % check; INV2 jog in position mode, then velocity mode (p.29–30) |
| 9 | ABORT TAKE-OFF | 1001 | transition | INV1/INV2 disabled → PREFLIGHT (p.39) |
| 4 | GENERATION | 0100 | Mode | INV2 provides counter-torque to the kite's pull: **traction / reel-out** (p.31) |
| 7 | TRANSITION RECOVERY | 0101 (spec "Transition Rec") | transition | torque handover to the recovery motor; brake released; INV2 velocity target = 0 (p.32–33) |
| 6 | RECOVERY | 0110 | Mode | recovery motor winds the tether back: **reel-in** (p.34) |
| 5 | TRANSITION TO GENERATION | 0111 (spec "Transition Gen") | transition | quick-stop the recovery motor; brake engaged; INV2 → torque mode (p.35–36) |
| 8 | LANDING | 1000 | Mode | like RECOVERY; exits on "Landing Complete" → PREFLIGHT (p.37–38) |
| 14 | BACK TO PREFLIGHT | – (not in spec) | transition | PLC-only (stateToText.pipe.ts) |
| 10 | SERVICE | 1010 | Mode | configuration and jog; both throttles enabled (p.28, p.40) |
| 31 | EMERGENCY | "10000" (16) | interrupt | from any state (SPEC p.8, Fig. 3) |
| −1 | "---" | – | – | no request (`REQUESTED_STATE`) |

Two discrepancies in the table:
- The transition codes 5 and 7 are **swapped** relative to the spec's numbering.
- The emergency code is 31 on the machine, not 16 as in the spec.

[DATA] The 2025-07-29 transition sequence matches the real numbering:
- SELF TEST → power-up: 0 → 15 → 1 → 2
- Flight: 2 → 3 → 4 → **7 → 6** → 8 → 2
- Shutdown: 2 → 13 → 15

The 2026-10-22 flight 1 sequence is 3 → 4 → 7 → 6.

### 2.2 Pumping phases vs PLC states: an important practical finding

- **In the spec**, GENERATION = reel-out and RECOVERY = reel-in, with Transition states in between. One pumping cycle would therefore be 4 → 7 → 6 → 5 → 4.
- **In practice [DATA]**, the operator flies whole pumping cycles **inside state 4 (GENERATION)**. RECOVERY (6) is entered only once, just before LANDING.
  - 2026-10-22: flight 1 is 14,035 of 14,100 samples in state 4; flights 2 and 3 are 100 % in state 4.
  - 2025-07-29: 13,626 samples in state 4 vs 10 in state 6.
  - In state 4 the drum alternates direction: drum rpm ranges from −126 to +68.
  - INV2 handles both directions: it generates on reel-out and motors on reel-in. INV1 stays idle.
  - The brake stays engaged: `BRAKE_CNT_EN` = 0 in state 4 and 1 in state 6, consistent with SPEC p.32/35 (the brake is released only when the recovery motor is engaged).
- **Consequence: reel-out and reel-in must be segmented from drum speed (or force), not from `ACTUAL_STATE`.** `ACTUAL_STATE` only marks the airborne window: states {3, 4, 5, 6, 7, 8}.
- **Control mode in practice [DATA].**
  - Spec: generation is torque-controlled, and only the "torque" throttle is active outside Service (SPEC p.29, p.31).
  - On 2026-10-22, `DsInverters.OperationMode[1..3]` = 3 throughout. That is profile velocity in CiA-402 numbering [ASSUMED]: 3 = PV, 4 = profile torque.
  - `Joystick_Percentual_Target_Torque` = 0 and `Torque_Target[2]` = 0 throughout.
  - The operator used speed throttles:
    - `Joystick_Percentual_Target_VelocityN` is active during **reel-out** (mean 49 % when rpm < −50).
    - `..._VelocityP` is active during **reel-in** (mean 26 % when rpm > +50).
    - `Velocity_Target[2]` follows them in signed rpm: −98 on reel-out, +52 on reel-in.
  - On 2025-07-29, `OperationMode[2]` was 4 (torque) in PREFLIGHT/DISABLE DC and 3 in GENERATION.
- **Other phases.**
  - Launch (TAKE-OFF 3): line is paid out from the full drum. `WindedLength` is at its maximum at launch (≈214 m wound on flight 1).
  - Landing (8): line is reeled in until "Landing Complete".
  - Parking/idle: PREFLIGHT (2) or POWERON (15), drum stopped.

---

## 3. Physical constants and conversions

| Quantity | Value | Evidence | Notes |
|---|---|---|---|
| Drum diameter used by the PLC | **D = 0.491 m** (r = 0.2455 m) | [DATA] `outTamburo_SpeedMSec / outTamburo_SpeedRPM` = π·0.491/60 exactly (median ratio → D = 0.4910). [CODE] HMI pilot view `RPM/60*0.491*3.14` (`hmi-pilot.component.html` L168, L174); FDP `process_catanzaro_flights.py` L132–133 | Whether this is the bare drum or an effective diameter including tether layers is unknown; ONB p.63 notes that r_d changes with the winding layer (TK100). The simulators' 0.30 m (PSIM `plant.py` L17) and 0.20 m (OSIM `tk10_default.yaml` L72) are explicitly illustrative/UNVERIFIED. |
| Tether speed | v [m/s] = RPM/60 · π · 0.491 = `outTamburo_SpeedMSec` | [DATA] | **Sign: negative = reel-out** (tether paid out, `WindedLength` decreasing); positive = reel-in. |
| Wound length | `outTamburo_WindedLength` [m] = **tether wound on the drum**, not tether out | [DATA] the integral of v correlates with WindedLength at r = 1.000; maximum (≈214 m) at take-off, minimum (≈28–50 m) at maximum reel-out. HMI gauge "Winded" 0–240 m, warning at 55 m (opc.config.ts L186–210) | Tether out ≈ L_wound(launch) − WindedLength (+ fixed length to the kite). The total tether length is not documented. |
| Drum angle / speed | `outTamburo_Angle` [deg], `outTamburo_SpeedRPM` [rpm] | names; PSIM `plcnext_map.py` L227–238 | The encoder resolution of 8192 counts in PSIM is a simulator assumption. |
| Load cells | `MeasureFloat[i]` in **kg (kgf)**; F [N] = 9.81 · `MeasureFloat_SUM` | [CODE] HMI unit "Kg", range −50..500, warning 400, alarm 450 (opc.config.ts L45–59); FDP formula ×9.81 | Four cells **under the drum** measure the drum reaction, not tether tension directly. [DATA] The load is very uneven: LC2 averages 50–86 kg (maximum 417) while LC4 is ≈0 or negative. Always use the SUM. On the ground the SUM reads ≈ −5 kg, so a tare offset is needed. [DATA] The plausibility check P_batt/P_mech = 0.82–0.89 on reel-out suggests SUM·g ≈ tether tension. |
| Generator speed | `Velocity_ActualValue[2]` ≈ drum rpm (ratio 1.0) | [DATA] | So no gear ratio needs to be applied for INV2 speed. |
| Generator torque | `Torque_ActualValue[2]`: HMI says **Nm** (0–250, opc.config.ts L164–173) | [DATA] **does not match Nm at the drum**: on reel-out, F·g·r / T2 ≈ 3.8–4.2; battery power / (T2·ω) ≈ 3.5–3.8, which is physically impossible if T2 were drum Nm | Likely CiA-402 per-mille of rated torque, or motor-side units [ASSUMED]. `process_catanzaro_flights.py` L136 applies an unexplained ×6, while `enhanced_catanzaro_analysis.py` L69 applies none. **Do not use T2 for absolute power until the scale is confirmed (Metronix rated torque).** `Current[2]` / `Torque_ActualValue[2]` = 5.46 constant. |
| Recovery motor torque | `Torque_ActualValue[1]`: HMI Nm, 0–50 | HMI L175–184 | Same caveat as T2. |
| Inverter voltage | `DsInverters.Voltage[i]` = HV bus in **0.1 V** | [DATA] Voltage[2] / HS voltage = 9.99 | |
| Inverter "Power" | `DsInverters.Power[i]` = `Voltage[i] × Current[i]` in raw units, **unsigned** | [DATA] ratio exactly 1.000; values ~1.5–3.6 × 10⁶ | **Not a power in W.** FDP uses it as "Pinv": avoid this. |
| HV DC bus | 600 V nominal (HSVC reference 600 V; pre-charge OK > 590 V; enable-DC OK > 585 V; HVDC max 621.5 V, min 508.5 V) | SPEC p.18–19, 22, 27 | [DATA] 612–621 V in flight (`CONV_MEAS_FB_HS_VOLTAGE_SCALED`) |
| Battery LS voltage | 43.5–53.5 V window; trickle ≤45 V, bulk 45–53 V, CV 53 V | SPEC p.21–22, 27; HMI vbatConfig L4–19 | [DATA] 48.6–51.6 V |
| Converter limits (HSVC mode) | 4280 W charge and discharge **per converter** (×4 ≈ 17.1 kW); LVDC current limit 80 A each | SPEC p.27 | Autonomous mode uses a 75 A limit (p.26). |
| Kite | LEI, 12 m² reference (range 3–17 m² tested) | DIAG p.2; ONB p.66; SYSD p.2 | Whether 12 m² is flat or projected area is UNVERIFIED (OSIM yaml). The MATLAB model uses S = 12 m², E = 5–6, tether 3.5 mm (PLOG L248–253). |
| Battery SoC gate | take-off requires every battery at SoC ≥ 25 % | SPEC p.29 | |

---

## 4. Power chain definitions and sign conventions

Recommended convention: **positive = energy flowing from the kite toward the battery (generation)**.

| Symbol | Definition (from fields) | Units | Sign in raw data | Notes |
|---|---|---|---|---|
| v_t | `−outTamburo_SpeedMSec`, or `−SpeedRPM/60·π·0.491` | m/s | raw is negative on reel-out; flip it | Reel-out positive after the flip. |
| F_t | `9.81 · (MeasureFloat_SUM − tare)` | N | ≥ 0 | Tare from the pre-launch PREFLIGHT window. |
| **P_mech** (drum / tether) | `F_t · v_t` | W | positive on reel-out | Equals FDP `PoMecD` with the minus sign used in `process_catanzaro_flights.py` L133. `enhanced_catanzaro_analysis.py` L66 omits the minus sign, so its sign is inverted: inconsistent. |
| P_shaft,INV2 | `k_T · T2 · V2 · 2π/60` | W | raw negative on reel-out | k_T is unknown, see §3. FDP `PoMecGen` (×6 in one script, ×1 in the other) is unreliable. |
| **P_bus** (HV side of converters) | `Σ_i ConvStruct.CONV_READ.CONV_MEAS_FB_HS_AVG_PWR_SCALED[i]` | W | positive = bus → batteries (charging) | [DATA] signed and consistent. Do **not** use `CONV_POWER_SCALED_HS[i]`: it stays positive (~13–16 kW) during discharge, so it is unsigned or broken. |
| **P_batt** (LS / battery side) | `CONV_MEAS_FB_LS_PWR_SCALED_CALC × 1000` = `Σ V_LS[i]·I_LS[i]` | W (field in kW) | positive = charging (generation) | [DATA] verified: +1.6 to +3.9 kW mean on reel-out, −2.2 to −2.7 kW on reel-in. It is computed in the Python backend, so the four terms may come from different PLC cycles (PCR R3). `Σ CONV_POWER_SCALED_LS[i]` (W) is equivalent. Per-battery current `CONV_MEAS_FB_LS_AVG_CURR_SCALED[i]` is positive when charging. |
| P_aux | standby draw with the drum stopped | W | negative | [DATA] P_batt ≈ −0.29 to −0.46 kW with the drum static (states 3, 6, 7): PLC, drives, thread guide, 24 V buses. OSIM assumes 300 W. |

ONB p.45–46 gives the same sign conventions: P_t > 0 in traction and < 0 in recovery; P_storage > 0 when charging.

**Efficiencies.** ONB p.32 gives the chain P_el = η_mech · η_machine · η_conv · P_t. Computable versions:

- η_conv (charging) = P_batt / P_bus during reel-out. [DATA] ≈ 1/1.10–1/1.16, i.e. **0.86–0.91**.
- η_conv (discharging) = P_bus / P_batt during reel-in. [DATA] ≈ **0.89**.
- η_gen,chain (drum → battery, reel-out) = E_batt,out / E_mech,out. [DATA] **0.82–0.89** on 2026-10-22, including converter losses and aux load.
- Reel-in cost ratio = |E_batt,in| / |E_mech,in|. [DATA] **3.6–4.4**: reel-in costs far more electrically than its mechanical work. Contributors are low-torque machine efficiency, aux load and acceleration.
  - P_mech is only about −0.6 kW on reel-in, because the force is low (~32–47 kg).
  - This is the dominant loss to attack.
- η_bus→machine = P_shaft / P_bus. Not computable until the T2 scale is known.
- **Cycle efficiency** (two variants):
  - η_cycle,mech = (E_mech,out − |E_mech,in|) / E_mech,out
  - η_cycle,el = E_batt,net / E_mech,out (electrical energy delivered per unit of mechanical energy harvested)

Flight-level sanity check [DATA, 2026-10-22]: net battery energy was **−155 / −128 / −204 Wh** for flights 1–3, and net drum mechanical energy was also negative. These were net-consuming test flights in light wind: mean wind 4.6 m/s on flight 1.

---

## 5. Recommended KPIs for a pumping-kite field-test analysis

**Preprocessing.**
1. Restrict to airborne states {3, 4, 5, 6, 7, 8}.
2. Tare the load cells.
3. Convert units per §3.
4. Low-pass the drum speed (e.g. 1 s).
5. Segment phases with hysteresis:
   - **reel-out**: v_t > +0.3 m/s for ≥ 3 s
   - **reel-in**: v_t < −0.3 m/s for ≥ 2 s
   - **transition**: in between
6. A **cycle** runs from one reel-out start to the next.
7. Optionally cross-check with the throttle (`VelocityN` = reel-out, `VelocityP` = reel-in) and, for spec-style operation, with states 4 / 7 / 6 / 5.
8. Integrate with the trapezoid rule on the real timestamps. The data is at ~10 Hz, so do not assume a fixed dt.

Per cycle:

| KPI | Formula / fields | Why |
|---|---|---|
| Cycle period T_c, reel-out / reel-in / transition durations, duty ratio T_out/T_c | phase segmentation | cycle shape |
| E_mech,out, E_mech,in, E_mech,net | ∫ P_mech dt per phase | harvested vs spent at the tether |
| E_batt,out, E_batt,in, **E_batt,net** | ∫ P_batt dt per phase | the energy actually stored, the headline number |
| **Mean cycle power** P̄_c = E_batt,net / T_c (also the mechanical version) | | primary performance metric (ONB p.30, 51) |
| Mean reel-out power P̄_out, mean reel-in power P̄_in, peak power | phase means and maxima of P_mech and P_batt | |
| η_cycle,mech, η_gen,chain, reel-in cost ratio, η_conv | §4 | where the losses are |
| Tether force: mean/max on reel-out, mean on reel-in, force ratio F_out/F_in | `MeasureFloat_SUM`·g | depower effectiveness. A high F_in kills the net energy. |
| Reel-out speed v_out (mean), reel-in speed v_in (mean) | `−SpeedMSec` | |
| **Reel-out factor f = v_out / v_w** | wind from `WIND_wind.speed`, time-aligned (merge_asof) and averaged over the phase | Loyd optimum ≈ 1/3 (ONB p.24–26; MATLAB f_out = 1/3, PLOG L251). The station measures at ~10 m height (UNVERIFIED), so this uses ground wind, not wind at kite altitude. |
| Tether length range (min/max tether out) and stroke per cycle | from `WindedLength` (wound) | operating envelope; stroke = Δ wound length on reel-out |
| Power curve: P̄_c vs v_w (binned) and P̄_out vs v_w | | comparison with the Loyd ∝ v_w³ trend and the MATLAB model |
| Force-speed operating points: F_out vs v_out scatter, coloured by v_w | | traction operating envelope |
| Aux / standby power | P_batt with the drum static | baseline loss |
| Battery health: per-battery current balance, SoC trend, LS voltage range | `CONV_MEAS_FB_LS_*[i]` | [DATA] BAT2 carries most of the discharge (mean I ≈ −11 to −16 A vs ≈ +1 to +5 A for the others) |
| Data quality: sample rate, gaps, load-cell imbalance, encoder faults | `DsAlarms.*`, Δt | |

Per flight:
- totals (Wh)
- number of complete cycles
- the share of airborne time spent generating
- net energy per km of tether stroke
- wind statistics

---

## 6. Open questions and conflicts

1. **State numbering.** The spec's 4-bit codes differ from the machine's codes.
   - Spec: POWER ON 0000, SELF TEST 1111, EMERGENCY "10000"; LANDING p.37 even lists Emergency as "0110".
   - Machine: SELF TEST 0, POWERON 15, EMERGENCY 31, plus BACK TO PREFLIGHT 14 (stateToText.pipe.ts; PLOG L84–89, L237–246).
   - The transition codes 5 and 7 are swapped.
   - Data analysis must use the HMI numbering.
2. **Generation control mode.**
   - Spec: GENERATION is torque-controlled through the torque throttle (p.29, 31). EnableDC STEP 2 puts INV1 in torque mode and INV2 in velocity mode (p.27), while Transition Gen puts INV2 in torque mode (p.36).
   - Data: INV2 runs in OperationMode 3 (velocity) during GENERATION. The operator uses the VelocityN/VelocityP throttles, and `Torque_Target[2]` = 0.
   - Which INV2 mode is intended in GENERATION? Confirm with the PLC programmer (Youcef).
3. **Recovery mode is not used for pumping.**
   - Pumping cycles run inside state 4, with INV2 motoring on reel-in.
   - INV1 and the brake are used only in the final RECOVERY → LANDING sequence.
   - This differs from the spec's cycle design (4 → 7 → 6 → 5).
4. **Direct drive vs gearbox.**
   - OVW p.1 says "Direct-drive drum" for the TK10 ground station. DIAG p.2 puts "Direct drive drum" under **K100**.
   - ONB p.59 describes an epicyclic gearbox coupling the two TK10 machines.
   - [DATA] INV2 rpm ≈ drum rpm.
   - Need: the actual ratios of M2→drum and M1→drum.
5. **Drum diameter 0.491 m.** It is used by the PLC, HMI and FDP. Is it the bare drum, or the effective diameter at a given layer? Is speed or length compensated for layer build-up? The simulators' 0.30 m and 0.20 m values are placeholders.
6. **Torque units and scale (`Torque_ActualValue`, `Torque_Target`).**
   - The HMI says Nm for the actual value and **A** for `Torque_Target[2]` (L107–116); it labels `Torque_Target[1]` as **RPM** (L97–106).
   - The data shows T2 is ~4× too small to be drum Nm.
   - The FDP scripts disagree with each other (×6 vs ×1, and opposite signs).
   - Need: the Metronix rated torque and rated current, and the CiA-402 scaling (per-mille?).
7. **`DsInverters.Power[i]` is unsigned V×I in raw units, not W.** `Current[i]` units are unknown (per-mille of rated current?). FDP "Pinv" should be dropped or rescaled.
8. **`CONV_POWER_SCALED_HS[i]` looks unsigned or wrapped** (positive during discharge). `CONV_MEAS_FB_HS_AVG_PWR_SCALED[i]` behaves correctly. Ask which is authoritative. PCR L38–48 has related open raw-vs-SCALED conflicts between Enrico and Youcef.
9. **Battery-power sign** (positive = charging) is verified empirically only. The HMI gauge is zero-centred and colours −10 kW as "alarm bright" (opc.config.ts L81–95), which agrees.
   - `CONV_STATUS_FB_POWER_DIRECTION` and `CONV_EXT_MODE_FB_POWER_FLOW_DIRECTION` could confirm it.
   - `LS_PWR_SCALED_CALC` is computed in Python, not in the PLC, so it is not cycle-synchronous (PCR R3).
10. **Load cells as tether tension.**
    - Their geometry (cells under the drum) and the tether exit angle are undocumented.
    - The tare and the drum weight are unknown.
    - The load distribution is strongly uneven (LC2 dominant).
    - Units are kg per the HMI. Is the conversion to N exactly ×9.81, and do the cells read reaction or tension?
    - Can the cells be sampled faster than 20 Hz (PCR Q1, still open)?
11. **Sampling.** The PLC tasks run at 20 Hz, but the archived OPC data is at 10 Hz (100 ms OPC poll, SYSD p.8). Historically ~5 Hz (PCR L11). Peak forces and power will be under-resolved, and timestamps are host receive time, not PLC SourceTimestamp (PCR R2).
12. **WindedLength reference.** This field is the tether wound on the drum. The total tether length, and the length between the drum and the POD, are not documented, so "tether out" requires a per-flight reference (e.g. the value at TAKE-OFF).
13. **Velocity sign convention on the HMI.** The pilot view shows an "up" arrow when RPM ≥ 0, i.e. reel-in in the data. Confirm the intended semantics.
14. **Kite area** (flat vs projected), depowered polar, and tether diameter/length/maximum force are not documented for TK10. The MATLAB and simulator values are marked UNVERIFIED (PLOG L265–275; OSIM yaml).
15. **Wind measurement height** is unknown (~10 m, UNVERIFIED), and no wind is measured at kite altitude. Reel-out factor and power-curve KPIs are therefore ground-wind based.
16. **Spec inconsistencies** that do not affect power analysis directly are listed in PSIM `README.md` "Spec modes". Examples: the Enable-DC STEP 5 target state, the CV branch setting "Bulk", and the Landing process table missing `manageStateMachine`.
