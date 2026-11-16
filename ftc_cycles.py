"""
Phases, cycles, power stages and statistics per flight for KTE_2030_FTC_0001.

Reads  <out>/<stem>/<stem>_analysis.csv  (20 Hz, written by ftc_extract.py)
Writes <out>/<stem>/<stem>_analysis.csv  with derived columns added after the time columns
       <out>/<stem>/<stem>_cycles.csv  one row per complete cycle ("Data per cycle")
       <out>/<stem>/<stem>_stats.json  "Data per flight"
       <out>/campaign/cycles_all.csv, flights_summary.csv

Definitions:
  cycle       = WindedLength (cable on the drum) peak -> trough -> next peak; generation = peak->trough (reel-out),
                recovery = trough->peak (reel-in). Peaks: prominence >= 5 m, >= 10 s apart.
  force       = raw load-cell sum in kg (as on the HMI); force_kn = 9.81*(SUM - tare)/1000 with a per-flight tare
                (quietest 30 s of the pre-flight padding). Tare is PROVISIONAL (open item 5).
  P_mech      = force_kn * line speed (kW); line speed = -drum_speed_ms (+ = reel-out)
  P_bus       = sum of the four DC/DC converter high-side powers (kW): power between the 624 V DC bus (all drives)
                and the battery converters, > 0 = towards the battery (electrical, 12 Nov)
  P_batt      = LS battery power (kW), + = charging
  energy      = integral of P in Wh (negative = consumed)
  energy ratio= (E_gen - E_rec) / E_gen * 100, with E_rec = consumed energy as a positive number

Usage: python ftc_cycles.py --all | --event <stem>
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks

from ftc_extract import CONFIG, write_csv

G = 9.81
DT_S = yaml.safe_load(open(CONFIG))["grid_ms"] / 1000   # grid step of the analysis table
PEAK_PROMINENCE_M = 5.0
PEAK_DISTANCE_S = 10.0
EXTREME_TOL_M = 3.5   # ~2 WindedLength steps (one drum turn = 1.54 m)
MOVING_MS = 0.2       # line speed above which the drum counts as moving
MIN_STROKE_M = 40.0   # smaller WindedLength swings are length corrections inside a phase, not phases (all 7 flights:
                      # corrections 12-29 m, real reel-outs / reel-ins 48-159 m)
BRAKE_WINDOW_S = 20.0   # brake engage searched up to this long after the end of a recovery
TARE_WINDOW_S = 30
STAGES = {"mech": "P_mech_kw", "motor": "P_motor_kw", "bus": "P_bus_kw", "batt": "P_batt_kw"}
# motor-shaft power per inverter: (multiplier, inverter index); formula from the electrical team (11 Nov),
# who give it as + = consumed; stored with the opposite sign so + = generated like the other stages
MOTOR_TERMS = ((6.0, 2), (4 * 6 / 7, 1), (1.0, 3))
DERIVED = ["phase", "cycle_id", "line_speed_ms", "force_kn", "P_mech_kw", "P_motor_kw", "P_bus_kw", "P_batt_kw",
           "kite_speed_ms", "kite_speed_src", "dist_gs_m", "datalink_state", "rtk_state", "plc_state_name"]
IPOK_DEVICE = {"orange": "IPOK orange", "blue": "IPOK blue"}
# DsStateMachine.ACTUAL_STATE codes (KTE_0010_TCN_0021_B, TK10 PGS state machine)
PLC_STATE = {0: "Power ON", 1: "Enable DC Power", 2: "Pre-flight", 3: "Take-off", 4: "Generation", 5: "Transition Rec",
             6: "Recovery", 7: "Transition Gen", 8: "Landing", 9: "Abort take-off", 10: "Service", 11: "Pre-charge",
             12: "Charge", 13: "Disable DC Power", 15: "Self test", 16: "Emergency"}


def load(out, stem):
    path = os.path.join(out, stem, f"{stem}_analysis.csv")
    df = pd.read_csv(path, low_memory=False)
    df = df.drop(columns=[c for c in DERIVED if c in df.columns])
    df.index = pd.to_datetime(df["time_gs"], utc=True)
    return df, path


def tare_kg(df):
    pre = df.loc[(df["t_s"] < 0), "OPC_force_sum_kg"].dropna()
    if len(pre) < TARE_WINDOW_S / DT_S:
        return None
    med = pre.rolling(f"{TARE_WINDOW_S}s").median()
    std = pre.rolling(f"{TARE_WINDOW_S}s").std()
    return round(float(med[std.idxmin()]), 2)


def ipok_device(cfg, stem):
    path = os.path.join(os.path.expanduser(cfg["ipok_dir"]), "KTE_2030_FTC_0001_IPOK_flight_manifest.csv")
    if not os.path.exists(path):
        return None
    m = pd.read_csv(path).set_index("stem")
    return m["device"].get(stem) if stem in m.index and isinstance(m["device"].get(stem), str) else None


def derive(df, tare):
    d = pd.DataFrame(index=df.index)
    d["line_speed_ms"] = -df["OPC_drum_speed_ms"]
    d["force_kn"] = G * (df["OPC_force_sum_kg"] - (tare or 0.0)) / 1000
    d["P_mech_kw"] = d["force_kn"] * d["line_speed_ms"]
    tq = [k * df.get(f"OPC_inv{i}_torque_actual", np.nan) * df.get(f"OPC_inv{i}_velocity_actual_rpm", np.nan)
          for k, i in MOTOR_TERMS]
    d["P_motor_kw"] = -sum(tq) * 2 * np.pi / 60 / 1000
    hs = df[[f"OPC_hs_power_{i}_w" for i in range(1, 5)]]
    # converter high side = the 624 V DC bus shared by all drives (not the generator alone); > 0 = towards the battery
    d["P_bus_kw"] = hs.sum(axis=1, min_count=4) / 1000
    d["P_batt_kw"] = df["OPC_batt_power_kw"]
    v_ipok = np.sqrt(df["IPOK_VelocityEast"] ** 2 + df["IPOK_VelocityNorth"] ** 2 + df["IPOK_VelocityUp"] ** 2)
    rel = df[["POD_velocity_az_ms", "POD_velocity_el_ms", "POD_velocity_r_ms"]].where(lambda x: x.abs() < 100)
    v_rel = np.sqrt((rel ** 2).sum(axis=1, min_count=3))
    d["kite_speed_ms"] = v_ipok.fillna(v_rel)
    d["kite_speed_src"] = np.where(v_ipok.notna(), "ipok", np.where(v_rel.notna(), "relay", ""))
    d["dist_gs_m"] = df["POD_radius_m"]
    d["plc_state_name"] = df["OPC_plc_state"].map(PLC_STATE).fillna("")
    up = df["LINK_gs_datalink_status"] == 1
    d["datalink_state"] = np.select([up & (df["LINK_link_source"] == 0), up & (df["LINK_link_source"] == 1)],
                                    ["main", "backup"], "not available")
    rtk = df["IPOK_RtkStatus"]
    no_gps = df["IPOK_GnssFix"].fillna(0).eq(0) | df["IPOK_Latitude"].isna()
    d["rtk_state"] = np.select([rtk.isna(), no_gps, rtk == 2, rtk == 1], ["", "no GPS", "fixed", "float"], "no RTK")
    return d


def find_cycles(df):
    """Cycles = reel-out (generation) then reel-in (recovery), from the turning points of WindedLength (cable on the
    drum: it falls during reel-out). This is the phase definition on every flight, because the motors were not used
    as designed (22/24 Oct the generator INV2 also reeled in; 28 Oct mostly the recovery motor INV1). The official
    window start and end count as turning points, so the take-off reel-out and the landing reel-in are kept (flagged).

    The recovery brake (OPC_BRAKE_CNT_EN = BrakeManualReq, 1 = released) is the holding brake of the recovery motor:
    released <=> INV1 turning. It only toggles on 28 Oct, where the operator released it by hand around each reel-in.
    It is reported per cycle as a cross-check, not used for the boundaries: it marks the operator's switch, which
    leads or lags the drum reversal by a few seconds, and 3-13 % of the 28 Oct reel-in was still done by INV2.
    Returns [(start, trough, end)] as positions in df, and one info dict per cycle."""
    inw = np.flatnonzero(df["in_window"].values == 1)
    wl = df["OPC_winded_length_m"].iloc[inw].interpolate(limit_direction="both")
    wl = wl.rolling(int(round(1 / DT_S)), center=True, min_periods=1).median().values
    dist = int(PEAK_DISTANCE_S / DT_S)
    peaks, _ = find_peaks(wl, prominence=PEAK_PROMINENCE_M, distance=dist)
    troughs, _ = find_peaks(-wl, prominence=PEAK_PROMINENCE_M, distance=dist)
    ext = sorted([(int(i), "P") for i in peaks] + [(int(i), "T") for i in troughs])
    if not ext:
        return [], []
    # window edges: a fall from the start (take-off reel-out) starts with a peak, a rise to the end (landing reel-in)
    # ends with one
    if ext[0][1] == "T" and wl[0] - wl[ext[0][0]] >= MIN_STROKE_M:
        ext.insert(0, (0, "P"))
    if ext[-1][1] == "T" and wl[-1] - wl[ext[-1][0]] >= MIN_STROKE_M:
        ext.append((len(wl) - 1, "P"))

    def alternate(pts):   # strict alternation: of two neighbours of the same kind keep the more extreme
        out = []
        for i, k in pts:
            if out and out[-1][1] == k:
                if (k == "P" and wl[i] > wl[out[-1][0]]) or (k == "T" and wl[i] < wl[out[-1][0]]):
                    out[-1] = (i, k)
            else:
                out.append((i, k))
        return out

    # drop length corrections: remove the smallest swing below MIN_STROKE_M (both of its turning points, or only the
    # inner one at a window edge), re-alternate, repeat
    alt = alternate(ext)
    while len(alt) > 2:
        sw = [abs(wl[alt[n + 1][0]] - wl[alt[n][0]]) for n in range(len(alt) - 1)]
        n = int(np.argmin(sw))
        if sw[n] >= MIN_STROKE_M:
            break
        drop = {n, n + 1} - {0, len(alt) - 1} if n in (0, len(alt) - 2) else {n, n + 1}
        alt = alternate([x for m, x in enumerate(alt) if m not in drop])
    # a phase starts when its sustained motion starts: find the last moment WindedLength is still within
    # EXTREME_TOL_M of the extreme (so pauses and corrections near it stay in the phase they interrupt), then step back
    # to where the continuous drum motion that leaves the extreme began (line speed, 0.5 s mean)
    v = pd.Series(-df["OPC_drum_speed_ms"].iloc[inw].values).interpolate(limit_direction="both") \
        .rolling(int(round(0.5 / DT_S)), center=True, min_periods=1).mean().values
    for n, (i, k) in enumerate(alt):
        if i in (0, len(wl) - 1):
            continue
        seg = wl[i:alt[n + 1][0] if n + 1 < len(alt) else len(wl)]
        near = np.flatnonzero(seg <= wl[i] + EXTREME_TOL_M) if k == "T" else np.flatnonzero(seg >= wl[i] - EXTREME_TOL_M)
        j = i + int(near[-1])
        moving = v[i:j + 1] < -MOVING_MS if k == "T" else v[i:j + 1] > MOVING_MS   # reel-in after T, reel-out after P
        still = np.flatnonzero(~moving)
        alt[n] = (min(len(wl) - 1, i + (int(still[-1]) + 1 if len(still) else 0)), k)
    brake = df["OPC_brake_cnt_en"].iloc[inw].fillna(0).values if "OPC_brake_cnt_en" in df else np.zeros(len(inw))
    rel = np.flatnonzero(np.diff(brake) > 0) + 1
    eng = np.flatnonzero(np.diff(brake) < 0) + 1
    v = df["line_speed_ms"].iloc[inw].values if "line_speed_ms" in df else -df["OPC_drum_speed_ms"].iloc[inw].values
    rpm1 = df["OPC_inv1_velocity_actual_rpm"].iloc[inw].abs().values > 10
    rpm2 = df["OPC_inv2_velocity_actual_rpm"].iloc[inw].abs().values > 10

    def offset(edges, i, lo, hi):
        e = edges[(edges >= lo) & (edges <= hi)]
        return round(float((e[np.argmin(np.abs(e - i))] - i) * DT_S), 1) if len(e) else None

    cycles, info = [], []
    for n in range(len(alt) - 2):
        (a, ka), (t, kt), (b, kb) = alt[n:n + 3]
        if (ka, kt, kb) != ("P", "T", "P") or not a < t < b:
            continue
        kind = "takeoff" if a == 0 else ("landing" if b == len(wl) - 1 else "")
        reel_in = v[t:b] < -0.3
        n_in = max(1, reel_in.sum())
        by1, by2 = (rpm1[t:b] & reel_in).sum() / n_in, (rpm2[t:b] & reel_in).sum() / n_in
        drive = "INV1 recovery motor" if by1 >= 0.8 else ("INV2 generator" if by2 >= 0.8 else "mixed")
        cycles.append((inw[a], inw[t], inw[b]))
        info.append({"cycle_kind": kind, "reel_in_drive": drive,
                     "reel_in_by_inv1_pct": round(float(by1 * 100), 1),
                     "brake_released_s": round(float(brake[t:b].sum() * DT_S), 1),
                     # brake edge minus WindedLength turning point (s): < 0 = the operator switched before the drum
                     "brake_release_vs_trough_s": offset(rel, t, a, b),
                     "brake_engage_vs_end_s": offset(eng, b, t, min(len(wl) - 1, b + int(BRAKE_WINDOW_S / DT_S)))})
    return cycles, info


def energy_wh(p_kw):
    return float(np.nansum(p_kw.values) * DT_S / 3.6)  # kW * s / 3600 * 1000 = Wh


def mm(s, how=("mean", "max")):
    s = s.dropna()
    if s.empty:
        return {h: None for h in how}
    return {h: round(float(getattr(s, h)()), 3) for h in how}


def ratio(e_gen, e_rec):
    if e_gen is None or e_gen <= 0:
        return None
    return round((e_gen - abs(min(e_rec, 0.0))) / e_gen * 100, 1)


def cycle_rows(df, d, cycles, info, stem):
    rows = []
    wl = df["OPC_winded_length_m"]
    for k, (a, t, b) in enumerate(cycles, start=1):
        g, r, full = slice(a, t), slice(t, b), slice(a, b)
        row = {"stem": stem, "cycle_id": k, **info[k - 1], "start_gs": df["time_gs"].iloc[a], "trough_gs": df["time_gs"].iloc[t],
               "end_gs": df["time_gs"].iloc[b],
               "gen_duration_s": round((t - a) * DT_S, 1), "rec_duration_s": round((b - t) * DT_S, 1),
               "cycle_duration_s": round((b - a) * DT_S, 1),
               "unwound_length_m": round(float(wl.iloc[a] - wl.iloc[t]), 2),
               "wound_length_m": round(float(wl.iloc[b] - wl.iloc[t]), 2)}
        w = mm(df["WIND_speed_ms"].iloc[full])
        row.update({"wind_mean_ms": w["mean"], "wind_max_ms": w["max"]})
        for ph, sl in (("gen", g), ("rec", r)):
            f = mm(df["OPC_force_sum_kg"].iloc[sl])
            row.update({f"{ph}_force_mean_kg": f["mean"], f"{ph}_force_max_kg": f["max"]})
            fk = mm(d["force_kn"].iloc[sl])
            row.update({f"{ph}_force_mean_kn": fk["mean"], f"{ph}_force_max_kn": fk["max"]})
            v = d["line_speed_ms"].iloc[sl]
            v = v if ph == "gen" else -v
            vs = mm(v)
            row.update({f"{ph}_{'reelout' if ph == 'gen' else 'reelin'}_speed_mean_ms": vs["mean"],
                        f"{ph}_{'reelout' if ph == 'gen' else 'reelin'}_speed_max_ms": vs["max"]})
            ks = mm(d["kite_speed_ms"].iloc[sl])
            row.update({f"{ph}_kite_speed_mean_ms": ks["mean"], f"{ph}_kite_speed_max_ms": ks["max"]})
            row[f"{ph}_pod_power_mean_w"] = mm(df["POD_power_w"].iloc[sl])["mean"]
            row[f"{ph}_wind_mean_ms"] = mm(df["WIND_speed_ms"].iloc[sl])["mean"]
            for st, col in STAGES.items():
                p = d[col].iloc[sl]
                pm = mm(p, ("mean", "max", "min"))
                row.update({f"{ph}_{st}_P_mean_kw": pm["mean"], f"{ph}_{st}_P_max_kw": pm["max"],
                            f"{ph}_{st}_P_min_kw": pm["min"], f"{ph}_{st}_E_wh": round(energy_wh(p), 2),
                            f"{ph}_{st}_coverage_pct": round(float(p.notna().mean() * 100), 1)})
        for st, col in STAGES.items():
            row[f"cycle_{st}_P_mean_kw"] = mm(d[col].iloc[full])["mean"]
            row[f"cycle_{st}_E_wh"] = round(energy_wh(d[col].iloc[full]), 2)
            row[f"cycle_{st}_energy_ratio_pct"] = ratio(row[f"gen_{st}_E_wh"], row[f"rec_{st}_E_wh"])
        rows.append(row)
    return pd.DataFrame(rows)


def pct(series, order):
    s = series[series != ""]
    if s.empty:
        return None
    return {k: round(float((s == k).mean() * 100), 1) for k in order}


def flight_stats(cfg, stem, df, d, cycles, tare, meta):
    w = df["in_window"] == 1
    dw, fw = d[w], df[w]
    stats = {"stem": stem, "window_utc": meta["window_utc"], "window_rome": meta["window_rome"],
             "duration_s": round(float(w.sum() * DT_S), 1), "n_cycles": len(cycles),
             "pod": meta["pod"], "tare_kg_provisional": tare,
             "wind_ms": mm(fw["WIND_speed_ms"]),
             "force_kg_raw": mm(fw["OPC_force_sum_kg"], ("mean", "min", "max")),
             "force_kn_tared": mm(dw["force_kn"], ("mean", "min", "max")),
             "power_kw": {st: mm(dw[col], ("mean", "min", "max")) for st, col in STAGES.items()},
             "energy_wh": {st: round(energy_wh(dw[col]), 1) for st, col in STAGES.items()},
             "power_coverage_pct": {st: round(float(dw[col].notna().mean() * 100), 1) for st, col in STAGES.items()}}
    ch = fw["POD_charge_pct"].dropna()
    stats["pod_battery"] = {"energy_used_wh": round(energy_wh(fw["POD_power_w"] / 1000), 1),
                            "charge_start_pct": None if ch.empty else float(ch.iloc[0]),
                            "charge_end_pct": None if ch.empty else float(ch.iloc[-1]),
                            "power_mean_w": mm(fw["POD_power_w"])["mean"],
                            "voltage_min_v": mm(fw["POD_voltage_v"], ("min",))["min"]}
    stats["pod_current_a"] = {**mm(fw["POD_current_a"], ("mean", "max")),
                              "p99": None if fw["POD_current_a"].dropna().empty
                              else round(float(fw["POD_current_a"].quantile(0.99)), 2),
                              "note": "total pack current; per-motor current is not in the telemetry"}
    stats["pod_torque_nm"] = {"left": mm(fw["POD_servo_left_torque_nm"].abs(), ("mean", "max")),
                              "right": mm(fw["POD_servo_right_torque_nm"].abs(), ("mean", "max"))}
    dev = ipok_device(cfg, stem)
    stats["rtk_status_pct"] = {"ipok": dev, "coverage_pct": round(float((dw["rtk_state"] != "").mean() * 100), 1),
                               "pct_of_covered_time": pct(dw["rtk_state"], ["fixed", "float", "no RTK", "no GPS"])}
    stats["datalink_status_pct"] = pct(dw["datalink_state"], ["main", "backup", "not available"])
    stats["kite_speed_source_pct"] = pct(dw["kite_speed_src"], ["ipok", "relay"])
    if cycles:
        a, b = cycles[0][0], cycles[-1][2]
        stats["cycles_span_gs"] = [df["time_gs"].iloc[a], df["time_gs"].iloc[b]]
    return stats


def process(cfg, out, stem):
    df, path = load(out, stem)
    meta = json.load(open(os.path.join(out, stem, f"{stem}_meta.json")))
    tare = tare_kg(df)
    d = derive(df, tare)
    cycles, info = find_cycles(df)
    d["phase"], d["cycle_id"] = "", np.nan
    for k, (a, t, b) in enumerate(cycles, start=1):
        d.iloc[a:t, d.columns.get_loc("phase")] = "generation"
        d.iloc[t:b, d.columns.get_loc("phase")] = "recovery"
        d.iloc[a:b, d.columns.get_loc("cycle_id")] = k
    cyc = cycle_rows(df, d, cycles, info, stem)
    cyc.to_csv(os.path.join(out, stem, f"{stem}_cycles.csv"), index=False, float_format="%.6g")
    stats = flight_stats(cfg, stem, df, d, cycles, tare, meta)
    with open(os.path.join(out, stem, f"{stem}_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)
    # derived columns go right after the time columns
    head = ["time_gs", "time_utc", "time_rome", "t_s", "in_window"]
    full = pd.concat([df[head], d[DERIVED], df.drop(columns=head)], axis=1)
    write_csv(full, path, index=False)
    print(f"{stem}: {len(cycles)} cycles, tare {tare} kg, E_batt {stats['energy_wh']['batt']} Wh, "
          f"datalink {stats['datalink_status_pct']}, rtk {stats['rtk_status_pct']['pct_of_covered_time']}")
    return cyc, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="processed/KTE_2030_FTC_0001")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(CONFIG))
    stems = list(cfg["events"]) if args.all else (args.event or [])
    cycs, summary = [], []
    for stem in stems:
        cyc, st = process(cfg, args.out, stem)
        cycs.append(cyc)
        summary.append({"stem": stem, "duration_s": st["duration_s"], "n_cycles": st["n_cycles"],
                        "wind_mean_ms": st["wind_ms"]["mean"], "wind_max_ms": st["wind_ms"]["max"],
                        **{f"E_{k}_wh": v for k, v in st["energy_wh"].items()},
                        **{f"P_{k}_mean_kw": v["mean"] for k, v in st["power_kw"].items()},
                        "force_mean_kg": st["force_kg_raw"]["mean"], "force_max_kg": st["force_kg_raw"]["max"],
                        "tare_kg": st["tare_kg_provisional"], "pod_energy_wh": st["pod_battery"]["energy_used_wh"],
                        "pod_current_max_a": st["pod_current_a"]["max"],
                        "ipok": st["rtk_status_pct"]["ipok"],
                        **{f"rtk_{k}_pct": v for k, v in (st["rtk_status_pct"]["pct_of_covered_time"] or {}).items()},
                        **{f"link_{k}_pct": v for k, v in (st["datalink_status_pct"] or {}).items()}})
    if args.all:
        os.makedirs(os.path.join(args.out, "campaign"), exist_ok=True)
        pd.concat(cycs).to_csv(os.path.join(args.out, "campaign", "cycles_all.csv"), index=False, float_format="%.6g")
        pd.DataFrame(summary).to_csv(os.path.join(args.out, "campaign", "flights_summary.csv"), index=False)


if __name__ == "__main__":
    main()
