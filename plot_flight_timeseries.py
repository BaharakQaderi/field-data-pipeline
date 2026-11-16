"""
Interactive time series of one flight, read straight from the ground-station InfluxDB.

Panels share one time axis (zoom / pan in any panel zooms all of them):
  1. Tether force       OPC_DsLoadCells.MeasureFloat_SUM [kg] -> kN = 9.81·(SUM − tare)/1000, tare = lowest 30 s median before launch
  2. Drum line speed    OPC_DsEncoder.outTamburo_SpeedMSec [m/s], < 0 = reel-out
  3. Depower            GS_UPLINK_COMMAND_pitch (target sent to the pod) and FLIGHT_SEGMENT_pitch (measured) [%]
  4. Electrical power   OPC_ConvStruct.CONV_READ.CONV_MEAS_FB_LS_PWR_SCALED_CALC [kW], > 0 = charging, < 0 = consumption
  5. Wind speed         WIND_wind.speed [m/s]
  6. Motor speed        DsInverters.Velocity_Target[1] / Velocity_ActualValue[1] (INV1 brushless / recovery) and
                        DsEncoder.outTamburo_SpeedRPM (drum) [rpm]; INV2 generator shown in the legend, hidden by default
  7. Inverter status    DsInverters.InternalLimitActive[i] (True = IIT limit active), StateDrive_Fault[i] (True = fault)
  8. Pod motor torque   FLIGHT_SEGMENT_servo_a_torque / servo_b_torque [N·m]; pod V3 (wire code 2): A = left, B = right; V2 (code 1): A = pitch, B = roll

Field meanings / sign conventions: docs/data_dictionary/README.md.
InfluxQL (no Flux pivot) because every source is written as one point per snapshot.

Usage:
    python plot_flight_timeseries.py --date 2026-10-24 --start 16:00:00 --stop 16:13:00
Outputs go to flight_timeseries/<date>_<start>/ : HTML plot, CSV + Parquet per source, summary.json
"""

import argparse
import json
import os
import re
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TZ = "Europe/Rome"
G = 9.81
PAD_S = 120  # seconds of context shown before/after the flight window
TARE_S = 600  # look-back before the padded window for the load-cell tare (quietest 30 s median)
INVERTERS = {1: "INV1 brushless", 2: "INV2 generator", 3: "INV3 thread guide"}
COMPOSE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Kite-projects", "tk-hmi-monorepo",
                       "docker-compose.yml")

SIGNALS = {
    "OPC": {
        "force_sum_kg": "OPC_DsLoadCells.MeasureFloat_SUM",
        "drum_speed_ms": "OPC_DsEncoder.outTamburo_SpeedMSec",
        "winded_length_m": "OPC_DsEncoder.outTamburo_WindedLength",
        "battery_power_kw": "OPC_ConvStruct.CONV_READ.CONV_MEAS_FB_LS_PWR_SCALED_CALC",
        "drum_speed_rpm": "OPC_DsEncoder.outTamburo_SpeedRPM",
        **{f"inv{i}_velocity_target_rpm": f"OPC_DsInverters.Velocity_Target[{i}]" for i in (1, 2)},
        **{f"inv{i}_velocity_actual_rpm": f"OPC_DsInverters.Velocity_ActualValue[{i}]" for i in (1, 2)},
        **{f"inv{i}_iit_active": f"OPC_DsInverters.InternalLimitActive[{i}]" for i in INVERTERS},
        **{f"inv{i}_fault": f"OPC_DsInverters.StateDrive_Fault[{i}]" for i in INVERTERS},
    },
    "GS_UPLINK_COMMAND": {"depower_target_pct": "GS_UPLINK_COMMAND_pitch"},
    "FLIGHT_SEGMENT": {"depower_measured_pct": "FLIGHT_SEGMENT_pitch", "pod_version": "FLIGHT_SEGMENT_pod_version",
                       "servo_a_torque_nm": "FLIGHT_SEGMENT_servo_a_torque",
                       "servo_b_torque_nm": "FLIGHT_SEGMENT_servo_b_torque"},
    "WIND": {"wind_speed_ms": "WIND_wind.speed", "wind_direction_deg": "WIND_wind.direction"},
}

COL = {"force": "#2a78d6", "speed": "#1f9e89", "target": "#eb6834", "measured": "#6a4fb3",
       "gen": "#2a78d6", "cons": "#eb6834", "wind": "#52514e", "flight": "rgba(42,120,214,0.07)",
       "drum": "#52514e", "iit": "#e8a200", "fault": "#d03b3b", "servo_a": "#6a4fb3", "servo_b": "#1f9e89"}


def default_token():
    if os.environ.get("INFLUX_TOKEN"):
        return os.environ["INFLUX_TOKEN"]
    m = re.search(r"DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=(\S+)", open(COMPOSE).read())
    return m.group(1)


def influxql(args, query):
    body = urllib.parse.urlencode({"db": args.bucket, "epoch": "ns", "q": query}).encode()
    req = urllib.request.Request(f"{args.url.rstrip('/')}/query", data=body, headers={
        "Authorization": f"Token {args.token}", "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())["results"][0]
    if "error" in result:
        raise RuntimeError(result["error"])
    series = result.get("series", [])
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series[0]["values"], columns=series[0]["columns"])


def fetch(args, source, fields, start, stop):
    sel = ", ".join(f'"{f}" AS "{name}"' for name, f in fields.items())
    df = influxql(args, f"SELECT {sel} FROM tk WHERE source='{source}' "
                        f"AND time >= '{start.isoformat()}' AND time < '{stop.isoformat()}'")
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="ns", utc=True)
    df = df.set_index("time").apply(pd.to_numeric, errors="coerce").dropna(how="all")
    df.insert(0, "time_local", df.index.tz_convert(TZ))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="local date YYYY-MM-DD (Europe/Rome)")
    ap.add_argument("--start", required=True, help="flight start, local HH:MM:SS")
    ap.add_argument("--stop", required=True, help="flight stop, local HH:MM:SS")
    ap.add_argument("--name", default=None, help="title, default 'Flight <date> <start>-<stop>'")
    ap.add_argument("--url", default=os.environ.get("INFLUX_URL", "http://192.168.1.100:8086"))
    ap.add_argument("--token", default=None)
    ap.add_argument("--bucket", default=os.environ.get("INFLUX_BUCKET", "gs"))
    ap.add_argument("--tare-kg", type=float, default=None, help="load-cell tare, default: from data before launch")
    ap.add_argument("--out", default="flight_timeseries")
    args = ap.parse_args()
    args.token = args.token or default_token()

    t0 = pd.Timestamp(f"{args.date} {args.start}", tz=TZ)
    t1 = pd.Timestamp(f"{args.date} {args.stop}", tz=TZ)
    w0, w1 = t0 - pd.Timedelta(seconds=PAD_S), t1 + pd.Timedelta(seconds=PAD_S)
    name = args.name or f"Flight {args.date} {args.start[:5]}–{args.stop[:5]}"
    out_dir = os.path.join(args.out, f"{args.date}_{args.start.replace(':', '')[:4]}")
    os.makedirs(out_dir, exist_ok=True)

    data = {}
    for source, fields in SIGNALS.items():
        df = fetch(args, source, fields, w0.tz_convert("UTC"), w1.tz_convert("UTC"))
        print(f"{source:18s} {len(df):6d} rows")
        data[source] = df
        if not df.empty:
            df.to_csv(os.path.join(out_dir, f"{source}.csv"), index_label="time_utc")
            df.to_parquet(os.path.join(out_dir, f"{source}.parquet"))

    opc = data["OPC"]
    tare_df = fetch(args, "OPC", {"f": SIGNALS["OPC"]["force_sum_kg"]},
                    (w0 - pd.Timedelta(seconds=TARE_S)).tz_convert("UTC"), w0.tz_convert("UTC"))
    if args.tare_kg is not None:
        tare_kg = args.tare_kg
    else:
        tare_kg = float(tare_df["f"].rolling("30s").median().min()) if not tare_df.empty else 0.0
    opc["tether_force_kn"] = G * (opc["force_sum_kg"] - tare_kg) / 1000

    # summary over the flight window only
    fl = opc[(opc.index >= t0) & (opc.index < t1)]
    dt_h = fl.index.to_series().diff().dt.total_seconds().fillna(0) / 3600
    p = fl["battery_power_kw"]
    wind = data["WIND"]
    wind_fl = wind[(wind.index >= t0) & (wind.index < t1)]["wind_speed_ms"] if not wind.empty else pd.Series(dtype=float)
    summary = {
        "flight": name, "start_local": str(t0), "stop_local": str(t1), "tare_kg": round(tare_kg, 2),
        "force_max_kn": round(float(fl["tether_force_kn"].max()), 2),
        "force_mean_kn": round(float(fl["tether_force_kn"].mean()), 2),
        "drum_speed_min_ms": round(float(fl["drum_speed_ms"].min()), 2),
        "drum_speed_max_ms": round(float(fl["drum_speed_ms"].max()), 2),
        "power_mean_kw": round(float(p.mean()), 3),
        "energy_charged_kwh": round(float((p.clip(lower=0) * dt_h).sum()), 4),
        "energy_consumed_kwh": round(float((-p.clip(upper=0) * dt_h).sum()), 4),
        "energy_net_kwh": round(float((p * dt_h).sum()), 4),
        "wind_mean_ms": round(float(wind_fl.mean()), 2) if len(wind_fl) else None,
        "wind_max_ms": round(float(wind_fl.max()), 2) if len(wind_fl) else None,
        "inv1_speed_max_rpm": int(fl["inv1_velocity_actual_rpm"].abs().max()),
        **{f"inv{i}_iit_active_pct": round(100 * float(fl[f"inv{i}_iit_active"].astype(float).mean()), 1)
           for i in INVERTERS},
        **{f"inv{i}_fault_samples": int(fl[f"inv{i}_fault"].astype(float).sum()) for i in INVERTERS},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    fs = data["FLIGHT_SEGMENT"]
    # pod_version is the wire code (FPA common/datalink/enums.py: 1 = V2, 2 = V3); the October campaign flew V3
    pod_v2 = fs["pod_version"].dropna().iloc[-1] == 1 if "pod_version" in fs and fs["pod_version"].notna().any() else False
    servo_names = ("Servo A – pitch/depower", "Servo B – roll/steering") if pod_v2 else ("Servo A – left", "Servo B – right")
    fig = make_subplots(rows=8, cols=1, shared_xaxes=True, vertical_spacing=0.018,
                        row_heights=[1, 1, 1, 1, 0.8, 1.2, 0.9, 1], subplot_titles=(
        f"Tether force [kN]  (load-cell sum, tare {tare_kg:.1f} kg)",
        "Drum line speed [m/s]  (< 0 = reel-out)",
        "Depower [%]  (target = uplink command, measured = pod)",
        "Battery-side electrical power [kW]  (> 0 charging, < 0 consumption)",
        "Wind speed [m/s]",
        "Motor speed [rpm]  (INV1 brushless target / actual, drum)",
        "Inverter status  (raised = True: IIT limit active / drive fault)",
        f"Pod motor torque [N·m]  (pod {'V2' if pod_v2 else 'V3'})"))
    x = opc["time_local"]
    fig.add_trace(go.Scattergl(x=x, y=opc["tether_force_kn"], name="Tether force", line=dict(color=COL["force"], width=1)), 1, 1)
    fig.add_trace(go.Scattergl(x=x, y=opc["drum_speed_ms"], name="Drum speed", line=dict(color=COL["speed"], width=1)), 2, 1)
    for src, col, label, color in [("GS_UPLINK_COMMAND", "depower_target_pct", "Depower target", COL["target"]),
                                   ("FLIGHT_SEGMENT", "depower_measured_pct", "Depower measured", COL["measured"])]:
        df = data[src]
        if not df.empty and col in df:
            d = df[col].dropna()
            fig.add_trace(go.Scattergl(x=df.loc[d.index, "time_local"], y=d, name=label,
                                       line=dict(color=color, width=1.2, shape="hv")), 3, 1)
    fig.add_trace(go.Scattergl(x=x, y=opc["battery_power_kw"], name="Battery power",
                               line=dict(color=COL["cons"], width=1)), 4, 1)
    fig.add_trace(go.Scattergl(x=x, y=opc["battery_power_kw"].rolling("5s").mean(), name="Battery power (5 s mean)",
                               line=dict(color=COL["gen"], width=2)), 4, 1)
    if not wind.empty:
        fig.add_trace(go.Scattergl(x=wind["time_local"], y=wind["wind_speed_ms"], name="Wind speed",
                                   line=dict(color=COL["wind"], width=1)), 5, 1)
    for col, label, color, dash, visible in [
            ("inv1_velocity_target_rpm", "INV1 brushless target", COL["target"], None, True),
            ("inv1_velocity_actual_rpm", "INV1 brushless actual", COL["force"], None, True),
            ("drum_speed_rpm", "Drum speed (rpm)", COL["drum"], "dot", True),
            ("inv2_velocity_target_rpm", "INV2 generator target", COL["iit"], None, "legendonly"),
            ("inv2_velocity_actual_rpm", "INV2 generator actual", COL["servo_b"], None, "legendonly")]:
        fig.add_trace(go.Scattergl(x=x, y=opc[col], name=label, visible=visible,
                                   line=dict(color=color, width=1.2, dash=dash)), 6, 1)

    # boolean status: one lane per (inverter, bit), raised by 0.7 when True
    lanes = [(i, kind) for kind in ("iit_active", "fault") for i in INVERTERS]
    for k, (i, kind) in enumerate(lanes):
        b = opc[f"inv{i}_{kind}"].astype(float)
        label = f"{INVERTERS[i]} {'IIT' if kind == 'iit_active' else 'FAULT'}"
        fig.add_trace(go.Scattergl(x=x, y=k + 0.7 * b, name=label, showlegend=False,
                                   customdata=b.map({1.0: "True", 0.0: "False"}),
                                   hovertemplate=f"{label}: %{{customdata}}<extra></extra>",
                                   line=dict(color=COL["iit" if kind == "iit_active" else "fault"], width=1.5,
                                             shape="hv"), fill=None), 7, 1)
    fig.update_yaxes(tickvals=list(range(len(lanes))), range=[-0.3, len(lanes) - 0.1],
                     ticktext=[f"INV{i} {'IIT' if kind == 'iit_active' else 'FAULT'}" for i, kind in lanes],
                     row=7, col=1)

    for col, label, color in [("servo_a_torque_nm", servo_names[0], COL["servo_a"]),
                              ("servo_b_torque_nm", servo_names[1], COL["servo_b"])]:
        if col in fs:
            d = fs[col].dropna()
            fig.add_trace(go.Scattergl(x=fs.loc[d.index, "time_local"], y=d, name=label,
                                       line=dict(color=color, width=1)), 8, 1)

    for r in range(1, 9):
        fig.add_vrect(x0=t0, x1=t1, fillcolor=COL["flight"], line_width=0, row=r, col=1)
    for r in (2, 4, 6, 8):
        fig.add_hline(y=0, line=dict(color="#999", width=0.8, dash="dot"), row=r, col=1)
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, spikecolor="#999")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04), row=8, col=1)
    fig.update_layout(title=f"{name}  —  shaded = flight window", height=2200, template="plotly_white",
                      hovermode="x unified", legend=dict(orientation="h", y=1.04, x=0),
                      margin=dict(l=60, r=20, t=110, b=40))
    html = os.path.join(out_dir, "timeseries.html")
    fig.write_html(html, include_plotlyjs="cdn")
    print(f"\nwrote {html}")


if __name__ == "__main__":
    main()
