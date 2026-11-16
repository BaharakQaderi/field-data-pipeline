"""
Pod steering/depower command vs measured position vs servo torque, for short time windows, from InfluxDB.

Per window one interactive HTML page with three stacked panels on a shared time axis:
  1. Roll (steering, %):  GS_UPLINK_COMMAND_roll (2 Hz snapshot of what the GS sends), FS_MAINTENANCE_roll
     (command events, ~20 Hz while the stick moves), TARANIS_Roll (stick), FLIGHT_SEGMENT_roll (measured position)
  2. Pitch (depower, %):  same sources for pitch
  3. Servo torques (N·m): FLIGHT_SEGMENT_servo_a_torque / servo_b_torque (V3 pod: A = left motor, B = right motor)
Field meanings: docs/data_dictionary/03_flight_segment_pod_fields.md

Usage:
  python plot_pod_command_response.py --date 2026-10-22 \
      --window "Attempt 1=16:44:00-16:47:00" --window "Attempt 2=16:50:00-16:55:00"
"""

import argparse
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import export_influx_day
from export_influx_day import MEASUREMENT, TZ, Influx, quote_field, rfc3339

export_influx_day.RETRIES = 1

SOURCES = {
    "GS_UPLINK_COMMAND": ["GS_UPLINK_COMMAND_roll", "GS_UPLINK_COMMAND_pitch"],
    "FS_MAINTENANCE": ["FS_MAINTENANCE_roll", "FS_MAINTENANCE_pitch"],
    "TARANIS": ["TARANIS_Roll", "TARANIS_Pitch"],
    "FLIGHT_SEGMENT": ["FLIGHT_SEGMENT_roll", "FLIGHT_SEGMENT_pitch", "FLIGHT_SEGMENT_servo_a_torque",
                       "FLIGHT_SEGMENT_servo_b_torque", "FLIGHT_SEGMENT_pod_version"],
}
SERVO_NAMES = {1: ("A = pitch motor", "B = roll motor"), 2: ("A = left motor", "B = right motor")}  # pod_version 1=V2, 2=V3

COL = {"uplink": "#9a9890", "maint": "#eb6834", "taranis": "#eda100", "measured": "#2a78d6",
       "servo_a": "#2a78d6", "servo_b": "#1baf7a", "text": "#52514e", "grid": "#e6e5e1"}


def fetch(db, start, stop):
    out = {}
    for source, fields in SOURCES.items():
        sel = ",".join(quote_field(f) for f in fields)
        df = db.influxql(f"SELECT {sel} FROM {MEASUREMENT} WHERE source='{source}' "
                         f"AND time >= '{rfc3339(start)}' AND time < '{rfc3339(stop)}'")
        if df.empty:
            out[source] = pd.DataFrame(columns=["time"] + fields)
            continue
        df["time"] = pd.to_datetime(df["time"].astype("int64"), utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
        for f in fields:
            df[f] = pd.to_numeric(df[f]) if f in df else pd.NA
        out[source] = df
    return out


def series(df, col):
    s = df[["time", col]].dropna() if col in df else pd.DataFrame(columns=["time", col])
    return s["time"], s[col]


def figure(name, data, start, stop):
    fs = data["FLIGHT_SEGMENT"]
    version = int(fs["FLIGHT_SEGMENT_pod_version"].dropna().mode().iloc[0]) if fs["FLIGHT_SEGMENT_pod_version"].notna().any() else 2
    a_name, b_name = SERVO_NAMES.get(version, ("A", "B"))

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.36, 0.28, 0.36],
                        subplot_titles=("Roll / steering (%, −100 = full left): command vs measured position",
                                        "Pitch / depower (%): command vs measured position",
                                        f"Pod servo torque (N·m) — V{version + 1} pod: servo {a_name}, servo {b_name}"))
    for row, axis in ((1, "roll"), (2, "pitch")):
        legend = row == 1
        x, y = series(data["GS_UPLINK_COMMAND"], f"GS_UPLINK_COMMAND_{axis}")
        fig.add_trace(go.Scatter(x=x, y=y, name="Uplink command (2 Hz)", legendgroup="uplink", showlegend=legend,
                                 line=dict(color=COL["uplink"], width=2, shape="hv"),
                                 hovertemplate="%{y:.0f} % uplink<extra></extra>"), row, 1)
        x, y = series(data["FS_MAINTENANCE"], f"FS_MAINTENANCE_{axis}")
        fig.add_trace(go.Scatter(x=x, y=y, name="FS_MAINTENANCE command events", legendgroup="maint", showlegend=legend,
                                 mode="lines+markers", line=dict(color=COL["maint"], width=1.5, shape="hv"),
                                 marker=dict(size=4), hovertemplate="%{y:.0f} % command<extra></extra>"), row, 1)
        x, y = series(data["TARANIS"], f"TARANIS_{axis.capitalize()}")
        fig.add_trace(go.Scatter(x=x, y=y, name="Taranis stick", legendgroup="taranis", showlegend=legend,
                                 visible="legendonly", line=dict(color=COL["taranis"], width=1.5, shape="hv"),
                                 hovertemplate="%{y:.0f} % stick<extra></extra>"), row, 1)
        x, y = series(fs, f"FLIGHT_SEGMENT_{axis}")
        fig.add_trace(go.Scatter(x=x, y=y, name="Measured position (FLIGHT_SEGMENT)", legendgroup="measured",
                                 showlegend=legend, line=dict(color=COL["measured"], width=2),
                                 hovertemplate="%{y:.1f} % measured<extra></extra>"), row, 1)
    x, y = series(fs, "FLIGHT_SEGMENT_servo_a_torque")
    fig.add_trace(go.Scatter(x=x, y=y, name=f"Servo {a_name}", line=dict(color=COL["servo_a"], width=1.5),
                             hovertemplate="%{y:.2f} N·m servo A<extra></extra>"), 3, 1)
    x, y = series(fs, "FLIGHT_SEGMENT_servo_b_torque")
    fig.add_trace(go.Scatter(x=x, y=y, name=f"Servo {b_name}", line=dict(color=COL["servo_b"], width=1.5),
                             hovertemplate="%{y:.2f} N·m servo B<extra></extra>"), 3, 1)
    for row in (1, 2, 3):
        fig.add_hline(y=0, line=dict(color=COL["text"], width=1), row=row, col=1)

    fig.update_layout(
        title=dict(text=f"<b>{name}</b> — pod command vs response, {start:%Y-%m-%d %H:%M:%S}–{stop:%H:%M:%S} local",
                   font=dict(size=15)),
        template="plotly_white", height=900, hovermode="x unified", font=dict(color=COL["text"]),
        legend=dict(orientation="h", y=-0.06), margin=dict(l=70, r=30, t=80, b=60),
    )
    fig.update_xaxes(gridcolor=COL["grid"], tickformat="%H:%M:%S")
    fig.update_yaxes(gridcolor=COL["grid"], zeroline=False)
    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text="N·m", row=3, col=1)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--window", action="append", required=True, help='"Name=HH:MM:SS-HH:MM:SS" local time')
    ap.add_argument("--url", default="http://192.168.1.100:8086")
    ap.add_argument("--org", default="kitenergy")
    ap.add_argument("--bucket", default="gs")
    ap.add_argument("--token", default=os.environ.get(
        "INFLUX_TOKEN", "h4ABAleq01agLd3K5kCI6v5mL74Utcg4Bna8UmMiQ7Y0fnyAwQuYJMQGdczQN2m1xfSizaoFmdghdfPMta1zA"))
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    db = Influx(args.url, args.token, args.org, args.bucket, timeout=120)
    out_dir = os.path.join(args.out, args.date, "pod_command_response")
    os.makedirs(out_dir, exist_ok=True)
    for spec in args.window:
        name, span = spec.split("=", 1)
        t_from, t_to = span.split("-")
        start = pd.Timestamp(f"{args.date} {t_from.strip()}", tz=TZ)
        stop = pd.Timestamp(f"{args.date} {t_to.strip()}", tz=TZ)
        data = fetch(db, start, stop)
        slug = name.strip().replace(" ", "_").lower()
        merged = pd.concat([df.assign(source=src) for src, df in data.items()], ignore_index=True).sort_values("time")
        merged.to_csv(os.path.join(out_dir, f"{slug}_data.csv"), index=False)
        path = os.path.join(out_dir, f"{slug}.html")
        figure(name.strip(), data, start.tz_localize(None), stop.tz_localize(None)).write_html(path, include_plotlyjs="inline")
        counts = ", ".join(f"{src} {len(df)}" for src, df in data.items())
        print(f"{name.strip()}: {counts} rows -> {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
