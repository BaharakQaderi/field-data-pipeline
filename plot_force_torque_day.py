"""
Whole-day scatters of the generator torque against the load-cell force and the winded length, coloured by time of day.

Figure 1, force_torque_scatter.html
  x  tether force   OPC_DsLoadCells.MeasureFloat_SUM [kg] -> kN = 9.81·SUM/1000 (raw, no tare: the load-cell
                    zero drifts during the day, e.g. −55 kg on the morning of 2026-10-24)
  y  generator      OPC_DsInverters.Torque_ActualValue[2] (INV2 = generator), drive value; HMI says Nm but the
     torque         scale is unconfirmed (docs/data_dictionary/README.md, "Do not use ... for absolute power")

Two traces, toggled from the legend: "active" samples (drum turning or |torque| > ACTIVE_TORQUE) at full 10 Hz,
and "idle" samples decimated to 1 Hz. A from/to time slider above the plot filters both traces by time of day;
the colour scale stays fixed to the whole day so a colour always means the same time.

Figure 2, ratio_length_scatter.html
  x  winded length  OPC_DsEncoder.outTamburo_WindedLength [m] (HMI "WINDED LINE")
  y  torque / force Torque_ActualValue[2] / (9.81·SUM) — if the torque is in N·m this is the lever arm in m, i.e.
                    the effective drum radius (bare drum 0.2455 m). Active samples only, and only where
                    SUM > MIN_FORCE_KG so the ratio does not blow up near zero force.

Usage:
    python plot_force_torque_day.py --date 2026-10-24
Outputs go to flight_timeseries/<date>_day/ : force_torque_scatter.html, ratio_length_scatter.html,
OPC_force_torque.parquet
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plot_flight_timeseries import G, TZ, default_token, fetch

ACTIVE_RPM = 1.0  # |drum rpm| above this = drum turning
ACTIVE_TORQUE = 2.0  # |generator torque| above this = drive loaded
IDLE_EVERY = 10  # keep every 10th idle sample (10 Hz -> 1 Hz)
MIN_FORCE_KG = 30.0  # ratio only above this load-cell sum
DRUM_RADIUS_M = 0.491 / 2

FIELDS = {
    "force_sum_kg": "OPC_DsLoadCells.MeasureFloat_SUM",
    "gen_torque": "OPC_DsInverters.Torque_ActualValue[2]",
    "drum_speed_rpm": "OPC_DsEncoder.outTamburo_SpeedRPM",
    "winded_length_m": "OPC_DsEncoder.outTamburo_WindedLength",
}

# from/to sliders: filter every trace by hour of day (kept in customdata[0]) and restyle
SLIDER_JS = """
const gd = document.getElementById('{plot_id}');
const orig = gd.data.map(t => ({x: Array.from(t.x), y: Array.from(t.y), c: Array.from(t.marker.color),
                                 h: t.customdata.map(r => r[0]), s: t.customdata.map(r => r[1])}));
const lo = Math.floor(orig.reduce((m, o) => o.h.reduce((a, h) => Math.min(a, h), m), Infinity) * 60);
const hi = Math.ceil(orig.reduce((m, o) => o.h.reduce((a, h) => Math.max(a, h), m), -Infinity) * 60);
const fmt = m => String(Math.floor(m / 60)).padStart(2, '0') + ':' + String(Math.round(m % 60)).padStart(2, '0');
const box = document.createElement('div');
box.style.cssText = 'font:14px sans-serif;margin:8px 60px;display:flex;gap:16px;align-items:center;flex-wrap:wrap';
box.innerHTML = `<b>Time window</b>
  <label>from <input id="t0" type="range" min="${lo}" max="${hi}" value="${lo}" style="width:300px"></label>
  <label>to <input id="t1" type="range" min="${lo}" max="${hi}" value="${hi}" style="width:300px"></label>
  <span id="tl"></span> <button id="treset">whole day</button>`;
gd.parentNode.insertBefore(box, gd);
const t0 = document.getElementById('t0'), t1 = document.getElementById('t1'), tl = document.getElementById('tl');
function update() {
  let a = +t0.value, b = +t1.value; if (a > b) [a, b] = [b, a];
  tl.textContent = fmt(a) + ' – ' + fmt(b);
  const xs = [], ys = [], cs = [], cd = [];
  orig.forEach(o => {
    const idx = o.h.map((h, i) => [h * 60, i]).filter(([m]) => m >= a && m <= b).map(([, i]) => i);
    xs.push(idx.map(i => o.x[i])); ys.push(idx.map(i => o.y[i])); cs.push(idx.map(i => o.c[i]));
    cd.push(idx.map(i => [o.h[i], o.s[i]]));
  });
  Plotly.restyle(gd, {x: xs, y: ys, 'marker.color': cs, customdata: cd});
}
t0.oninput = t1.oninput = update;
document.getElementById('treset').onclick = () => { t0.value = lo; t1.value = hi; update(); };
update();
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="local date YYYY-MM-DD (Europe/Rome)")
    ap.add_argument("--url", default=os.environ.get("INFLUX_URL", "http://192.168.1.100:8086"))
    ap.add_argument("--token", default=None)
    ap.add_argument("--bucket", default=os.environ.get("INFLUX_BUCKET", "gs"))
    ap.add_argument("--out", default="flight_timeseries")
    args = ap.parse_args()
    args.token = args.token or default_token()

    d0 = pd.Timestamp(args.date, tz=TZ)
    out_dir = os.path.join(args.out, f"{args.date}_day")
    os.makedirs(out_dir, exist_ok=True)

    df = fetch(args, "OPC", FIELDS, d0.tz_convert("UTC"), (d0 + pd.Timedelta(days=1)).tz_convert("UTC"))
    df = df.dropna(subset=["force_sum_kg", "gen_torque"])
    df.to_parquet(os.path.join(out_dir, "OPC_force_torque.parquet"))
    df["force_kn"] = G * df["force_sum_kg"] / 1000
    df["hour"] = (df["time_local"] - d0).dt.total_seconds() / 3600
    df["active"] = (df["drum_speed_rpm"].abs() > ACTIVE_RPM) | (df["gen_torque"].abs() > ACTIVE_TORQUE)
    print(f"{len(df)} samples, {int(df['active'].sum())} active")

    df["ratio"] = df["gen_torque"] / (G * df["force_sum_kg"])
    h_range = (np.floor(df["hour"].min()), np.ceil(df["hour"].max()))
    active = df[df["active"]]

    fig = scatter([("Active (drum turning or |T| > 2), 10 Hz", active, 4),
                   ("Idle, 1 Hz", df[~df["active"]].iloc[::IDLE_EVERY], 3)],
                  "force_kn", "gen_torque", h_range, "force %{x:.2f} kN<br>torque %{y:.1f}")
    fig.update_layout(
        title=f"Tether force vs generator torque — {args.date} (whole day, colour = time of day)",
        xaxis_title="Tether force [kN]  (9.81 · load-cell sum, no tare)",
        yaxis_title="Generator torque  Torque_ActualValue[2]  (drive value, scale unconfirmed)")
    write(fig, os.path.join(out_dir, "force_torque_scatter.html"))

    loaded = active[active["force_sum_kg"] > MIN_FORCE_KG]
    print(f"ratio: {len(loaded)} samples with load-cell sum > {MIN_FORCE_KG:.0f} kg, "
          f"median {loaded['ratio'].median():.3f}, IQR {loaded['ratio'].quantile(0.25):.3f}–{loaded['ratio'].quantile(0.75):.3f}")
    fig = scatter([(f"Active, load-cell sum > {MIN_FORCE_KG:.0f} kg, 10 Hz", loaded, 4)],
                  "winded_length_m", "ratio", h_range, "length %{x:.1f} m<br>torque/force %{y:.3f}")
    fig.add_hline(y=DRUM_RADIUS_M, line=dict(color="#52514e", width=1, dash="dot"),
                  annotation_text=f"bare drum radius {DRUM_RADIUS_M:.4f} m", annotation_position="top left")
    fig.update_layout(
        title=f"Generator torque / tether force vs winded length — {args.date} (colour = time of day)",
        xaxis_title="Winded length [m]  (outTamburo_WindedLength)",
        yaxis_title="Torque_ActualValue[2] / (9.81 · load-cell sum)  [m if torque is N·m]")
    write(fig, os.path.join(out_dir, "ratio_length_scatter.html"))


def scatter(parts, x, y, h_range, hover):
    ticks = np.arange(h_range[0], h_range[1] + 1)
    fig = go.Figure()
    for label, part, size in parts:
        fig.add_trace(go.Scattergl(
            # plain lists, not numpy: the slider JS reads gd.data back as JS arrays
            x=part[x].round(4).tolist(), y=part[y].round(4).tolist(), mode="markers", name=label,
            customdata=list(zip(part["hour"].round(6), part["time_local"].dt.strftime("%H:%M:%S.%f").str[:-5])),
            hovertemplate="%{customdata[1]}<br>" + hover + "<extra></extra>",
            marker=dict(size=size, opacity=0.6, color=part["hour"].round(6).tolist(), cmin=h_range[0],
                        cmax=h_range[1], colorscale="Turbo",
                        colorbar=dict(title="time", tickvals=ticks, ticktext=[f"{int(t):02d}:00" for t in ticks]))))
    fig.update_layout(template="plotly_white", height=800, showlegend=True,
                      legend=dict(orientation="h", y=1.06, x=0), margin=dict(l=70, r=20, t=110, b=60))
    return fig


def write(fig, html):
    fig.write_html(html, include_plotlyjs="cdn", post_script=SLIDER_JS)
    print(f"wrote {html}  ({os.path.getsize(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
