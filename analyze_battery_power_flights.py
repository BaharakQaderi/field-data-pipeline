"""
Battery-side power analysis of flights, read straight from the ground-station InfluxDB.

Signal: OPC_ConvStruct.CONV_READ.CONV_MEAS_FB_LS_PWR_SCALED_CALC  [kW]
Computed by the HMI (tk-hmi-monorepo/python/restapi/workers/managers/opc.py) as
    sum_i( CONV_MEAS_FB_LS_VOLTAGE_SCALED[i] * CONV_MEAS_FB_LS_AVG_CURR_SCALED[i] ) / 1000,  i = 1..4
i.e. total low-side (battery) power of the 4 DC/DC converters.
Sign: > 0 = power delivered to the battery (generation), < 0 = drawn from it (consumption).

Usage:
    python analyze_battery_power_flights.py
Outputs go to battery_power_flights/<date>/ : HTML report, PNG, summary CSV, raw CSV per flight.
"""

import io
import os
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INFLUX_URL = "http://192.168.1.100:8086"
INFLUX_ORG = "kitenergy"
INFLUX_BUCKET = "gs"
INFLUX_TOKEN = (
    "h4ABAleq01agLd3K5kCI6v5mL74Utcg4Bna8UmMiQ7Y0fnyAwQuYJMQGdczQN2m1xfSizaoFmdghdfPMta1zA"
)
TZ = "Europe/Rome"

PWR = "OPC_ConvStruct.CONV_READ.CONV_MEAS_FB_LS_PWR_SCALED_CALC"
VOLT = "OPC_ConvStruct.CONV_READ.CONV_MEAS_FB_LS_VOLTAGE_SCALED[{}]"
CURR = "OPC_ConvStruct.CONV_READ.CONV_MEAS_FB_LS_AVG_CURR_SCALED[{}]"

DATE = "2026-10-22"
FLIGHTS = {
    "Flight 1": ("10:15:59", "10:39:29"),
    "Flight 2": ("11:59:20", "12:20:00"),
    "Flight 3": ("15:32:26", "15:53:00"),
}

SMOOTH_S = 5  # rolling-mean window for the trend line
OUT_DIR = os.path.join("battery_power_flights", DATE)

COL_GEN = "#2a78d6"
COL_CONS = "#eb6834"
COL_TEXT = "#52514e"
COL_GRID = "#e6e5e1"


def query_flight(start_local, stop_local):
    start = pd.Timestamp(f"{DATE} {start_local}", tz=TZ).tz_convert("UTC")
    stop = pd.Timestamp(f"{DATE} {stop_local}", tz=TZ).tz_convert("UTC")
    flux = f"""
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {start.strftime('%Y-%m-%dT%H:%M:%SZ')}, stop: {stop.strftime('%Y-%m-%dT%H:%M:%SZ')})
  |> filter(fn: (r) => r._measurement == "tk" and
       r._field =~ /CONV_MEAS_FB_LS_(PWR_SCALED_CALC|VOLTAGE_SCALED|AVG_CURR_SCALED)/)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> drop(columns: ["_start", "_stop", "result", "table"])
"""
    req = urllib.request.Request(
        f"{INFLUX_URL}/api/v2/query?org={INFLUX_ORG}",
        data=flux.encode(),
        headers={
            "Authorization": f"Token {INFLUX_TOKEN}",
            "Content-Type": "application/vnd.flux",
            "Accept": "application/csv",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        df = pd.read_csv(io.StringIO(resp.read().decode()))
    df["time"] = pd.to_datetime(df["_time"], format="ISO8601").dt.tz_convert(TZ)
    df = df.sort_values("time").reset_index(drop=True)
    df["t_s"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
    return df


def flight_stats(name, df):
    p = df[PWR].to_numpy()
    t = df["t_s"].to_numpy()
    gen_kwh = np.trapezoid(np.clip(p, 0, None), t) / 3600
    cons_kwh = -np.trapezoid(np.clip(p, None, 0), t) / 3600
    duration = t[-1] - t[0]
    return {
        "Flight": name,
        "Start (local)": df["time"].iloc[0].strftime("%H:%M:%S"),
        "End (local)": df["time"].iloc[-1].strftime("%H:%M:%S"),
        "Duration (min)": duration / 60,
        "Samples": len(df),
        "Mean power (kW)": p.mean(),
        "Median power (kW)": np.median(p),
        "Peak generation (kW)": p.max(),
        "Peak consumption (kW)": p.min(),
        "Mean while generating (kW)": p[p > 0].mean() if (p > 0).any() else 0.0,
        "Mean while consuming (kW)": p[p < 0].mean() if (p < 0).any() else 0.0,
        "Time generating (%)": (p > 0).mean() * 100,
        "Energy generated (Wh)": gen_kwh * 1000,
        "Energy consumed (Wh)": cons_kwh * 1000,
        "Net energy (Wh)": (gen_kwh - cons_kwh) * 1000,
        "Net mean power (W)": (gen_kwh - cons_kwh) * 1000 * 3600 / duration,
        "Gen/Cons ratio": gen_kwh / cons_kwh if cons_kwh else np.nan,
    }


def check_calc(df):
    """Recompute the HMI formula from the raw channels to make sure the stored value is consistent."""
    calc = sum(df[VOLT.format(i)] * df[CURR.format(i)] for i in range(1, 5)) / 1000
    return float((calc - df[PWR]).abs().max())


def flight_figure(name, df, stats):
    p = df[PWR]
    smooth = df.set_index("time")[PWR].rolling(f"{SMOOTH_S}s", center=True).mean().to_numpy()
    t = df["time"].dt.tz_localize(None)
    dt = df["t_s"].diff().fillna(0).to_numpy()
    pv = p.to_numpy()
    cum_gen = np.cumsum(np.clip(pv, 0, None) * dt) / 3.6  # Wh
    cum_cons = np.cumsum(np.clip(pv, None, 0) * dt) / 3.6
    cum_net = cum_gen + cum_cons

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4],
        subplot_titles=(
            "Battery-side power (kW) — positive = generation",
            "Cumulative energy (Wh)",
        ),
    )
    fig.add_trace(go.Scatter(x=t, y=np.clip(pv, 0, None), fill="tozeroy", mode="none",
                             fillcolor="rgba(42,120,214,0.35)", name="Generating (raw)",
                             hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=np.clip(pv, None, 0), fill="tozeroy", mode="none",
                             fillcolor="rgba(235,104,52,0.35)", name="Consuming (raw)",
                             hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=pv, mode="lines", line=dict(color="rgba(82,81,78,0.35)", width=1),
                             name="Power 10 Hz", hovertemplate="%{x|%H:%M:%S.%L}<br>%{y:.2f} kW<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=smooth, mode="lines", line=dict(color="#0b0b0b", width=2),
                             name=f"{SMOOTH_S}s mean", hovertemplate="%{x|%H:%M:%S}<br>%{y:.2f} kW (5s mean)<extra></extra>"), 1, 1)

    fig.add_trace(go.Scatter(x=t, y=cum_gen, mode="lines", line=dict(color=COL_GEN, width=2),
                             name="Generated", hovertemplate="%{y:.1f} Wh generated<extra></extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=cum_cons, mode="lines", line=dict(color=COL_CONS, width=2),
                             name="Consumed", hovertemplate="%{y:.1f} Wh consumed<extra></extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=cum_net, mode="lines", line=dict(color="#0b0b0b", width=2, dash="dot"),
                             name="Net", hovertemplate="%{y:.1f} Wh net<extra></extra>"), 2, 1)

    for row in (1, 2):
        fig.add_hline(y=0, line=dict(color=COL_TEXT, width=1), row=row, col=1)

    fig.update_layout(
        title=dict(text=(f"<b>{name}</b>  {stats['Start (local)']}–{stats['End (local)']} "
                         f"({stats['Duration (min)']:.1f} min)  ·  generated {stats['Energy generated (Wh)']:.0f} Wh, "
                         f"consumed {stats['Energy consumed (Wh)']:.0f} Wh, net {stats['Net energy (Wh)']:+.0f} Wh"),
                   font=dict(size=15)),
        template="plotly_white", height=620, hovermode="x unified",
        legend=dict(orientation="h", y=-0.12), margin=dict(l=60, r=30, t=80, b=60),
        font=dict(color=COL_TEXT),
    )
    fig.update_xaxes(gridcolor=COL_GRID, tickformat="%H:%M")
    fig.update_yaxes(gridcolor=COL_GRID, zeroline=False)
    return fig


def comparison_figure(summary):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=summary["Flight"], y=summary["Energy generated (Wh)"], name="Generated",
                         marker_color=COL_GEN, texttemplate="%{y:.0f} Wh", textposition="outside"))
    fig.add_trace(go.Bar(x=summary["Flight"], y=-summary["Energy consumed (Wh)"], name="Consumed",
                         marker_color=COL_CONS, texttemplate="%{y:.0f} Wh", textposition="outside"))
    fig.add_trace(go.Scatter(x=summary["Flight"], y=summary["Net energy (Wh)"], name="Net",
                             mode="markers+text", marker=dict(color="#0b0b0b", size=12, symbol="diamond"),
                             text=[f"net {v:+.0f} Wh" for v in summary["Net energy (Wh)"]],
                             textposition="middle right"))
    fig.add_hline(y=0, line=dict(color=COL_TEXT, width=1))
    fig.update_layout(title="<b>Energy balance per flight</b> (battery side)", barmode="relative",
                      template="plotly_white", height=420, yaxis_title="Energy (Wh)",
                      bargap=0.5, font=dict(color=COL_TEXT), margin=dict(l=60, r=30, t=60, b=40))
    fig.update_yaxes(gridcolor=COL_GRID)
    return fig


def distribution_figure(data):
    fig = go.Figure()
    for (name, df), color in zip(data.items(), [COL_GEN, "#1baf7a", "#6250d6"]):
        fig.add_trace(go.Histogram(x=df[PWR], name=name, xbins=dict(size=0.25), histnorm="percent",
                                   marker_color=color, opacity=0.6))
    fig.add_vline(x=0, line=dict(color=COL_TEXT, width=1))
    fig.update_layout(title="<b>Power distribution</b> (% of flight time per 0.25 kW bin)", barmode="overlay",
                      template="plotly_white", height=380, xaxis_title="Battery-side power (kW)",
                      yaxis_title="% of time", font=dict(color=COL_TEXT), margin=dict(l=60, r=30, t=60, b=50))
    return fig


def detect_cycles(df, min_gen_s=15):
    """Split a flight into pumping cycles: each cycle starts at a generation phase
    (5 s mean > 0 for at least min_gen_s) and ends where the next one starts."""
    smooth = df.set_index("time")[PWR].rolling(f"{SMOOTH_S}s", center=True).mean().to_numpy()
    t = df["t_s"].to_numpy()
    pos = smooth > 0
    edges = np.flatnonzero(np.diff(pos.astype(int)) == 1) + 1
    if pos[0]:
        edges = np.r_[0, edges]
    starts = []
    for e in edges:
        end = e + np.argmax(~pos[e:]) if (~pos[e:]).any() else len(pos)
        if t[end - 1] - t[e] >= min_gen_s:
            starts.append(e)
    rows = []
    p = df[PWR].to_numpy()
    for k, (a, b) in enumerate(zip(starts, starts[1:] + [len(df)])):
        seg_p, seg_t = p[a:b], t[a:b]
        if len(seg_t) < 2:
            continue
        gen = np.trapezoid(np.clip(seg_p, 0, None), seg_t) * 1000 / 3600
        cons = -np.trapezoid(np.clip(seg_p, None, 0), seg_t) * 1000 / 3600
        rows.append({
            "Cycle": k + 1,
            "Start (local)": df["time"].iloc[a].strftime("%H:%M:%S"),
            "Duration (s)": seg_t[-1] - seg_t[0],
            "Generating (s)": float(np.sum(np.diff(seg_t)[smooth[a:b - 1] > 0])),
            "Peak (kW)": seg_p.max(),
            "Generated (Wh)": gen,
            "Consumed (Wh)": cons,
            "Net (Wh)": gen - cons,
            "Net mean power (W)": (gen - cons) * 3600 / (seg_t[-1] - seg_t[0]),
            "Complete": b < len(df),
        })
    return pd.DataFrame(rows)


def table_html(summary, transpose=True):
    fmt = summary.copy()
    for c in fmt.columns:
        if fmt[c].dtype.kind == "f":
            fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"{v:,.2f}")
    fmt = fmt.fillna("")
    if transpose:
        return fmt.set_index("Flight").T.to_html(classes="summary", border=0)
    return fmt.to_html(classes="summary", border=0, index=False)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data, rows = {}, []
    for name, (start, stop) in FLIGHTS.items():
        df = query_flight(start, stop)
        err = check_calc(df)
        print(f"{name}: {len(df)} samples, max recompute error {err:.2e} kW")
        df.drop(columns=["_time"]).to_csv(os.path.join(OUT_DIR, f"{name.replace(' ', '_').lower()}_raw.csv"), index=False)
        data[name] = df
        rows.append(flight_stats(name, df))

    summary = pd.DataFrame(rows)
    total = {
        "Flight": "Total",
        "Duration (min)": summary["Duration (min)"].sum(),
        "Samples": summary["Samples"].sum(),
        "Energy generated (Wh)": summary["Energy generated (Wh)"].sum(),
        "Energy consumed (Wh)": summary["Energy consumed (Wh)"].sum(),
        "Net energy (Wh)": summary["Net energy (Wh)"].sum(),
        "Net mean power (W)": summary["Net energy (Wh)"].sum() * 3600 / (summary["Duration (min)"].sum() * 60),
    }
    summary_with_total = pd.concat([summary, pd.DataFrame([total])], ignore_index=True)
    summary_with_total.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print(summary_with_total.set_index("Flight").T.to_string())

    cycle_html = ""
    for name, df in data.items():
        cycles = detect_cycles(df)
        cycles.insert(0, "Flight", name)
        cycles.to_csv(os.path.join(OUT_DIR, f"{name.replace(' ', '_').lower()}_cycles.csv"), index=False)
        print(f"\n{name} cycles:\n{cycles.round(1).to_string(index=False)}")
        cycle_html += f"<h3>{name} — pumping cycles</h3>" + table_html(cycles.drop(columns="Flight"), transpose=False)

    figs = [flight_figure(n, data[n], s) for n, s in zip(data, rows)]
    figs += [comparison_figure(summary), distribution_figure(data)]

    for i, fig in enumerate(figs):
        try:
            fig.write_image(os.path.join(OUT_DIR, f"fig{i + 1}.png"), width=1300, height=fig.layout.height, scale=2)
        except Exception as e:  # kaleido not installed
            print(f"PNG export skipped: {e}")
            break

    parts = [figs[0].to_html(full_html=False, include_plotlyjs="inline")]
    parts += [f.to_html(full_html=False, include_plotlyjs=False) for f in figs[1:]]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Flight Power Report {DATE}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, sans-serif; background:#fcfcfb; color:#0b0b0b; max-width:1300px; margin:auto; padding:16px; }}
h1 {{ font-size:22px; margin-bottom:4px; }} h2 {{ font-size:18px; margin-top:32px; }} h3 {{ font-size:15px; margin:16px 0 0; }} p.note {{ color:#52514e; font-size:13px; }}
table.summary {{ border-collapse:collapse; font-size:14px; margin:12px 0 24px; }}
table.summary th, table.summary td {{ padding:6px 14px; border-bottom:1px solid #e6e5e1; text-align:right; }}
table.summary th:first-child, table.summary tbody th {{ text-align:left; color:#52514e; font-weight:500; }}
</style></head><body>
<h1>Battery-side power — flights of {DATE}</h1>
<p class="note">Signal <code>{PWR}</code> = Σ<sub>i=1..4</sub> V<sub>LS,i</sub>·I<sub>LS,i</sub> / 1000 [kW], 10 Hz from InfluxDB {INFLUX_URL}.
Positive = power delivered to the battery (generation), negative = drawn from the battery (consumption).
Energies are trapezoid integrals of the 10 Hz signal.</p>
{table_html(summary_with_total)}
{''.join(parts)}
<h2>Per-cycle breakdown</h2>
<p class="note"><b>Superseded:</b> this simple power-sign split can cut one reel-out into several pieces (e.g. the +8.5 Wh "cycle" at 10:30:00 is part of one reel-out).
Use the drum-speed based cycles in <code>reports/2026-10-22/flight_analysis_report.html</code>.</p>
<p class="note">A cycle starts when the 5 s mean power turns positive for at least 15 s (reel-out) and ends at the next such start
(so it includes the reel-in that follows). The last cycle of each flight may be cut off by the end of the time window (Complete = False).</p>
{cycle_html}
</body></html>"""
    path = os.path.join(OUT_DIR, "battery_power_report.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"Report: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
