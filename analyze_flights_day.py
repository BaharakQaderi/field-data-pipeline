"""
Full engineering analysis of one test day of the TK10 pumping kite, from the local archive made by
export_influx_day.py (no InfluxDB needed). Field meanings and sign conventions: docs/data_dictionary/.

Per flight and per pumping cycle: phase segmentation, mechanical vs battery energy, efficiencies,
tether force/speed, reel-out factor, kite trajectory, pilot inputs, battery balance, data quality.

Conventions (verified on 2026-10-22 data):
  v_t   = -DsEncoder.outTamburo_SpeedMSec          [m/s]  > 0 reel-out
  F     = 9.81 * (DsLoadCells.MeasureFloat_SUM - tare) [N]  tare estimated per flight (see estimate_tare)
  P_mech = F * v_t                                   [W]    > 0 kite -> drum
  P_batt = CONV_MEAS_FB_LS_PWR_SCALED_CALC * 1000    [W]    > 0 charging
  P_bus  = sum_i CONV_MEAS_FB_HS_AVG_PWR_SCALED[i]   [W]    > 0 bus -> batteries
Pumping happens inside PLC state 4, so phases come from the drum speed, not from the state machine.

Usage:
  python analyze_flights_day.py --date 2026-10-22 \
      --flight "Flight 1=10:15:59-10:39:29" --flight "Flight 2=11:59:20-12:20:00" \
      --flight "Flight 3=15:32:26-15:53:00"
"""

import argparse
import html
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TZ = "Europe/Rome"
G = 9.81

V_PHASE = 0.3        # m/s dead-band for reel-out / reel-in
MIN_OUT_S = 3.0      # shortest reel-out run kept
MIN_IN_S = 2.0       # shortest reel-in run kept
MERGE_GAP_S = 2.0    # same-sign runs separated by a shorter pause are merged
MIN_STROKE_M = 20.0  # tether paid out / reeled in that makes a pumping cycle
ETA_GEN_ASSUMED = 0.85   # drum -> battery on reel-out, used only for the power-balance tare estimate
AUX_ASSUMED_KW = 0.35
TETHER_TOTAL_M = 270.0   # drum to pod (team, 2026-10-23)
BRIDLE_M = 37.0          # pod to kite lines (estimate)

PHASE_NAMES = {1: "reel-out", -1: "reel-in", 0: "transition"}
COL = {"reel-out": "#2a78d6", "reel-in": "#eb6834", "transition": "#9a9890",
       "Flight 1": "#2a78d6", "Flight 2": "#1baf7a", "Flight 3": "#6250d6",
       "text": "#52514e", "grid": "#e6e5e1", "ink": "#0b0b0b"}

OPC = {
    "speed": "OPC_DsEncoder.outTamburo_SpeedMSec",
    "wound": "OPC_DsEncoder.outTamburo_WindedLength",
    "lc_sum": "OPC_DsLoadCells.MeasureFloat_SUM",
    "p_batt": "OPC_ConvStruct.CONV_READ.CONV_MEAS_FB_LS_PWR_SCALED_CALC",
    "state": "OPC_DsStateMachine.ACTUAL_STATE",
    "thr_n": "OPC_DsExternalData.Joystick_Percentual_Target_VelocityN",
    "thr_p": "OPC_DsExternalData.Joystick_Percentual_Target_VelocityP",
}
HS_PWR = [f"OPC_ConvStruct.CONV_READ.CONV_MEAS_FB_HS_AVG_PWR_SCALED[{i}]" for i in range(1, 5)]
LS_CURR = [f"OPC_ConvStruct.CONV_READ.CONV_MEAS_FB_LS_AVG_CURR_SCALED[{i}]" for i in range(1, 5)]
LS_VOLT = [f"OPC_ConvStruct.CONV_READ.CONV_MEAS_FB_LS_VOLTAGE_SCALED[{i}]" for i in range(1, 5)]
LC = [f"OPC_DsLoadCells.MeasureFloat[{i}]" for i in range(1, 5)]


# ----------------------------------------------------------------------------------------- loading

def read_source(flight_dir, source, columns=None):
    path = os.path.join(flight_dir, f"{source}.parquet")
    return pd.read_parquet(path, columns=columns) if os.path.exists(path) else pd.DataFrame()


def sparse_series(df, col):
    """Event-style sources store one field per row; keep only rows where this field is present."""
    if df.empty or col not in df:
        return pd.DataFrame(columns=["time", col])
    return df[["time", col]].dropna().sort_values("time")


def attach(base, df, col, name, tolerance):
    s = sparse_series(df, col)
    out = pd.merge_asof(base[["time"]], s, on="time", direction="backward",
                        tolerance=pd.Timedelta(tolerance) if tolerance else None)
    return out[col].to_numpy()


def load_flight(flight_dir):
    opc = read_source(flight_dir, "OPC").sort_values("time").reset_index(drop=True)
    d = pd.DataFrame({"time": opc["time"]})
    d["t_s"] = (d["time"] - d["time"].iloc[0]).dt.total_seconds()
    d["v"] = -opc[OPC["speed"]].astype(float)
    d["wound"] = opc[OPC["wound"]].astype(float)
    d["tether_out"] = TETHER_TOTAL_M - d["wound"]
    d["lc_sum"] = opc[OPC["lc_sum"]].astype(float)
    d["p_batt"] = opc[OPC["p_batt"]].astype(float) * 1000
    d["p_bus"] = opc[HS_PWR].astype(float).sum(axis=1)
    d["state"] = opc[OPC["state"]]
    d["thr_out"] = opc[OPC["thr_n"]].astype(float)
    d["thr_in"] = opc[OPC["thr_p"]].astype(float)
    for i in range(4):
        d[f"i_bat{i + 1}"] = opc[LS_CURR[i]].astype(float)
        d[f"u_bat{i + 1}"] = opc[LS_VOLT[i]].astype(float)
        d[f"lc{i + 1}"] = opc[LC[i]].astype(float)

    wind = read_source(flight_dir, "WIND")
    d["wind"] = attach(d, wind, "WIND_wind.speed", "wind", "2s")
    d["wind_dir"] = attach(d, wind, "WIND_wind.direction", "wind_dir", "2s")

    fs = read_source(flight_dir, "FLIGHT_SEGMENT")
    for col, name in [("FLIGHT_SEGMENT_elevation", "elev"), ("FLIGHT_SEGMENT_azimuth", "az"),
                      ("FLIGHT_SEGMENT_radius", "radius"), ("FLIGHT_SEGMENT_velocity_r", "vel_r"),
                      ("FLIGHT_SEGMENT_roll_angle", "roll_angle"), ("FLIGHT_SEGMENT_ipok_status", "ipok"),
                      ("FLIGHT_SEGMENT_charge", "pod_charge")]:
        d[name] = attach(d, fs, col, name, "0.5s" if name != "pod_charge" else None)
    # IPOK distance must be close to tether + lines (sag makes it a bit shorter, anchor offset a bit longer)
    implausible = (d["radius"] - (d["tether_out"] + BRIDLE_M)).abs() > 40
    bad = implausible | (d["elev"] < 0) | ~pd.Series(d["ipok"]).isin([1, 2]).to_numpy()
    d.loc[bad, ["elev", "az", "radius", "vel_r"]] = np.nan

    tar = read_source(flight_dir, "TARANIS")
    d["cmd_roll"] = attach(d, tar, "TARANIS_Roll", "cmd_roll", None)
    d["cmd_pitch"] = attach(d, tar, "TARANIS_Pitch", "cmd_pitch", None)

    br = read_source(flight_dir, "BRIDGE")
    d["downlink_per"] = attach(d, br, "BRIDGE_main_downlink_per", "per", "2s")
    d["lost_packets"] = attach(d, br, "BRIDGE_lost_packets", "lost", None)
    return d


# ------------------------------------------------------------------------------------ processing

def estimate_tare(d):
    """The load-cell zero drifts during the day (sum reads -5..-33 kg unloaded), so it is estimated per flight.
    central: power balance on reel-out with assumed eta and aux load:  P_batt = eta*g*(S - tare)*v - aux
    floor:   1st percentile of the in-flight 1 s sum (tether tension cannot be negative -> tare <= floor)
    """
    s = d.set_index("time")[["lc_sum", "v", "p_batt"]].resample("1s").mean().dropna()
    ro = s[s["v"] > 0.8]
    central = float(np.median(ro["lc_sum"] - (ro["p_batt"] + AUX_ASSUMED_KW * 1000) / (ETA_GEN_ASSUMED * G * ro["v"])))
    floor = float(s["lc_sum"].quantile(0.01))
    central = min(central, floor)
    return {"central": central, "floor": floor}


def segment_phases(d):
    v = d.set_index("time")["v"].rolling("1s", center=True).mean().to_numpy()
    raw = np.where(v > V_PHASE, 1, np.where(v < -V_PHASE, -1, 0))
    t = d["t_s"].to_numpy()

    def runs(lab):
        edges = np.flatnonzero(np.diff(lab)) + 1
        starts = np.r_[0, edges]
        ends = np.r_[edges, len(lab)]
        return list(zip(starts, ends, lab[starts]))

    lab = raw.copy()
    for a, b, s in runs(lab):          # drop short reel-out/in blips
        dur = t[b - 1] - t[a]
        if (s == 1 and dur < MIN_OUT_S) or (s == -1 and dur < MIN_IN_S):
            lab[a:b] = 0
    rr = runs(lab)
    for k in range(1, len(rr) - 1):    # merge same-sign runs split by a short pause
        a, b, s = rr[k]
        if s == 0 and rr[k - 1][2] == rr[k + 1][2] != 0 and t[b - 1] - t[a] < MERGE_GAP_S:
            lab[a:b] = rr[k - 1][2]
    d["phase_id"] = lab
    d["phase"] = pd.Series(lab).map(PHASE_NAMES).to_numpy()

    # A pumping cycle starts at a reel-out that pays out >= MIN_STROKE_M after >= MIN_STROKE_M of reel-in since
    # the previous cycle start; smaller step-outs (launch) and pauses stay inside the current cycle.
    wound = d["wound"].to_numpy()
    starts, reeled_in, out_start, out_len = [], np.inf, None, 0.0
    for a, b, s in runs(lab):
        if s == 1:
            if out_start is None:
                out_start, out_len = a, 0.0
            out_len += wound[a] - wound[b - 1]
            if out_len >= MIN_STROKE_M and reeled_in >= MIN_STROKE_M:
                starts.append(out_start)
                reeled_in = 0.0
                out_start = None
        elif s == -1:
            reeled_in += wound[b - 1] - wound[a]
            if wound[b - 1] - wound[a] > 5:
                out_start = None
    cycle = np.zeros(len(d), dtype=int)
    for k, a in enumerate(starts):
        cycle[a:] = k + 1
    d["cycle"] = cycle                 # 0 = launch segment before the first reel-out
    return len(starts)


def integrate(t, p, mask=None):
    """Trapezoid on the real timestamps, Wh. With a mask only intervals whose start sample is in the mask count."""
    dt = np.diff(t)
    mid = (p[1:] + p[:-1]) / 2
    if mask is not None:
        mid = np.where(mask[:-1], mid, 0.0)
    return float(np.nansum(mid * dt) / 3600)


def cycle_kpis(d, flight, tare, complete_last):
    rows = []
    n_cycles = d["cycle"].max()
    for c in range(1, n_cycles + 1):
        seg = d[d["cycle"] == c]
        if len(seg) < 20:
            continue
        t = seg["t_s"].to_numpy()
        out = (seg["phase_id"] == 1).to_numpy()
        inn = (seg["phase_id"] == -1).to_numpy()
        # cycle 1 starts at launch (no preceding reel-in) and the last one is cut by the flight window,
        # so only the cycles in between are regular pumping cycles
        kind = "first (from launch)" if c == 1 else ("cut at window end" if c == n_cycles and not complete_last
                                                       else "regular")
        row = {"Flight": flight, "Cycle": c, "Start": seg["time"].iloc[0].tz_convert(TZ).strftime("%H:%M:%S"),
               "Type": kind, "Complete": kind == "regular"}
        row["T cycle (s)"] = t[-1] - t[0]
        row["T out (s)"] = float(np.sum(np.diff(t)[out[:-1]]))
        row["T in (s)"] = float(np.sum(np.diff(t)[inn[:-1]]))
        pb = seg["p_batt"].to_numpy()
        row["E batt out (Wh)"] = integrate(t, pb, out)
        row["E batt in (Wh)"] = integrate(t, pb, inn)
        row["E batt net (Wh)"] = integrate(t, pb)
        trans = ~(out | inn)
        row["T trans (s)"] = float(np.sum(np.diff(t)[trans[:-1]]))
        row["E batt trans (Wh)"] = integrate(t, pb, trans)
        row["Net mean power (W)"] = row["E batt net (Wh)"] * 3600 / row["T cycle (s)"]
        for key, tr in [("", tare["central"]), (" [floor tare]", tare["floor"])]:
            pm = G * (seg["lc_sum"].to_numpy() - tr) * seg["v"].to_numpy()
            row[f"E mech out (Wh){key}"] = integrate(t, pm, out)
            row[f"E mech in (Wh){key}"] = integrate(t, pm, inn)
        row["E mech net (Wh)"] = integrate(t, G * (seg["lc_sum"].to_numpy() - tare["central"]) * seg["v"].to_numpy())
        f_kg = seg["lc_sum"] - tare["central"]
        row["F out mean (kg)"] = f_kg[out].mean()
        row["F out max (kg)"] = f_kg[out].max()
        row["F in mean (kg)"] = f_kg[inn].mean()
        row["F ratio out/in"] = row["F out mean (kg)"] / row["F in mean (kg)"] if inn.any() else np.nan
        row["v out mean (m/s)"] = seg["v"][out].mean()
        row["v in mean (m/s)"] = -seg["v"][inn].mean()
        row["Wind mean (m/s)"] = seg["wind"].mean()
        row["Reel-out factor"] = row["v out mean (m/s)"] / seg["wind"][out].mean()
        row["Stroke (m)"] = seg["wound"].max() - seg["wound"].min()
        row["Tether out max (m)"] = seg["tether_out"].max()
        row["Elevation out (deg)"] = seg["elev"][out].mean()
        row["Gen chain eff out"] = row["E batt out (Wh)"] / row["E mech out (Wh)"] if row["E mech out (Wh)"] > 0 else np.nan
        row["Reel-in cost ratio"] = (row["E batt in (Wh)"] / row["E mech in (Wh)"]
                                     if row["E mech in (Wh)"] < 0 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def flight_kpis(d, flight, tare, cycles):
    t = d["t_s"].to_numpy()
    pb = d["p_batt"].to_numpy()
    out = (d["phase_id"] == 1).to_numpy()
    inn = (d["phase_id"] == -1).to_numpy()
    pm = G * (d["lc_sum"].to_numpy() - tare["central"]) * d["v"].to_numpy()
    static = (d.set_index("time")["v"].abs().rolling("5s").max() < 0.05).to_numpy()
    bus = d["p_bus"].to_numpy()
    e_bus_out, e_batt_out = integrate(t, bus, out), integrate(t, pb, out)
    e_bus_in, e_batt_in = integrate(t, bus, inn), integrate(t, pb, inn)
    full = cycles[cycles["Complete"]] if len(cycles) else cycles
    return {
        "Flight": flight,
        "Start": d["time"].iloc[0].tz_convert(TZ).strftime("%H:%M:%S"),
        "Duration (min)": t[-1] / 60,
        "Pumping cycles (complete)": f"{len(cycles)} ({len(full)})",
        "Wind mean / p90 (m/s)": f"{np.nanmean(d['wind']):.1f} / {np.nanpercentile(d['wind'], 90):.1f}",
        "Wind direction (deg, from)": f"{np.nanmedian(d['wind_dir']):.0f}",
        "Time reel-out / reel-in / transition (%)": " / ".join(
            f"{100 * np.mean(d['phase_id'] == k):.0f}" for k in (1, -1, 0)),
        "E batt generated (Wh)": integrate(t, np.clip(pb, 0, None)),
        "E batt consumed (Wh)": -integrate(t, np.clip(pb, None, 0)),
        "E batt net (Wh)": integrate(t, pb),
        "Net mean power (W)": integrate(t, pb) * 3600 / t[-1],
        "E batt reel-out phases (Wh)": e_batt_out,
        "E batt reel-in phases (Wh)": e_batt_in,
        "E mech reel-out (Wh)": integrate(t, pm, out),
        "E mech reel-in (Wh)": integrate(t, pm, inn),
        "Best complete cycle net (Wh)": full["E batt net (Wh)"].max() if len(full) else np.nan,
        "Net mean power complete cycles (W)": (full["E batt net (Wh)"].sum() * 3600 / full["T cycle (s)"].sum()
                                            if len(full) else np.nan),
        "Gen chain eff reel-out": e_batt_out / integrate(t, pm, out),
        "Reel-in cost ratio": e_batt_in / integrate(t, pm, inn),
        "Converter eff charging": e_batt_out / e_bus_out if e_bus_out > 0 else np.nan,
        "Converter eff discharging": e_bus_in / e_batt_in if e_batt_in < 0 else np.nan,
        "Draw with drum stopped in flight (kW)": -np.median(pb[static]) / 1000 if static.any() else np.nan,
        "Tether force max (kg)": (d["lc_sum"] - tare["central"]).max(),
        "Tare central / floor (kg)": f"{tare['central']:.1f} / {tare['floor']:.1f}",
        "Tether out min-max (m)": f"{d['tether_out'].min():.0f}-{d['tether_out'].max():.0f}",
        "Kite range check: radius vs tether+bridle (m)": np.nanmedian(d["radius"] - (d["tether_out"] + BRIDLE_M)),
        "Kite position valid (%)": 100 * np.mean(np.isfinite(d["elev"])),
        "Downlink PER mean (%)": np.nanmean(d["downlink_per"]),
        "Lost packets in flight": (np.nanmax(d["lost_packets"]) - np.nanmin(d["lost_packets"])),
    }


# ------------------------------------------------------------------------------------- figures

def base_layout(fig, title, height):
    fig.update_layout(title=dict(text=title, font=dict(size=15)), template="plotly_white", height=height,
                      font=dict(color=COL["text"]), margin=dict(l=70, r=30, t=70, b=50),
                      legend=dict(orientation="h", y=-0.08))
    fig.update_xaxes(gridcolor=COL["grid"])
    fig.update_yaxes(gridcolor=COL["grid"], zeroline=False)
    return fig


def decimate(d, hz=2):
    step = max(1, int(round(10 / hz)))
    return d.iloc[::step]


def phase_shapes(fig, d, rows):
    t_local = d["time"].dt.tz_convert(TZ).dt.tz_localize(None)
    ids = d["phase_id"].to_numpy()
    edges = np.flatnonzero(np.diff(ids)) + 1
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(ids)]):
        if ids[a] == 0:
            continue
        for r in rows:
            fig.add_vrect(x0=t_local.iloc[a], x1=t_local.iloc[b - 1], row=r, col=1, line_width=0, layer="below",
                          fillcolor=COL["reel-out"] if ids[a] == 1 else COL["reel-in"], opacity=0.08)


def fig_flight_overview(d, flight, kpi, tare):
    dd = decimate(d, 5)
    t = dd["time"].dt.tz_convert(TZ).dt.tz_localize(None)
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        row_heights=[0.17, 0.17, 0.24, 0.27, 0.15],
                        subplot_titles=(f"Tether out (m) = {TETHER_TOTAL_M:.0f} − wound length; kite ≈ +{BRIDLE_M:.0f} m lines; dotted = IPOK radius",
                                        "Tether speed v_t (m/s) — positive = reel-out",
                                        f"Tether force (kg) — tare {tare['central']:.1f} kg, band = floor tare {tare['floor']:.1f} kg",
                                        "Power (kW) — battery (> 0 charging) and mechanical at drum (> 0 kite → drum)",
                                        "Ground wind speed (m/s)"))
    fig.add_trace(go.Scatter(x=t, y=dd["tether_out"], line=dict(color=COL["ink"], width=1.5), name="Tether out",
                             hovertemplate="%{y:.1f} m tether out<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=dd["radius"], line=dict(color=COL["ink"], width=1, dash="dot"),
                             name="Kite distance (IPOK)", hovertemplate="%{y:.0f} m radius<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=dd["v"], line=dict(color=COL["ink"], width=1), name="v_t",
                             hovertemplate="%{y:.2f} m/s<extra></extra>"), 2, 1)
    f_c = dd["lc_sum"] - tare["central"]
    f_f = dd["lc_sum"] - tare["floor"]
    fig.add_trace(go.Scatter(x=t, y=f_c, line=dict(width=0), showlegend=False, hoverinfo="skip"), 3, 1)
    fig.add_trace(go.Scatter(x=t, y=f_f, line=dict(width=0), fill="tonexty", fillcolor="rgba(82,81,78,0.18)",
                             name="Tare uncertainty", hoverinfo="skip"), 3, 1)
    fig.add_trace(go.Scatter(x=t, y=f_c, line=dict(color=COL["ink"], width=1), name="Force (central tare)",
                             hovertemplate="%{y:.0f} kg<extra></extra>"), 3, 1)
    pm = G * (dd["lc_sum"] - tare["central"]) * dd["v"] / 1000
    fig.add_trace(go.Scatter(x=t, y=pm.rolling(5, center=True).mean(), line=dict(color="#9a9890", width=1.2),
                             name="P mech (1 s mean)", hovertemplate="%{y:.2f} kW mech<extra></extra>"), 4, 1)
    fig.add_trace(go.Scatter(x=t, y=(dd["p_batt"] / 1000).rolling(5, center=True).mean(),
                             line=dict(color=COL["reel-out"], width=1.5), name="P battery (1 s mean)",
                             hovertemplate="%{y:.2f} kW batt<extra></extra>"), 4, 1)
    fig.add_trace(go.Scatter(x=t, y=dd["wind"], line=dict(color=COL["ink"], width=1), name="Wind",
                             hovertemplate="%{y:.1f} m/s<extra></extra>"), 5, 1)
    for r in (2, 4):
        fig.add_hline(y=0, line=dict(color=COL["text"], width=1), row=r, col=1)
    phase_shapes(fig, d, rows=[1, 2, 3, 4, 5])
    title = (f"<b>{flight}</b> {kpi['Start']} · {kpi['Duration (min)']:.1f} min · {kpi['Pumping cycles (complete)']} cycles · "
             f"battery net {kpi['E batt net (Wh)']:+.0f} Wh · wind {kpi['Wind mean / p90 (m/s)']} m/s "
             f"<span style='font-size:12px'>(blue band = reel-out, orange = reel-in)</span>")
    fig = base_layout(fig, title, 980)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(tickformat="%H:%M")
    return fig


def fig_cycle_energy(cycles):
    labels = [f"F{r.Flight[-1]}·C{r.Cycle}{'' if r.Complete else '*'}" for r in cycles.itertuples()]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=cycles["E batt out (Wh)"], name="Battery, reel-out phase",
                         marker_color=COL["reel-out"], hovertemplate="%{y:.1f} Wh<extra>reel-out</extra>"))
    fig.add_trace(go.Bar(x=labels, y=cycles["E batt in (Wh)"], name="Battery, reel-in phase",
                         marker_color=COL["reel-in"], hovertemplate="%{y:.1f} Wh<extra>reel-in</extra>"))
    other = cycles["E batt net (Wh)"] - cycles["E batt out (Wh)"] - cycles["E batt in (Wh)"]
    fig.add_trace(go.Bar(x=labels, y=other, name="Battery, transitions/stops", marker_color=COL["transition"],
                         hovertemplate="%{y:.1f} Wh<extra>transition</extra>"))
    fig.add_trace(go.Scatter(x=labels, y=cycles["E batt net (Wh)"], mode="markers", name="Net per cycle",
                             marker=dict(color=COL["ink"], size=10, symbol="diamond"),
                             hovertemplate="net %{y:+.1f} Wh<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=COL["text"], width=1))
    fig = base_layout(fig, "<b>Battery energy per pumping cycle</b> (* = first cycle from launch, or cut by end of flight window)", 430)
    fig.update_layout(barmode="relative", bargap=0.35, yaxis_title="Wh")
    return fig


def fig_scatter_grid(cycles):
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                        subplot_titles=("Mean cycle battery power vs wind", "Reel-out force vs reel-out speed",
                                        "Reel-out factor v_out / v_wind vs wind"))
    for flight, g in cycles[cycles["Complete"]].groupby("Flight"):
        c = COL[flight]
        hov = [f"{flight} C{r.Cycle}" for r in g.itertuples()]
        fig.add_trace(go.Scatter(x=g["Wind mean (m/s)"], y=g["Net mean power (W)"], mode="markers", name=flight,
                                 marker=dict(color=c, size=10, line=dict(color="white", width=2)), text=hov,
                                 hovertemplate="%{text}<br>wind %{x:.1f} m/s<br>%{y:+.0f} W<extra></extra>"), 1, 1)
        fig.add_trace(go.Scatter(x=g["v out mean (m/s)"], y=g["F out mean (kg)"], mode="markers", showlegend=False,
                                 marker=dict(color=c, size=10, line=dict(color="white", width=2)), text=hov,
                                 hovertemplate="%{text}<br>%{x:.2f} m/s<br>%{y:.0f} kg<extra></extra>"), 1, 2)
        fig.add_trace(go.Scatter(x=g["Wind mean (m/s)"], y=g["Reel-out factor"], mode="markers", showlegend=False,
                                 marker=dict(color=c, size=10, line=dict(color="white", width=2)), text=hov,
                                 hovertemplate="%{text}<br>wind %{x:.1f}<br>f = %{y:.2f}<extra></extra>"), 1, 3)
    fig.add_hline(y=0, line=dict(color=COL["text"], width=1), row=1, col=1)
    fig.add_hline(y=1 / 3, line=dict(color=COL["text"], width=1, dash="dot"), row=1, col=3,
                  annotation_text="Loyd optimum 1/3", annotation_position="top left")
    fig = base_layout(fig, "<b>Cycle operating points</b> (complete cycles)", 450)
    fig.update_layout(legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    fig.update_xaxes(title_text="wind (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="W", row=1, col=1)
    fig.update_xaxes(title_text="v_out (m/s)", row=1, col=2)
    fig.update_yaxes(title_text="kg", row=1, col=2)
    fig.update_xaxes(title_text="wind (m/s)", row=1, col=3)
    return fig


def fig_drivers(cycles):
    full = cycles[cycles["Complete"]]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("Net battery energy per cycle vs force ratio F_out / F_in",
                                        "Reel-in: battery power vs mechanical power (kW)"))
    for flight, g in full.groupby("Flight"):
        c = COL[flight]
        hov = [f"{flight} C{r.Cycle}" for r in g.itertuples()]
        fig.add_trace(go.Scatter(x=g["F ratio out/in"], y=g["E batt net (Wh)"], mode="markers", name=flight,
                                 marker=dict(color=c, size=11, line=dict(color="white", width=2)), text=hov,
                                 hovertemplate="%{text}<br>ratio %{x:.2f}<br>net %{y:+.1f} Wh<extra></extra>"), 1, 1)
        p_mech = g["E mech in (Wh)"] * 3.6 / g["T in (s)"]
        p_batt = g["E batt in (Wh)"] * 3.6 / g["T in (s)"]
        fig.add_trace(go.Scatter(x=-p_mech, y=-p_batt, mode="markers", showlegend=False,
                                 marker=dict(color=c, size=11, line=dict(color="white", width=2)), text=hov,
                                 hovertemplate="%{text}<br>mech %{x:.2f} kW<br>battery %{y:.2f} kW<extra></extra>"), 1, 2)
    fig.add_hline(y=0, line=dict(color=COL["text"], width=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1.2], y=[0, 1.2], mode="lines", line=dict(color=COL["text"], dash="dot", width=1),
                             name="battery = mechanical", hoverinfo="skip"), 1, 2)
    fig = base_layout(fig, "<b>What drives the cycle result</b> (complete cycles)", 430)
    fig.update_xaxes(title_text="F_out / F_in", row=1, col=1)
    fig.update_yaxes(title_text="Wh", row=1, col=1)
    fig.update_xaxes(title_text="mechanical power to reel in (kW)", row=1, col=2)
    fig.update_yaxes(title_text="battery power drawn (kW)", row=1, col=2)
    fig.update_layout(legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    return fig


def fig_trajectory(flights):
    fig = make_subplots(rows=1, cols=len(flights), shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=list(flights))
    for k, (flight, d) in enumerate(flights.items()):
        dd = decimate(d.dropna(subset=["elev", "az"]), 2)
        downwind = (np.nanmedian(d["wind_dir"]) + 180) % 360
        for ph in ("reel-out", "reel-in", "transition"):
            g = dd[dd["phase"] == ph]
            fig.add_trace(go.Scatter(x=g["az"], y=g["elev"], mode="markers", name=ph, showlegend=k == 0,
                                       legendgroup=ph, marker=dict(color=COL[ph], size=4, opacity=0.6),
                                       hovertemplate="az %{x:.0f}°, el %{y:.0f}°<extra>" + ph + "</extra>"),
                          1, k + 1)
        fig.add_vline(x=downwind, line=dict(color=COL["text"], width=1, dash="dot"), row=1, col=k + 1,
                      annotation_text=f"downwind {downwind:.0f}°", annotation_position="bottom right")
        fig.update_xaxes(title_text="azimuth (° from N)", row=1, col=k + 1)
    fig.update_yaxes(title_text="elevation (°)", row=1, col=1)
    fig = base_layout(fig, "<b>Kite position seen from the ground station</b> (IPOK, 2 Hz, invalid/stale fixes removed)", 460)
    fig.update_layout(legend=dict(orientation="h", y=1.14, x=1, xanchor="right"))
    return fig


def fig_pilot_and_force(flights, tares):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("Depower command (Taranis pitch, %) by phase",
                                        "Tether force (kg) by phase"))
    for ph in ("reel-out", "reel-in"):
        xs, ps, fs = [], [], []
        for flight, d in flights.items():
            g = decimate(d[d["phase"] == ph], 2)
            xs += [flight] * len(g)
            ps += g["cmd_pitch"].tolist()
            fs += (g["lc_sum"] - tares[flight]["central"]).tolist()
        fig.add_trace(go.Box(x=xs, y=ps, name=ph, marker_color=COL[ph], boxpoints=False, legendgroup=ph), 1, 1)
        fig.add_trace(go.Box(x=xs, y=fs, name=ph, marker_color=COL[ph], boxpoints=False, legendgroup=ph,
                             showlegend=False), 1, 2)
    fig = base_layout(fig, "<b>Depower effectiveness</b>: pilot depower input and resulting tether force", 420)
    fig.update_layout(boxmode="group")
    return fig


def fig_battery(flights):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("Mean battery current by phase (A, > 0 charging)",
                                        "Battery voltage range in flight (V)"))
    for i in range(1, 5):
        x, y = [], []
        for flight, d in flights.items():
            for ph in ("reel-out", "reel-in"):
                x.append(f"{flight[-1]}·{ph}")
                y.append(d.loc[d["phase"] == ph, f"i_bat{i}"].mean())
        fig.add_trace(go.Bar(x=x, y=y, name=f"BAT{i}", marker_color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"][i - 1],
                             hovertemplate="%{y:.1f} A<extra>BAT" + str(i) + "</extra>"), 1, 1)
        lo = [flights[f][f"u_bat{i}"].min() for f in flights]
        hi = [flights[f][f"u_bat{i}"].max() for f in flights]
        fig.add_trace(go.Bar(x=list(flights), y=np.subtract(hi, lo), base=lo, showlegend=False,
                             marker_color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"][i - 1],
                             hovertemplate="%{base:.1f}–%{y:.1f} V span<extra>BAT" + str(i) + "</extra>"), 1, 2)
    fig.add_hline(y=0, line=dict(color=COL["text"], width=1), row=1, col=1)
    fig = base_layout(fig, "<b>Battery balance</b> across the 4 converter/battery strings", 420)
    fig.update_layout(barmode="group", bargap=0.25)
    return fig


# ---------------------------------------------------------------------------------------- report

def phase_power_table(cycles):
    """Per cycle: battery energy and average power of generation (reel-out), recovery (reel-in) and transitions."""
    def w(e, t):
        return e * 3600 / t if t > 0 else np.nan
    rows = []
    for _, r in cycles.iterrows():
        rows.append({
            "Flight": r["Flight"], "Cycle": int(r["Cycle"]), "Start": r["Start"], "Type": r["Type"],
            "Gen time (s)": r["T out (s)"], "Gen energy (Wh)": r["E batt out (Wh)"],
            "Gen mean power (W)": w(r["E batt out (Wh)"], r["T out (s)"]),
            "Rec time (s)": r["T in (s)"], "Rec energy (Wh)": r["E batt in (Wh)"],
            "Rec mean power (W)": w(r["E batt in (Wh)"], r["T in (s)"]),
            "Trans time (s)": r["T trans (s)"], "Trans energy (Wh)": r["E batt trans (Wh)"],
            "Trans mean power (W)": w(r["E batt trans (Wh)"], r["T trans (s)"]),
            "Cycle time (s)": r["T cycle (s)"], "Cycle net energy (Wh)": r["E batt net (Wh)"],
            "Cycle net mean power (W)": r["Net mean power (W)"],
        })
    return pd.DataFrame(rows)


def table_html(df, transpose=False, digits=2):
    fmt = df.copy()
    for c in fmt.columns:
        if fmt[c].dtype.kind == "f":
            fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"{v:,.{digits}f}")
    if transpose:
        return fmt.set_index(fmt.columns[0]).T.to_html(classes="tbl", border=0, escape=True)
    return fmt.to_html(classes="tbl", border=0, index=False, escape=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--flight", action="append", required=True, help='"Flight 1=HH:MM:SS-HH:MM:SS"')
    ap.add_argument("--archive", default="data_archive")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--findings", help="optional HTML/markdown-free text file with the findings section")
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.date)
    os.makedirs(out_dir, exist_ok=True)
    flights, tares, kpis, all_cycles = {}, {}, [], []
    for spec in args.flight:
        name = spec.split("=", 1)[0].strip()
        fdir = os.path.join(args.archive, args.date, "flights", name.replace(" ", "_").lower())
        d = load_flight(fdir)
        tare = estimate_tare(d)
        segment_phases(d)
        cycles = cycle_kpis(d, name, tare, complete_last=False)
        flights[name], tares[name] = d, tare
        all_cycles.append(cycles)
        kpis.append(flight_kpis(d, name, tare, cycles))
        print(f"{name}: tare central {tare['central']:.1f} kg, floor {tare['floor']:.1f} kg, {len(cycles)} cycles")

    cycles = pd.concat(all_cycles, ignore_index=True)
    summary = pd.DataFrame(kpis)
    cycles.to_csv(os.path.join(out_dir, "cycles.csv"), index=False)
    phase_tbl = phase_power_table(cycles)
    phase_tbl.to_csv(os.path.join(out_dir, "cycles_power_by_phase.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "flights_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summary.set_index("Flight").T.to_string())
    print(cycles.round(2).to_string(index=False))

    figs = [fig_flight_overview(flights[n], n, k, tares[n]) for n, k in zip(flights, kpis)]
    figs += [fig_cycle_energy(cycles), fig_drivers(cycles), fig_scatter_grid(cycles), fig_pilot_and_force(flights, tares),
             fig_trajectory(flights), fig_battery(flights)]
    parts = [figs[0].to_html(full_html=False, include_plotlyjs="inline")]
    parts += [f.to_html(full_html=False, include_plotlyjs=False) for f in figs[1:]]

    findings = ""
    if args.findings and os.path.exists(args.findings):
        with open(args.findings) as fh:
            findings = fh.read()

    cycle_cols = ["Flight", "Cycle", "Start", "Type", "T cycle (s)", "T out (s)", "T in (s)",
                  "E batt out (Wh)", "E batt in (Wh)", "E batt net (Wh)", "Net mean power (W)",
                  "E mech out (Wh)", "E mech in (Wh)", "E mech net (Wh)", "Gen chain eff out", "Reel-in cost ratio",
                  "F ratio out/in", "F out mean (kg)", "F out max (kg)", "F in mean (kg)", "v out mean (m/s)", "v in mean (m/s)",
                  "Wind mean (m/s)", "Reel-out factor", "Stroke (m)", "Tether out max (m)", "Elevation out (deg)"]
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TK10 Flight Analysis {html.escape(args.date)}</title>
<style>
body {{ font-family:-apple-system,Segoe UI,sans-serif; background:#fcfcfb; color:#0b0b0b; max-width:1400px; margin:auto; padding:16px; line-height:1.45; }}
h1 {{ font-size:24px; margin:8px 0 4px; }} h2 {{ font-size:19px; margin:36px 0 8px; border-bottom:1px solid #e6e5e1; padding-bottom:4px; }}
p.note, li.note {{ color:#52514e; font-size:13px; }}
.tbl {{ border-collapse:collapse; font-size:13px; margin:10px 0 20px; }}
.tbl th, .tbl td {{ padding:5px 10px; border-bottom:1px solid #e6e5e1; text-align:right; white-space:nowrap; }}
.tbl tbody th, .tbl th:first-child {{ text-align:left; color:#52514e; font-weight:500; }}
.scroll {{ overflow-x:auto; }}
.findings li {{ margin-bottom:6px; }}
code {{ font-size:12px; background:#f0efec; padding:1px 4px; border-radius:3px; }}
</style></head><body>
<h1>TK10 field test {html.escape(args.date)} — flight analysis</h1>
<p class="note">Internal engineering report. Source: verified archive from InfluxDB mirror <code>gs_mirror</code>
(export_influx_day.py). Conventions and field meanings: <code>docs/data_dictionary/</code>. All times local (CEST).</p>
{findings}
<h2>Flight summary</h2>
<div class="scroll">{table_html(summary, transpose=True)}</div>
<h2>Per-flight time series</h2>
{''.join(parts[:len(flights)])}
<h2>Pumping cycles</h2>
<p class="note">Phases from drum speed (1 s mean, dead-band ±{V_PHASE} m/s, reel-out ≥ {MIN_OUT_S:.0f} s, reel-in ≥ {MIN_IN_S:.0f} s).
A new cycle starts at a reel-out that pays out ≥ {MIN_STROKE_M:.0f} m after ≥ {MIN_STROKE_M:.0f} m of reel-in since the previous start, and runs to the next start
(one reel-out + one reel-in). Every start matches a maximum of the wound-length saw-tooth within 1–5 s.
Cycle 1 of each flight starts at launch and the last one is cut by the flight window; both are excluded from the "regular cycle" statistics.
The PLC stays in GENERATION throughout, so it cannot mark the phases.</p>
<h3>Battery energy and average power per phase (W)</h3>
<p class="note">Generation = reel-out (tether out, battery charging), recovery = reel-in (motor pulls tether in, battery discharging),
transition = slow/stopped drum between them. Mean power = phase energy / phase time; generation + recovery + transition energy = cycle net energy.
Positive = into the battery.</p>
<div class="scroll">{table_html(phase_tbl, digits=1)}</div>
{''.join(parts[len(flights):len(flights) + 3])}
<div class="scroll">{table_html(cycles[cycle_cols])}</div>
<h2>Depower, trajectory and batteries</h2>
{''.join(parts[len(flights) + 3:])}
<h2>Method and caveats</h2>
<ul>
<li class="note"><b>Load-cell tare.</b> The unloaded sum drifts from about +6 kg to −33 kg over the day (mainly LC3), so a fixed zero is wrong.
Central tare per flight: power balance on reel-out, P_batt = {ETA_GEN_ASSUMED}·g·(S − tare)·v − {AUX_ASSUMED_KW} kW, solved for the tare (median over 1 s samples with v_t &gt; 0.8 m/s).
The floor tare (1st percentile of the in-flight sum, since tension cannot be negative) is the other end of the band.
Over a complete cycle the tare error cancels in the net mechanical energy; it only shifts energy between the reel-out and reel-in shares.
Because the central tare assumes η = {ETA_GEN_ASSUMED}, the reel-out generation-chain efficiency is not an independent measurement — the reel-in cost ratio and all battery-side numbers are.</li>
<li class="note"><b>Tether force</b> = 9.81 × (sum of the 4 load cells under the drum − tare). It is the drum reaction, taken as tether tension (no exit-angle correction).</li>
<li class="note"><b>Not used:</b> <code>DsInverters.Power</code> (unsigned raw V×I), <code>CONV_POWER_SCALED_HS</code> (unsigned), <code>Torque_ActualValue[2]</code> (scale unconfirmed, ≈4× too small for drum Nm).</li>
<li class="note"><b>Wind</b> is the ground station anemometer (height unknown), so the reel-out factor uses ground wind, not wind at kite altitude.</li>
<li class="note"><b>Kite position</b> comes from IPOK (RTK GPS + IMU) relative to the surveyed GS anchor; samples with invalid radius, negative elevation or stale/no RTK are removed. Azimuth is degrees from North (clockwise); the dotted line is the downwind direction from the ground anemometer.</li>
<li class="note"><b>Tether geometry:</b> {TETHER_TOTAL_M:.0f} m tether to the pod plus ≈{BRIDLE_M:.0f} m of lines to the kite; tether out = {TETHER_TOTAL_M:.0f} m − wound length (≈50 m stays on the drum at full reel-out). The IPOK straight-line distance should be slightly shorter than tether + lines because of sag; the median difference is listed in the summary as a consistency check.</li>
<li class="note"><b>Timing:</b> all timestamps are ground-backend receive times; OPC is a 10 Hz snapshot of a 20 Hz PLC, so short force/power peaks are under-resolved.</li>
</ul>
</body></html>"""
    path = os.path.join(out_dir, "flight_analysis_report.html")
    with open(path, "w") as fh:
        fh.write(page)
    print(f"Report: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
