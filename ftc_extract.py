"""
Per-flight extraction for flight-test campaign KTE_2030_FTC_0001.

For each event it writes to <out>/<stem>/:
  <stem>_native.parquet   every source at its own rate, long-wide (column `source`, time_gs index)
  <stem>_full_source.csv  20 Hz, FLIGHT_SEGMENT + selected OPC + IPOK + TARANIS + THROTTLE + WIND, original column names
  <stem>_analysis.csv     same 20 Hz grid, all sources incl. commands and links, short names (input of ftc_cycles.py)
  <stem>_meta.json        windows, measured clock offsets, pod identity, row counts, coverage, config snapshot
  figures/<stem>_alignment.png   proof of the clock alignment

Sources: Influx mirror (InfluxQL, no Flux pivot), Fra's per-flight IPOK files, GS sent-packet log (20 Hz commands).
Field lists, windows and staleness limits: config/fields_KTE_2030_FTC_0001.yaml.

Clock alignment (reference = GS/Influx clock):
  IPOK GNSS utcTime -> GS: cross-correlation of IPOK Pitch/Roll with the pod-relayed angles (accept corr >= 0.9,
              or a consistent lag over 5-min sub-windows); no IPOK -> offset of the nearest same-day event
  GS sent log -> GS: cross-correlation with GS_UPLINK_COMMAND pitch/roll (same machine, expect ~0)
  OPC vs pod: reel-out speed vs relayed range rate; reported only, NOT applied. A lag here can be downlink latency
              of the relayed data or physics (range rate leading the drum), so it needs a team decision.

Usage:
    INFLUX_TOKEN=... python ftc_extract.py --event KTE_2030_FTC_0001_20261024_S4E1
    INFLUX_TOKEN=... python ftc_extract.py --all
"""

import argparse
import glob
import json
import os
import re
import ssl
import subprocess
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

TZ = "Europe/Rome"
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "fields_KTE_2030_FTC_0001.yaml")
IPOK_MIN_CORR = 0.9
SUB_MIN_CORR = 0.6       # sub-window fallback: >= 3 windows above this, all within SUB_MAX_SPREAD_S of the lag
SUB_MAX_SPREAD_S = 0.25
# order of the source blocks in the unified CSV
CSV_ORDER = ["OPC", "FLIGHT_SEGMENT", "IPOK", "CMD", "TARANIS", "BRIDGE", "WIND", "THROTTLE", "FS_MAINTENANCE"]


class Influx:
    def __init__(self, cfg, token):
        ic = cfg["influx"]
        self.url, self.bucket, self.token = ic["url"].rstrip("/"), ic["bucket"], token
        self.ctx = ssl.create_default_context(cafile=os.path.expanduser(ic["cafile"]))
        self.ctx.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN

    def query(self, q):
        body = urllib.parse.urlencode({"db": self.bucket, "epoch": "ns", "q": q}).encode()
        req = urllib.request.Request(f"{self.url}/query", data=body, headers={
            "Authorization": f"Token {self.token}", "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"})
        with urllib.request.urlopen(req, context=self.ctx, timeout=600) as resp:
            result = json.loads(resp.read())["results"][0]
        if "error" in result:
            raise RuntimeError(result["error"])
        series = result.get("series", [])
        if not series:
            return pd.DataFrame()
        return pd.DataFrame(series[0]["values"], columns=series[0]["columns"])

    def fetch(self, source, fields, start, stop):
        sel = ", ".join(f'"{f}" AS "{name}"' for name, f in fields.items())
        df = self.query(f"SELECT {sel} FROM tk WHERE source='{source}' "
                        f"AND time >= '{iso(start)}' AND time < '{iso(stop)}'")
        if df.empty:
            return empty_frame(fields)
        df.index = pd.to_datetime(df.pop("time"), unit="ns", utc=True).rename("time_gs")
        df = df.map(lambda v: float(v) if isinstance(v, bool) else v)
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        return clean_index(df)

    def last_values(self, source, fields, start, stop):
        out = {}
        for f in fields:
            df = self.query(f'SELECT LAST("{f}") FROM tk WHERE source=\'{source}\' '
                            f"AND time >= '{iso(start)}' AND time < '{iso(stop)}'")
            out[f] = None if df.empty else df.iloc[0, 1]
        return out


def iso(t):
    return pd.Timestamp(t).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def empty_frame(columns):
    return pd.DataFrame(columns=list(columns), dtype=float, index=pd.DatetimeIndex([], tz="UTC", name="time_gs"))


def clean_index(df):
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def read_sent_log(cfg, ev, w0, w1):
    spec = cfg["logs"]["gs_sent"]
    root = os.path.join(os.path.expanduser(cfg["raw_root"]), ev["day_dir"])
    parts = []
    for f in sorted(glob.glob(os.path.join(root, spec["glob"]), recursive=True)):
        if os.path.getsize(f) == 0:
            continue
        d = pd.read_csv(f, usecols=lambda c: c in spec["columns"] + [spec["time_col"]], low_memory=False)
        t = pd.to_numeric(d.pop(spec["time_col"]), errors="coerce")
        d.index = pd.to_datetime(t, unit="s", utc=True).rename("time_gs")
        d = d[(d.index >= w0) & (d.index < w1)]
        if len(d):
            parts.append(d.apply(pd.to_numeric, errors="coerce"))
    return clean_index(pd.concat(parts)) if parts else empty_frame(spec["columns"])


IPOK_TEXT = ["FormattedUtcTime", "deviceUtc"]   # text timestamps in the raw file; utcTime carries the same time


def ipok_columns(cfg):
    """Numeric columns of the per-flight IPOK files (header of any file), so flights without IPOK keep the schema."""
    cols = cfg["logs"]["ipok"]["csv_columns"]
    if cols != "all":
        return cols
    files = sorted(glob.glob(os.path.join(os.path.expanduser(cfg["ipok_dir"]), "*_ipok_raw.csv")))
    return [c for c in pd.read_csv(files[0], nrows=0).columns if c not in IPOK_TEXT] if files else ["utcTime"]


def read_ipok(cfg, stem):
    path = os.path.join(os.path.expanduser(cfg["ipok_dir"]), f"{stem}_ipok_raw.csv")
    cols = ipok_columns(cfg)
    if not os.path.exists(path):
        return empty_frame(cols).rename_axis("time_utc"), None
    d = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False).apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=["utcTime"])
    d.index = pd.to_datetime(d["utcTime"], unit="s", utc=True).rename("time_utc")
    return clean_index(d), path


def lag(a, b, t0, t1, dt=0.05, max_s=20.0):
    """Seconds to ADD to b's clock so that b lines up with a (positive = b's clock is behind), and the correlation."""
    g = pd.date_range(t0, t1, freq=f"{int(dt * 1000)}ms")

    def on(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        s = s[~s.index.duplicated()]
        if len(s) < 50:
            return None
        return np.interp(g.asi8, s.index.asi8, s.values.astype(float))

    x, y = on(a), on(b)
    if x is None or y is None or x.std() == 0 or y.std() == 0:
        return None, None
    x, y = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    n = int(max_s / dt)
    if len(x) <= 4 * n:
        return None, None
    c = [np.dot(x[n:-n], np.roll(y, k)[n:-n]) / (len(x) - 2 * n) for k in range(-n, n + 1)]
    k = int(np.argmax(c))
    return round((k - n) * dt, 3), round(float(c[k]), 3)


def best_lag(pairs, t0, t1, **kw):
    """Run lag() on several (name, a, b) pairs and keep the one with the highest correlation."""
    results = {name: lag(a, b, t0, t1, **kw) for name, a, b in pairs}
    ok = {k: v for k, v in results.items() if v[1] is not None}
    best = max(ok, key=lambda k: ok[k][1]) if ok else None
    return best, results


# status words, flags, modes, counters, ids: never interpolated
DISCRETE = re.compile(r"status|state|fault|fix|connected|counter|mode|version|source|_id$|enable|active|engaged|"
                      r"lights|piswth|requested|lost_packets|clipping|timepulse|numsv|satcount|packet|rtcm|word|brake",
                      re.I)
GAP_PERIODS = 2.5   # linear interpolation never bridges a gap longer than this many native sample periods
MIN_GAP_S = 0.1


def native_period_s(s):
    dt = np.diff(s.index.asi8) / 1e9
    return float(np.median(dt)) if len(dt) else None


def update_period_s(s):
    """True update period of a column that the source re-logs faster than it changes (e.g. converter values updated
    at 2 Hz in the 10 Hz OPC stream, wind at 1 Hz in a 5 Hz stream), or None. Requires the value changes to come at
    regular multiples of one period, so quantised or merely slow signals are not mistaken for it."""
    t = s.index.asi8 / 1e9
    if len(t) < 50:
        return None
    logged = np.median(np.diff(t))
    iv = np.diff(t[np.r_[True, np.diff(s.values) != 0]])
    if len(iv) < 20:
        return None
    p = float(np.median(iv))
    if p < 1.8 * logged:
        return None
    k = np.round(iv / p)
    regular = np.mean((k >= 1) & (np.abs(iv - k * p) <= 0.15 * p + 0.6 * logged))
    return p if regular >= 0.8 else None


def update_points(s, period):
    """Keep the first sample of each new value (the update instant) and, for a value that stays longer than 1.5
    update periods, its last sample too, so a real plateau stays flat instead of becoming a ramp."""
    t, v = s.index.asi8, s.values
    start = np.r_[True, v[1:] != v[:-1]]
    keep = start.copy()
    idx = np.flatnonzero(start)
    run_end = np.r_[idx[1:] - 1, len(v) - 1]
    long_run = (t[run_end] - t[idx]) > 1.5 * period * 1e9
    keep[run_end[long_run]] = True
    return s[keep]


def column_kind(src, name, s, cfg):
    """'linear' for continuous signals of sampled sources, else 'hold'."""
    if src in cfg["hold_sources"] or DISCRETE.search(name):
        return "hold"
    v = s.dropna().values
    if len(v) and np.all(v == np.round(v)) and len(np.unique(v)) <= 16:   # integer codes
        return "hold"
    return "linear"


def resample(grid, s, kind, max_age_s=None, wrap=False, period=None):
    """One column onto the grid. linear: time interpolation between the two neighbouring native samples, NaN if they
    are more than GAP_PERIODS native periods apart or the grid point is outside the data. hold: last value, at most
    max_age_s old (None = held without limit, for event-based sources)."""
    s = s.dropna()
    s = s[~s.index.duplicated(keep="last")]
    out = np.full(len(grid), np.nan)
    if s.empty:
        return out
    t, g = s.index.asi8, grid.asi8
    if kind == "hold":
        m = pd.merge_asof(pd.DataFrame(index=grid), s.rename("v").to_frame(), left_index=True, right_index=True,
                          direction="backward",
                          tolerance=pd.Timedelta(seconds=max_age_s) if max_age_s is not None else None)
        return m["v"].values.astype(float)
    # data gaps are judged on the logged rows, before any thinning to update instants
    if len(t) < 2:
        return out
    j = np.clip(np.searchsorted(t, g, side="left"), 1, len(t) - 1)
    gap = ((t[j] - t[j - 1]) > max(MIN_GAP_S, GAP_PERIODS * native_period_s(s)) * 1e9) & (t[j] != g)
    if period:   # interpolate between the real update instants, not the repeated log rows
        s = update_points(s, period)
    t, v = s.index.asi8, s.values.astype(float)
    if wrap:
        v = np.degrees(np.unwrap(np.radians(v)))
    y = np.interp(g, t, v, left=np.nan, right=np.nan) if len(t) > 1 else np.full(len(g), np.nan)
    y[gap] = np.nan
    if wrap:   # back to the source's own range: [0, 360) or (-180, 180]
        y = np.mod(y, 360) if s.min() >= 0 else np.mod(y + 180, 360) - 180
    return y


def on_grid(grid, df, src, prefix, cfg, max_age_s, overrides=None):
    """Every column of one source on the grid (per column: a source row carries only the fields of that frame).
    Returns the table (prefix + short name) and the method used per column."""
    out, how = {}, {}
    for col in df.columns:
        name = prefix + col
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        kind = column_kind(src, col, s, cfg)
        age = (overrides or {}).get(col, max_age_s)
        if kind == "hold" and src not in cfg["hold_sources"] and len(s) > 1:
            age = max(age or 0, GAP_PERIODS * native_period_s(s))   # discrete column of a sampled source
        wrap = name in cfg["wrap_deg"]
        period = update_period_s(s) if kind == "linear" else None
        out[name] = resample(grid, s, kind, age, wrap, period)
        how[name] = kind + (f" between {period:.2f} s updates" if period else "") + (" (unwrapped)" if wrap else "")
    return pd.DataFrame(out, index=grid), how


def write_csv(table, path, index=True):
    """%.8g for most columns; epoch-like columns and lat/lon written with fixed decimals so they keep precision."""
    t = table.copy()
    for c in t.columns:
        if not pd.api.types.is_float_dtype(t[c]):
            continue
        x = t[c]
        if re.search(r"latitude|longitude", c, re.I):
            t[c] = x.map(lambda v: "" if pd.isna(v) else f"{v:.9f}")
        elif x.abs().max() >= 1e7:
            t[c] = x.map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
    t.to_csv(path, index=index, float_format="%.8g")


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:
        return None


def alignment_figure(path, stem, fs, ipok_gs, cmd, upl, opc, a):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax = axes[0]
    if not ipok_gs.empty:
        ax.plot(ipok_gs.index, ipok_gs["Roll"], lw=0.6, label=f"IPOK Roll (shifted {a['ipok_to_gs_s']} s)")
    ax.plot(fs.index, fs["kite_roll_deg"], lw=0.8, label="Pod-relayed roll_angle")
    ax.set_ylabel("roll [deg]"), ax.legend(loc="upper right", fontsize=8)
    ax = axes[1]
    if not cmd.empty:
        ax.plot(cmd.index, cmd["pitch"], lw=0.6, label="CMD pitch (sent log, 20 Hz)")
    if not upl.empty:
        ax.plot(upl.index, upl["pitch_pct"], ".", ms=2, label="GS_UPLINK_COMMAND pitch (2 Hz)")
    ax.plot(fs.index, fs["pitch_pct"], lw=0.6, label="Pod measured pitch")
    ax.set_ylabel("depower [%]"), ax.legend(loc="upper right", fontsize=8)
    ax = axes[2]
    ax.plot(opc.index, -opc["drum_speed_ms"], lw=0.6, label="Line speed (OPC, + = reel-out)")
    vr = fs["velocity_r_ms"].where(fs["velocity_r_ms"].abs() < 100)
    ax.plot(vr.index, vr, lw=0.6, label="Pod-relayed range rate")
    ax.set_ylabel("[m/s]"), ax.legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].set_title(f"{stem} — clock alignment (GS clock)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def process(cfg, influx, stem, out_root, fallback=None):
    """fallback = (stem, lag_s) of a same-day event, used when this event has no accepted IPOK alignment."""
    ev = cfg["events"][stem]
    t0, t1 = pd.Timestamp(ev["start"]), pd.Timestamp(ev["stop"])
    pad = pd.Timedelta(seconds=cfg["pad_s"])
    w0, w1 = t0 - pad, t1 + pad
    lookback = pd.Timedelta(seconds=cfg["event_lookback_s"])
    out_dir = os.path.join(out_root, stem)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    print(f"== {stem}  {t0:%Y-%m-%d %H:%M:%S}–{t1:%H:%M:%S} UTC")

    # 1. Influx sources
    data = {}
    for src, spec in cfg["sources"].items():
        start = w0 - lookback if spec["max_age_s"] is None else w0
        data[src] = influx.fetch(src, spec["fields"], start, w1)
        print(f"   {src:18s} {len(data[src]):7d} rows")
    fs, opc = data["FLIGHT_SEGMENT"], data["OPC"]

    # pod identity must be constant inside the window
    fs_win = fs[(fs.index >= t0) & (fs.index < t1)]
    pod = {k: sorted(fs_win[k].dropna().unique().astype(int).tolist()) for k in ("fs_id", "pod_version")}
    if len(pod["fs_id"]) != 1 or len(pod["pod_version"]) != 1:
        raise RuntimeError(f"{stem}: pod identity changes inside the window: {pod}")

    # 2. Logs
    cmd = read_sent_log(cfg, ev, w0, w1)
    ipok, ipok_path = read_ipok(cfg, stem)
    print(f"   {'GS sent log':18s} {len(cmd):7d} rows")
    print(f"   {'IPOK':18s} {len(ipok):7d} rows" + ("" if ipok_path else "  (no file)"))

    # 3. Alignment
    align = {"reference": "GS/Influx backend clock"}
    if not ipok.empty:
        i0, i1 = max(t0, ipok.index[0] + pd.Timedelta(seconds=30)), min(t1, ipok.index[-1] - pd.Timedelta(seconds=30))
        best, res = best_lag([("roll", fs["kite_roll_deg"], ipok["Roll"]),
                              ("pitch", fs["kite_pitch_deg"], ipok["Pitch"])], i0, i1)
        lag_s, corr = res[best] if best else (None, None)
        align["ipok"] = {"method": "xcorr IPOK Pitch/Roll vs pod-relayed angles", "signal": best,
                         "all": {k: {"lag_s": v[0], "corr": v[1]} for k, v in res.items()},
                         "lag_s": lag_s, "corr": corr, "accepted": bool(corr is not None and corr >= IPOK_MIN_CORR)}
        if lag_s is not None and not align["ipok"]["accepted"]:
            # gaps in the IPOK file lower the full-window correlation; accept if 5-min sub-windows agree on the lag
            a, b = (fs["kite_pitch_deg"], ipok["Pitch"]) if best == "pitch" else (fs["kite_roll_deg"], ipok["Roll"])
            subs = [lag(a, b, s0, s0 + pd.Timedelta(minutes=5)) for s0 in pd.date_range(i0, i1, freq="5min")[:-1]]
            good = [l for l, c in subs if c is not None and c >= SUB_MIN_CORR]
            ok = len(good) >= 3 and max(abs(l - lag_s) for l in good) <= SUB_MAX_SPREAD_S
            align["ipok"]["subwindows"] = [{"lag_s": l, "corr": c} for l, c in subs]
            if ok:
                align["ipok"]["accepted"] = True
                align["ipok"]["accepted_by"] = "sub-window lag consistency"
    else:
        align["ipok"] = {"accepted": False, "note": "no IPOK file for this event"}
    ipok_lag = align["ipok"].get("lag_s") if align["ipok"]["accepted"] else None
    if ipok_lag is None and fallback:
        ipok_lag = fallback[1]
        align["ipok"]["fallback"] = {"from": fallback[0], "lag_s": fallback[1],
                                     "note": "GS clock offset borrowed from the nearest same-day event"}
    align["ipok_to_gs_s"] = ipok_lag
    # GS clock minus true UTC: the GS was ahead by the lag added to IPOK
    align["gs_minus_utc_s"] = ipok_lag

    upl = data["GS_UPLINK_COMMAND"]
    if not cmd.empty and not upl.empty:
        best, res = best_lag([("pitch", upl["pitch_pct"], cmd["pitch"]), ("roll", upl["roll_pct"], cmd["roll"])],
                             t0, t1, dt=0.1, max_s=5)
        align["cmd_log"] = {"method": "xcorr sent log vs GS_UPLINK_COMMAND (2 Hz)", "signal": best,
                            "all": {k: {"lag_s": v[0], "corr": v[1]} for k, v in res.items()}}
    align["cmd_to_gs_s"] = 0.0  # same machine; the xcorr above is the check

    vr = fs["velocity_r_ms"].where(fs["velocity_r_ms"].abs() < 100)
    lag_s, corr = lag(vr, -opc["drum_speed_ms"], t0, t1, dt=0.1, max_s=10)
    align["opc_vs_pod"] = {"method": "xcorr OPC line speed vs pod-relayed range rate", "lag_s": lag_s, "corr": corr,
                           "applied_s": 0.0, "note": "not applied: relay latency or physics, team to decide"}
    ct = (fs.index.astype("int64") / 1e9 - fs["controller_time"]).dropna()
    align["pod_controller_minus_gs_s"] = round(-float(ct.median()), 3) if len(ct) else None
    print(f"   align: IPOK→GS {align['ipok_to_gs_s']} s (corr {align['ipok'].get('corr')}), "
          f"OPC vs pod {lag_s} s (corr {corr}, not applied)")

    ipok_gs = ipok.copy()
    if ipok_lag is not None:
        ipok_gs.index = (ipok_gs.index + pd.Timedelta(seconds=ipok_lag)).rename("time_gs")
    else:
        ipok_gs = pd.DataFrame()
    if ipok_gs.empty:
        ipok_gs = pd.DataFrame(columns=ipok_columns(cfg), dtype=float,
                               index=pd.DatetimeIndex([], tz="UTC", name="time_gs"))

    # 4. Native parquet (every source at its own rate, GS clock)
    blocks = {src: (data[src], cfg["sources"][src]["prefix"]) for src in cfg["sources"]}
    blocks["CMD"] = (cmd, cfg["logs"]["gs_sent"]["prefix"])
    blocks["IPOK"] = (ipok_gs, cfg["logs"]["ipok"]["prefix"])
    native = []
    for src, (df, prefix) in blocks.items():
        if df.empty:
            continue
        d = df[(df.index >= w0) | (cfg["sources"].get(src, {}).get("max_age_s", 0) is None)].add_prefix(prefix)
        d.insert(0, "source", src)
        native.append(d)
    nat = pd.concat(native).sort_index()
    nat.index.name = "time_gs"
    nat.to_parquet(os.path.join(out_dir, f"{stem}_native.parquet"))

    # 5. Unified 20 Hz tables (same grid, same resampling): full source (original names) and analysis (short names)
    grid = pd.date_range(w0, w1, freq=f"{cfg['grid_ms']}ms", inclusive="left", name="time_gs")
    max_age = {src: spec["max_age_s"] for src, spec in cfg["sources"].items()}
    max_age.update({"CMD": cfg["logs"]["gs_sent"]["max_age_s"], "IPOK": cfg["logs"]["ipok"]["max_age_s"]})
    gridded, methods = {}, {}
    for src in dict.fromkeys(CSV_ORDER + cfg["full_source"]):
        df, prefix = blocks[src]
        gridded[src], how = on_grid(grid, df, src, prefix, cfg, max_age[src],
                                    cfg["logs"]["ipok"].get("max_age_override") if src == "IPOK" else None)
        methods.update(how)
    head = pd.DataFrame(index=grid)
    head["time_utc"] = (grid - pd.Timedelta(seconds=ipok_lag)).strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] \
        if ipok_lag is not None else ""
    head["time_rome"] = grid.tz_convert(TZ).strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3]
    head["t_s"] = ((grid - t0).total_seconds()).round(2)
    head["in_window"] = ((grid >= t0) & (grid < t1)).astype(int)
    stamp = grid.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3]

    original = {}   # short name -> original name (Influx field, or IPOK_<raw column>)
    for src, spec in cfg["sources"].items():
        original.update({spec["prefix"] + k: v for k, v in spec["fields"].items()})
    full = pd.concat([head] + [gridded[src] for src in cfg["full_source"]], axis=1).rename(columns=original)
    full.index = stamp
    full.index.name = "time_gs"
    write_csv(full, os.path.join(out_dir, f"{stem}_full_source.csv"))

    fs_drop = [cfg["sources"]["FLIGHT_SEGMENT"]["prefix"] + c for c in ("controller_time", "fs_id", "pod_version", "counter")]
    table = pd.concat([head] + [gridded[src] for src in CSV_ORDER], axis=1).drop(columns=fs_drop)
    table.index = stamp
    table.index.name = "time_gs"
    csv_path = os.path.join(out_dir, f"{stem}_analysis.csv")
    write_csv(table, csv_path)
    for old in ("_data.csv",):   # earlier name of the analysis table
        if os.path.exists(os.path.join(out_dir, stem + old)):
            os.remove(os.path.join(out_dir, stem + old))

    # 6. Meta
    inw = table[table["in_window"] == 1]
    coverage = {}
    for src in CSV_ORDER:
        prefix = blocks[src][1]
        cols = [c for c in table.columns if c.startswith(prefix)]
        coverage[src] = round(float(inw[cols].notna().any(axis=1).mean() * 100), 1) if cols else 0.0
    gaps = {}
    for src, (df, _) in blocks.items():
        d = df[(df.index >= t0) & (df.index < t1)]
        gaps[src] = round(float(d.index.to_series().diff().dt.total_seconds().max()), 2) if len(d) > 1 else None
    meta = {
        "stem": stem, "campaign": cfg["campaign"], "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "code_rev": git_rev(),
        "window_utc": [iso(t0), iso(t1)], "window_rome": [str(t0.tz_convert(TZ)), str(t1.tz_convert(TZ))],
        "window_approx": bool(ev.get("approx", False)), "pad_s": cfg["pad_s"],
        "pod": {"fs_id": pod["fs_id"][0], "pod_version_code": pod["pod_version"][0],
                "pod_version": {1: "V2", 2: "V3"}.get(pod["pod_version"][0], "?"),
                "servo_a": "left", "servo_b": "right"} if pod["pod_version"][0] == 2 else
               {"fs_id": pod["fs_id"][0], "pod_version_code": pod["pod_version"][0], "pod_version": "V2",
                "servo_a": "pitch", "servo_b": "roll"},
        "alignment": align,
        "rows_native": {src: int(len(df)) for src, (df, _) in blocks.items()},
        "coverage_pct_in_window": coverage,
        "max_gap_s_in_window": gaps,
        "ipok_file": os.path.basename(ipok_path) if ipok_path else None,   # name only: no local paths in outputs
        "csv": {"rate_hz": 1000 / cfg["grid_ms"], "rows": len(table), "columns": len(table.columns) + 1,
                "full_source_columns": len(full.columns) + 1, "hold_max_age_s": max_age,
                "linear_max_gap": f"{GAP_PERIODS} native periods (min {MIN_GAP_S} s)",
                "method_per_column": methods},
        "config_snapshot": influx.last_values("OPC_CONFIG", cfg["config_snapshot"]["OPC_CONFIG"],
                                              t1 - pd.Timedelta(days=7), t1),
        "fields": {src: spec["fields"] for src, spec in cfg["sources"].items()},
    }
    with open(os.path.join(out_dir, f"{stem}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    alignment_figure(os.path.join(out_dir, "figures", f"{stem}_alignment.png"), stem,
                     fs[(fs.index >= t0) & (fs.index < t1)], ipok_gs, cmd, upl, data["OPC"], align)
    print(f"   wrote {stem}_full_source.csv ({len(full)} rows × {len(full.columns) + 1} cols) and _analysis.csv "
          f"({len(table.columns) + 1} cols)")
    print(f"   coverage in window: {coverage}")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", action="append", help="event stem (repeatable)")
    ap.add_argument("--all", action="store_true", help="process every event in the config")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--out", default="processed/KTE_2030_FTC_0001")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    token = os.environ.get("INFLUX_TOKEN")
    if not token:
        raise SystemExit("set INFLUX_TOKEN (mirror token); it is never read from or written to a file")
    stems = list(cfg["events"]) if args.all else (args.event or [])
    if not stems:
        raise SystemExit("give --event <stem> or --all")
    influx = Influx(cfg, token)
    metas = {stem: process(cfg, influx, stem, args.out) for stem in stems}
    # events without an accepted IPOK alignment: re-run with the GS clock offset of the nearest same-day event
    for stem, m in list(metas.items()):
        if m["alignment"]["ipok_to_gs_s"] is not None:
            continue
        t = pd.Timestamp(cfg["events"][stem]["start"])
        same_day = [(abs((pd.Timestamp(cfg["events"][s]["start"]) - t).total_seconds()), s) for s, o in metas.items()
                    if s != stem and s[:26] == stem[:26] and o["alignment"]["ipok_to_gs_s"] is not None]
        if same_day:
            ref = min(same_day)[1]
            print(f"   {stem}: re-run with GS clock offset from {ref}")
            metas[stem] = process(cfg, influx, stem, args.out,
                                  fallback=(ref, metas[ref]["alignment"]["ipok_to_gs_s"]))
    rows = []
    for stem, m in metas.items():
        a = m["alignment"]
        rows.append({"stem": stem, "start_utc": m["window_utc"][0], "stop_utc": m["window_utc"][1],
                     "approx": m["window_approx"], "fs_id": m["pod"]["fs_id"], "pod_version": m["pod"]["pod_version"],
                     "ipok_to_gs_s": a["ipok_to_gs_s"], "ipok_corr": a["ipok"].get("corr"),
                     "opc_vs_pod_lag_s": a["opc_vs_pod"]["lag_s"], "opc_vs_pod_corr": a["opc_vs_pod"]["corr"],
                     "pod_controller_minus_gs_s": a["pod_controller_minus_gs_s"],
                     **{f"cov_{k}": v for k, v in m["coverage_pct_in_window"].items()}})
    events_csv = os.path.join(args.out, "events.csv")
    new = pd.DataFrame(rows).set_index("stem")
    if os.path.exists(events_csv):
        old = pd.read_csv(events_csv, index_col="stem")
        new = pd.concat([old.drop(index=new.index, errors="ignore"), new]).sort_index()
    new.to_csv(events_csv)


if __name__ == "__main__":
    main()
