"""
Raw OPC + OPC_CONFIG export of KTE_2030_FTC_0001: one CSV per flight, every OPC field at its native rate (~10 Hz),
with the OPC_CONFIG values in force at each sample. Separate from the analysis outputs of ftc_extract.py.

Why it is fast and does not crash the server (unlike a Flux export with pivot()):
  * InfluxQL returns one row per timestamp with each field as a column, so nothing has to be pivoted.
  * Only fields that have data in the flight are selected (one COUNT(*) per source, aggregated on the server).
    SELECT * would return every field of the whole `tk` measurement, mostly empty.
  * The flight is fetched in short time chunks (default 60 s) and appended to the CSV as it arrives; neither
    the server nor this machine ever holds the whole flight. A chunk that fails is split in two and retried.

Config: OPC_CONFIG is logged sparsely, so the last value of each config field before the export start is fetched
(LAST(*), within --config-lookback-h) and carried forward; each OPC row gets the config in force at that moment.

Columns: time_utc, time_rome, in_window, OPC_* (original names), OPC_CONFIG_* (original names).
Range: the official window from config/fields_KTE_2030_FTC_0001.yaml +- pad_s (in_window = 1 inside the window).
Check: at the end the non-empty count of every OPC column is compared with the server's COUNT; the result is in
<stem>_opc_config_meta.json together with the time range, row and column counts and the query time.

Usage (token only via the environment, never on the command line or in files):
    read -s INFLUX_TOKEN && export INFLUX_TOKEN
    python ftc_opc_export.py --all                    # all 7 flights
    python ftc_opc_export.py --event KTE_2030_FTC_0001_20261022_S1E1 [--gzip] [--chunk-s 60] [--force]
Output: processed/KTE_2030_FTC_0001_opc/<stem>_opc_config.csv[.gz] (+ _meta.json)
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import yaml

from ftc_extract import CONFIG, TZ, Influx, iso

OPC, CONF = "OPC", "OPC_CONFIG"


def q(name):
    return '"' + name.replace('"', '\\"') + '"'


def field_counts(influx, source, t0, t1):
    """{field: number of points} for every field of `source` with data in [t0, t1) (server-side COUNT)."""
    df = influx.query(f"SELECT COUNT(*) FROM tk WHERE source='{source}' AND time >= '{iso(t0)}' AND time < '{iso(t1)}'")
    if df.empty:
        return {}
    row = df.drop(columns="time").iloc[0]
    prefix = source + "_"
    out = {c[len("count_"):]: int(v) for c, v in row.items() if pd.notna(v) and v > 0}
    # a field key belongs to its source by prefix; OPC_ must not swallow OPC_CONFIG_
    return {f: n for f, n in sorted(out.items())
            if f.startswith(prefix) and not (source == OPC and f.startswith(CONF + "_"))}


def select(influx, source, fields, t0, t1):
    df = influx.query(f"SELECT {', '.join(q(f) for f in fields)} FROM tk WHERE source='{source}' "
                      f"AND time >= '{iso(t0)}' AND time < '{iso(t1)}'")
    if df.empty:
        return pd.DataFrame(columns=fields, index=pd.DatetimeIndex([], tz="UTC"))
    df.index = pd.to_datetime(df.pop("time"), unit="ns", utc=True)
    return df.reindex(columns=fields)


def select_chunked(influx, source, fields, t0, t1, retries=3):
    """select(), splitting the interval in two when the server times out or fails."""
    for attempt in range(retries):
        try:
            return select(influx, source, fields, t0, t1)
        except Exception as e:
            print(f"      {iso(t0)} +{(t1 - t0).total_seconds():.0f} s failed ({type(e).__name__}: {str(e)[:80]})")
            if (t1 - t0).total_seconds() > 5:
                mid = t0 + (t1 - t0) / 2
                return pd.concat([select_chunked(influx, source, fields, t0, mid, retries),
                                  select_chunked(influx, source, fields, mid, t1, retries)])
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{source} {iso(t0)}..{iso(t1)} failed {retries} times")


def config_state(influx, fields, t0, t1, lookback_h):
    """Config changes in [t0, t1), preceded by the last value of each field before t0; forward-filled."""
    df = influx.query(f"SELECT LAST(*) FROM tk WHERE source='{CONF}' "
                      f"AND time >= '{iso(t0 - pd.Timedelta(hours=lookback_h))}' AND time < '{iso(t0)}'")
    first = pd.DataFrame(columns=fields, index=pd.DatetimeIndex([t0 - pd.Timedelta(microseconds=1)]))
    if not df.empty:
        row = df.drop(columns="time").iloc[0]
        for f in fields:
            first.loc[first.index[0], f] = row.get("last_" + f)
    changes = select_chunked(influx, CONF, fields, t0, t1) if fields else first.iloc[:0]
    state = pd.concat([first, changes]).sort_index()
    state = state[~state.index.duplicated(keep="last")].infer_objects().ffill()
    missing = [f for f in fields if state[f].isna().all()]
    return state, missing


def export(influx, cfg, stem, out_dir, chunk_s, gzip, force, lookback_h):
    ev = cfg["events"][stem]
    pad = pd.Timedelta(seconds=cfg["pad_s"])
    w0, w1 = pd.Timestamp(ev["start"]), pd.Timestamp(ev["stop"])
    t0, t1 = w0 - pad, w1 + pad
    path = os.path.join(out_dir, f"{stem}_opc_config.csv" + (".gz" if gzip else ""))
    if os.path.exists(path) and not force:
        print(f"{stem}: exists, skipped (--force to redo)")
        return
    tic = time.time()
    print(f"{stem}: {iso(t0)} .. {iso(t1)}")
    opc_n = field_counts(influx, OPC, t0, t1)
    conf_n = field_counts(influx, CONF, t0 - pd.Timedelta(hours=lookback_h), t1)
    opc_fields, conf_fields = list(opc_n), list(conf_n)
    print(f"   {len(opc_fields)} OPC fields, {len(conf_fields)} OPC_CONFIG fields")
    state, conf_missing = config_state(influx, conf_fields, t0, t1, lookback_h)
    header = ["time_utc", "time_rome", "in_window"] + opc_fields + conf_fields
    tmp = path + ".part"
    rows, got = 0, {f: 0 for f in opc_fields}
    edges = list(pd.date_range(t0, t1, freq=f"{chunk_s}s")) + [t1]
    edges = sorted(set(edges))
    with (pd.io.common.get_handle(tmp, "w", compression="gzip" if gzip else None)) as h:
        h.handle.write(",".join(header) + "\n")
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]), start=1):
            d = select_chunked(influx, OPC, opc_fields, a, b)
            if d.empty:
                continue
            d = d[~d.index.duplicated(keep="last")].sort_index()
            for f in opc_fields:
                got[f] += int(d[f].notna().sum())
            if conf_fields:
                c = pd.merge_asof(pd.DataFrame(index=d.index), state, left_index=True, right_index=True,
                                  direction="backward")   # config in force at each OPC sample
                d = pd.concat([d, c[conf_fields]], axis=1)
            d.insert(0, "in_window", ((d.index >= w0) & (d.index <= w1)).astype(int))
            d.insert(0, "time_rome", d.index.tz_convert(TZ).strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3])
            d.insert(0, "time_utc", d.index.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z")
            d.to_csv(h.handle, header=False, index=False, lineterminator="\n")
            rows += len(d)
            print(f"\r   chunk {i}/{len(edges) - 1}, {rows} rows", end="", flush=True)
    print()
    mismatch = {f: {"server": opc_n[f], "csv": got[f]} for f in opc_fields if got[f] != opc_n[f]}
    os.replace(tmp, path)
    meta = {"stem": stem, "file": os.path.basename(path), "influx_bucket": cfg["influx"]["bucket"],
            "window_utc": [iso(w0), iso(w1)], "export_utc": [iso(t0), iso(t1)], "pad_s": cfg["pad_s"],
            "rows": rows, "opc_fields": len(opc_fields), "config_fields": len(conf_fields),
            "config_lookback_h": lookback_h, "config_fields_without_value": conf_missing,
            "config_changes_in_range": int(len(state) - 1),
            "count_check": "ok" if not mismatch else "mismatch", "count_mismatch": mismatch,
            "chunk_s": chunk_s, "seconds": round(time.time() - tic, 1),
            "size_mb": round(os.path.getsize(path) / 1e6, 1)}
    with open(os.path.join(out_dir, f"{stem}_opc_config_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"   {rows} rows x {len(header)} columns, {meta['size_mb']} MB, {meta['seconds']} s, "
          f"count check {meta['count_check']}" + (f" ({len(mismatch)} columns differ)" if mismatch else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--event", action="append")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="processed/KTE_2030_FTC_0001_opc")
    ap.add_argument("--chunk-s", type=int, default=60)
    ap.add_argument("--config-lookback-h", type=float, default=24)
    ap.add_argument("--gzip", action="store_true", help="write .csv.gz (pandas reads it directly)")
    ap.add_argument("--force", action="store_true", help="redo flights that already have a file")
    args = ap.parse_args()
    token = os.environ.get("INFLUX_TOKEN")
    if not token:
        sys.exit("INFLUX_TOKEN is not set (read -s INFLUX_TOKEN && export INFLUX_TOKEN)")
    cfg = yaml.safe_load(open(CONFIG))
    influx = Influx(cfg, token)
    os.makedirs(args.out, exist_ok=True)
    for stem in (list(cfg["events"]) if args.all else args.event):
        export(influx, cfg, stem, args.out, args.chunk_s, args.gzip, args.force, args.config_lookback_h)


if __name__ == "__main__":
    main()
