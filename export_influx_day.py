"""
Export one day of ground-station data from InfluxDB to Parquet (whole day) and CSV + Parquet (flights).

Why InfluxQL and not Flux: Flux pivot() builds the whole wide table in server memory, which hangs on
OPC (~400 fields at 10 Hz). Every source is written as one point per snapshot, so InfluxQL
`SELECT <fields> FROM tk WHERE source='X'` returns already-aligned wide rows without any pivot.

Pipeline
  1. discover   sources (tag `source`), their field keys for the day, field types (SHOW FIELD KEYS)
  2. count      SELECT COUNT(*) ... GROUP BY time(1h) per source = expected non-null values per field/hour
  3. export     one Parquet per source per UTC hour; skipped if already present and verified (resumable)
  4. verify     non-null count of every column == server count, otherwise the file is rejected
  5. flights    slice each source to the flight windows -> CSV (UTC + local time) and Parquet
  6. manifest   manifest.json with rows, columns, time span, size, sha256, gaps, check results

Usage
  export INFLUX_URL="http://192.168.15.99:8086"
  export INFLUX_TOKEN="..."
  python export_influx_day.py --date 2026-10-22 \
      --flight "Flight 1=10:15:59-10:39:29" --flight "Flight 2=11:59:20-12:20:00" \
      --flight "Flight 3=15:32:26-15:53:00"
"""

import argparse
import hashlib
import io
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

TZ = "Europe/Rome"
MEASUREMENT = "tk"
GAP_THRESHOLD_S = 1.0
RETRIES = 3

PANDAS_TYPES = {"float": "float64", "integer": "Int64", "boolean": "boolean", "string": "string"}


class Influx:
    def __init__(self, url, token, org, bucket, timeout=600):
        self.url, self.token, self.org, self.bucket, self.timeout = url.rstrip("/"), token, org, bucket, timeout

    def _post(self, path, data, content_type, params=None, accept="application/csv"):
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        req = urllib.request.Request(
            f"{self.url}{path}{query}", data=data,
            headers={"Authorization": f"Token {self.token}", "Content-Type": content_type, "Accept": accept},
        )
        for attempt in range(1, RETRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read().decode()
            except Exception as e:
                if attempt == RETRIES:
                    raise
                print(f"    retry {attempt}/{RETRIES - 1} after error: {e}")
                time.sleep(5 * attempt)

    def flux(self, query):
        text = self._post("/api/v2/query", query.encode(), "application/vnd.flux", {"org": self.org})
        return pd.read_csv(io.StringIO(text)) if text.strip() else pd.DataFrame()

    def influxql(self, query):
        """JSON rather than CSV output: CSV cannot tell an empty string from a missing value."""
        body = urllib.parse.urlencode({"db": self.bucket, "epoch": "ns", "q": query}).encode()
        result = json.loads(self._post("/query", body, "application/x-www-form-urlencoded", accept="application/json"))
        result = result["results"][0]
        if "error" in result:
            raise RuntimeError(f"InfluxQL error: {result['error']}")
        series = result.get("series", [])
        if not series:
            return pd.DataFrame()
        return pd.concat([pd.DataFrame(sr["values"], columns=sr["columns"], dtype=object) for sr in series],
                         ignore_index=True)


def rfc3339(ts):
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def quote_field(name):
    return '"' + name.replace('"', '\\"') + '"'


def discover(db, start, stop):
    flux = f"""import "influxdata/influxdb/schema"
schema.tagValues(bucket: "{db.bucket}", tag: "source", start: {rfc3339(start)}, stop: {rfc3339(stop)})"""
    sources = sorted(db.flux(flux)["_value"].dropna().tolist())

    fields = {}
    for source in sources:
        flux = f"""import "influxdata/influxdb/schema"
schema.fieldKeys(bucket: "{db.bucket}", predicate: (r) => r._measurement == "{MEASUREMENT}" and r.source == "{source}",
                 start: {rfc3339(start)}, stop: {rfc3339(stop)})"""
        fields[source] = sorted(db.flux(flux)["_value"].dropna().tolist())

    types_df = db.influxql(f"SHOW FIELD KEYS FROM {MEASUREMENT}")
    types = dict(zip(types_df["fieldKey"], types_df["fieldType"]))
    return sources, fields, types


def hourly_counts(db, source, fields, start, stop):
    """Expected non-null values per field per UTC hour, straight from the server."""
    df = db.influxql(
        f"SELECT COUNT(*) FROM {MEASUREMENT} WHERE source='{source}' "
        f"AND time >= '{rfc3339(start)}' AND time < '{rfc3339(stop)}' GROUP BY time(1h)"
    )
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df["time"].astype("int64"), utc=True)
    cols = {f"count_{f}": f for f in fields if f"count_{f}" in df.columns}
    counts = df[list(cols)].rename(columns=cols).apply(pd.to_numeric).fillna(0).astype("int64")
    return counts[counts.sum(axis=1) > 0]


def fetch_hour(db, source, fields, types, hour_start):
    hour_stop = hour_start + pd.Timedelta(hours=1)
    select = ",".join(quote_field(f) for f in fields)
    df = db.influxql(
        f"SELECT {select} FROM {MEASUREMENT} WHERE source='{source}' "
        f"AND time >= '{rfc3339(hour_start)}' AND time < '{rfc3339(hour_stop)}'"
    )
    cols = {"time": pd.to_datetime(df["time"].astype("int64"), utc=True)}
    for f in fields:
        raw = df[f] if f in df.columns else pd.Series([None] * len(df), index=df.index, dtype=object)
        kind = types.get(f, "float")
        if kind == "float":
            cols[f] = pd.to_numeric(raw).astype("float64")
        else:
            cols[f] = pd.Series(pd.array(raw.tolist(), dtype=PANDAS_TYPES[kind]), index=df.index)
    return pd.DataFrame(cols).sort_values("time").reset_index(drop=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gaps(times, threshold_s=GAP_THRESHOLD_S):
    dt = times.diff().dt.total_seconds()
    idx = np.flatnonzero(dt.to_numpy() > threshold_s)
    return [{"from": times.iloc[i - 1].isoformat(), "to": times.iloc[i].isoformat(), "seconds": round(dt.iloc[i], 3)}
            for i in idx]


def file_entry(path, df):
    return {
        "file": path,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "start": df["time"].iloc[0].isoformat() if len(df) else None,
        "end": df["time"].iloc[-1].isoformat() if len(df) else None,
        "bytes": os.path.getsize(path),
        "sha256": sha256(path),
    }


def export_day(db, date, out_dir, only_sources=None):
    start = pd.Timestamp(date, tz=TZ)
    stop = start + pd.Timedelta(days=1)
    print(f"Day {date} local = {rfc3339(start)} .. {rfc3339(stop)}")

    sources, fields, types = discover(db, start, stop)
    if only_sources:
        sources = [s for s in sources if s in only_sources]
    print("Sources: " + ", ".join(f"{s} ({len(fields[s])} fields)" for s in sources))

    manifest = {"day": {}, "fields": fields, "types": {s: {f: types.get(f) for f in fields[s]} for s in sources}}
    for source in sources:
        src_dir = os.path.join(out_dir, "day", source)
        os.makedirs(src_dir, exist_ok=True)
        counts = hourly_counts(db, source, fields[source], start, stop)
        entries = []
        for hour, expected in counts.iterrows():
            path = os.path.join(src_dir, f"{source}_{hour.strftime('%Y%m%dT%H%MZ')}.parquet")
            if os.path.exists(path):
                df = pd.read_parquet(path)
                if (df[expected.index].notna().sum() == expected).all():
                    print(f"  {source} {hour:%H}Z  exists, verified ({len(df)} rows)")
                    entries.append({**file_entry(path, df), "verified": True})
                    continue
            t0 = time.time()
            df = fetch_hour(db, source, fields[source], types, hour)
            got = df[expected.index].notna().sum()
            bad = got[got != expected]
            if len(bad):
                raise RuntimeError(f"{source} {hour}: count mismatch on {len(bad)} fields, e.g. "
                                   f"{bad.index[0]} got {bad.iloc[0]} expected {expected[bad.index[0]]}")
            tmp = path + ".tmp"
            df.to_parquet(tmp, compression="zstd", index=False)
            os.replace(tmp, path)
            print(f"  {source} {hour:%H}Z  {len(df):>6} rows x {df.shape[1]} cols  "
                  f"{os.path.getsize(path) / 1e6:6.1f} MB  {time.time() - t0:5.1f}s  verified")
            entries.append({**file_entry(path, df), "verified": True})
        manifest["day"][source] = entries
    return manifest, sources


def load_day(out_dir, source):
    src_dir = os.path.join(out_dir, "day", source)
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(os.path.join(src_dir, f)) for f in files], ignore_index=True)


def export_flights(out_dir, date, flights, sources):
    result = {}
    day = {s: load_day(out_dir, s) for s in sources}
    for name, (t_from, t_to) in flights.items():
        a = pd.Timestamp(f"{date} {t_from}", tz=TZ)
        b = pd.Timestamp(f"{date} {t_to}", tz=TZ)
        fl_dir = os.path.join(out_dir, "flights", name.replace(" ", "_").lower())
        os.makedirs(fl_dir, exist_ok=True)
        entries = {"start_local": a.isoformat(), "stop_local": b.isoformat(), "files": {}}
        for source, df in day.items():
            if df.empty:
                continue
            seg = df[(df["time"] >= a) & (df["time"] < b)].reset_index(drop=True)
            if seg.empty:
                continue
            pq = os.path.join(fl_dir, f"{source}.parquet")
            seg.to_parquet(pq, compression="zstd", index=False)
            csv = os.path.join(fl_dir, f"{source}.csv")
            seg_csv = seg.copy()
            seg_csv.insert(1, "time_local", seg_csv["time"].dt.tz_convert(TZ))
            seg_csv.to_csv(csv, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f%z")
            entries["files"][source] = {
                "parquet": file_entry(pq, seg), "csv": file_entry(csv, seg),
                "gaps_over_1s": gaps(seg["time"]) if source == "OPC" else None,
            }
            print(f"  {name} {source:<18} {len(seg):>6} rows")
        result[name] = entries
    return result


def parse_flight(text):
    name, span = text.split("=", 1)
    t_from, t_to = span.split("-")
    return name.strip(), (t_from.strip(), t_to.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="local date, YYYY-MM-DD (Europe/Rome)")
    ap.add_argument("--flight", action="append", default=[], help='"Name=HH:MM:SS-HH:MM:SS" local time, repeatable')
    ap.add_argument("--source", action="append", help="limit to these sources (default: all)")
    ap.add_argument("--out", default="data_archive")
    ap.add_argument("--url", default=os.environ.get("INFLUX_URL", "http://192.168.15.99:8086"))
    ap.add_argument("--token", default=os.environ.get("INFLUX_TOKEN"))
    ap.add_argument("--org", default=os.environ.get("INFLUX_ORG", "Kitenergy"))
    ap.add_argument("--bucket", default=os.environ.get("INFLUX_BUCKET", "gs_mirror"))
    args = ap.parse_args()
    if not args.token:
        ap.error("set INFLUX_TOKEN or pass --token")

    out_dir = os.path.join(args.out, args.date)
    os.makedirs(out_dir, exist_ok=True)
    db = Influx(args.url, args.token, args.org, args.bucket)

    t0 = time.time()
    manifest, sources = export_day(db, args.date, out_dir, args.source)
    flights = dict(parse_flight(f) for f in args.flight)
    if flights:
        print("Flights:")
        manifest["flights"] = export_flights(out_dir, args.date, flights, sources)
    manifest["export"] = {
        "url": args.url, "org": args.org, "bucket": args.bucket, "measurement": MEASUREMENT,
        "date_local": args.date, "timezone": TZ,
        "created_utc": datetime.now(timezone.utc).isoformat(), "duration_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    print(f"Done in {time.time() - t0:.0f}s -> {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
