"""
Check a local day archive (export_influx_day.py) against an InfluxDB server without downloading raw data.

The server computes per source, per UTC hour and per field COUNT (all fields) and SUM / MIN / MAX (numeric
fields); the same statistics are computed from the archive's Parquet files and compared. This is light enough
for the ground-station host: one aggregate query at a time, with a pause between queries, aborting if a query
gets slow. Rows the server has that the archive lacks (e.g. data written after the last mirror sync) show up
as count differences.

Usage:
  python compare_server_aggregates.py data_archive/2026-10-22 --url http://192.168.1.100:8086 \
      --org kitenergy --bucket gs --token ...
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

import export_influx_day
from export_influx_day import MEASUREMENT, TZ, Influx, quote_field, rfc3339

export_influx_day.RETRIES = 1  # never repeat a heavy query against a struggling server

PAUSE_S = 1.0
SLOW_S = 60.0


def load_source(day_dir, source):
    files = sorted(glob.glob(os.path.join(day_dir, "day", source, "*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def local_stats(df, fields, numeric):
    hour = df["time"].dt.floor("h")
    g = df.groupby(hour)
    out = {"count": g[fields].count()}
    for agg in ("sum", "min", "max"):
        out[agg] = getattr(g[numeric], agg)() if numeric else pd.DataFrame(index=out["count"].index)
    return out


def timed(db, query):
    t0 = time.time()
    df = db.influxql(query)
    dt = time.time() - t0
    if dt > SLOW_S:
        raise RuntimeError(f"query took {dt:.0f}s, stopping to protect the server")
    time.sleep(PAUSE_S)
    return df, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive_day", help="e.g. data_archive/2026-10-22")
    ap.add_argument("--url", required=True)
    ap.add_argument("--org", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--token", default=os.environ.get("INFLUX_TOKEN"))
    args = ap.parse_args()

    db = Influx(args.url, args.token, args.org, args.bucket, timeout=120)
    date = os.path.basename(os.path.normpath(args.archive_day))
    start = pd.Timestamp(date, tz=TZ)
    stop = start + pd.Timedelta(days=1)

    rows, details = [], []
    for source in sorted(os.listdir(os.path.join(args.archive_day, "day"))):
        local = load_source(args.archive_day, source)
        fields = [c for c in local.columns if c != "time"]
        numeric = [f for f in fields if pd.api.types.is_numeric_dtype(local[f]) and not pd.api.types.is_bool_dtype(local[f])]
        loc = local_stats(local, fields, numeric)

        # server: counts for every field of the measurement (also reveals fields the archive lacks)
        cnt, dt = timed(db, f"SELECT COUNT(*) FROM {MEASUREMENT} WHERE source='{source}' AND time >= '{rfc3339(start)}' "
                            f"AND time < '{rfc3339(stop)}' GROUP BY time(1h)")
        cnt.index = pd.to_datetime(cnt["time"].astype("int64"), utc=True)
        cnt = cnt.drop(columns="time").apply(pd.to_numeric).fillna(0).astype("int64")
        cnt.columns = [c.removeprefix("count_") for c in cnt.columns]
        cnt = cnt.loc[cnt.sum(axis=1) > 0, cnt.sum(axis=0) > 0]
        server_only_fields = sorted(set(cnt.columns) - set(fields))

        hours = cnt.index.union(loc["count"].index)
        n_diff_count = n_diff_val = 0
        for hour in hours:
            exp = cnt.reindex(index=[hour], columns=fields).fillna(0).astype("int64").iloc[0]
            got = loc["count"].reindex(index=[hour], columns=fields).fillna(0).astype("int64").iloc[0]
            for f in fields:
                if exp[f] != got[f]:
                    n_diff_count += 1
                    details.append({"source": source, "hour_utc": hour, "field": f, "check": "count",
                                    "server": exp[f], "archive": got[f]})
            if not numeric or got.sum() == 0 or exp.sum() == 0:
                continue
            sel = ",".join(f"{agg}({quote_field(f)}) AS \"{agg}_{f}\"" for f in numeric for agg in ("SUM", "MIN", "MAX"))
            agg_df, dt_h = timed(db, f"SELECT {sel} FROM {MEASUREMENT} WHERE source='{source}' AND time >= "
                                     f"'{rfc3339(hour)}' AND time < '{rfc3339(hour + pd.Timedelta(hours=1))}'")
            if agg_df.empty:
                continue
            srv = agg_df.iloc[0]
            for f in numeric:
                for agg in ("sum", "min", "max"):
                    a = srv.get(f"{agg.upper()}_{f}")
                    b = loc[agg].at[hour, f] if hour in loc[agg].index and got[f] > 0 else None  # pandas sums empty to 0
                    a = None if a is None or pd.isna(a) else float(a)
                    b = None if b is None or pd.isna(b) else float(b)
                    if a is None and b is None:
                        continue
                    if a is None or b is None or not np.isclose(a, b, rtol=1e-9, atol=1e-6):
                        n_diff_val += 1
                        details.append({"source": source, "hour_utc": hour, "field": f, "check": agg,
                                        "server": a, "archive": b})
        rows.append({
            "source": source, "fields (archive)": len(fields), "fields only on server": ", ".join(server_only_fields),
            "hours compared": len(hours), "rows archive": len(local),
            "count differences": n_diff_count, "sum/min/max differences": n_diff_val,
            "status": "identical" if not (n_diff_count or n_diff_val or server_only_fields) else "differs",
        })
        print(f"{source:<18} hours {len(hours):>2}  count diffs {n_diff_count:>4}  value diffs {n_diff_val:>4}  "
              f"server-only fields {len(server_only_fields)}  (count query {dt:.1f}s)", flush=True)

    out_dir = args.archive_day
    tag = args.url.split("//")[-1].split(":")[0].replace(".", "_")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"check_vs_{tag}.csv"), index=False)
    pd.DataFrame(details).to_csv(os.path.join(out_dir, f"check_vs_{tag}_details.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
