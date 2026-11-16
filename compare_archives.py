"""
Compare two day archives made by export_influx_day.py (e.g. field InfluxDB vs company mirror) value by value.

For every source: field sets, timestamps only on one side, and per field the number of differing values on
the shared timestamps (exact comparison, missing == missing). Writes comparison.csv next to the report.

Usage:
  python compare_archives.py data_archive/2026-10-22 data_archive_primary/2026-10-22 --names mirror primary
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

TZ = "Europe/Rome"


def load(day_dir, source):
    files = sorted(glob.glob(os.path.join(day_dir, "day", source, "*.parquet")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.drop_duplicates("time").set_index("time").sort_index()


def span(idx):
    if len(idx) == 0:
        return ""
    a, b = idx.min().tz_convert(TZ), idx.max().tz_convert(TZ)
    return f"{a:%H:%M:%S}-{b:%H:%M:%S} ({len(idx)} rows)"


def compare_source(a, b):
    fields_a, fields_b = set(a.columns), set(b.columns)
    common_t = a.index.intersection(b.index)
    only_a, only_b = a.index.difference(b.index), b.index.difference(a.index)
    diffs = {}
    for f in sorted(fields_a & fields_b):
        x, y = a.loc[common_t, f], b.loc[common_t, f]
        same = (x == y).fillna(False).to_numpy(dtype=bool) | (x.isna() & y.isna()).to_numpy()
        n = int((~same).sum())
        if n:
            diffs[f] = n
    return {
        "fields_only_a": sorted(fields_a - fields_b), "fields_only_b": sorted(fields_b - fields_a),
        "common_rows": len(common_t), "only_a": only_a, "only_b": only_b, "diffs": diffs,
        "values_compared": len(common_t) * len(fields_a & fields_b),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    args = ap.parse_args()
    na, nb = args.names

    sources = sorted({os.path.basename(p) for d in (args.a, args.b) for p in glob.glob(os.path.join(d, "day", "*"))})
    rows = []
    for source in sources:
        a, b = load(args.a, source), load(args.b, source)
        if a.empty or b.empty:
            rows.append({"source": source, "status": f"only in {na if b.empty else nb}",
                         f"{na} span": span(a.index), f"{nb} span": span(b.index)})
            continue
        r = compare_source(a, b)
        ok = not (r["fields_only_a"] or r["fields_only_b"] or len(r["only_a"]) or len(r["only_b"]) or r["diffs"])
        rows.append({
            "source": source, "status": "identical" if ok else "differs",
            f"{na} span": span(a.index), f"{nb} span": span(b.index),
            "shared rows": r["common_rows"], "values compared": r["values_compared"],
            "differing values": sum(r["diffs"].values()),
            f"rows only in {na}": f"{len(r['only_a'])} {span(r['only_a'])}".strip(),
            f"rows only in {nb}": f"{len(r['only_b'])} {span(r['only_b'])}".strip(),
            f"fields only in {na}": ", ".join(r["fields_only_a"]),
            f"fields only in {nb}": ", ".join(r["fields_only_b"]),
            "fields with differences": "; ".join(f"{k} ({v})" for k, v in list(r["diffs"].items())[:10]),
        })
    out = pd.DataFrame(rows)
    path = os.path.join(args.b, f"comparison_vs_{na}.csv")
    out.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print(out.T.to_string())
    print(f"\nWritten {path}")


if __name__ == "__main__":
    main()
