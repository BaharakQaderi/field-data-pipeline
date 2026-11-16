"""
Does putting every source on the 20 Hz grid change the signals? Compares, inside the official window, each column
that ftc_extract.py interpolated linearly with its own native samples (from <stem>_native.parquet).

Per column:
  The native side is the data as logged, each row weighted by the time until the next row, so both sides are
  time-weighted. For columns the source re-logs between real updates the shape is compared at the updates.
  mean        native mean vs grid mean, difference as % of the native std
  std         grid std / native std
  min, max    difference as % of the native range (linear interpolation can only cut peaks, never add them)
  pctl        largest difference of the 1/5/50/95/99th percentiles, % of range
  ks          Kolmogorov-Smirnov distance between the native and grid value distributions (0 = identical)
  shape       grid series interpolated back to the native timestamps, RMS error / native std (0 = same shape)
  energy      for power columns, native trapezoid integral vs grid sum, % difference

Writes <out>/<stem>/<stem>_resample_check.csv, <out>/campaign/resample_check_all.csv and
<out>/<stem>/figures/<stem>_resample_check.png (native samples vs 20 Hz for a few signals).
Usage: python ftc_resample_check.py [--event <stem> ...]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ftc_extract import CONFIG, update_period_s, update_points
from ftc_figures import DPI_FLIGHT   # also applies the repo's standard PNG text sizes

POWER = {"OPC_batt_power_kw": "kW", "POD_power_w": "W"}
SHOW = [("OPC_force_sum_kg", "force [kg]  OPC 10 Hz"), ("OPC_drum_speed_ms", "drum speed [m/s]  OPC 10 Hz"),
        ("OPC_batt_power_kw", "battery power [kW]  OPC 10 Hz"), ("WIND_speed_ms", "wind [m/s]  5 Hz"),
        ("POD_roll_pct", "pod roll [%]  20 Hz"), ("POD_servo_left_torque_nm", "pod left torque [N·m]  10 Hz"),
        ("IPOK_Roll", "IPOK roll [deg]  100 Hz")]
BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e6e5e0"


def wquantile(v, w, q):
    o = np.argsort(v)
    c = np.cumsum(w[o]) / w.sum()
    return np.interp(np.asarray(q) / 100, c, v[o])


def wks(v, w, yv):
    """KS distance between the time-weighted native distribution and the (uniform) grid distribution."""
    xs = np.unique(np.r_[v, yv])
    o = np.argsort(v)
    fx = np.interp(xs, v[o], np.cumsum(w[o]) / w.sum(), left=0, right=1)
    fy = np.searchsorted(np.sort(yv), xs, side="right") / len(yv)
    return float(np.max(np.abs(fx - fy)))


def check(x, y, wrap, pts):
    """x: native rows as logged, each weighted by the time until the next row (= the value as the source states it);
    y: grid values; pts: the native samples the shape is compared at (the real updates for re-logged columns)."""
    t = x.index.asi8 / 1e9
    xv, yv, pv = x.values.astype(float), y.values.astype(float), pts.values.astype(float)
    dt = np.diff(t)
    w = np.r_[np.minimum(dt, 5 * np.median(dt)), np.median(dt)]   # a data gap does not inflate one value's weight
    mean = np.average(xv, weights=w)
    sd = np.sqrt(np.average((xv - mean) ** 2, weights=w))
    rng = xv.max() - xv.min()
    q = [1, 5, 50, 95, 99]
    tp = pts.index.asi8 / 1e9
    if wrap:   # angles: interpolate on the unwrapped grid series, compare on the circle
        back = np.interp(tp, y.index.asi8 / 1e9, np.degrees(np.unwrap(np.radians(yv))))
        err = np.mod(back - pv + 180, 360) - 180
    else:
        back = np.interp(tp, y.index.asi8 / 1e9, yv)
        err = back - pv
    inside = (tp >= y.index.asi8[0] / 1e9) & (tp <= y.index.asi8[-1] / 1e9)
    pct = lambda d: round(float(d / rng * 100), 3) if rng > 0 else 0.0
    return {
        "native_n": len(xv), "native_hz": round(len(xv) / (t[-1] - t[0]), 1), "update_n": len(pv), "grid_n": len(yv),
        "native_mean": round(float(mean), 4), "grid_mean": round(float(yv.mean()), 4),
        "mean_diff_pct_std": round(float((yv.mean() - mean) / sd * 100), 3) if sd > 0 else 0.0,
        "std_ratio": round(float(yv.std() / sd), 4) if sd > 0 else 1.0,
        "min_diff_pct_range": pct(yv.min() - xv.min()), "max_diff_pct_range": pct(yv.max() - xv.max()),
        "pctl_max_diff_pct_range": pct(np.max(np.abs(np.percentile(yv, q) - wquantile(xv, w, q)))),
        "ks": round(wks(xv, w, yv), 4),
        "shape_nrmse": round(float(np.sqrt(np.mean(err[inside] ** 2)) / sd), 4) if sd > 0 else 0.0,
    }


def flight(out, stem, cfg):
    meta = json.load(open(os.path.join(out, stem, f"{stem}_meta.json")))
    t0, t1 = (pd.Timestamp(x) for x in meta["window_utc"])
    nat = pd.read_parquet(os.path.join(out, stem, f"{stem}_native.parquet"))
    grid = pd.read_csv(os.path.join(out, stem, f"{stem}_analysis.csv"), low_memory=False)
    grid.index = pd.to_datetime(grid["time_gs"], utc=True)
    grid = grid[grid["in_window"] == 1]
    nat = nat[(nat.index >= t0) & (nat.index < t1)]
    rows = []
    for col, how in meta["csv"]["method_per_column"].items():
        if not how.startswith("linear") or col not in nat or col not in grid:
            continue
        x = pd.to_numeric(nat[col], errors="coerce").dropna()
        x = x[~x.index.duplicated(keep="last")]
        period = update_period_s(x) if "updates" in how else None
        pts = update_points(x, period) if period else x   # shape: compare at the real updates, not repeated rows
        y = pd.to_numeric(grid[col], errors="coerce").dropna()
        if len(x) < 10 or len(y) < 10:
            continue
        r = {"flight": stem[18:], "column": col, "method": how, "source": nat.loc[x.index[0], "source"],
             **check(x, y, "unwrapped" in how, pts)}
        if col in POWER:
            k = 1000.0 if POWER[col] == "W" else 1.0
            e_nat = np.trapezoid(x.values, x.index.asi8 / 1e9) / k / 3.6
            e_grid = y.sum() * cfg["grid_ms"] / 1000 / k / 3.6
            r.update(energy_native_wh=round(float(e_nat), 2), energy_grid_wh=round(float(e_grid), 2),
                     energy_diff_pct=round(float((e_grid - e_nat) / abs(e_nat) * 100), 3) if e_nat else None)
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, stem, f"{stem}_resample_check.csv"), index=False)
    figure(os.path.join(out, stem, "figures", f"{stem}_resample_check.png"), stem, nat, grid, t0, t1)
    return res


def figure(path, stem, nat, grid, t0, t1):
    show = [(c, lab) for c, lab in SHOW if c in nat and nat[c].notna().sum() > 100]
    fig, axes = plt.subplots(len(show), 2, figsize=(13, 2.8 * len(show)), gridspec_kw=dict(width_ratios=[2.2, 1]))
    axes = np.atleast_2d(axes)
    for (c, lab), (a1, a2) in zip(show, axes):
        x = nat[c].dropna()
        y = grid[c].dropna()
        # zoom: the 8 s around the largest swing of the signal in the middle of the window
        mid = x[(x.index > t0 + (t1 - t0) * 0.3) & (x.index < t1 - (t1 - t0) * 0.3)]
        tc = mid.diff().abs().rolling(20).sum().idxmax() if len(mid) > 40 else x.index[len(x) // 2]
        z0, z1 = tc - pd.Timedelta(seconds=4), tc + pd.Timedelta(seconds=4)
        xz, yz = x[(x.index >= z0) & (x.index <= z1)], y[(y.index >= z0) & (y.index <= z1)]
        a1.plot(yz.index, yz.values, "-", color=ORANGE, lw=2, label="20 Hz grid (linear)")
        a1.plot(xz.index, xz.values, "o", color=BLUE, ms=5, label="native samples")
        a1.set_ylabel(lab, color=INK2)
        lo, hi = np.nanpercentile(x, 0.5), np.nanpercentile(x, 99.5)
        bins = np.linspace(lo, hi, 50)
        a2.hist(x.clip(lo, hi), bins=bins, density=True, histtype="step", color=BLUE, lw=2, label="native")
        a2.hist(y.clip(lo, hi), bins=bins, density=True, histtype="step", color=ORANGE, lw=2, label="20 Hz")
        for a in (a1, a2):
            a.grid(color=GRID, lw=0.6)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
    axes[0, 0].legend(loc="upper left")
    axes[0, 1].legend(loc="upper right")
    axes[0, 0].set_title("8 s zoom at the largest swing", loc="left")
    axes[0, 1].set_title("value distribution in the window", loc="left")
    fig.suptitle(f"{stem[18:]} — native samples vs the 20 Hz grid", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI_FLIGHT)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", action="append")
    ap.add_argument("--out", default="processed/KTE_2030_FTC_0001")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(CONFIG))
    allres = pd.concat([flight(args.out, s, cfg) for s in (args.event or list(cfg["events"]))])
    os.makedirs(os.path.join(args.out, "campaign"), exist_ok=True)
    allres.to_csv(os.path.join(args.out, "campaign", "resample_check_all.csv"), index=False)
    worst = allres.groupby("source").agg(
        columns=("column", "nunique"), mean_diff_pct_std=("mean_diff_pct_std", lambda v: v.abs().max()),
        std_ratio_min=("std_ratio", "min"), max_cut_pct_range=("max_diff_pct_range", "min"),
        pctl_diff_pct_range=("pctl_max_diff_pct_range", "max"), ks_max=("ks", "max"), shape_nrmse_max=("shape_nrmse", "max"))
    print(worst.to_string())
    if "energy_diff_pct" in allres:
        print(allres.dropna(subset=["energy_diff_pct"])[["flight", "column", "energy_native_wh", "energy_grid_wh",
                                                          "energy_diff_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
