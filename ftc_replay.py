"""
Browser replay of the KTE_2030_FTC_0001 flights: one self-contained HTML page (3D kite + tether, play/scrub,
live readouts, linked charts) for the technical team and for business presentations. No install needed.

Reads <out>/<stem>/<stem>_analysis.csv and _stats.json (ftc_extract.py + ftc_cycles.py), resamples to 5 Hz, and writes
<out>/replay/KTE_2030_FTC_0001_replay.html (template: replay_template.html next to this script).

Kite position: pod-relayed azimuth / elevation / radius from the GS anchor (IPOK-based), converted to
East-North-Up. Flights without IPOK (S1E5) replay without the 3D kite.
Usage: python ftc_replay.py [--event <stem> ...]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

from ftc_extract import CONFIG

RATE_HZ = 5
HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "replay_template.html")
PHASE = {"generation": 1, "recovery": 2}
LINK = {"main": 0, "backup": 1, "not available": 2}
RTK = {"fixed": 0, "float": 1, "no RTK": 2, "no GPS": 3}

# column name in the replay -> (source column or callable, decimals)
COLS = {
    "force": ("OPC_force_sum_kg", 1), "v": ("line_speed_ms", 2), "wl": ("OPC_winded_length_m", 1),
    "pb": ("P_batt_kw", 2), "pm": ("P_mech_kw", 2), "pg": ("P_bus_kw", 2),
    "wind": ("WIND_speed_ms", 1), "wdir": ("WIND_direction_deg", 0),
    "cp": ("CMD_pitch", 1), "cr": ("CMD_roll", 1), "mp": ("POD_pitch_pct", 1), "mr": ("POD_roll_pct", 1),
    "tl": ("POD_servo_left_torque_nm", 1), "tr": ("POD_servo_right_torque_nm", 1),
    "podw": ("POD_power_w", 0), "ks": ("kite_speed_ms", 1),
}


def cfg_dt():
    return yaml.safe_load(open(CONFIG))["grid_ms"] / 1000


def arr(s, nd):
    s = pd.to_numeric(s, errors="coerce").round(nd)
    return [None if pd.isna(x) else (int(x) if nd == 0 else float(x)) for x in s]


def flight(out, stem):
    df = pd.read_csv(os.path.join(out, stem, f"{stem}_analysis.csv"), low_memory=False)
    st = json.load(open(os.path.join(out, stem, f"{stem}_stats.json")))
    w = df[df.in_window == 1].reset_index(drop=True)
    dt = cfg_dt()
    step = int(round(1 / dt / RATE_HZ))
    # cumulative battery energy at full rate, then resample
    w["eb"] = (w["P_batt_kw"].fillna(0) * dt / 3.6).cumsum()
    w["em"] = (w["P_mech_kw"].fillna(0) * dt / 3.6).cumsum()
    s = w.iloc[::step].reset_index(drop=True)
    r, az, el = s["POD_radius_m"], np.radians(s["POD_azimuth_deg"]), np.radians(s["POD_elevation_deg"])
    cols = {k: arr(s[c], nd) for k, (c, nd) in COLS.items()}
    cols["E"] = arr(r * np.cos(el) * np.sin(az), 1)
    cols["N"] = arr(r * np.cos(el) * np.cos(az), 1)
    cols["U"] = arr(r * np.sin(el), 1)
    cols["eb"], cols["em"] = arr(s["eb"], 1), arr(s["em"], 1)
    cols["ph"] = [PHASE.get(x, 0) for x in s["phase"].fillna("")]
    cols["cy"] = [0 if pd.isna(x) else int(x) for x in s["cycle_id"]]
    cols["link"] = [LINK.get(x, 2) for x in s["datalink_state"].fillna("not available")]
    cols["rtk"] = [RTK.get(x, -1) for x in s["rtk_state"].fillna("")]
    has_pos = sum(x is not None for x in cols["E"]) > 0.2 * len(s)
    day = stem[18:26]
    return {
        "stem": stem, "code": stem[27:], "date": f"{day[6:8]}.{day[4:6]}.{day[:4]}",
        "start_rome": s["time_rome"].iloc[0][11:19], "start_ms": int(pd.Timestamp(s["time_rome"].iloc[0]).value // 10**6),
        "rate": RATE_HZ, "n": len(s), "has_pos": bool(has_pos),
        "pod": f"POD {st['pod']['fs_id']} ({st['pod']['pod_version']})",
        "summary": {"duration_min": round(st["duration_s"] / 60, 1), "cycles": st["n_cycles"],
                    "wind_mean": st["wind_ms"]["mean"], "wind_max": st["wind_ms"]["max"],
                    "force_max": st["force_kg_raw"]["max"], "e_batt": st["energy_wh"]["batt"],
                    "e_mech": st["energy_wh"]["mech"], "pod_wh": st["pod_battery"]["energy_used_wh"],
                    "link_main": (st["datalink_status_pct"] or {}).get("main"),
                    "rtk_fixed": ((st["rtk_status_pct"] or {}).get("pct_of_covered_time") or {}).get("fixed")},
        "cols": cols,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", action="append")
    ap.add_argument("--out", default="processed/KTE_2030_FTC_0001")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(CONFIG))
    stems = args.event or list(cfg["events"])
    data = {"campaign": cfg["campaign"], "flights": [flight(args.out, s) for s in stems]}
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    html = open(TEMPLATE).read().replace("/*__DATA__*/null", payload)
    os.makedirs(os.path.join(args.out, "replay"), exist_ok=True)
    path = os.path.join(args.out, "replay", "KTE_2030_FTC_0001_replay.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB, {len(stems)} flights)")


if __name__ == "__main__":
    main()
