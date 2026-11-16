"""
Load cells of KTE_2030_FTC_0001: how the four cells (OPC_DsLoadCells.MeasureFloat[1..4]) share the total
(MeasureFloat_SUM, the exact sum of the four) along the tether length on the drum (outTamburo_WindedLength).
All flights on one page: campaign/loadcells.html.

Two views of the cells (switch on the page):
  raw   = cells as logged. A share below 0 % or above 100 % points to a cell offset, not to a real load.
  tared = each cell minus its reading at the pre-flight slack moment: the steadiest 10 s with the drum stopped in the
          2 min extracted before the window. That reading may hold some line tension at rest; shares are unaffected
          as long as the cells split it like the flight load.
Offsets are also taken after the window (drift = after - before), and each cell is fitted in flight as
  cell = a + b*sum + c*sum*(WindedLength - 150 m)/100       (sum > 50 kg)
b is the share at 150 m and c its change per 100 m, both independent of any constant offset. If the pre-flight tare
holds through the flight, a = o - b*O (o = cell offset, O = summed offsets); a larger mismatch means the offsets
drifted during the flight and only the regression shares are reliable.

Only rows with the load-cell sum above a threshold (50 and 100 kg, both shown) inside the official window.
Drum direction from line_speed_ms: reel-out > +0.2 m/s, reel-in < -0.2 m/s, else stopped (in "all" only).
Each curve is the median per 5 m bin of WindedLength with the 25-75 % band; bins with < 1 s of data are left out.

Writes campaign/loadcells.html, campaign/KTE_2030_FTC_0001_loadcells_*.png, campaign/loadcell_bins.csv and
campaign/loadcell_offsets.csv.
Usage: python ftc_loadcells.py [--out processed/KTE_2030_FTC_0001]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ftc_figures import C1, C2, C3, C4, CFG, DAY, DPI_CAMPAIGN, DT_S, GRID, INK, INK2, SURF

CELLS = [f"OPC_force_{i}_kg" for i in range(1, 5)]
CELL = {c: (f"cell {i}", col) for i, (c, col) in enumerate(zip(CELLS, (C1, C2, C3, C4)), start=1)}
THRESHOLDS = (50, 100)
LEN_BIN_M, FORCE_BIN_KG = 5, 25
MIN_N = int(round(1 / DT_S))           # 1 s of data per bin
MOVING_MS = 0.2
DASH = {"reel-out": "solid", "reel-in": "dash", "all": "solid"}
FLIGHT_DASH = ("solid", "dash", "dot")   # flights of the same day in view B
SLACK_S, SLACK_STOP_MS = 10, 0.05         # slack moment: 10 s with |line speed| < 0.05 m/s, least sum variation
N_SLACK = int(round(SLACK_S / DT_S))
FIT_KG, WL_REF_M = 50, 150                # regression: sum > 50 kg, share referenced to 150 m
DRIFT_KG = 8                              # tare check: max |predicted - fitted intercept| above this = drifted
# cell column and sum column per view
TARE = {"raw": (lambda c: c, "OPC_force_sum_kg", "raw cells"),
        "tared": (lambda c: f"tared_{c}", "tared_sum_kg", "tared with the pre-flight slack moment")}


def slack_moments(r):
    """Steadiest SLACK_S with the drum stopped before and after the window; cell readings there = offsets."""
    sd = r["OPC_force_sum_kg"].rolling(N_SLACK, min_periods=N_SLACK).std()
    still = (r["line_speed_ms"].abs().rolling(N_SLACK, min_periods=N_SLACK).max() < SLACK_STOP_MS).to_numpy()
    win = r.index[r["in_window"] == 1]
    res = {}
    for side, mask in (("before", r.index < win[0]), ("after", r.index > win[-1])):
        cand = sd[mask & still].dropna()
        if cand.empty:
            res[side] = None
            continue
        k = r.index.get_loc(cand.idxmin())
        seg = r.iloc[k - N_SLACK + 1:k + 1]
        res[side] = {"end": seg.index[-1], "sum": seg["OPC_force_sum_kg"].mean(), "sd": seg["OPC_force_sum_kg"].std(),
                     "cells": seg[CELLS].mean().to_numpy()}
    return res


def load(out, stem):
    cols = ["time_rome", "in_window", "OPC_winded_length_m", "line_speed_ms", "OPC_force_sum_kg", *CELLS]
    r = pd.read_csv(os.path.join(out, stem, f"{stem}_analysis.csv"), usecols=cols, low_memory=False)
    r.index = pd.to_datetime(r.pop("time_rome"))
    slack = slack_moments(r)
    d = r[r["in_window"] == 1].dropna(subset=["OPC_winded_length_m", "OPC_force_sum_kg"]).copy()
    v = d["line_speed_ms"]
    d["dir"] = np.select([v > MOVING_MS, v < -MOVING_MS], ["reel-out", "reel-in"], "stopped")
    o = slack["before"]["cells"] if slack["before"] else np.zeros(len(CELLS))
    for c, oc in zip(CELLS, o):
        d[f"tared_{c}"] = d[c] - oc
    d["tared_sum_kg"] = d["OPC_force_sum_kg"] - o.sum()
    for tare, (cc, sc, _) in TARE.items():
        for c in CELLS:
            d[f"share_{tare}_{c}"] = d[cc(c)] / d[sc] * 100
    return d, slack


def regression(d):
    """cell = a + b*sum + c*sum*(WL - WL_REF)/100 per cell: a [kg], b [%], c [% per 100 m], R2."""
    w = d[d["OPC_force_sum_kg"] > FIT_KG].dropna(subset=CELLS)
    s = w["OPC_force_sum_kg"].to_numpy()
    X = np.c_[np.ones(len(w)), s, s * (w["OPC_winded_length_m"].to_numpy() - WL_REF_M) / 100]
    out = []
    for c in CELLS:
        y = w[c].to_numpy()
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out.append((coef[0], coef[1] * 100, coef[2] * 100, r2))
    return out


def offsets_table(flights, slacks):
    """One row per flight: slack moments, offsets before/after, regression and the tare check."""
    rows = []
    for stem, d in flights.items():
        sl, fit = slacks[stem], regression(d)
        r = {"stem": stem}
        for side in ("before", "after"):
            m = sl[side]
            r[f"{side}_end"] = m["end"].strftime("%H:%M:%S") if m else None
            r[f"{side}_sum_kg"] = round(m["sum"], 1) if m else None
            r[f"{side}_sd_kg"] = round(m["sd"], 2) if m else None
            for i, c in enumerate(CELLS):
                r[f"{side}_{CELL[c][0]}_kg"] = round(m["cells"][i], 1) if m else None
        for i, c in enumerate(CELLS):
            a, b, k, q = fit[i]
            n = CELL[c][0]
            r.update({f"fit_{n}_a_kg": round(a, 1), f"fit_{n}_share_pct": round(b, 1),
                      f"fit_{n}_per100m": round(k, 1), f"fit_{n}_r2": round(q, 2)})
            if sl["before"]:
                o = sl["before"]["cells"]
                r[f"pred_{n}_a_kg"] = round(o[i] - b / 100 * o.sum(), 1)
        if sl["before"]:
            r["tare_mismatch_kg"] = round(max(abs(r[f"pred_{CELL[c][0]}_a_kg"] - r[f"fit_{CELL[c][0]}_a_kg"])
                                              for c in CELLS), 1)
            r["tare_ok"] = r["tare_mismatch_kg"] <= DRIFT_KG
        else:
            r["tare_mismatch_kg"], r["tare_ok"] = None, False
        rows.append(r)
    return pd.DataFrame(rows)


def binned(d, x, edges, ycol):
    b = pd.cut(d[x], edges)
    g = d.groupby(b, observed=True)[ycol]
    t = pd.DataFrame({"n": g.size(), "median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75)})
    t = t.reindex(pd.IntervalIndex.from_breaks(edges))   # every bin, so a line breaks where data is missing
    t["x"] = [i.mid for i in t.index]
    t.loc[t["n"].fillna(0) < MIN_N, ["median", "q25", "q75"]] = np.nan
    t["n"] = t["n"].fillna(0).astype(int)
    first, last = t["median"].first_valid_index(), t["median"].last_valid_index()
    return t.loc[first:last].reset_index(drop=True) if first is not None else t.iloc[:0].reset_index(drop=True)


def tables(flights):
    """Long table: tare (raw / tared), view (length / force), metric (share / kg), threshold, flight, dir, cell, x, n,
    median, q25, q75. The threshold applies to the sum of the same view (raw or tared)."""
    lmax = max(d["OPC_winded_length_m"].max() for d in flights.values())
    ledges = np.arange(0, lmax + LEN_BIN_M, LEN_BIN_M)
    fmax = max(max(d["OPC_force_sum_kg"].max(), d["tared_sum_kg"].max()) for d in flights.values())
    fedges = np.arange(FORCE_BIN_KG, fmax + FORCE_BIN_KG, FORCE_BIN_KG)
    rows = []
    for tare, (cc, sc, _) in TARE.items():
        for stem, d in flights.items():
            for thr in THRESHOLDS:
                h = d[d[sc] > thr]
                for dr in ("reel-out", "reel-in", "all"):
                    hd = h if dr == "all" else h[h["dir"] == dr]
                    for c in CELLS:
                        for metric, ycol in (("share", f"share_{tare}_{c}"), ("kg", cc(c))):
                            t = binned(hd, "OPC_winded_length_m", ledges, ycol)
                            rows.append(t.assign(tare=tare, view="length", metric=metric, threshold=thr, stem=stem,
                                                 dir=dr, cell=c))
            h = d[d[sc] > FORCE_BIN_KG]
            for c in CELLS:
                t = binned(h, sc, fedges, f"share_{tare}_{c}")
                rows.append(t.assign(tare=tare, view="force", metric="share", threshold=FORCE_BIN_KG, stem=stem,
                                     dir="all", cell=c))
    t = pd.concat(rows, ignore_index=True)
    return t[["tare", "view", "metric", "threshold", "stem", "dir", "cell", "x", "n", "median", "q25", "q75"]]


def summary(flights, t, tare):
    """Per flight and threshold: time above it, median share per cell, and its trend in % per 100 m of WindedLength."""
    _, sc, _ = TARE[tare]
    rows = []
    for stem, d in flights.items():
        for thr in THRESHOLDS:
            h = d[d[sc] > thr]
            r = {"stem": stem, "threshold": thr, "above_pct": round(len(h) / len(d) * 100, 1)}
            for c in CELLS:
                r[f"{CELL[c][0]}_median"] = round(float(h[f"share_{tare}_{c}"].median()), 1)
                b = t[(t.view == "length") & (t.metric == "share") & (t.threshold == thr) & (t.stem == stem)
                      & (t.dir == "all") & (t.cell == c)]
                r[f"{CELL[c][0]}_per100m"] = round(float(np.polyfit(b["x"], b["median"], 1, w=np.sqrt(b["n"]))[0] * 100), 1) \
                    if len(b) > 2 else None
            rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- layout: one row per test day


def grid_positions(stems):
    days = list(DAY)
    pos, col = {}, {d: 0 for d in days}
    for s in stems:
        day = s[18:26]
        pos[s] = (days.index(day) + 1, col[day] + 1)
        col[day] += 1
    return pos, len(days), max(col.values())


def label(stem):
    return f"{DAY[stem[18:26]][0]} · {stem[27:]}"


def band(fig, b, color, name, legend, row, col, dash, hover):
    fig.add_trace(go.Scatter(x=b["x"], y=b["q75"], mode="lines", line=dict(width=0), hoverinfo="skip",
                             showlegend=False, legendgroup=name), row=row, col=col)
    fig.add_trace(go.Scatter(x=b["x"], y=b["q25"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=hex_rgba(color, 0.12), hoverinfo="skip", showlegend=False, legendgroup=name),
                  row=row, col=col)
    fig.add_trace(go.Scatter(x=b["x"], y=b["median"].round(1), mode="lines", name=name, legendgroup=name,
                             showlegend=legend, line=dict(color=color, width=2, dash=dash),
                             customdata=np.stack([b["q25"].round(1), b["q75"].round(1), b["n"] * DT_S], axis=1),
                             hovertemplate=hover), row=row, col=col)


def hex_rgba(h, a):
    return f"rgba({int(h[1:3], 16)},{int(h[3:5], 16)},{int(h[5:7], 16)},{a})"


def style(fig, height):
    fig.update_layout(template="plotly_white", paper_bgcolor=SURF, plot_bgcolor=SURF, height=height,
                      font=dict(family="Inter, Helvetica, Arial, sans-serif", size=12, color=INK2),
                      legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0, xanchor="left"),
                      margin=dict(l=60, r=20, t=60, b=50), hovermode="closest")
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=INK2, ticks="outside", showticklabels=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=INK2, ticks="outside", showticklabels=True)
    for a in fig.layout.annotations:   # subplot titles
        a.font = dict(size=12, color=INK)
    return fig


def fig_per_flight(t, stems, view, metric, thr, dirs):
    """View A / C / D: one panel per flight (rows = test days), the four cells in each."""
    pos, nr, nc = grid_positions(stems)
    specs = [[{} if (r, c) in pos.values() else None for c in range(1, nc + 1)] for r in range(1, nr + 1)]
    order = sorted(stems, key=lambda s: pos[s])
    fig = make_subplots(rows=nr, cols=nc, specs=specs, subplot_titles=[label(s) for s in order],
                        horizontal_spacing=0.05, vertical_spacing=0.09)
    unit = "%" if metric == "share" else "kg"
    xl = "WindedLength" if view == "length" else "sum"
    xu = "m" if view == "length" else "kg"
    first = True
    for s in order:
        r, c = pos[s]
        for cell in CELLS:
            name, color = CELL[cell]
            for dr in dirs:
                b = t[(t.view == view) & (t.metric == metric) & (t.threshold == thr) & (t.stem == s) & (t.dir == dr)
                      & (t.cell == cell)]
                if b.empty:
                    continue
                band(fig, b, color, name, first and dr == dirs[0], r, c, DASH[dr],
                     f"{xl} %{{x:.0f}} {xu}: median %{{y:.1f}} {unit}<br>25-75 %: %{{customdata[0]}} to "
                     f"%{{customdata[1]}} {unit}<br>%{{customdata[2]:.0f}} s of data<extra>{name} {dr if dr != 'all' else ''}</extra>")
        first = False
        if metric == "share":
            for y in (0, 100):
                fig.add_hline(y=y, line=dict(color=INK2, width=1, dash="dot"), row=r, col=c)
        if view == "force":
            for x in THRESHOLDS:
                fig.add_vline(x=x, line=dict(color=INK2, width=1, dash="dot"), row=r, col=c)
    if len(dirs) > 1:   # legend entries for the line style
        for dr in dirs:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=dr, line=dict(color=INK2, width=2, dash=DASH[dr])),
                          row=1, col=1)
    fig.update_xaxes(matches="x")
    fig.update_yaxes(matches="y")
    for s in order:
        r, c = pos[s]
        if r == nr or (r + 1, c) not in pos.values():
            fig.update_xaxes(title_text="WindedLength [m]" if view == "length" else "load-cell sum [kg]", row=r, col=c)
        if c == 1:
            fig.update_yaxes(title_text="cell / sum [%]" if metric == "share" else "cell [kg]", row=r, col=c)
    if metric == "share":
        fig.update_yaxes(range=[-40, 130])
    return style(fig, 300 * nr + 110)


def fig_per_cell(t, stems, thr):
    """View B: one panel per cell, every flight in it (colour = test day, line style = flight of that day)."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=[CELL[c][0] for c in CELLS], shared_xaxes=True,
                        horizontal_spacing=0.06, vertical_spacing=0.1)
    k = {}
    for i, cell in enumerate(CELLS):
        r, c = i // 2 + 1, i % 2 + 1
        for s in stems:
            day = s[18:26]
            j = k.setdefault(s, sum(1 for x in k if x[18:26] == day))
            b = t[(t.view == "length") & (t.metric == "share") & (t.threshold == thr) & (t.stem == s) & (t.dir == "all")
                  & (t.cell == cell)]
            fig.add_trace(go.Scatter(x=b["x"], y=b["median"].round(1), mode="lines", name=label(s), legendgroup=s,
                                     showlegend=i == 0, line=dict(color=DAY[day][1], width=2, dash=FLIGHT_DASH[j % 3]),
                                     customdata=np.stack([b["q25"].round(1), b["q75"].round(1)], axis=1),
                                     hovertemplate="WindedLength %{x:.0f} m: median %{y:.1f} %<br>25-75 %: "
                                                   f"%{{customdata[0]}} to %{{customdata[1]}} %<extra>{label(s)}</extra>"),
                          row=r, col=c)
        for y in (0, 100):
            fig.add_hline(y=y, line=dict(color=INK2, width=1, dash="dot"), row=r, col=c)
        if r == 2:
            fig.update_xaxes(title_text="WindedLength [m]", row=r, col=c)
        if c == 1:
            fig.update_yaxes(title_text="cell / sum [%]", row=r, col=c)
    fig.update_yaxes(range=[-40, 130])
    return style(fig, 720)


# --------------------------------------------------------------------------- static PNG (matplotlib, same tables)


def png_per_flight(t, stems, view, metric, thr, dirs, title, path):
    pos, nr, nc = grid_positions(stems)
    f, axes = plt.subplots(nr, nc, figsize=(5.4 * nc, 4.0 * nr), sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for s in stems:
        r, c = pos[s]
        ax = axes[r - 1, c - 1]
        ax.set_visible(True)
        for cell in CELLS:
            name, color = CELL[cell]
            for dr in dirs:
                b = t[(t.view == view) & (t.metric == metric) & (t.threshold == thr) & (t.stem == s) & (t.dir == dr)
                      & (t.cell == cell)]
                ax.fill_between(b["x"], b["q25"], b["q75"], color=color, alpha=0.12, lw=0)
                ax.plot(b["x"], b["median"], color=color, lw=2, ls="-" if DASH[dr] == "solid" else "--",
                        label=f"{name}" + (f" {dr}" if len(dirs) > 1 else ""))
        if metric == "share":
            for y in (0, 100):
                ax.axhline(y, color=INK2, lw=0.8, ls=":")
            ax.set_ylim(-40, 130)
        if view == "force":
            for x in THRESHOLDS:
                ax.axvline(x, color=INK2, lw=0.8, ls=":")
        ax.set_title(label(s), fontsize=13, color=INK, loc="left")
        ax.grid(color=GRID, lw=0.6)
        ax.set_facecolor(SURF)
        ax.tick_params(labelbottom=True, labelleft=True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if c == 1:
            ax.set_ylabel("cell / sum [%]" if metric == "share" else "cell [kg]", color=INK2)
        ax.set_xlabel("WindedLength [m]" if view == "length" else "load-cell sum [kg]", color=INK2)
    h, l = axes[0, 0].get_legend_handles_labels()
    f.suptitle(title, x=0.01, y=0.995, ha="left", color=INK, fontsize=14)
    f.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.96), ncol=8, frameon=False)
    f.patch.set_facecolor(SURF)
    f.tight_layout(rect=(0, 0, 1, 0.935))
    f.savefig(path, dpi=DPI_CAMPAIGN)
    plt.close(f)


def png_per_cell(t, stems, thr, title, path):
    f, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    k = {}
    for ax, cell in zip(axes.flat, CELLS):
        for s in stems:
            day = s[18:26]
            j = k.setdefault(s, sum(1 for x in k if x[18:26] == day))
            b = t[(t.view == "length") & (t.metric == "share") & (t.threshold == thr) & (t.stem == s) & (t.dir == "all")
                  & (t.cell == cell)]
            ax.plot(b["x"], b["median"], color=DAY[day][1], lw=2, ls=("-", "--", ":")[j % 3], label=label(s))
        for y in (0, 100):
            ax.axhline(y, color=INK2, lw=0.8, ls=":")
        ax.set_ylim(-40, 130)
        ax.set_title(CELL[cell][0], color=INK, loc="left")
        ax.grid(color=GRID, lw=0.6)
        ax.set_facecolor(SURF)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("WindedLength [m]", color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("cell / sum [%]", color=INK2)
    axes[0, 1].legend(frameon=True, facecolor=SURF, edgecolor=GRID, loc="upper left", bbox_to_anchor=(1.01, 1))
    f.suptitle(title, x=0.01, ha="left", color=INK, fontsize=14)
    f.patch.set_facecolor(SURF)
    f.tight_layout()
    f.savefig(path, dpi=DPI_CAMPAIGN)
    plt.close(f)


# --------------------------------------------------------------------------- offsets (view E)


def fig_offsets(off, stems):
    """One panel per cell: offset at the slack moment before (circle) and after (diamond) each flight."""
    fig = make_subplots(rows=1, cols=4, subplot_titles=[CELL[c][0] for c in CELLS], shared_yaxes=True,
                        horizontal_spacing=0.03)
    labels = [label(s) for s in stems]
    for i, c in enumerate(CELLS):
        n = CELL[c][0]
        for s, lab in zip(stems, labels):
            r = off[off.stem == s].iloc[0]
            y = [r[f"before_{n}_kg"], r[f"after_{n}_kg"]]
            col = DAY[s[18:26]][1]
            fig.add_trace(go.Scatter(x=[lab, lab], y=y, mode="lines+markers", showlegend=False,
                                     line=dict(color=col, width=2),
                                     marker=dict(symbol=["circle", "diamond"], size=10, color=col,
                                                 line=dict(color=SURF, width=2)),
                                     customdata=[["before", r["before_end"]], ["after", r["after_end"]]],
                                     hovertemplate="%{customdata[0]} (%{customdata[1]}): %{y:.1f} kg"
                                                   f"<extra>{lab} {n}</extra>"), row=1, col=i + 1)
        fig.add_hline(y=0, line=dict(color=INK2, width=1, dash="dot"), row=1, col=i + 1)
    for name, sym in (("before the flight", "circle"), ("after the flight", "diamond")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(symbol=sym, size=10, color=INK2)), row=1, col=1)
    fig.update_yaxes(title_text="cell at rest [kg]", row=1, col=1)
    fig.update_xaxes(tickangle=-45)
    return style(fig, 420)


def png_offsets(off, stems, path):
    f, axes = plt.subplots(1, 4, figsize=(16, 5.6), sharey=True)
    labels = [label(s) for s in stems]
    for ax, c in zip(axes, CELLS):
        n = CELL[c][0]
        for k, (s, lab) in enumerate(zip(stems, labels)):
            r = off[off.stem == s].iloc[0]
            col = DAY[s[18:26]][1]
            ax.plot([k, k], [r[f"before_{n}_kg"], r[f"after_{n}_kg"]], color=col, lw=2)
            ax.plot(k, r[f"before_{n}_kg"], "o", color=col, ms=10, mec=SURF, mew=1.5, label="before" if k == 0 else None)
            ax.plot(k, r[f"after_{n}_kg"], "D", color=col, ms=9, mec=SURF, mew=1.5, label="after" if k == 0 else None)
        ax.axhline(0, color=INK2, lw=0.8, ls=":")
        ax.set_xticks(range(len(stems)), labels, rotation=45, ha="right", fontsize=10)
        ax.set_title(n, color=INK, loc="left")
        ax.grid(color=GRID, lw=0.6)
        ax.set_facecolor(SURF)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("cell at rest [kg]", color=INK2)
    axes[0].legend(frameon=False)
    f.suptitle("Load-cell offsets at the slack moments (10 s, drum stopped) before (circle) and after (diamond) "
               "each flight", x=0.01, ha="left", color=INK, fontsize=14)
    f.patch.set_facecolor(SURF)
    f.tight_layout()
    f.savefig(path, dpi=DPI_CAMPAIGN)
    plt.close(f)


# --------------------------------------------------------------------------- page


def page(out, figs, pngs, summ, off, stems):
    drifted = set(off.loc[~off["tare_ok"].astype(bool), "stem"])

    def cell_td(r, c):
        med, slope = r[f"{CELL[c][0]}_median"], r[f"{CELL[c][0]}_per100m"]
        bad = "bad" if not -5 <= med <= 105 else ""
        trend = "" if pd.isna(slope) else f"{slope:+.0f}"
        return f"<td class='{bad}'>{med:.0f}<span>{trend}</span></td>"

    def flight_td(stem, tare):
        warn = "<em class='warn'>offset drifted</em>" if tare == "tared" and stem in drifted else ""
        return f"<td>{label(stem)}{warn}</td>"

    def table(tare, thr):
        s = summ[tare][summ[tare].threshold == thr]
        head = "".join(f"<th>{CELL[c][0]}</th>" for c in CELLS)
        body = "".join(f"<tr>{flight_td(r.stem, tare)}<td>{r.above_pct:.0f}</td>"
                       + "".join(cell_td(r, c) for c in CELLS) + "</tr>" for _, r in s.iterrows())
        return (f"<div class='t'><table><thead><tr><th>flight</th><th>time above [%]</th>{head}</tr></thead>"
                f"<tbody>{body}</tbody></table></div>")

    def offsets_html():
        head = "".join(f"<th>{CELL[c][0]}</th>" for c in CELLS)
        body = ""
        for _, r in off.iterrows():
            def cells(side):
                return "".join(f"<td>{r[f'{side}_{CELL[c][0]}_kg']:+.1f}</td>" for c in CELLS)
            drift = "".join(
                f"<td class='{'bad' if abs(d) >= 10 else ''}'>{d:+.1f}</td>"
                for d in (r[f"after_{CELL[c][0]}_kg"] - r[f"before_{CELL[c][0]}_kg"] for c in CELLS))
            body += (f"<tr><td rowspan='3'>{label(r.stem)}</td><td class='l'>before · {r.before_end}</td>"
                     f"<td>{r.before_sum_kg:.1f}</td>{cells('before')}</tr>"
                     f"<tr><td class='l'>after · {r.after_end}</td><td>{r.after_sum_kg:.1f}</td>{cells('after')}</tr>"
                     f"<tr class='sub'><td class='l'>drift</td><td></td>{drift}</tr>")
        return (f"<div class='t'><table><thead><tr><th>flight</th><th class='l'>moment · end time</th>"
                f"<th>sum [kg]</th>{head}</tr></thead><tbody>{body}</tbody></table></div>")

    def regression_html():
        head = "".join(f"<th>{CELL[c][0]}</th>" for c in CELLS)
        body = ""
        for _, r in off.iterrows():
            cells = "".join(
                f"<td>{r[f'fit_{CELL[c][0]}_share_pct']:.0f}<span>{r[f'fit_{CELL[c][0]}_per100m']:+.0f}</span>"
                f"<small>R² {r[f'fit_{CELL[c][0]}_r2']:.2f}</small></td>" for c in CELLS)
            ok = r["tare_ok"]
            verdict = (f"<td class='{'ok' if ok else 'bad'}'>{r.tare_mismatch_kg:.0f} kg · "
                       f"{'tare holds' if ok else 'offset drifted, use regression'}</td>")
            body += f"<tr><td>{label(r.stem)}</td>{cells}{verdict}</tr>"
        return (f"<div class='t'><table><thead><tr><th>flight</th>{head}<th>tare check</th></tr></thead>"
                f"<tbody>{body}</tbody></table></div>")

    def section(key):
        return f"{figs[key]}<p class='png'><a href='{pngs[key]}'>PNG for the report</a></p>"

    blocks = ""
    for tare, (_, _, desc) in TARE.items():
        for thr in THRESHOLDS:
            k = f"{tare}_{thr}"
            hide = "" if (tare, thr) == ("raw", THRESHOLDS[0]) else "hidden"
            blocks += f"""<div class="blk" data-tare="{tare}" data-thr="{thr}" {hide}>
<h2>Summary, sum &gt; {thr} kg, {desc}</h2>
<p class="note">Median share of each cell [%]; small number = change in percentage points per 100 m of WindedLength.
Shares outside 0–100 % are marked: a cell cannot carry more than the total, so these point to cell offsets.</p>
{table(tare, thr)}
<h2>A. Share of each cell along WindedLength, per flight</h2>
<p class="note">Line = median per {LEN_BIN_M} m bin, band = 25–75 %. Solid = reel-out, dashed = reel-in.</p>
{section(f'A_{k}')}
<h2>B. Each cell across all flights</h2>
<p class="note">Median share per {LEN_BIN_M} m bin, all drum directions. Colour = test day, line style = flight of that day.</p>
{section(f'B_{k}')}
<h2>C. Cell load in kg along WindedLength, per flight</h2>
<p class="note">Same as A in kg instead of share: shows whether a share changes because load moves between cells or
because of an offset (an offset is a constant kg shift).</p>
{section(f'C_{k}')}
</div>"""
        blocks += f"""<div class="blk" data-tare="{tare}" {'hidden' if tare != 'raw' else ''}>
<h2>D. Share of each cell vs total load, per flight ({desc})</h2>
<p class="note">Does the split depend on how hard the tether pulls? Median per {FORCE_BIN_KG} kg bin of the sum, all
drum directions and lengths. Dotted lines = the 50 and 100 kg thresholds; below them offsets dominate the shares.</p>
{section(f'D_{tare}')}
</div>"""

    def radios(name, items, first):
        return "".join(f"<label><input type='radio' name='{name}' value='{v}' {'checked' if v == first else ''}> {t}</label>"
                       for v, t in items)

    bar = (f"<span class='grp'><b>Cells:</b>{radios('tare', [('raw', 'raw'), ('tared', 'tared (pre-flight slack)')], 'raw')}</span>"
           f"<span class='grp'><b>Threshold:</b>{radios('thr', [(t, f'sum &gt; {t} kg') for t in THRESHOLDS], THRESHOLDS[0])}</span>")
    drifted_txt = ", ".join(label(s) for s in stems if s in drifted) or "none"
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>KTE_2030_FTC_0001 load cells</title>
<script src="plotly.min.js"></script>
<style>body{{font-family:Inter,Helvetica,Arial,sans-serif;background:{SURF};color:{INK};margin:0;padding:24px 16px;max-width:1500px}}
h1{{font-size:21px;margin:0 0 6px}} h2{{font-size:16px;margin:28px 0 4px}} p{{color:{INK2};max-width:950px;line-height:1.45}}
.note{{font-size:13px;margin:2px 0 8px}} a{{color:{C1}}} .png{{font-size:12px;margin:0}}
.bar{{position:sticky;top:0;background:{SURF};padding:8px 0;border-bottom:1px solid {GRID};z-index:5;display:flex;gap:28px;flex-wrap:wrap}}
.grp{{display:flex;gap:14px;flex-wrap:wrap;align-items:center}} .bar label{{font-size:14px;cursor:pointer}}
.t{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}}
th,td{{padding:5px 12px;border-bottom:1px solid {GRID};text-align:right;vertical-align:top}}
th:first-child,td:first-child,.l{{text-align:left}} th{{color:{INK2};font-weight:600}}
td span{{color:{INK2};font-size:11px;margin-left:6px}} td small{{display:block;color:{INK2};font-size:10px}}
tr.sub td{{color:{INK2};font-size:12px}} td.bad,tr.sub td.bad{{color:#c0392b;font-weight:600}} td.ok{{color:#1e7d4f}}
em.warn{{display:block;font-style:normal;font-size:11px;color:#c0392b}}
.key{{border-left:3px solid {C1};padding:6px 12px;margin:10px 0;max-width:950px;color:{INK}}}
.key li{{margin:3px 0;line-height:1.4}}</style>
</head><body>
<h1>Load cells — share of the total along the tether length</h1>
<p>Each of the four load cells (MeasureFloat[1..4]) as a share of their sum (MeasureFloat_SUM) against the tether on
the drum (outTamburo_WindedLength), all {len(stems)} flights, inside the official window and only when the sum is above
the selected threshold. The physical position of cells 1–4 is still to be checked.</p>
<div class="key"><b>Key points</b><ul>
<li>The cells split the load the same way on all three days: from the regression, cell 2 carries about three quarters
and cell 3 about a fifth at 150 m, and load moves from cell 3 to cell 2 as the tether winds on. Differences between days
in the raw shares are offsets.</li>
<li>Offsets at rest are small and stable on 28 Oct, large and drifting on 22 Oct (cell 3 from +14 to −37 kg over the day).</li>
<li>The pre-flight tare holds through the flight except for: {drifted_txt}. For those, use the regression shares.</li>
</ul></div>
<h2>E. Offsets at the slack moments</h2>
<p class="note">Steadiest {SLACK_S} s with the drum stopped (|line speed| &lt; {SLACK_STOP_MS} m/s) in the 2 min extracted
before and after each window. The reading there is offset plus any line tension at rest. The tared view subtracts the
<i>before</i> values. Drift of 10 kg or more is marked.</p>
{section('E')}
{offsets_html()}
<h2>Regression: shares without offsets</h2>
<p class="note">Each cell fitted in flight as <i>cell = a + b·sum + c·sum·(WindedLength − {WL_REF_M} m)/100</i>, sum &gt;
{FIT_KG} kg. Big number = b, the share at {WL_REF_M} m [%]; small number = c, its change per 100 m. Neither depends on a
constant offset. Tare check = largest gap between the intercept a predicted from the pre-flight offsets and the fitted
one; above {DRIFT_KG} kg the offsets moved during the flight. Cells 1 and 4 carry little load, so their R² is low.</p>
{regression_html()}
<div class="bar">{bar}</div>
{blocks}
<p class="note">Data: <a href="loadcell_bins.csv">loadcell_bins.csv</a> (per bin) and
<a href="loadcell_offsets.csv">loadcell_offsets.csv</a> (offsets and regression per flight).</p>
<script>
function show() {{
  const tare = document.querySelector("input[name=tare]:checked").value;
  const thr = document.querySelector("input[name=thr]:checked").value;
  document.querySelectorAll(".blk").forEach(d => {{
    d.hidden = d.dataset.tare !== tare || (d.dataset.thr !== undefined && d.dataset.thr !== thr);
    if (!d.hidden) d.querySelectorAll(".js-plotly-plot").forEach(g => Plotly.Plots.resize(g));
  }});
}}
document.querySelectorAll(".bar input").forEach(r => r.addEventListener("change", show));
</script></body></html>"""
    with open(os.path.join(out, "campaign", "loadcells.html"), "w") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="processed/KTE_2030_FTC_0001")
    args = ap.parse_args()
    stems = list(CFG["events"])
    flights, slacks = {}, {}
    for s in stems:
        flights[s], slacks[s] = load(args.out, s)
    t = tables(flights)
    camp = os.path.join(args.out, "campaign")
    t.round(3).to_csv(os.path.join(camp, "loadcell_bins.csv"), index=False)
    off = offsets_table(flights, slacks)
    off.to_csv(os.path.join(camp, "loadcell_offsets.csv"), index=False)
    summ = {tare: summary(flights, t[t["tare"] == tare], tare) for tare in TARE}
    figs, pngs = {}, {}
    cfg = dict(include_plotlyjs=False, full_html=False, config={"displaylogo": False, "responsive": True})
    pre = "KTE_2030_FTC_0001_loadcells"
    for tare, (_, _, desc) in TARE.items():
        tt = t[t["tare"] == tare]
        suf = "" if tare == "raw" else "_tared"
        for thr in THRESHOLDS:
            k, sub = f"{tare}_{thr}", f"sum > {thr} kg, {desc}"
            figs[f"A_{k}"] = fig_per_flight(tt, stems, "length", "share", thr, ("reel-out", "reel-in")).to_html(**cfg)
            pngs[f"A_{k}"] = f"{pre}_share_vs_length_per_flight_{thr}kg{suf}.png"
            png_per_flight(tt, stems, "length", "share", thr, ("reel-out", "reel-in"),
                           f"Load-cell share along WindedLength per flight ({sub}); solid = reel-out, dashed = reel-in",
                           os.path.join(camp, pngs[f"A_{k}"]))
            figs[f"B_{k}"] = fig_per_cell(tt, stems, thr).to_html(**cfg)
            pngs[f"B_{k}"] = f"{pre}_share_vs_length_per_cell_{thr}kg{suf}.png"
            png_per_cell(tt, stems, thr, f"Each load cell across all flights ({sub})", os.path.join(camp, pngs[f"B_{k}"]))
            figs[f"C_{k}"] = fig_per_flight(tt, stems, "length", "kg", thr, ("reel-out", "reel-in")).to_html(**cfg)
            pngs[f"C_{k}"] = f"{pre}_kg_vs_length_per_flight_{thr}kg{suf}.png"
            png_per_flight(tt, stems, "length", "kg", thr, ("reel-out", "reel-in"),
                           f"Load-cell kg along WindedLength per flight ({sub}); solid = reel-out, dashed = reel-in",
                           os.path.join(camp, pngs[f"C_{k}"]))
        figs[f"D_{tare}"] = fig_per_flight(tt, stems, "force", "share", FORCE_BIN_KG, ("all",)).to_html(**cfg)
        pngs[f"D_{tare}"] = f"{pre}_share_vs_sum_per_flight{suf}.png"
        png_per_flight(tt, stems, "force", "share", FORCE_BIN_KG, ("all",),
                       f"Load-cell share vs total load per flight ({desc})", os.path.join(camp, pngs[f"D_{tare}"]))
    figs["E"] = fig_offsets(off, stems).to_html(**cfg)
    pngs["E"] = f"{pre}_offsets.png"
    png_offsets(off, stems, os.path.join(camp, pngs["E"]))
    page(args.out, figs, pngs, summ, off, stems)
    for tare in TARE:
        print(f"--- {tare}\n" + summ[tare].to_string(index=False))
    print(off[["stem", "tare_mismatch_kg", "tare_ok"]].to_string(index=False))


if __name__ == "__main__":
    main()
