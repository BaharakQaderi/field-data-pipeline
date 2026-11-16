"""
Figures for KTE_2030_FTC_0001 (Data_flights_req.xlsx "Figures per flight" + campaign figures + an overview).

Each figure is built once as a plotly figure and written twice:
  figures/<stem>_<name>.html  interactive (zoom, hover), for the technical team; plotly.js saved next to it (offline)
  figures/<stem>_<name>.png   static, for the report (rendered with matplotlib from the same traces)
plus figures/index.html (gallery) per flight and campaign/index.html.

Reads the outputs of ftc_extract.py and ftc_cycles.py. Force is the raw load-cell sum in kg (as on the HMI).
Usage: python ftc_figures.py --all | --event <stem>
"""

import argparse
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots

from ftc_extract import CONFIG

# palette (light mode): categorical in fixed order, status reserved for RTK states
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
PHASE = {"generation": C1, "recovery": C2}
STAGE = {"P_mech_kw": ("Mechanical (F·v)", C1), "P_motor_kw": ("Motor shaft (T·ω)", C5),
         "P_bus_kw": ("DC bus (converter high side)", C2),
         "P_batt_kw": ("Battery", C3)}
DAY = {"20261022": ("22 Oct", C1), "20261024": ("24 Oct", C2), "20261028": ("28 Oct", C3)}
RTK = {"fixed": "#0ca30c", "float": "#fab219", "no RTK": "#ec835a", "no GPS": "#d03b3b"}
LINK = {"main": C1, "backup": C2}
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = [[0, C2], [0.5, "#d9d8d4"], [1, C1]]   # consumption (orange) <- 0 -> generation (blue)
CFG = yaml.safe_load(open(CONFIG))
DT_S = CFG["grid_ms"] / 1000

# PNG text sizes: the repo's standard block ("Enhanced plotting parameters for better quality", as in
# vaie_flights_analysis.py / catanzaro_torque_analysis.py). Also applies to ftc_loadcells.py, which imports this module.
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
})
DPI_CAMPAIGN, DPI_FLIGHT = 300, 200    # repo standard is 300; per-flight PNGs at 200 to keep the viewer zip smaller
SPARSE = 500                           # marker traces with fewer points are drawn large (one marker = one cycle/flight)


def png_dpi(path):
    return DPI_CAMPAIGN if os.sep + "campaign" + os.sep in os.path.abspath(path) else DPI_FLIGHT

# --------------------------------------------------------------------------- rendering


def style(fig, title, height=None):
    fig.update_layout(template="plotly_white", title=dict(text=title, x=0.01, font=dict(size=15, color=INK)),
                      font=dict(family="Inter, Helvetica, Arial, sans-serif", size=12, color=INK2),
                      paper_bgcolor=SURF, plot_bgcolor=SURF, hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1, xanchor="right"),
                      margin=dict(l=70, r=30, t=80, b=55), height=height)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=INK2, ticks="outside")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=INK2, ticks="outside")
    return fig


def _axis_title(layout, key):
    ax = layout[key] if key in layout else None
    return ax.title.text if ax is not None and ax.title is not None and ax.title.text else ""


def to_png(fig, path):
    """Render the plotly figure's scatter/bar traces with matplotlib (one subplot per y-axis, stacked)."""
    lay = fig.layout
    dpi = png_dpi(path)
    if not fig.data:  # empty figure: write the explanation instead of a plot
        f, ax = plt.subplots(figsize=(12, 3))
        msg = lay.annotations[0].text if lay.annotations else "no data"
        ax.text(0.5, 0.5, msg, ha="center", va="center", color=INK2, fontsize=12, transform=ax.transAxes)
        ax.axis("off")
        f.suptitle(lay.title.text.replace("<br>", "\n").replace("<sub>", "").replace("</sub>", "")
                   if lay.title and lay.title.text else "", x=0.01, ha="left", color=INK, fontsize=14)
        f.savefig(path, dpi=dpi)
        plt.close(f)
        return
    yaxes = sorted({t.yaxis or "y" for t in fig.data}, key=lambda a: int(a[1:] or 1))
    n = len(yaxes)
    heights = []
    for a in yaxes:
        dom = lay["yaxis" + a[1:]].domain if lay["yaxis" + a[1:]].domain else (0, 1)
        heights.append(dom[1] - dom[0])
    shared = n > 1
    timex = any(t.x is not None and len(t.x) and isinstance(t.x[0], str) and "T" in t.x[0] for t in fig.data)
    if n >= 4 and not timex:   # campaign scatters with 4+ panels: 2 columns, so sparse points are not stretched thin
        nr = -(-n // 2)
        f, grid = plt.subplots(nr, 2, figsize=(14, 4.6 * nr), sharex=True, squeeze=False)
        for ax in grid.flat[n:]:
            ax.set_visible(False)
        axes = grid.flat[:n]
        bottom = [axes[max(i for i in range(n) if i % 2 == c)] for c in range(2)]   # lowest panel per column
        for ax in axes:
            ax.tick_params(labelbottom=True)
    else:
        fig_h = max(4.8, (lay.height or 450) / 100 * 1.0)
        f, grid = plt.subplots(n, 1, figsize=(12, fig_h), sharex=shared, squeeze=False,
                               gridspec_kw=dict(height_ratios=heights) if n > 1 else None)
        axes = grid[:, 0]
        bottom = [axes[-1]]
    rowof = {a: i for i, a in enumerate(yaxes)}
    stacked = lay.barmode == "stack"
    bottoms = {}
    grouped = [t for t in fig.data if t.type == "bar"] if lay.barmode == "group" else []
    for t in fig.data:
        ax = axes[rowof[t.yaxis or "y"]]
        x = pd.to_datetime(t.x) if t.x is not None and len(t.x) and isinstance(t.x[0], str) and "T" in t.x[0] \
            else np.asarray(t.x)
        y = np.asarray(t.y, dtype=float) if t.y is not None else None
        label = t.name if t.showlegend is not False and t.name else None
        if t.type == "bar":
            key = (rowof[t.yaxis or "y"])
            b = bottoms.get(key, np.zeros(len(y))) if stacked else None
            w = 0.8 * (np.min(np.diff(x)) if len(x) > 1 and np.issubdtype(np.asarray(x).dtype, np.number) else 0.8)
            if t in grouped:   # side by side within each category, like plotly's barmode="group"
                j, nb = grouped.index(t), len(grouped)
                w = 0.8 / nb
                ax.set_xticks(np.arange(len(x)), [str(v).replace("<br>", "\n") for v in x])
                x = np.arange(len(x)) + (j - (nb - 1) / 2) * w
            ax.bar(x, y, width=w, bottom=b, color=t.marker.color, label=label, edgecolor=SURF, linewidth=1)
            if stacked:
                bottoms[key] = b + np.nan_to_num(y)
            continue
        mode = t.mode or "lines"
        color = t.line.color if "lines" in mode else t.marker.color
        if "markers" in mode and t.marker is not None and isinstance(t.marker.color, (list, tuple, np.ndarray)):
            cs = t.marker.colorscale
            cmap = LinearSegmentedColormap.from_list("c", [c for _, c in cs]) if cs else "viridis"
            sc = ax.scatter(x, y, c=np.asarray(t.marker.color, dtype=float), cmap=cmap, s=6, linewidths=0,
                            vmin=t.marker.cmin, vmax=t.marker.cmax, rasterized=True)
            cb = f.colorbar(sc, ax=ax, pad=0.01)
            if t.marker.colorbar is not None and t.marker.colorbar.title is not None:
                cb.set_label(t.marker.colorbar.title.text or "", color=INK2)
        elif "markers" in mode:
            if "lines" in mode:
                ax.plot(x, y, color=color, lw=2, label=None)
            big = len(y) < SPARSE
            sym = np.asarray(t.marker.symbol) if isinstance(t.marker.symbol, (list, tuple, np.ndarray)) \
                else np.full(len(y), t.marker.symbol or "circle")
            for k, (sname, m) in enumerate((("circle", "o"), ("triangle-down", "v"), ("square", "s"), ("diamond", "D"))):
                sel = sym == sname
                if sel.any():
                    ax.scatter(np.asarray(x)[sel], y[sel], s=70 if big else max(6, (t.marker.size or 4) ** 2 / 2),
                               color=color, marker=m, label=label if k == 0 or not (sym == "circle").any() else None,
                               edgecolors=SURF if big else "none", linewidths=1.2 if big else 0,
                               alpha=t.marker.opacity or 1, rasterized=not big, zorder=3)
        else:
            dash = {"dash": "--", "dot": ":"}.get(t.line.dash, "-") if t.line is not None else "-"
            draw = ax.step if t.line is not None and t.line.shape == "hv" else ax.plot
            kw = dict(where="post") if draw == ax.step else {}
            draw(x, y, color=color, lw=1.4 if (t.line.width or 2) >= 2 else 0.9, ls=dash, label=label, **kw)
            if t.fill == "tozeroy":
                ax.fill_between(x, 0, y, color=color, alpha=0.15, step="post" if draw == ax.step else None)
    for sh in lay.shapes or []:  # vrect / vline spanning the whole figure height
        if sh.yref and "domain" not in sh.yref and sh.yref != "paper":
            continue
        # make_subplots(row="all") adds one shape per subplot: draw each only on its own axis, or the bands stack up
        yr = (sh.yref or "").replace(" domain", "")
        for ax in ([axes[rowof[yr]]] if yr in rowof else axes):
            if sh.type == "rect":
                ax.axvspan(pd.Timestamp(sh.x0), pd.Timestamp(sh.x1), color=sh.fillcolor, alpha=sh.opacity or 0.1,
                           lw=0, zorder=0)
            elif sh.type == "line" and sh.x0 == sh.x1:
                ax.axvline(pd.Timestamp(sh.x0), color=sh.line.color or INK2, lw=0.8, ls=":")
    for sh in lay.shapes or []:  # hline (zero lines), drawn on its own axis
        if sh.type == "line" and sh.y0 == sh.y1 and sh.x0 != sh.x1 and "domain" in (sh.xref or "") + " domain":
            yr = sh.yref or "y"
            if yr in rowof:
                axes[rowof[yr]].axhline(sh.y0, color=sh.line.color or INK2, lw=0.8, zorder=1)
    for a, ax in zip(yaxes, axes):
        ax.set_axisbelow(True)
        ax.set_ylabel(_axis_title(lay, "yaxis" + a[1:]), color=INK2)
        ax.grid(color=GRID, lw=0.6)
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if lay["yaxis" + a[1:]].range:
            ax.set_ylim(lay["yaxis" + a[1:]].range)
        if len(ax.get_legend_handles_labels()[0]) > (1 if n == 1 else 0):   # one series in a one-panel figure: the title names it
            if ax.patches:   # bars fill the plot: legend outside, right
                ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.005, 1))
            else:
                ax.legend(frameon=True, facecolor=SURF, edgecolor=GRID, framealpha=0.9, loc="best")
    xa = "xaxis" + (fig.data[0].xaxis or "x")[1:]
    xlab = _axis_title(lay, xa) or next((_axis_title(lay, k) for k in lay if k.startswith("xaxis") and _axis_title(lay, k)), "")
    for ax in bottom:
        ax.set_xlabel(xlab, color=INK2)
    if lay[xa].range and not shared:
        axes[-1].set_xlim(lay[xa].range)
    if lay[xa].tickvals:
        axes[-1].set_xticks(list(lay[xa].tickvals))
    title = lay.title.text if lay.title and lay.title.text else ""
    lines = title.replace("<sub>", "").replace("</sub>", "").split("<br>")
    wrap = 125 if f.get_figwidth() > 13 else 110
    f.suptitle("\n".join(textwrap.fill(ln, wrap) for ln in lines), x=0.01, ha="left", color=INK, fontsize=14)
    f.patch.set_facecolor(SURF)
    f.tight_layout()
    f.savefig(path, dpi=dpi)
    plt.close(f)


class Writer:
    def __init__(self, folder, prefix):
        self.folder, self.prefix, self.items = folder, prefix, []
        os.makedirs(folder, exist_ok=True)

    def save(self, fig, name, caption, req=None):
        base = os.path.join(self.folder, f"{self.prefix}_{name}")
        fig.write_html(base + ".html", include_plotlyjs="directory", full_html=True,
                       config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}})
        try:
            to_png(fig, base + ".png")
            png = True
        except Exception as e:  # keep the HTML even if the static render fails
            print(f"   PNG failed for {name}: {e}")
            png = False
        self.items.append({"name": name, "caption": caption, "req": req, "png": png,
                           "file": f"{self.prefix}_{name}"})

    def index(self, title, intro):
        cards = "\n".join(
            f'<figure><a href="{i["file"]}.html">' +
            (f'<img src="{i["file"]}.png" alt="{i["caption"]}" loading="lazy">' if i["png"] else "") +
            f'</a><figcaption><b>{i["caption"]}</b>' + (f'<span>{i["req"]}</span>' if i["req"] else "") +
            f'<a href="{i["file"]}.html">interactive</a>' + (f' · <a href="{i["file"]}.png">PNG</a>' if i["png"] else "") +
            "</figcaption></figure>" for i in self.items)
        html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font-family:Inter,Helvetica,Arial,sans-serif;background:{SURF};color:{INK};margin:0;padding:24px 16px}}
h1{{font-size:20px;margin:0 0 4px}} p{{color:{INK2};max-width:900px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:18px}}
figure{{margin:0;background:#fff;border:1px solid {GRID};border-radius:8px;padding:10px}}
img{{width:100%;border-radius:4px}} figcaption{{font-size:13px;color:{INK2};display:flex;flex-direction:column;gap:3px}}
figcaption b{{color:{INK}}} figcaption span{{font-size:12px}} a{{color:{C1}}}</style></head><body>
<h1>{title}</h1><p>{intro}</p><div class="grid">{cards}</div></body></html>"""
        with open(os.path.join(self.folder, "index.html"), "w") as f:
            f.write(html)


# --------------------------------------------------------------------------- helpers


def load(out, stem):
    df = pd.read_csv(os.path.join(out, stem, f"{stem}_analysis.csv"), low_memory=False)
    df["t"] = pd.to_datetime(df["time_rome"])
    stats = json.load(open(os.path.join(out, stem, f"{stem}_stats.json")))
    cyc = pd.read_csv(os.path.join(out, stem, f"{stem}_cycles.csv"))
    return df, stats, cyc


def pdf_line(values, bins):
    v = values.dropna()
    if v.empty:
        return [], []
    h, e = np.histogram(v, bins=bins, density=True)
    return list(e[:-1]), list(h)


# averaging windows of the "max of moving average" analysis (0 = highest single sample)
WINDOWS_S = [0, 1, 2, 3, 5, 7, 10, 15, 20]
MAXAVG = [  # (name, column, |value|, phases, caption)
    ("pod_torque_left", "POD_servo_left_torque_nm", True, ("generation", "recovery"), "pod left motor |torque| [N·m]"),
    ("pod_torque_right", "POD_servo_right_torque_nm", True, ("generation", "recovery"), "pod right motor |torque| [N·m]"),
    ("pod_current", "POD_current_a", False, ("generation", "recovery"), "pod current [A]"),
    ("generator_torque", "OPC_inv2_torque_actual", True, ("generation", "recovery"), "generator |torque| (INV2) [N·m]"),
    ("brushless_torque", "OPC_inv1_torque_actual", True, ("recovery",), "brushless |torque| (INV1) [N·m]")]


def max_moving_avg(x, col, phase, absval):
    """Highest moving average of `col` over each window in WINDOWS_S, using only `phase` data. x: native samples with a
    'phase' column. The window never spans two phases or a data gap (> 2 s): each phase segment is averaged on its own
    and a window counts only when it fits completely inside the segment."""
    v = x[col].abs() if absval else x[col]
    gap = x.index.to_series().diff() > pd.Timedelta(seconds=2)
    seg = ((x["phase"] != x["phase"].shift()) | gap).cumsum()
    rows = []
    for w in WINDOWS_S:
        best = (np.nan, None)
        for _, s in v[x["phase"] == phase].groupby(seg):
            if w == 0:
                m = s
            else:
                step = s.index.to_series().diff().median()
                m = s.rolling(f"{w}s").mean()
                m = m[m.index >= s.index[0] + pd.Timedelta(seconds=w) - step]   # full windows only
            if len(m) and (np.isnan(best[0]) or m.max() > best[0]):
                best = (float(m.max()), m.idxmax())
        rows.append({"window_s": w, "value": best[0], "time_gs": best[1]})
    return pd.DataFrame(rows)


def maxavg_table(nv, stem):
    rows = []
    for name, col, absval, phases, lab in MAXAVG:
        for ph in phases:
            r = max_moving_avg(nv[col], col, ph, absval)
            rows.append(r.assign(stem=stem, signal=name, column=col, abs=absval, phase=ph))
    t = pd.concat(rows, ignore_index=True)
    t["time_rome"] = pd.to_datetime(t["time_gs"], utc=True).dt.tz_convert("Europe/Rome").dt.strftime("%H:%M:%S.%f").str[:-4]
    return t[["stem", "signal", "column", "abs", "phase", "window_s", "value", "time_gs", "time_rome"]]


def fig_maxavg(series, title, ylab, panels=None):
    """series: (name, table rows for one signal/phase, colour, marker symbol[, panel]). panels: one y label per panel
    (small multiples, e.g. left / right motor) instead of one plot."""
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.08) if panels else go.Figure()
    for name, t, color, symbol, *panel in series:
        if t["value"].isna().all():
            continue
        row = dict(row=panel[0] + 1, col=1) if panels else {}
        fig.add_trace(go.Scatter(x=t["window_s"], y=t["value"].round(3), name=name, mode="lines+markers",
                                 showlegend=not panels or panel[0] == 0,
                                 line=dict(color=color, width=2), marker=dict(size=8, color=color, symbol=symbol,
                                                                              line=dict(color=SURF, width=1.5)),
                                 customdata=t[["label"]].values,
                                 hovertemplate=f"window %{{x}} s: %{{y:.2f}}<br>%{{customdata[0]}}<extra>{name}</extra>"),
                      **row)
    fig.update_xaxes(tickvals=WINDOWS_S, range=[-0.5, 20.5])
    fig.update_xaxes(title_text="averaging window [s]  (0 = highest single sample)", **(dict(row=len(panels), col=1) if panels else {}))
    for i, lab in enumerate(panels or [ylab]):
        fig.update_yaxes(title_text=lab, **(dict(row=i + 1, col=1) if panels else {}))
    return style(fig, title, height=760 if panels else None)


def native(out, stem, df, cols):
    """Native samples of `cols` (as logged, not interpolated) inside the window, each tagged with the phase of the
    nearest grid row: distributions and max moving averages use these, so the 20 Hz interpolation cannot reshape them."""
    nat = pd.read_parquet(os.path.join(out, stem, f"{stem}_native.parquet"), columns=cols)
    g = df[["time_gs", "phase", "in_window"]].copy()
    g.index = pd.to_datetime(g.pop("time_gs"), utc=True)
    res = {}
    for c in cols:
        x = nat[c].dropna()
        x = x[~x.index.duplicated(keep="last")].rename(c).to_frame()
        m = pd.merge_asof(x, g, left_index=True, right_index=True, direction="nearest",
                          tolerance=pd.Timedelta(milliseconds=CFG["grid_ms"]))
        res[c] = m[m["in_window"] == 1]
    return res


def thin(df, n=20000):
    return df if len(df) <= n else df.iloc[:: int(np.ceil(len(df) / n))]


def phase_note(stats):
    return f"{stats['n_cycles']} complete cycles"


# --------------------------------------------------------------------------- per-flight figures


def fig_pdf(df, col, bins, title, xlab, by_phase=True):
    fig = go.Figure()
    groups = [("generation", df[df.phase == "generation"]), ("recovery", df[df.phase == "recovery"])] if by_phase \
        else [("all", df)]
    for name, g in groups:
        x, y = pdf_line(g[col], bins)
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(shape="hv", width=2,
                                 color=PHASE.get(name, C1)), fill="tozeroy",
                                 hovertemplate=f"{xlab}: %{{x:.2f}}<br>density %{{y:.3f}}<extra>{name}</extra>"))
    fig.update_xaxes(title=xlab)
    fig.update_yaxes(title="probability density")
    return style(fig, title)


MAXAVG_FIGS = [  # (file name, title, y label, [(signal, legend, colour, marker)], phases)
    ("pod_torque", "pod motor torque", "|torque| [N·m]",
     [("pod_torque_left", "left motor", C1, "circle"), ("pod_torque_right", "right motor", C2, "square")],
     ("generation", "recovery")),
    ("pod_current", "pod current (total pack, no per-motor current)", "pod current [A]",
     [("pod_current", "pod current", C1, "circle")], ("generation", "recovery")),
    ("generator_torque", "generator torque (INV2, Torque_ActualValue[2], per motor)", "|torque| [N·m]",
     [("generator_torque", "generator", C1, "circle")], ("generation", "recovery")),
    ("brushless_torque", "brushless torque (INV1, Torque_ActualValue[1])", "|torque| [N·m]",
     [("brushless_torque", "brushless", C1, "circle")], ("recovery",))]


def shade_phases(fig, df):
    ph = df[["t", "phase"]].copy()
    ph["blk"] = (ph.phase != ph.phase.shift()).cumsum()
    for _, b in ph[ph.phase.isin(["generation", "recovery"])].groupby("blk"):
        fig.add_vrect(x0=b.t.iloc[0], x1=b.t.iloc[-1], fillcolor=PHASE[b.phase.iloc[0]], opacity=0.07, line_width=0)


def fig_pod_temps(df, short):
    w = df[df.in_window == 1]
    fig = go.Figure()
    for col, name, color in (("POD_servo_left_temp_c", "left motor", C1), ("POD_servo_right_temp_c", "right motor", C2)):
        fig.add_trace(go.Scatter(x=w["t"], y=w[col], name=name, mode="lines", line=dict(color=color, width=2),
                                 hovertemplate=f"%{{y:.0f}} °C<extra>{name}</extra>"))
    shade_phases(fig, w)
    fig.update_xaxes(title="time (Rome)")
    fig.update_yaxes(title="motor temperature [°C]  (1 °C steps, 5 Hz)")
    return style(fig, f"{short} — pod motor temperatures<br><sub>blue band = generation, orange band = recovery</sub>")


def flight_figures(out, stem):
    df, stats, cyc = load(out, stem)
    w = df[df.in_window == 1].copy()
    short = stem.replace("KTE_2030_FTC_0001_", "")
    W = Writer(os.path.join(out, stem, "figures"), stem)
    gen, rec = w[w.phase == "generation"], w[w.phase == "recovery"]
    note = phase_note(stats)
    nv = native(out, stem, df, ["WIND_speed_ms", "OPC_force_sum_kg", "POD_servo_left_torque_nm",
                                "POD_servo_right_torque_nm", "POD_current_a", "OPC_inv2_torque_actual",
                                "OPC_inv1_torque_actual"])
    ph_of = lambda c, ph: nv[c].loc[nv[c]["phase"] == ph, c]

    # overview (interactive time series, the main HTML for the team)
    W.save(fig_overview(df, stats, short), "overview", "Flight overview: all key signals on one time axis")

    W.save(fig_pdf(nv["WIND_speed_ms"], "WIND_speed_ms", np.arange(0, 16, 0.5), f"{short} — wind speed distribution",
                   "wind speed [m/s]", by_phase=False), "wind_pdf", "Wind speed probability distribution")
    fv = nv["OPC_force_sum_kg"]["OPC_force_sum_kg"]
    fmin = np.floor(float(np.nanpercentile(fv, 0.1)) / 10) * 10
    fmax = max(fmin + 50, float(np.nanpercentile(fv, 99.9)) + 20)
    W.save(fig_pdf(nv["OPC_force_sum_kg"], "OPC_force_sum_kg", np.arange(fmin, fmax, 10),
                   f"{short} — force (load-cell sum) distribution, generation vs recovery<br><sub>{note}; raw kg as on the HMI</sub>",
                   "force [kg]"), "force_pdf", "Force probability distribution, generation and recovery")
    for ph in ("generation", "recovery"):
        fig = go.Figure()
        for side, color in (("left", C1), ("right", C2)):
            x, y = pdf_line(ph_of(f"POD_servo_{side}_torque_nm", ph), np.arange(-12, 12.25, 0.25))
            fig.add_trace(go.Scatter(x=x, y=y, name=f"{side} motor", mode="lines",
                                     line=dict(shape="hv", width=2, color=color), fill="tozeroy"))
        fig.update_xaxes(title="pod motor torque [N·m]")
        fig.update_yaxes(title="probability density")
        W.save(style(fig, f"{short} — pod motor torque distribution, {ph}<br><sub>POD 1 (V3): servo A = left, B = right</sub>"),
               f"pod_torque_pdf_{ph}", f"Pod motor torque distribution, {ph}")
    W.save(fig_pod_temps(df, short), "pod_motor_temperatures", "Pod motor temperatures")
    t = maxavg_table(nv, stem)
    t.to_csv(os.path.join(out, stem, f"{stem}_max_moving_average.csv"), index=False)
    t["label"] = "at " + t["time_rome"].fillna("–") + " Rome"
    for name, title, ylab, sigs, phases in MAXAVG_FIGS:
        for ph in phases:
            series = [(lab, t[(t.signal == sig) & (t.phase == ph)], color, sym) for sig, lab, color, sym in sigs]
            W.save(fig_maxavg(series, f"{short} — {title}, all {ph} data<br><sub>highest moving average over each "
                                      f"window; windows never span two phases; {note}</sub>", ylab),
                   f"{name}_max_moving_avg_{ph}", f"Max moving average of {title}, {ph}")
    W.save(fig_cum_energy(w, short), "cumulative_energy", "Cumulative energy per stage [Wh]")
    W.save(fig_trajectory(w, short), "trajectory_power", "Kite trajectory coloured by instantaneous battery power")
    W.save(fig_force_speed(w, short), "force_vs_drum_speed", "Tether force vs drum velocity")
    W.save(fig_commands(df, short), "pod_commands_vs_force", "Pod commands, measured position and torques vs GS force")
    W.save(fig_force_torque(w, short), "force_vs_generator_torque", "Tether force vs generator torque, coloured by tether length")
    W.save(fig_per_distance(w, short), "datalink_per_vs_distance", "Datalink PER (main and backup) vs distance from GS")
    for col, bins, lab, name in (("POD_kite_roll_deg", np.arange(-180, 181, 20), "kite roll angle [deg] (XSens roll as logged; mounting offset vs 'antenna up' to confirm)", "rtk_vs_roll"),
                                 ("kite_speed_ms", np.arange(0, 42, 3), "kite speed [m/s]", "rtk_vs_speed")):
        W.save(fig_rtk(w, col, bins, lab, short, stats), name, f"RTK status vs {lab.split(' [')[0]}")
    W.index(f"{short} — figures",
            f"Window {stats['window_rome'][0][11:19]}–{stats['window_rome'][1][11:19]} Rome, "
            f"{stats['duration_s'] / 60:.1f} min, {stats['n_cycles']} cycles, POD {stats['pod']['fs_id']} "
            f"({stats['pod']['pod_version']}). Click a figure for the interactive version. "
            "Force = raw load-cell sum (kg). Power + = generation / charging.")
    print(f"{stem}: {len(W.items)} figures")


def fig_overview(df, stats, short):
    d = thin(df, 20000)
    rows = [("force [kg]", [("OPC_force_sum_kg", "load-cell sum", C1)]),
            ("line speed [m/s]", [("line_speed_ms", "+ reel-out / − reel-in", C1)]),
            ("tether on drum [m]", [("OPC_winded_length_m", "WindedLength", C1)]),
            ("power [kW]", [(k, v[0], v[1]) for k, v in STAGE.items() if df[k].notna().any()]),
            ("pitch [%] (+ = powered)", [("CMD_pitch", "command (20 Hz)", C1), ("POD_pitch_pct", "measured", C2)]),
            ("steering [%]", [("CMD_roll", "command (20 Hz)", C1), ("POD_roll_pct", "measured", C2)]),
            ("pod torque [N·m]", [("POD_servo_left_torque_nm", "left motor", C1), ("POD_servo_right_torque_nm", "right motor", C2)]),
            ("wind [m/s]", [("WIND_speed_ms", "wind speed", C1)])]
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.012)
    for i, (lab, series) in enumerate(rows, start=1):
        for col, name, color in series:
            fig.add_trace(go.Scattergl(x=d["t"], y=d[col], name=name, mode="lines", line=dict(color=color, width=1.2),
                                       legendgroup=lab, showlegend=True,
                                       hovertemplate=f"%{{y:.2f}}<extra>{name}</extra>"), row=i, col=1)
        fig.update_yaxes(title=dict(text=lab, font=dict(size=11)), row=i, col=1)
    shade_phases(fig, df)
    win = df[df.in_window == 1]
    for x in (win.t.iloc[0], win.t.iloc[-1]):
        fig.add_vline(x=x, line=dict(color=INK2, width=1, dash="dot"))
    fig.update_xaxes(title_text="time (Rome)", row=len(rows), col=1)
    return style(fig, f"{short} — overview<br><sub>blue band = generation, orange band = recovery; dotted = official "
                      f"window; {stats['n_cycles']} cycles</sub>", height=1500)


def fig_cum_energy(w, short):
    fig = go.Figure()
    for col, (name, color) in STAGE.items():
        if not w[col].notna().any():   # e.g. motor-shaft power before inverter 3 is extracted
            continue
        e = (w[col].fillna(0) * DT_S / 3.6).cumsum()
        fig.add_trace(go.Scatter(x=w["t"], y=e, name=name, mode="lines", line=dict(color=color, width=2),
                                 hovertemplate=f"%{{y:.1f}} Wh<extra>{name}</extra>"))
    e = (w["POD_power_w"].fillna(0) / 1000 * DT_S / 3.6).cumsum() * -1
    fig.add_trace(go.Scatter(x=w["t"], y=e, name="Pod battery (consumed)", mode="lines",
                             line=dict(color=C4, width=2, dash="dash"), hovertemplate="%{y:.1f} Wh<extra>pod</extra>"))
    fig.update_xaxes(title="time (Rome)")
    fig.update_yaxes(title="cumulative energy [Wh]  (negative = consumed)")
    return style(fig, f"{short} — cumulative energy per stage")


def fig_trajectory(w, short):
    r, az, el = w["POD_radius_m"], np.radians(w["POD_azimuth_deg"]), np.radians(w["POD_elevation_deg"])
    ok = r.notna() & az.notna() & el.notna()
    p = w["P_batt_kw"]
    lim = float(np.nanpercentile(p.abs(), 98)) if p.notna().any() else 1
    fig = go.Figure()
    if ok.any():
        d = thin(w[ok], 15000)
        fig.add_trace(go.Scattergl(
            x=d["POD_azimuth_deg"], y=d["POD_elevation_deg"], mode="markers", name="kite",
            marker=dict(size=4, color=d["P_batt_kw"], colorscale=DIV, cmin=-lim, cmax=lim,
                        colorbar=dict(title="battery power [kW]")),
            customdata=np.stack([d["time_rome"].str[11:19], d["P_batt_kw"], d["POD_radius_m"]], axis=1),
            hovertemplate="%{customdata[0]}<br>az %{x:.1f}°, el %{y:.1f}°, r %{customdata[2]} m"
                          "<br>P_batt %{customdata[1]} kW<extra></extra>"))
    else:
        fig.add_annotation(text="no kite position for this flight (no IPOK)", showarrow=False, x=0.5, y=0.5,
                           xref="paper", yref="paper")
    fig.update_xaxes(title="azimuth seen from the GS [deg, 0 = North]")
    fig.update_yaxes(title="elevation [deg]")
    return style(fig, f"{short} — kite trajectory vs instantaneous power<br><sub>blue = charging/generation, "
                      f"orange = consumption; 3D view and time playback in the replay</sub>")


def fig_force_speed(w, short):
    fig = go.Figure()
    for ph in ("generation", "recovery"):
        d = thin(w[w.phase == ph], 10000)
        fig.add_trace(go.Scattergl(x=d["line_speed_ms"], y=d["OPC_force_sum_kg"], mode="markers", name=ph,
                                   marker=dict(size=4, color=PHASE[ph], opacity=0.5),
                                   hovertemplate="v %{x:.2f} m/s<br>F %{y:.0f} kg<extra>" + ph + "</extra>"))
    fig.update_xaxes(title="line speed [m/s]  (+ reel-out)")
    fig.update_yaxes(title="force [kg]  (load-cell sum)")
    return style(fig, f"{short} — tether force vs drum velocity")


def fig_commands(df, short):
    d = thin(df, 20000)
    rows = [("force [kg]", [("OPC_force_sum_kg", "GS force (load-cell sum)", C1)]),
            ("pitch [%] (+ = powered)", [("TARANIS_pitch_pct", "Taranis stick", C3), ("CMD_pitch", "command sent (20 Hz)", C1),
                             ("POD_pitch_pct", "measured position", C2)]),
            ("steering [%]", [("TARANIS_roll_pct", "Taranis stick", C3), ("CMD_roll", "command sent (20 Hz)", C1),
                              ("POD_roll_pct", "measured position", C2)]),
            ("pod torque [N·m]", [("POD_servo_left_torque_nm", "left motor", C1),
                                  ("POD_servo_right_torque_nm", "right motor", C2)])]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03)
    for i, (lab, series) in enumerate(rows, start=1):
        for col, name, color in series:
            fig.add_trace(go.Scattergl(x=d["t"], y=d[col], name=name, mode="lines", line=dict(color=color, width=1.2),
                                       legendgroup=name, showlegend=i in (1, 2, 4),
                                       hovertemplate=f"%{{y:.1f}}<extra>{name}</extra>"), row=i, col=1)
        fig.update_yaxes(title=dict(text=lab, font=dict(size=11)), row=i, col=1)
    fig.update_xaxes(title_text="time (Rome)", row=4, col=1)
    return style(fig, f"{short} — pod commands vs GS force<br><sub>stick → command sent → measured position → motor "
                      f"torque (POD 1, V3: left/right motors)</sub>", height=900)


def fig_force_torque(w, short):
    d = thin(w.dropna(subset=["OPC_force_sum_kg", "OPC_inv2_torque_actual"]), 15000)
    wl = d["OPC_winded_length_m"]
    fig = go.Figure(go.Scattergl(
        x=d["OPC_force_sum_kg"], y=d["OPC_inv2_torque_actual"], mode="markers", name="samples",
        marker=dict(size=4, color=wl, colorscale=[[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)],
                    cmin=float(np.nanmin(wl)) if len(wl) else 0, cmax=float(np.nanmax(wl)) if len(wl) else 1,
                    colorbar=dict(title="tether on drum [m]")),
        customdata=np.stack([d["time_rome"].str[11:19], wl], axis=1),
        hovertemplate="%{customdata[0]}<br>F %{x:.0f} kg, T %{y:.1f}<br>on drum %{customdata[1]:.0f} m<extra></extra>"))
    fig.update_xaxes(title="force [kg]  (load-cell sum)")
    fig.update_yaxes(title="generator torque Torque_ActualValue[2] [N·m]")
    return style(fig, f"{short} — tether force vs generator torque at different tether lengths<br><sub>colour = "
                      f"WindedLength (cable on the drum)</sub>")


def fig_per_distance(w, short):
    fig = go.Figure()
    bins = np.arange(0, max(50, float(np.nanmax(w["dist_gs_m"])) if w["dist_gs_m"].notna().any() else 50) + 25, 25)
    for ch, col in (("main", "LINK_main_downlink_per_pct"), ("backup", "LINK_backup_downlink_per_pct")):
        d = w.dropna(subset=["dist_gs_m", col])
        if d.empty:
            continue
        g = d.groupby(pd.cut(d["dist_gs_m"], bins), observed=True)[col]
        mid = [b.mid for b in g.mean().index]
        fig.add_trace(go.Scatter(x=mid, y=g.mean().values, name=f"{ch} (mean)", mode="lines+markers",
                                 line=dict(color=LINK[ch], width=2), marker=dict(size=8),
                                 customdata=np.stack([g.quantile(0.95).values, g.size().values], axis=1),
                                 hovertemplate="%{x:.0f} m: mean %{y:.1f} %, p95 %{customdata[0]:.0f} %"
                                               "<br>%{customdata[1]} samples<extra>" + ch + "</extra>"))
        fig.add_trace(go.Scatter(x=mid, y=g.quantile(0.95).values, name=f"{ch} (95th pct)", mode="lines",
                                 line=dict(color=LINK[ch], width=1, dash="dot"), hoverinfo="skip"))
    if not fig.data:
        fig.add_annotation(text="no distance data (no kite position)", showarrow=False, x=0.5, y=0.5,
                           xref="paper", yref="paper")
    fig.update_xaxes(title="distance kite ↔ GS [m]  (25 m bins)")
    fig.update_yaxes(title="downlink packet error rate [%]")
    return style(fig, f"{short} — datalink PER vs distance from GS")


def fig_rtk(w, col, bins, lab, short, stats):
    d = w[(w["rtk_state"].notna()) & (w["rtk_state"] != "")].dropna(subset=[col])
    fig = go.Figure()
    if d.empty:
        fig.add_annotation(text="no IPOK data for this flight", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    else:
        cut = pd.cut(d[col], bins)
        tab = pd.crosstab(cut, d["rtk_state"], normalize="index") * 100
        n = cut.value_counts().reindex(tab.index)
        mid = [b.mid for b in tab.index]
        for st, color in RTK.items():
            y = tab[st] if st in tab else pd.Series(0, index=tab.index)
            fig.add_trace(go.Bar(x=mid, y=y.values, name=st, marker=dict(color=color),
                                 customdata=n.values,
                                 hovertemplate=f"%{{x}}: %{{y:.1f}} % {st}<br>%{{customdata}} samples<extra></extra>"))
        fig.update_layout(barmode="stack", bargap=0.08)
    fig.update_xaxes(title=lab)
    fig.update_yaxes(title="share of time in bin [%]", range=[0, 100])
    dev = stats["rtk_status_pct"]["ipok"] or "none"
    return style(fig, f"{short} — RTK status vs {lab.split(' [')[0]} (IPOK {dev})")


# --------------------------------------------------------------------------- campaign figures


def campaign_figures(out, stems):
    c = pd.read_csv(os.path.join(out, "campaign", "cycles_all.csv"))
    s = pd.read_csv(os.path.join(out, "campaign", "flights_summary.csv"))
    c["day"] = c["stem"].str[18:26]
    c["flight"] = c["stem"].str[27:]
    # regular pumping cycles only: take-off, landing and interrupted ("short") cycles stay in cycles_all.csv
    n_all = len(c)
    c_all = c   # every cycle incl. take-off / landing (as in the replay), for the POD consumption figure
    c = c[c["cycle_kind"].isna()] if "cycle_kind" in c else c
    excl = f"{n_all - len(c)} take-off / landing / interrupted cycles excluded"
    W = Writer(os.path.join(out, "campaign"), "KTE_2030_FTC_0001")
    for st, (lab, _) in (("batt", STAGE["P_batt_kw"]), ("mech", STAGE["P_mech_kw"]), ("motor", STAGE["P_motor_kw"]),
                         ("bus", STAGE["P_bus_kw"])):
        for xcol, xlab, name in (("wind_mean_ms", "average wind speed over the cycle [m/s]", "wind"),
                                 ("cycle_duration_s", "cycle duration [s]", "duration")):
            fig = go.Figure()
            ycol = f"cycle_{st}_energy_ratio_pct"
            clipped = 0
            for day, (dlab, color) in DAY.items():
                d = c[(c.day == day) & c[ycol].notna()]
                y = d[ycol].clip(lower=-200)
                clipped += int((d[ycol] < -200).sum())
                fig.add_trace(go.Scatter(
                    x=d[xcol], y=y, mode="markers", name=dlab,
                    marker=dict(size=10, color=color, line=dict(color=SURF, width=2),
                                symbol=np.where(d[ycol] < -200, "triangle-down", "circle")),
                    customdata=np.stack([d["flight"], d["cycle_id"], d[ycol].round(1)], axis=1),
                    hovertemplate="%{customdata[0]} cycle %{customdata[1]}<br>ratio %{customdata[2]} %"
                                  "<br>%{x:.1f}<extra></extra>"))
            fig.update_xaxes(title=xlab)
            fig.update_yaxes(title=f"cycle energy ratio [%] — {lab}", range=[-210, 105])
            sub = f"{len(c[c[ycol].notna()])} regular cycles from 7 flights ({excl}); ratio = (E_gen − E_rec)/E_gen"
            if clipped:
                sub += f"; {clipped} cycles below −200 % drawn as ▼ at −200 %"
            nn = int(c[ycol].isna().sum())
            if nn:
                sub += f"; {nn} cycles without net generation omitted"
            W.save(style(fig, f"Cycle energy ratio vs {name.replace('wind', 'average wind')} — {lab}<br><sub>{sub}</sub>"),
                   f"energy_ratio_vs_{name}_{st}", f"Cycle energy ratio vs {name} ({lab})")
    # net energy per cycle vs wind: easier to read than the ratio when generation is small
    for st, (lab, _) in (("batt", STAGE["P_batt_kw"]), ("mech", STAGE["P_mech_kw"]), ("motor", STAGE["P_motor_kw"])):
        fig = go.Figure()
        for day, (dlab, color) in DAY.items():
            d = c[c.day == day]
            fig.add_trace(go.Scatter(
                x=d["wind_mean_ms"], y=d[f"cycle_{st}_E_wh"], mode="markers", name=dlab,
                marker=dict(size=10, color=color, line=dict(color=SURF, width=2)),
                customdata=np.stack([d["flight"], d["cycle_id"], d[f"gen_{st}_E_wh"], d[f"rec_{st}_E_wh"]], axis=1),
                hovertemplate="%{customdata[0]} cycle %{customdata[1]}<br>net %{y:.1f} Wh"
                              "<br>generation %{customdata[2]} Wh, recovery %{customdata[3]} Wh<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=INK2, width=1))
        fig.update_xaxes(title="average wind speed over the cycle [m/s]")
        fig.update_yaxes(title=f"net energy per cycle [Wh] — {lab}")
        W.save(style(fig, f"Net energy per cycle vs average wind — {lab}<br><sub>above 0 = the cycle produced "
                          f"net energy; {len(c)} regular cycles ({excl})</sub>"),
               f"cycle_net_energy_vs_wind_{st}", f"Net energy per cycle vs wind ({lab})", None)
    # each phase on its own: energy (one panel per stage) and duration vs the average wind of that phase
    stages = [("mech", "P_mech_kw"), ("motor", "P_motor_kw"), ("bus", "P_bus_kw"), ("batt", "P_batt_kw")]
    for ph, phname in (("gen", "generation"), ("rec", "recovery")):
        fig = make_subplots(rows=len(stages), cols=1, shared_xaxes=True, vertical_spacing=0.03)
        for i, (st, col) in enumerate(stages, start=1):
            for day, (dlab, color) in DAY.items():
                d = c[c.day == day]
                fig.add_trace(go.Scatter(
                    x=d[f"{ph}_wind_mean_ms"], y=d[f"{ph}_{st}_E_wh"], mode="markers", name=dlab, showlegend=i == 1,
                    marker=dict(size=9, color=color, line=dict(color=SURF, width=2)),
                    customdata=np.stack([d["flight"], d["cycle_id"], d[f"{ph}_duration_s"]], axis=1),
                    hovertemplate="%{customdata[0]} cycle %{customdata[1]}<br>%{y:.1f} Wh in %{customdata[2]} s"
                                  "<br>wind %{x:.1f} m/s<extra></extra>"), row=i, col=1)
            fig.add_hline(y=0, line=dict(color=INK2, width=1), row=i, col=1)
            fig.update_yaxes(title=dict(text=f"{STAGE[col][0]} [Wh]", font=dict(size=11)), row=i, col=1)
        fig.update_xaxes(title_text=f"average wind speed over the {phname} phase [m/s]", row=len(stages), col=1)
        W.save(style(fig, f"Energy in each {phname} phase vs wind<br><sub>{len(c)} regular cycles ({excl}); "
                          f"+ = generated, − = consumed</sub>", height=1100),
               f"{phname}_phase_energy_vs_wind", f"Energy in each {phname} phase vs wind (all stages)", None)
        fig = go.Figure()
        for day, (dlab, color) in DAY.items():
            d = c[c.day == day]
            fig.add_trace(go.Scatter(
                x=d[f"{ph}_wind_mean_ms"], y=d[f"{ph}_duration_s"], mode="markers", name=dlab,
                marker=dict(size=10, color=color, line=dict(color=SURF, width=2)),
                customdata=np.stack([d["flight"], d["cycle_id"]], axis=1),
                hovertemplate="%{customdata[0]} cycle %{customdata[1]}<br>%{y:.0f} s at %{x:.1f} m/s<extra></extra>"))
        fig.update_xaxes(title=f"average wind speed over the {phname} phase [m/s]")
        fig.update_yaxes(title=f"{phname} phase duration [s]")
        W.save(style(fig, f"Duration of each {phname} phase vs wind<br><sub>{len(c)} regular cycles ({excl})</sub>"),
               f"{phname}_phase_duration_vs_wind", f"Duration of each {phname} phase vs wind", None)
    # POD battery: average power drawn in generation and in recovery (JN), duration-weighted = pod energy / phase time
    def pod_mean(d, ph):
        ok = d[f"{ph}_pod_power_mean_w"].notna() & d[f"{ph}_duration_s"].notna()
        w = d.loc[ok, f"{ph}_duration_s"]
        return (d.loc[ok, f"{ph}_pod_power_mean_w"] * w).sum() / w.sum() if w.sum() else np.nan, int(ok.sum())

    groups = [(f"{DAY[s[18:26]][0]} {s[27:]}", c_all[c_all.stem == s]) for s in stems] + [("all flights", c_all)]
    groups = [(f"{g}<br>{len(d)} cycle{'s' if len(d) != 1 else ''}", d) for g, d in groups]
    fig = go.Figure()
    for ph, phname in (("gen", "generation"), ("rec", "recovery")):
        vals = [pod_mean(d, ph) for _, d in groups]
        fig.add_trace(go.Bar(x=[g for g, _ in groups], y=[round(v, 1) for v, _ in vals], name=phname,
                             marker=dict(color=PHASE[phname], line=dict(color=SURF, width=2)),
                             customdata=[n for _, n in vals],
                             hovertemplate=f"%{{y:.0f}} W average in {phname}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_xaxes(title="flight")
    fig.update_yaxes(title="average POD battery power [W]", rangemode="tozero")
    W.save(style(fig, f"POD battery consumption: average power in generation and in recovery<br><sub>all cycles incl. "
                      f"take-off and landing; weighted by phase duration (pod energy ÷ time in the phase)</sub>"),
           "pod_power_per_phase", "POD battery: average power in generation and recovery", None)
    # max moving average, all flights of a day together (= the highest of that day's flights for each window)
    t = pd.concat([pd.read_csv(os.path.join(out, st, f"{st}_max_moving_average.csv")) for st in stems])
    t["day"] = t["stem"].str[18:26]
    best = t.sort_values("value", ascending=False).drop_duplicates(["day", "signal", "phase", "window_s"])
    best = best.sort_values(["day", "signal", "phase", "window_s"])
    best["label"] = best["stem"].str[27:] + " at " + best["time_rome"].fillna("–") + " Rome"
    best.drop(columns="label").to_csv(os.path.join(out, "campaign", "max_moving_average_per_day.csv"), index=False)
    for name, title, ylab, sigs, phases in MAXAVG_FIGS:
        for ph in phases:
            series = [(DAY[day][0], best[(best.day == day) & (best.signal == sig) & (best.phase == ph)], DAY[day][1],
                       "circle", k) for k, (sig, lab, _, _) in enumerate(sigs) for day in DAY]
            fig = fig_maxavg(series, f"Campaign — {title}, all {ph} data per test day<br><sub>highest moving average "
                                     f"over each window across the day's flights; windows never span two phases</sub>",
                             ylab, panels=[f"{lab} {ylab}" for _, lab, _, _ in sigs] if len(sigs) > 1 else None)
            W.save(fig, f"{name}_max_moving_avg_{ph}", f"Max moving average of {title}, {ph}, per day", None)
    # energy per flight per stage (grouped bars)
    fig = go.Figure()
    s["flight"] = s["stem"].str[27:] + " · " + s["stem"].str[24:26] + " Oct"
    for k, (lab, color) in (("mech", STAGE["P_mech_kw"]), ("motor", STAGE["P_motor_kw"]), ("bus", STAGE["P_bus_kw"]),
                            ("batt", STAGE["P_batt_kw"])):
        if f"E_{k}_wh" not in s:
            continue
        fig.add_trace(go.Bar(x=s["flight"], y=s[f"E_{k}_wh"], name=lab, marker=dict(color=color),
                             hovertemplate="%{x}: %{y:.0f} Wh<extra>" + lab + "</extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(title="net energy [Wh]  (− = consumed)")
    W.save(style(fig, "Net energy per flight and stage"), "energy_per_flight", "Net energy per flight and stage", None)
    W.index("KTE_2030_FTC_0001 — campaign figures",
            "Cross-flight figures. Per-flight figures are in each flight's figures/ folder.")
    print(f"campaign: {len(W.items)} figures")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="processed/KTE_2030_FTC_0001")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(CONFIG))
    stems = list(cfg["events"]) if args.all else (args.event or [])
    for stem in stems:
        flight_figures(args.out, stem)
    if args.all:
        campaign_figures(args.out, stems)
        top_index(args.out, stems)


def top_index(out, stems):
    s = pd.read_csv(os.path.join(out, "campaign", "flights_summary.csv")).set_index("stem")
    rows = "".join(
        f"<tr><td><a href='{st}/figures/index.html'>{st[18:]}</a></td><td>{s.at[st, 'duration_s'] / 60:.1f}</td>"
        f"<td>{int(s.at[st, 'n_cycles'])}</td><td>{s.at[st, 'wind_mean_ms']:.1f} / {s.at[st, 'wind_max_ms']:.1f}</td>"
        f"<td>{s.at[st, 'force_max_kg']:.0f}</td><td>{s.at[st, 'E_mech_wh']:.0f}</td><td>{s.at[st, 'E_batt_wh']:.0f}</td>"
        f"<td>{s.at[st, 'pod_energy_wh']:.0f}</td><td><a href='{st}/{st}_full_source.csv'>full source</a> · <a href='{st}/{st}_analysis.csv'>analysis</a></td></tr>" for st in stems)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KTE_2030_FTC_0001 results</title><style>body{{font-family:Helvetica,Arial,sans-serif;background:{SURF};color:{INK};margin:0;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 6px}} p{{color:{INK2};max-width:900px}} a{{color:{C1}}}
.t{{overflow-x:auto}} table{{border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}}
th,td{{padding:6px 12px;border-bottom:1px solid {GRID};text-align:right}} th:first-child,td:first-child{{text-align:left}}
th{{color:{INK2};font-weight:600}}</style></head><body>
<h1>KTE_2030_FTC_0001 — processed flights</h1>
<p><a href="replay/KTE_2030_FTC_0001_replay.html">Flight replay (3D)</a> · <a href="campaign/index.html">Campaign figures</a> · <a href="campaign/loadcells.html">Load cells</a> ·
<a href="events.csv">events.csv</a> (windows, clock offsets, coverage) · <a href="campaign/cycles_all.csv">all cycles</a></p>
<p>Report figures (PDF for LaTeX, PNG 600 dpi): <a href="campaign/report/KTE_2030_FTC_0001_pod_power_per_phase.pdf">POD power
per phase</a> · <a href="campaign/report/KTE_2030_FTC_0001_recovery_phase_duration_vs_wind.pdf">recovery duration vs wind</a> ·
<a href="campaign/report/KTE_2030_FTC_0001_recovery_phase_battery_energy_vs_wind.pdf">recovery battery energy vs wind</a></p>
<div class="t"><table><thead><tr><th>Flight (figures)</th><th>min</th><th>cycles</th><th>wind avg / max [m/s]</th><th>peak force [kg]</th>
<th>E mech [Wh]</th><th>E battery [Wh]</th><th>pod used [Wh]</th><th>data</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>Force = raw load-cell sum. Energy: negative = consumed.</p></body></html>"""
    with open(os.path.join(out, "index.html"), "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
