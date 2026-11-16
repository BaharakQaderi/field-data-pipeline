"""
Report versions of selected KTE_2030_FTC_0001 campaign figures, sized for a LaTeX page.

The figures are drawn at the size they are printed (\\textwidth ~ 6.3 in), so text and markers keep their size in the
PDF: include them with \\includegraphics[width=\\textwidth]{...}. The PDF is vector with embedded TrueType fonts; the
PNG (600 dpi) is for Word / slides.

  recovery_phase_duration_vs_wind   duration of each recovery phase vs the average wind of that phase
  recovery_phase_battery_energy_vs_wind   battery energy in each recovery phase (bottom-right panel of
                                          KTE_2030_FTC_0001_recovery_phase_energy_vs_wind.png) on its own
  pod_power_per_phase   POD battery: average power in generation and in recovery per flight, all cycles incl.
                        take-off and landing (as in the replay), weighted by phase duration

Reads campaign/cycles_all.csv (ftc_cycles.py --all); the two recovery figures use regular cycles only, as in the
campaign figures.
Writes campaign/report/KTE_2030_FTC_0001_<name>.pdf and .png.
Usage: python ftc_report_figures.py [--out processed/KTE_2030_FTC_0001]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ftc_figures import GRID, INK, INK2

# same hues as the other figures (C1, C2, C3), one step darker so they print strongly
DAY = {"20261022": ("22 Oct", "#1f5fb0"), "20261024": ("24 Oct", "#d0501c"), "20261028": ("28 Oct", "#128a5f")}
WIDTH_IN, HEIGHT_IN = 6.3, 4.0
PNG_DPI = 600

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,          # embedded TrueType: text stays text in the PDF
    "ps.fonttype": 42,
})


def scatter(c, ycol, title, subtitle, ylabel, path, zero=False):
    f, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))
    for day, (lab, color) in DAY.items():
        d = c[c["day"] == day]
        ax.scatter(d["rec_wind_mean_ms"], d[ycol], s=95, color=color, edgecolors="white", linewidths=1.2,
                   label=lab, zorder=3)
    if zero:
        ax.axhline(0, color=INK2, lw=1, zorder=1)
    ax.set_xlabel("average wind speed over the recovery phase [m/s]", color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(f"{title}\n", loc="left", color=INK, fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK2, fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID, framealpha=1, handletextpad=0.3, borderpad=0.4)
    f.tight_layout()
    for ext in ("pdf", "png"):
        f.savefig(f"{path}.{ext}", dpi=PNG_DPI, facecolor="white")
    plt.close(f)
    print(path + ".pdf / .png")


def pod_power_bars(c, path):
    """Grouped bars per flight: average POD power in generation / recovery = sum(mean power x duration) / sum(duration)."""
    def avg(d, ph):
        ok = d[f"{ph}_pod_power_mean_w"].notna() & d[f"{ph}_duration_s"].notna()
        w = d.loc[ok, f"{ph}_duration_s"]
        return float((d.loc[ok, f"{ph}_pod_power_mean_w"] * w).sum() / w.sum()) if w.sum() else np.nan

    stems = list(dict.fromkeys(c["stem"]))
    groups = [(f"{DAY[s[18:26]][0]}\n{s[27:]}", c[c["stem"] == s]) for s in stems] + [("all\nflights", c)]
    x = np.arange(len(groups))
    w = 0.38
    f, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))
    for k, (ph, lab, color) in enumerate((("gen", "generation", DAY["20261022"][1]),
                                         ("rec", "recovery", DAY["20261024"][1]))):
        vals = [avg(d, ph) for _, d in groups]
        bars = ax.bar(x + (k - 0.5) * w, vals, width=w, color=color, edgecolor="white", linewidth=1, label=lab,
                      zorder=3)
        ax.bar_label(bars, fmt="%.0f", fontsize=8.5, padding=1.5, color=INK)
    ax.axvline(len(groups) - 1.5, color=GRID, lw=1.2, zorder=1)          # separates the campaign total
    ax.set_xticks(x, [g for g, _ in groups], fontsize=9.5)
    ax.set_ylabel("average POD battery power [W]", color=INK)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    ax.set_title("POD battery consumption per phase\n", loc="left", color=INK, fontweight="bold")
    ax.text(0, 1.02, f"average power in generation and recovery; all {len(c)} cycles", transform=ax.transAxes,
            color=INK2, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID, framealpha=1, loc="upper left", ncol=2,
              handlelength=1.2, borderpad=0.4)
    f.tight_layout()
    for ext in ("pdf", "png"):
        f.savefig(f"{path}.{ext}", dpi=PNG_DPI, facecolor="white")
    plt.close(f)
    print(path + ".pdf / .png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="processed/KTE_2030_FTC_0001")
    args = ap.parse_args()
    c_all = pd.read_csv(os.path.join(args.out, "campaign", "cycles_all.csv"))
    c = c_all[c_all["cycle_kind"].isna()].copy() if "cycle_kind" in c_all else c_all.copy()   # regular cycles
    c["day"] = c["stem"].str[18:26]
    sub = f"{len(c)} regular cycles"
    folder = os.path.join(args.out, "campaign", "report")
    os.makedirs(folder, exist_ok=True)
    pre = os.path.join(folder, "KTE_2030_FTC_0001_")
    scatter(c, "rec_duration_s", "Duration of each recovery phase vs wind", sub,
            "recovery phase duration [s]", pre + "recovery_phase_duration_vs_wind")
    scatter(c, "rec_batt_E_wh", "Battery energy in each recovery phase vs wind", f"{sub}; − = drawn from the battery",
            "battery energy [Wh]", pre + "recovery_phase_battery_energy_vs_wind", zero=True)
    pod_power_bars(c_all, pre + "pod_power_per_phase")


if __name__ == "__main__":
    main()
