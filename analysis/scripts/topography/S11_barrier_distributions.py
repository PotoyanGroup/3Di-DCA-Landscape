"""Supplementary Figure S11: cross-family Langevin barrier distributions per representation.

Displays the all-pairs kinetic-barrier data computed upstream by step 03j
(Globin and TRPM all-pairs barriers) and step 03f (MDH binary AA-vs-3Di
cohort). The figure is a 6-violin panel, one violin per (family, representation)
condition ordered MDH/TRPM/Globin x AA/3Di. Each violin pools per-trajectory
reduced barriers (deltaH^/sigma_H,nat); black jittered dots mark per-pair
medians and a thick horizontal bar marks the pooled median. AA and 3Di violins
are colored blue and red. Family groups are annotated with their regime label
(consistent / intermediate / complementary). No MSM or Kramers transform is
applied: this is the raw kinetic-barrier evidence for the regime split. A
group-level summary table (n_traj, n_pairs, median, IQR) is also printed.

Inputs (paths resolved through analyses/scripts/_paths.py unless noted):
  - OUT_DATA / "03j_all_pairs_barriers_raw.csv" : per-trajectory barriers
        (deltaH_in_sigma_nat) keyed by family/rep for Globin and TRPM
  - OUT_DATA / "03j_all_pairs_barriers_summary.csv" : per-pair median barriers
        keyed by family/rep for Globin and TRPM
  - OUT_DATA / "03f_langevin_comparison_barriers.csv" : MDH barriers; the
        family=="MDH", method=="uniform" rows split by alphabet (AA / 3Di)

Outputs:
  - OUT_FIG / "03o_barrier_distributions_si.pdf" : the S11 violin figure (vector)
  - OUT_FIG / "03o_barrier_distributions_si.png" : the S11 violin figure (160 dpi)

Usage:
  python analyses/scripts/topography/S11_barrier_distributions.py
  No command-line arguments.
"""
from pathlib import Path
import sys
import csv
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from _paths import OUT_FIG, OUT_DATA

ALL_PAIRS_RAW = OUT_DATA / "03j_all_pairs_barriers_raw.csv"
ALL_PAIRS_SUMMARY = OUT_DATA / "03j_all_pairs_barriers_summary.csv"
MDH_BARRIERS = OUT_DATA / "03f_langevin_comparison_barriers.csv"


def load_globin_trpm():
    by_cond_all = {}
    by_cond_pair = {}
    with open(ALL_PAIRS_RAW) as fh:
        for r in csv.DictReader(fh):
            v = float(r["deltaH_in_sigma_nat"])
            by_cond_all.setdefault((r["family"], r["rep"]), []).append(v)
    with open(ALL_PAIRS_SUMMARY) as fh:
        for r in csv.DictReader(fh):
            try:
                med = float(r["median"])
            except (ValueError, KeyError):
                continue
            if np.isfinite(med):
                by_cond_pair.setdefault((r["family"], r["rep"]),
                                          []).append(med)
    return by_cond_all, by_cond_pair


def load_mdh():
    aa, di = [], []
    with open(MDH_BARRIERS) as fh:
        for r in csv.DictReader(fh):
            if r["family"] == "MDH" and r["method"] == "uniform":
                v = float(r["deltaH_in_sigma_nat"])
                if r["alphabet"] == "AA":
                    aa.append(v)
                elif r["alphabet"] == "3Di":
                    di.append(v)
    return aa, di


def main():
    by_all, by_pair = load_globin_trpm()
    mdh_aa, mdh_3di = load_mdh()
    by_all[("MDH", "AA")] = mdh_aa
    by_all[("MDH", "3Di")] = mdh_3di
    by_pair[("MDH", "AA")] = [float(np.median(mdh_aa))]
    by_pair[("MDH", "3Di")] = [float(np.median(mdh_3di))]

    order = [("MDH", "AA"), ("MDH", "3Di"),
             ("TRPM", "AA"), ("TRPM", "3Di"),
             ("Globin", "AA"), ("Globin", "3Di")]

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    x = np.arange(len(order))
    bw = 0.72
    data = [np.asarray(by_all[k]) for k in order]

    parts = ax.violinplot(data, positions=x, widths=bw,
                          showmedians=False, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        rep = order[i][1]
        pc.set_facecolor("#3a6fb0" if rep == "AA" else "#cc4a4a")
        pc.set_edgecolor("black")
        pc.set_linewidth(0.7)
        pc.set_alpha(0.55)

    rng = np.random.default_rng(0)
    for i, k in enumerate(order):
        medians = np.asarray(by_pair[k])
        jitter = rng.uniform(-0.14, 0.14, size=len(medians))
        ax.scatter(x[i] + jitter, medians,
                   color="black", s=32, zorder=5,
                   edgecolor="white", linewidth=0.9)

    for i, k in enumerate(order):
        med = float(np.median(by_all[k]))
        ax.plot([x[i] - 0.32, x[i] + 0.32], [med, med],
                color="black", lw=2.8, zorder=6, solid_capstyle="round")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}\n{r}" for f, r in order], fontsize=11)
    ax.set_ylabel(r"$\Delta H^\ddagger\ /\ \sigma_{H,\mathrm{nat}}$",
                  fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(d.max() for d in data) * 1.10)

    handles = [
        Patch(facecolor="#3a6fb0", edgecolor="black", alpha=0.55,
              label="AA representation"),
        Patch(facecolor="#cc4a4a", edgecolor="black", alpha=0.55,
              label="3Di representation"),
        Line2D([0], [0], color="black", lw=2.8,
               label="pooled median"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="black", markeredgecolor="white",
               markersize=8, label="per-pair median",
               linestyle="None"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)

    # Regime annotations across family groups
    ymax = ax.get_ylim()[1]
    regime_y = ymax * 0.96
    family_groups = [("MDH", 0.5, "consistent"),
                     ("TRPM", 2.5, "intermediate"),
                     ("Globin", 4.5, "complementary")]
    for _, xc, label in family_groups:
        ax.text(xc, regime_y, label, ha="center", va="top",
                fontsize=9, fontstyle="italic", color="0.30",
                bbox=dict(boxstyle="round,pad=0.30",
                          fc="white", ec="0.55", lw=0.5, alpha=0.85))

    ax.set_title("Cross-family Langevin barrier distributions per "
                 "representation",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    out_pdf = OUT_FIG / "03o_barrier_distributions_si.pdf"
    out_png = OUT_FIG / "03o_barrier_distributions_si.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")

    print("\nGroup-level summary:")
    print(f"{'condition':<14}{'n_traj':>8}{'n_pairs':>8}{'median':>8}"
          f"{'IQR':>10}")
    for k in order:
        v = np.asarray(by_all[k])
        npair = len(by_pair[k])
        q25, q75 = np.percentile(v, [25, 75])
        print(f"{k[0]} {k[1]:<10}{len(v):>8}{npair:>8}"
              f"{np.median(v):>8.2f}{q75 - q25:>10.2f}")


if __name__ == "__main__":
    main()
