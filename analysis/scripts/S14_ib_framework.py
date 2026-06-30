"""Supplementary Figure S14: information-bottleneck framework showing when AA- and
3Di-aware representations are complementary across protein families.

Loads four precomputed per-family summary tables (information-bottleneck
correlations, cross-latent mutual information, sequence/structure diversity, and
k-NN overlap), merges them on family, computes Pearson correlations and linear
fits, and assembles a 2x2 figure. Panel A: the information-bottleneck plane with
AA-VAE and 3Di-VAE points joined by AA -> 3Di shift segments. Panel B: per-family
cross-latent mutual information (nats) as sorted bars. Panel C: the predictive law
relating mutual information to raw 3Di-Hamming structural diversity, with a linear
fit and Pearson/Spearman statistics. Panel D: cross-metric validation showing that
information-theoretic mutual information and geometric mean k-NN overlap agree on
the cross-latent regime.

Inputs (paths resolved through analyses/scripts/_paths.py unless noted):
  - OUT_DATA / "22_information_bottleneck.csv" : per-family IB-plane correlations
    (rho_AA_seq, rho_AA_struct, rho_3Di_seq, rho_3Di_struct, rho_AA_vs_3Di_latent)
  - OUT_DATA / "24_mutual_information.csv" : per-family cross-latent mutual
    information (MI_nats) with bootstrap CIs (MI_ci_lo, MI_ci_hi)
  - OUT_DATA / "15_diversity_summary.csv" : per-family diversity, providing
    "mean 3Di-Hamming" (renamed to h3di) structural diversity
  - OUT_DATA / "30_knn_overlap_summary.csv" : per-family mean k-NN overlap
    (mean_O5, mean_O10, mean_O20)

Outputs:
  - OUT_MS / "Fig8_IB_framework_v3.pdf" : the 2x2 information-bottleneck figure
    (OUT_MS = OUT.parent / "figures_for_manuscript" / "Fig8_IB_framework")
  - OUT_MS / "Fig8_IB_framework_v3.png" : PNG copy of the same figure
  - OUT_DATA / "31_fig8_v3_data.csv" : merged per-family table used for the figure

Usage:
  python analyses/scripts/S14_ib_framework.py
  No command-line arguments.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

from _paths import OUT_DATA, OUT

OUT_MS = OUT.parent / "figures_for_manuscript" / "Fig8_IB_framework"
OUT_MS.mkdir(parents=True, exist_ok=True)

ib  = pd.read_csv(OUT_DATA / "22_information_bottleneck.csv")
mi  = pd.read_csv(OUT_DATA / "24_mutual_information.csv")
div = pd.read_csv(OUT_DATA / "15_diversity_summary.csv")
knn = pd.read_csv(OUT_DATA / "30_knn_overlap_summary.csv")

df = ib.merge(mi[["family", "MI_nats", "MI_ci_lo", "MI_ci_hi"]], on="family")
df = df.merge(div[["family", "mean 3Di-Hamming"]], on="family")
df = df.merge(knn[["family", "mean_O5", "mean_O10", "mean_O20"]], on="family")
df = df.rename(columns={"mean 3Di-Hamming": "h3di"})
df_sorted = df.sort_values("MI_nats", ascending=False).reset_index(drop=True)
print(df.to_string(index=False))

# Correlations
r_mi_h, p_mi_h = pearsonr(df["MI_nats"], df["h3di"])
r_o_mi, p_o_mi = pearsonr(df["mean_O10"], df["MI_nats"])
r_rho_h, p_rho_h = pearsonr(df["rho_AA_vs_3Di_latent"], df["h3di"])
slope_mi, intercept_mi = np.polyfit(df["h3di"], df["MI_nats"], 1)

print(f"\nMI vs 3Di-Hamming    : r={r_mi_h:.3f}  p={p_mi_h:.3f}")
print(f"O10 vs MI            : r={r_o_mi:.3f}  p={p_o_mi:.3f}")
print(f"Spearman ρ vs 3Di-Ham: r={r_rho_h:.3f}  p={p_rho_h:.3f}")


# ============================================================
# Build 2x2 figure
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11),
                          gridspec_kw=dict(wspace=0.30, hspace=0.36))

# --- A: IB plane ---
ax = axes[0, 0]
for _, r in df.iterrows():
    ax.plot([r["rho_AA_seq"], r["rho_3Di_seq"]],
            [r["rho_AA_struct"], r["rho_3Di_struct"]],
            "-", color="0.5", alpha=0.5, lw=1.2, zorder=1)
    ax.scatter(r["rho_AA_seq"], r["rho_AA_struct"],
                s=120, c="#1f77b4", edgecolor="white",
                linewidth=1.0, marker="o", zorder=3)
    ax.scatter(r["rho_3Di_seq"], r["rho_3Di_struct"],
                s=120, c="#ff7f0e", edgecolor="white",
                linewidth=1.0, marker="s", zorder=3)
    mx = 0.5*(r["rho_AA_seq"]+r["rho_3Di_seq"])
    my = 0.5*(r["rho_AA_struct"]+r["rho_3Di_struct"])
    ax.annotate(r["family"], (mx, my), xytext=(7, 7),
                 textcoords="offset points",
                 fontsize=10, fontweight="bold")
ax.axhline(0, ls=":", color="0.7", lw=0.7)
ax.axvline(0, ls=":", color="0.7", lw=0.7)
xlim = ax.get_xlim(); ylim = ax.get_ylim()
m_lim = max(xlim[1], ylim[1]); lo = min(xlim[0], ylim[0])
ax.plot([lo, m_lim], [lo, m_lim], "--", color="0.7", lw=0.7, zorder=0)
ax.legend([Line2D([0], [0], marker="o", color="w",
                    markerfacecolor="#1f77b4", markersize=10,
                    markeredgecolor="white", markeredgewidth=1.0),
           Line2D([0], [0], marker="s", color="w",
                    markerfacecolor="#ff7f0e", markersize=10,
                    markeredgecolor="white", markeredgewidth=1.0),
           Line2D([0], [0], color="0.5", lw=1.2)],
          ["AA-VAE", "3Di-VAE", "AA → 3Di shift"],
          loc="upper left", fontsize=9, framealpha=0.9)
ax.set_xlabel(r"$\rho_{\rm seq}$  (latent ↔ AA-Hamming)", fontsize=10)
ax.set_ylabel(r"$\rho_{\rm struct}$  (latent ↔ 3Di-Hamming)", fontsize=10)
ax.set_title("A.  Information-bottleneck plane",
              fontsize=12, fontweight="bold")

# --- B: MI bars (no error bars: point estimates only) ---
ax = axes[0, 1]
x = np.arange(len(df_sorted))
ax.bar(x, df_sorted["MI_nats"], color="#9467bd", alpha=0.85,
       edgecolor="black", linewidth=0.5)
for i, r in df_sorted.iterrows():
    ax.text(i, r["MI_nats"] + 0.07, f"{r['MI_nats']:.2f}",
            ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(df_sorted["family"])
ax.set_ylabel(r"$I(\mu_{AA}; \mu_{3Di})$  (nats)", fontsize=10)
ax.set_title("B.  Cross-latent mutual information",
              fontsize=12, fontweight="bold")
ymax = max(df_sorted["MI_nats"]) * 1.22
ax.set_ylim(0, ymax)
ax.axhline(0.5, ls=":", color="0.5", lw=0.6)
ax.text(0.02, 0.95, "redundant\nlatents",
         transform=ax.transAxes, fontsize=8, color="0.3",
         ha="left", va="top", style="italic",
         bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
ax.text(0.98, 0.18, "orthogonal\nlatents",
         transform=ax.transAxes, fontsize=8, color="0.3",
         ha="right", va="top", style="italic",
         bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

# --- C: predictive law (MI vs 3Di-Hamming) ---
ax = axes[1, 0]
ax.scatter(df_sorted["h3di"], df_sorted["MI_nats"],
            s=170, c="#2ca02c", edgecolor="black",
            linewidth=0.6, zorder=3)
for _, r in df_sorted.iterrows():
    ax.annotate(r["family"], (r["h3di"], r["MI_nats"]),
                 xytext=(7, 5), textcoords="offset points",
                 fontsize=10, fontweight="bold")
xs = np.linspace(df_sorted["h3di"].min(), df_sorted["h3di"].max(), 50)
ax.plot(xs, slope_mi*xs + intercept_mi, "--", color="0.4", lw=1.4)
ax.text(0.04, 0.05,
         f"Pearson r = {r_mi_h:.2f}  (p = {p_mi_h:.3f})\n"
         f"slope = {slope_mi:.1f} nats / unit Hamming\n"
         f"(Spearman ρ vs Hamming: r = {r_rho_h:.2f}, p = {p_rho_h:.3f})",
         transform=ax.transAxes, fontsize=9, va="bottom",
         bbox=dict(boxstyle="round", facecolor="white",
                    alpha=0.9, edgecolor="0.7"))
ax.set_xlabel("3Di-Hamming  (structural diversity)", fontsize=10)
ax.set_ylabel(r"$I(\mu_{AA}; \mu_{3Di})$  (nats)", fontsize=10)
ax.set_title("C.  Predicting the regime from\nraw structural diversity",
              fontsize=12, fontweight="bold")

# --- D: validation (MI vs mean k-NN overlap) ---
ax = axes[1, 1]
ax.scatter(df_sorted["mean_O10"], df_sorted["MI_nats"],
            s=170, c="#d62728", edgecolor="black",
            linewidth=0.6, zorder=3)
for _, r in df_sorted.iterrows():
    ax.annotate(r["family"], (r["mean_O10"], r["MI_nats"]),
                 xytext=(7, 5), textcoords="offset points",
                 fontsize=10, fontweight="bold")
slope_o, intercept_o = np.polyfit(df_sorted["mean_O10"],
                                    df_sorted["MI_nats"], 1)
xs_o = np.linspace(df_sorted["mean_O10"].min(),
                    df_sorted["mean_O10"].max(), 50)
ax.plot(xs_o, slope_o*xs_o + intercept_o, "--", color="0.4", lw=1.4)
ax.text(0.04, 0.95,
         f"Pearson r = {r_o_mi:.3f}  (p = {p_o_mi:.3f})\n"
         f"two independent metrics agree\n"
         f"on the cross-latent regime",
         transform=ax.transAxes, fontsize=9, va="top",
         bbox=dict(boxstyle="round", facecolor="white",
                    alpha=0.9, edgecolor="0.7"))
ax.set_xlabel(r"Mean $k$-NN overlap $\langle O_{10}(i) \rangle$  "
               "(geometric)", fontsize=10)
ax.set_ylabel(r"$I(\mu_{AA}; \mu_{3Di})$  (nats, info-theoretic)",
               fontsize=10)
ax.set_title("D.  Cross-metric validation:\n"
              "MI and neighborhood overlap agree",
              fontsize=12, fontweight="bold")

fig.suptitle("Information-bottleneck framework: when are AA- and "
              "3Di-aware representations complementary?",
              fontsize=13, fontweight="bold", y=1.005)
out_fig = OUT_MS / "Fig8_IB_framework_v3.pdf"
fig.savefig(out_fig, bbox_inches="tight", dpi=150)
fig.savefig(str(out_fig).replace(".pdf", ".png"),
             bbox_inches="tight", dpi=180)
print(f"\nSaved {out_fig}")

df.to_csv(OUT_DATA / "31_fig8_v3_data.csv", index=False)
print(f"Saved data to {OUT_DATA/'31_fig8_v3_data.csv'}")
