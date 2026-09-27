"""
PRAHARI — end-to-end evaluation. Produces the figure set for Review 1.

Runs the full four-layer chain over the corpus and writes:

    docs/figures/fig1_confusion.png            Layer 1
    docs/figures/fig2_severity.png             Layer 2
    docs/figures/fig3_restoration.png          Layer 3 (independent metrics)
    docs/figures/fig4_hr_error.png             Layer 4 — clinical impact
    docs/figures/fig5_gating_tradeoff.png      Layer 4 — THE headline figure
    docs/figures/results.json                  every number, for the paper

Usage
-----
    python -m ml.evaluate                 # full corpus
    python -m ml.evaluate --max 1500      # quick pass
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hub.contracts import CLASSES
from hub.pipeline import Prahari
from hub import restore as restore_mod
from hub import hr as hr_mod
from hub import gate as gate_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data", "synthetic", "corpus.npz")
FIGS = os.path.join(ROOT, "docs", "figures")

INK = "#111A18"
ACCENT = "#0B6E5F"
AMBER = "#8A5A0E"
CRIT = "#9B2A2A"
BLUE = "#2F4C77"
PALETTE = [ACCENT, BLUE, AMBER, "#6B3356", CRIT]


def style(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)
    ax.grid(alpha=0.18, lw=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--max", type=int, default=0, help="subsample for speed")
    args = ap.parse_args()

    os.makedirs(FIGS, exist_ok=True)
    d = np.load(args.corpus, allow_pickle=True)
    X, X_ref, y, sev = d["X"], d["X_ref"], d["y"], d["severity"]

    if args.max and args.max < len(y):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), args.max, replace=False)
        X, X_ref, y, sev = X[idx], X_ref[idx], y[idx], sev[idx]

    print(f"  evaluating {len(y)} windows through the full chain ...", flush=True)
    P = Prahari()

    pred, sti, sev_hat = [], [], []
    hr_err_raw, hr_err_restored = [], []
    rest_rows = []

    for i in range(len(y)):
        w, ref = X[i].astype(np.float64), X_ref[i].astype(np.float64)
        v = P.diagnose(w)
        pred.append(v.cls)
        sti.append(v.sti)
        sev_hat.append(v.severity)

        restored = P.restore(w, v) if v.cls != "Clean" else w

        hr_err_raw.append(hr_mod.hr_error(w, ref))
        hr_err_restored.append(hr_mod.hr_error(restored, ref))

        if y[i] != "Clean":
            m = restore_mod.evaluate(w, restored, ref,
                                     classifier_says_clean=(P.diagnose(restored).cls == "Clean"))
            m["true_class"] = str(y[i])
            m["true_severity"] = float(sev[i])
            rest_rows.append(m)

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(y)}", flush=True)

    pred = np.asarray(pred)
    sti = np.asarray(sti, dtype=float)
    sev_hat = np.asarray(sev_hat, dtype=float)
    hr_err_raw = np.asarray(hr_err_raw, dtype=float)
    hr_err_restored = np.asarray(hr_err_restored, dtype=float)

    results = {}

    # ------------------------------------------------------- fig 1 confusion
    cm = confusion_matrix(y, pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(cm, cmap="GnBu")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8.5)
    thresh = cm.max() / 2 if cm.max() else 1
    for r in range(len(CLASSES)):
        for c in range(len(CLASSES)):
            ax.text(c, r, cm[r, c], ha="center", va="center", fontsize=8,
                    color="white" if cm[r, c] > thresh else INK)
    ax.set_title("Layer 1 — degradation classification", fontsize=11, pad=10)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(f"{FIGS}/fig1_confusion.png", dpi=200); plt.close(fig)

    acc = float((pred == y).mean())
    results["layer1"] = {"accuracy": acc,
                         "confusion": cm.tolist(), "classes": CLASSES}
    print(f"\n  Layer 1 accuracy: {acc*100:.2f}%")

    # -------------------------------------------------------- fig 2 severity
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for k, c in enumerate([c for c in CLASSES if c != "Clean"]):
        m = (y == c)
        if not m.any():
            continue
        jitter = (np.random.default_rng(k).random(m.sum()) - 0.5) * 0.04
        ax.scatter(sev[m] + jitter, sev_hat[m], s=7, alpha=0.32,
                   color=PALETTE[k], label=c, edgecolors="none")
    ax.plot([0, 1], [0, 1], color=INK, lw=1, ls="--", label="ideal")
    style(ax, "Layer 2 — severity regression", "true severity", "predicted severity")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIGS}/fig2_severity.png", dpi=200); plt.close(fig)

    deg = y != "Clean"
    sev_mae = float(np.mean(np.abs(sev_hat[deg] - sev[deg])))
    results["layer2"] = {"severity_mae_degraded": sev_mae}
    print(f"  Layer 2 severity MAE (degraded windows): {sev_mae:.4f}")

    # ----------------------------------------------------- fig 3 restoration
    by_class = defaultdict(list)
    for r in rest_rows:
        by_class[r["true_class"]].append(r)

    faults = [c for c in CLASSES if c != "Clean"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    gains = [[r["snr_gain"] for r in by_class[c]] for c in faults]
    bp = axes[0].boxplot(gains, tick_labels=faults, patch_artist=True, widths=0.55)
    for patch, col in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(col); patch.set_alpha(0.55); patch.set_edgecolor(INK)
    for med in bp["medians"]:
        med.set_color(INK)
    axes[0].axhline(0, color=CRIT, lw=1, ls="--")
    style(axes[0], "Layer 3 — SNR improvement (independent of classifier)",
          "", "SNR gain (dB)")

    width = 0.38
    xs = np.arange(len(faults))
    before = [np.mean([r["corr_before"] for r in by_class[c]]) for c in faults]
    after = [np.mean([r["corr_after"] for r in by_class[c]]) for c in faults]
    axes[1].bar(xs - width / 2, before, width, label="degraded",
                color=AMBER, alpha=0.75, edgecolor=INK, lw=0.5)
    axes[1].bar(xs + width / 2, after, width, label="restored",
                color=ACCENT, alpha=0.9, edgecolor=INK, lw=0.5)
    axes[1].set_xticks(xs, faults)
    axes[1].set_ylim(0, 1.05)
    style(axes[1], "Correlation with pre-degradation window", "", "Pearson r")
    axes[1].legend(fontsize=8.5, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIGS}/fig3_restoration.png", dpi=200); plt.close(fig)

    results["layer3"] = {
        c: {
            "snr_gain_mean": float(np.mean([r["snr_gain"] for r in by_class[c]])),
            "corr_before_mean": float(before[i]),
            "corr_after_mean": float(after[i]),
            "reclassified_clean_rate": float(
                np.mean([r["reclassified_clean"] for r in by_class[c]])),
            "n": len(by_class[c]),
        } for i, c in enumerate(faults) if by_class[c]
    }
    print("\n  Layer 3 restoration (SNR gain dB / corr before->after):")
    for c in faults:
        if by_class[c]:
            r = results["layer3"][c]
            print(f"    {c:<12} {r['snr_gain_mean']:+6.2f} dB   "
                  f"{r['corr_before_mean']:.3f} -> {r['corr_after_mean']:.3f}   "
                  f"(reclassified clean: {r['reclassified_clean_rate']*100:.0f}%)")

    # ------------------------------------------------------- fig 4 HR error
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    levels = sorted(set(np.round(sev[deg], 2)))
    for k, c in enumerate(faults):
        ms, xs_ = [], []
        for lv in levels:
            m = (y == c) & (np.round(sev, 2) == lv) & np.isfinite(hr_err_raw)
            if m.sum() >= 3:
                ms.append(np.median(hr_err_raw[m])); xs_.append(lv)
        if xs_:
            ax.plot(xs_, ms, "o-", color=PALETTE[k], label=c, lw=1.6, ms=5)
    ax.axhline(gate_mod.MISLEADING_BPM, color=CRIT, ls="--", lw=1.2,
               label=f"misleading threshold ({gate_mod.MISLEADING_BPM:.0f} bpm)")
    style(ax, "Layer 4 — heart-rate error induced by degradation",
          "true severity", "median |HR error| (bpm)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIGS}/fig4_hr_error.png", dpi=200); plt.close(fig)

    # --------------------------------------------- fig 5 gating trade-off
    rows_raw = gate_mod.sweep(sti, hr_err_raw)
    rows_res = gate_mod.sweep(sti, hr_err_restored)
    summary = gate_mod.summarise(rows_raw)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot([r["availability"] for r in rows_raw],
            [r["misleading_released"] * 100 for r in rows_raw],
            "-o", ms=3.4, color=ACCENT, label="gated, no restoration")
    ax.plot([r["availability"] for r in rows_res],
            [r["misleading_released"] * 100 for r in rows_res],
            "-s", ms=3.4, color=BLUE, label="gated + restoration")
    base = rows_raw[0]
    ax.scatter([base["availability"]], [base["misleading_released"] * 100],
               s=90, color=CRIT, zorder=5, label="ungated device (baseline)")
    op = summary["operating_point"]
    ax.scatter([op["availability"]], [op["misleading_released"] * 100],
               s=90, marker="*", color=INK, zorder=5,
               label=f"operating point (STI>={op['threshold']})")
    style(ax, "Layer 4 — what gating buys",
          "availability (fraction of windows reporting an HR)",
          "clinically misleading readings released (%)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIGS}/fig5_gating_tradeoff.png", dpi=200); plt.close(fig)

    results["layer4"] = {
        "sweep_raw": rows_raw,
        "sweep_restored": rows_res,
        "summary": summary,
        "misleading_bpm": gate_mod.MISLEADING_BPM,
    }

    print(f"\n  Layer 4 — gating")
    print(f"    ungated baseline : {base['misleading_released']*100:.2f}% of windows "
          f"released a misleading HR")
    print(f"    operating point  : STI >= {op['threshold']}  ->  "
          f"{op['misleading_released']*100:.2f}% misleading, "
          f"{op['availability']*100:.1f}% availability")
    if np.isfinite(summary["misleading_reduction"]):
        print(f"    reduction        : {summary['misleading_reduction']*100:.1f}% "
              f"fewer misleading readings, at a cost of "
              f"{summary['availability_cost']*100:.1f} points of availability")
    print(f"    precision        : {op['precision']*100:.1f}% of released readings "
          f"are within tolerance (baseline {base['precision']*100:.1f}%)")
    res0 = rows_res[0]
    print(f"    restoration alone: {res0['misleading_released']*100:.2f}% misleading "
          f"at {res0['availability']*100:.0f}% availability "
          f"(vs {base['misleading_released']*100:.2f}% ungated)")
    results_restore_only = {
        "misleading_released": res0["misleading_released"],
        "availability": res0["availability"],
        "reduction_vs_ungated": (1 - res0["misleading_released"] / base["misleading_released"])
        if base["misleading_released"] > 0 else float("nan"),
    }
    print("    NOTE: this corpus is ~80% degraded by construction, so availability "
          "here is pessimistic.\n          Re-run the sweep on real recordings "
          "before quoting an operating point.")

    with open(f"{FIGS}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  wrote 5 figures + results.json to docs/figures/")


if __name__ == "__main__":
    main()
