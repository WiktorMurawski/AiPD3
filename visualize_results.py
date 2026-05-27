import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

C_BLUE   = "#3266AD"
C_RED    = "#C0533A"
C_AMBER  = "#D4953A"
C_GREEN  = "#4A8C3F"
C_GRAY   = "#888780"
C_LIGHT  = "#F1EFE8"


def load(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def short_label(label: str, n: int = 7) -> str:
    """Skraca etykietę do n znaków dla osi wykresów."""
    return label[:n]


def plot_confusion_matrix(
    cm:       list[list[int]],
    labels:   list[str],
    title:    str,
    ax:       plt.Axes,
    annotate: bool = True,
) -> None:
    cm_arr   = np.array(cm)
    n        = len(labels)
    row_sums = cm_arr.sum(axis=1, keepdims=True).clip(1)
    cm_norm  = cm_arr / row_sums

    img_data = np.zeros((n, n, 4))
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if i == j:
                img_data[i, j] = [1 - v*0.55, 1 - v*0.2, 1 - v*0.55, 1]
            elif cm_arr[i, j] > 0:
                img_data[i, j] = [1, 1 - v*0.7, 1 - v*0.7, 1]
            else:
                img_data[i, j] = [0.97, 0.97, 0.97, 1]

    ax.imshow(img_data, aspect="auto")

    if annotate:
        for i in range(n):
            for j in range(n):
                val = cm_arr[i, j]
                if val == 0:
                    continue
                pct = cm_norm[i, j] * 100
                color = "white" if (i == j and pct > 60) else "#222"
                ax.text(j, i, f"{val}\n({pct:.0f}%)",
                        ha="center", va="center",
                        fontsize=7.5, color=color, fontweight="bold" if i == j else "normal")

    short = [short_label(l) for l in labels]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticklabels(short)
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Prawdziwa klasa")
    ax.set_title(title)

    for i in range(n):
        ax.add_patch(mpatches.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor="#555", linewidth=0.8, linestyle="--"
        ))


def save_confusion_matrices(rep: dict, spk: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_confusion_matrix(
        rep["confusion_matrix"], rep["labels"],
        f"Tryb repetition — dokładność {rep['accuracy']*100:.1f}%",
        axes[0],
    )
    plot_confusion_matrix(
        spk["confusion_matrix"], spk["labels"],
        f"Tryb speaker-independent — dokładność {spk['accuracy']*100:.1f}%",
        axes[1],
    )

    fig.suptitle("Macierze pomyłek — MFCC + DTW", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = out_dir / "01_confusion_matrices.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Zapisano: {path}")


def save_f1_comparison(rep: dict, spk: dict, out_dir: Path) -> None:
    labels  = rep["labels"]
    n       = len(labels)
    f1_rep  = [rep["per_class"][l]["f1"]  for l in labels]
    f1_spk  = [spk["per_class"][l]["f1"]  for l in labels]
    prec_spk = [spk["per_class"][l]["precision"] for l in labels]
    rec_spk  = [spk["per_class"][l]["recall"]    for l in labels]

    x   = np.arange(n)
    w   = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bars1 = ax.bar(x - w/2, f1_rep, w, label="repetition",        color=C_BLUE,  alpha=0.85, zorder=3)
    bars2 = ax.bar(x + w/2, f1_spk, w, label="speaker-independent", color=C_RED, alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([short_label(l) for l in labels], rotation=45, ha="right")
    ax.set_ylim(0.6, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.axhline(1.0, color="#ccc", linewidth=0.8, linestyle="--")
    ax.set_ylabel("F1-score")
    ax.set_title("F1 per klasa — porównanie trybów")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    for bar, val in zip(bars2, f1_spk):
        if val < 0.99:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=C_RED)

    ax2 = axes[1]
    ax2.bar(x - w/2, prec_spk, w, label="precision", color=C_BLUE,  alpha=0.85, zorder=3)
    ax2.bar(x + w/2, rec_spk,  w, label="recall",    color=C_GREEN, alpha=0.85, zorder=3)

    ax2.set_xticks(x)
    ax2.set_xticklabels([short_label(l) for l in labels], rotation=45, ha="right")
    ax2.set_ylim(0.6, 1.05)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.axhline(1.0, color="#ccc", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Wartość metryki")
    ax2.set_title("Precision / Recall — tryb speaker-independent")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    path = out_dir / "02_f1_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Zapisano: {path}")


def save_accuracy_per_speaker(spk: dict, out_dir: Path) -> None:
    preds = spk["predictions"]

    from collections import defaultdict
    speaker_data: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for p in preds:
        spk_id = p["speaker"]
        speaker_data[spk_id]["total"]   += 1
        speaker_data[spk_id]["correct"] += int(p["correct"])

    speakers = sorted(speaker_data.keys())
    accs     = [speaker_data[s]["correct"] / speaker_data[s]["total"] for s in speakers]
    totals   = [speaker_data[s]["total"] for s in speakers]

    SPEAKER_FLAGS = {
        "01": ("clipping (przestery)", C_RED),
        "05": ("specyficzna wymowa",   C_AMBER),
    }
    colors = [SPEAKER_FLAGS.get(s, (None, C_BLUE))[1] for s in speakers]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    bars = ax.bar(speakers, [a * 100 for a in accs], color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5, zorder=3)

    mean_acc = sum(accs) / len(accs) * 100
    ax.axhline(mean_acc, color=C_GRAY, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(len(speakers) - 0.3, mean_acc + 0.5,
            f"średnia {mean_acc:.1f}%", ha="right", va="bottom",
            fontsize=9, color=C_GRAY)

    for bar, acc, total in zip(bars, accs, totals):
        correct = round(acc * total)
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.4,
                f"{acc*100:.0f}%\n({correct}/{total})",
                ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 115)
    ax.set_xlabel("Mówca (ID)")
    ax.set_ylabel("Dokładność (%)")
    ax.set_title("Dokładność per mówca — tryb speaker-independent")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    legend_patches = [
        mpatches.Patch(color=C_RED,   label="clipping (przestery) — mówca 01"),
        mpatches.Patch(color=C_AMBER, label="specyficzna wymowa — mówca 05"),
        mpatches.Patch(color=C_BLUE,  label="pozostali mówcy"),
    ]
    ax.legend(handles=legend_patches, frameon=False, fontsize=9,
              loc="lower right")

    plt.tight_layout()
    path = out_dir / "03_accuracy_per_speaker.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Zapisano: {path}")


def save_dtw_distributions(rep: dict, spk: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, data, title in [
        (axes[0], rep, "Tryb repetition"),
        (axes[1], spk, "Tryb speaker-independent"),
    ]:
        preds = data["predictions"]
        dists_ok  = [p["dist"] for p in preds if     p["correct"]]
        dists_err = [p["dist"] for p in preds if not p["correct"]]

        bins = np.linspace(
            min(dists_ok + dists_err) * 0.9,
            max(dists_ok + dists_err) * 1.05,
            30,
        )

        ax.hist(dists_ok,  bins=bins, alpha=0.7, color=C_BLUE,  label=f"poprawne (n={len(dists_ok)})",  zorder=3)
        ax.hist(dists_err, bins=bins, alpha=0.8, color=C_RED,   label=f"błędne   (n={len(dists_err)})", zorder=4)

        med_ok  = np.median(dists_ok)
        ax.axvline(med_ok,  color=C_BLUE,  linestyle="--", linewidth=1.2,
                   label=f"mediana poprawnych: {med_ok:.1f}")
        if dists_err:
            med_err = np.median(dists_err)
            ax.axvline(med_err, color=C_RED,   linestyle="--", linewidth=1.2,
                       label=f"mediana błędnych: {med_err:.1f}")

        ax.set_xlabel("Dystans DTW (znormalizowany)")
        ax.set_ylabel("Liczba nagrań")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8.5)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("Rozkład dystansów DTW — poprawne vs błędne klasyfikacje",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "04_dtw_distance_distributions.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Zapisano: {path}")


def save_dtw_per_class(rep: dict, out_dir: Path) -> None:
    labels = rep["labels"]
    preds  = rep["predictions"]

    from collections import defaultdict
    dists_by_class: dict[str, list] = defaultdict(list)
    for p in preds:
        if p["correct"]:
            dists_by_class[p["true"]].append(p["dist"])

    data = [dists_by_class.get(l, []) for l in labels]

    fig, ax = plt.subplots(figsize=(12, 5))

    bp = ax.boxplot(
        data,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=C_GRAY),
        capprops=dict(color=C_GRAY),
        flierprops=dict(marker="o", markersize=4, markerfacecolor=C_RED,
                        markeredgecolor=C_RED, alpha=0.6),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(C_BLUE)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([short_label(l) for l in labels], rotation=45, ha="right")
    ax.set_ylabel("Dystans DTW (znormalizowany)")
    ax.set_title("Rozkład dystansów DTW per cyfra — tryb repetition (poprawne klasyfikacje)")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    medians = [np.median(d) if d else 0 for d in data]
    worst_i = int(np.argmax(medians))
    ax.annotate(
        f"najwyższy medianowy\ndystans: {medians[worst_i]:.1f}",
        xy=(worst_i + 1, medians[worst_i]),
        xytext=(worst_i + 2.5, medians[worst_i] + 1),
        arrowprops=dict(arrowstyle="->", color=C_GRAY),
        fontsize=9, color=C_GRAY,
    )

    plt.tight_layout()
    path = out_dir / "05_dtw_distances_per_class.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Zapisano: {path}")


def save_summary_figure(rep: dict, spk: dict, out_dir: Path) -> None:
    labels  = rep["labels"]
    f1_rep  = [rep["per_class"][l]["f1"] for l in labels]
    f1_spk  = [spk["per_class"][l]["f1"] for l in labels]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Wyniki systemu rozpoznawania cyfr polskich — MFCC + DTW",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_confusion_matrix(rep["confusion_matrix"], labels,
                          f"Repetition — {rep['accuracy']*100:.1f}%", ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_confusion_matrix(spk["confusion_matrix"], labels,
                          f"Speaker-independent — {spk['accuracy']*100:.1f}%", ax2)

    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(labels))
    w = 0.35
    ax3.bar(x - w/2, f1_rep, w, label="repetition",          color=C_BLUE,  alpha=0.85)
    ax3.bar(x + w/2, f1_spk, w, label="speaker-independent", color=C_RED,   alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels([short_label(l) for l in labels], rotation=45, ha="right")
    ax3.set_ylim(0.6, 1.08)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax3.set_title("F1 per klasa")
    ax3.legend(frameon=False, fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    preds   = rep["predictions"]
    dists_ok  = [p["dist"] for p in preds if     p["correct"]]
    dists_err = [p["dist"] for p in preds if not p["correct"]]
    bins = np.linspace(min(dists_ok + dists_err)*0.9, max(dists_ok + dists_err)*1.05, 25)
    ax4.hist(dists_ok,  bins=bins, alpha=0.7, color=C_BLUE, label=f"poprawne (n={len(dists_ok)})")
    ax4.hist(dists_err, bins=bins, alpha=0.8, color=C_RED,  label=f"błędne (n={len(dists_err)})")
    ax4.axvline(np.median(dists_ok), color=C_BLUE, linestyle="--", linewidth=1.2)
    ax4.set_xlabel("Dystans DTW")
    ax4.set_ylabel("Liczba nagrań")
    ax4.set_title("Rozkład dystansów DTW (repetition)")
    ax4.legend(frameon=False, fontsize=8)
    ax4.grid(axis="y", alpha=0.3)

    path = out_dir / "00_summary.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Zapisano: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generuj wykresy wyników MFCC+DTW")
    parser.add_argument("--results-dir", default="results",
                        help="Folder z plikami JSON (domyślnie: results)")
    parser.add_argument("--out-dir",     default="figures",
                        help="Folder na wygenerowane PNG (domyślnie: figures)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    rep_path = results_dir / "results_repetition.json"
    spk_path = results_dir / "results_speaker.json"

    if not rep_path.exists():
        print(f"Brak pliku: {rep_path}")
        return
    if not spk_path.exists():
        print(f"Brak pliku: {spk_path}")
        return

    rep = load(rep_path)
    spk = load(spk_path)

    print(f"Generowanie wykresów → {out_dir}/")
    save_confusion_matrices(rep, spk, out_dir)
    save_f1_comparison(rep, spk, out_dir)
    save_accuracy_per_speaker(spk, out_dir)
    save_dtw_distributions(rep, spk, out_dir)
    save_dtw_per_class(rep, out_dir)
    save_summary_figure(rep, spk, out_dir)
    print(f"\nGotowe! Wygenerowano 6 plików PNG w katalogu '{out_dir}/'.")


if __name__ == "__main__":
    main()
