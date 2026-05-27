"""
evaluate.py

Ewaluacja klasyfikatora MFCC + DTW na zbiorze testowym.

Uruchomienie:
    python evaluate.py --data data/raw --split repetition
    python evaluate.py --data data/raw --split speaker --test-speakers 01 02 03
"""

import sys
import argparse
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from dataset import load_dataset, ALL_LABELS
from dtw import DTWClassifier


def confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> np.ndarray:
    """Zwraca macierz pomyłek (wiersze = prawdziwe, kolumny = predykowane)."""
    idx = {label: i for i, label in enumerate(labels)}
    n   = len(labels)
    cm  = np.zeros((n, n), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if true in idx and pred in idx:
            cm[idx[true], idx[pred]] += 1
    return cm


def print_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    """Wypisuje macierz pomyłek w czytelnym formacie tekstowym."""
    short = [l[:5].ljust(5) for l in labels]
    col_w = 6

    header = " " * 10 + "  ".join(f"{s:>{col_w}}" for s in short)
    print(header)
    print("-" * len(header))

    for i, label in enumerate(labels):
        row_vals = "  ".join(f"{cm[i, j]:>{col_w}}" for j in range(len(labels)))
        marker   = "  <-" if any(cm[i, j] > 0 for j in range(len(labels)) if j != i) else ""
        print(f"{label[:9]:<10}{row_vals}{marker}")


def per_class_stats(
    cm: np.ndarray,
    labels: list[str],
) -> dict[str, dict]:
    """
    Oblicza precision, recall i F1 dla każdej klasy.

    Precision = TP / (TP + FP)  - ile predykcji danej klasy jest trafnych
    Recall    = TP / (TP + FN)  - ile próbek danej klasy zostało wykrytych
    F1        = 2 * P * R / (P + R)
    """
    stats = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        stats[label] = {
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   int(cm[i, :].sum()),
        }
    return stats


def evaluate(
    data_dir:      str | Path,
    split:         str        = "repetition",
    test_rep:      int        = 2,
    test_speakers: list[str] | None = None,
    dtw_metric:    str        = "euclidean",
    dtw_window:    int        = 20,
    results_dir:   str | Path = "results",
    verbose:       bool       = True,
) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    templates, test_recs = load_dataset(
        data_dir,
        split=split,
        test_rep=test_rep,
        test_speakers=test_speakers,
        verbose=verbose,
    )

    clf = DTWClassifier(metric=dtw_metric, window=dtw_window)
    clf.fit(templates)

    print(f"\nKlasyfikacja {len(test_recs)} nagrań testowych...")
    t0 = time.time()

    y_true, y_pred, distances = [], [], []
    errors_by_label = defaultdict(list)

    for i, rec in enumerate(test_recs):
        label_pred, dist = clf.predict(rec.features)
        y_true.append(rec.label)
        y_pred.append(label_pred)
        distances.append(dist)

        if label_pred != rec.label:
            errors_by_label[rec.label].append(
                f"{rec.speaker}/{rec.path.name} → predykcja: '{label_pred}'"
            )

        if verbose and (i % 50 == 0 or i == len(test_recs) - 1):
            print(f"  {i+1}/{len(test_recs)} ({(i+1)/len(test_recs)*100:.0f}%)")

    elapsed = time.time() - t0

    cm      = confusion_matrix(y_true, y_pred, ALL_LABELS)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true) if y_true else 0.0
    stats   = per_class_stats(cm, ALL_LABELS)

    print(f"\n{'='*60}")
    print(f"  WYNIKI EWALUACJI")
    print(f"{'='*60}")
    print(f"  Podział:          {split}")
    print(f"  Metryka DTW:      {dtw_metric}")
    print(f"  Okno Sakoe-Chiba: {dtw_window}")
    print(f"  Nagrań testowych: {len(test_recs)}")
    print(f"  Czas klasyfikacji:{elapsed:.1f}s  ({elapsed/len(test_recs)*1000:.0f} ms/nagranie)")
    print(f"\n  *** DOKŁADNOŚĆ: {accuracy*100:.1f}% ({correct}/{len(y_true)}) ***")
    print(f"{'='*60}\n")

    print("Statystyki per klasa:")
    print(f"{'Klasa':<12} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Wsparcie':>9}")
    print("-" * 45)
    for label in ALL_LABELS:
        s = stats[label]
        print(f"{label:<12} {s['precision']:>6.3f} {s['recall']:>7.3f} "
              f"{s['f1']:>6.3f} {s['support']:>9}")

    print("\nMacierz pomyłek (wiersze=prawdziwe, kolumny=predykowane):")
    print_confusion_matrix(cm, ALL_LABELS)

    if errors_by_label:
        print(f"\nBłędne klasyfikacje ({sum(len(v) for v in errors_by_label.values())} łącznie):")
        for label in ALL_LABELS:
            if errors_by_label[label]:
                print(f"  '{label}':")
                for err in errors_by_label[label]:
                    print(f"    {err}")

    results = {
        "config": {
            "split":         split,
            "test_rep":      test_rep,
            "test_speakers": test_speakers,
            "dtw_metric":    dtw_metric,
            "dtw_window":    dtw_window,
        },
        "accuracy":     round(accuracy, 4),
        "n_correct":    correct,
        "n_total":      len(y_true),
        "elapsed_s":    round(elapsed, 2),
        "confusion_matrix": cm.tolist(),
        "labels":       ALL_LABELS,
        "per_class":    stats,
        "predictions":  [
            {"file":    str(r.path), "speaker": r.speaker,
             "true":    t,           "pred":    p,
             "dist":    round(d, 4), "correct": t == p}
            for r, t, p, d in zip(test_recs, y_true, y_pred, distances)
        ],
    }

    out_path = results_dir / f"results_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWyniki zapisane do: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Ewaluacja klasyfikatora cyfr MFCC+DTW"
    )
    parser.add_argument(
        "--data", default="data/raw",
        help="Folder z nagraniami (domyślnie: data/raw)"
    )
    parser.add_argument(
        "--split", choices=["repetition", "speaker"],
        default="repetition",
        help="Strategia podziału: repetition (domyślna) lub speaker"
    )
    parser.add_argument(
        "--test-rep", type=int, default=2,
        help="(split=repetition) Które nagranie do testu: 1 lub 2 (domyślnie: 2)"
    )
    parser.add_argument(
        "--test-speakers", nargs="+", default=None,
        help="(split=speaker) Nazwy mówców testowych, np. speaker_1 speaker_2"
    )
    parser.add_argument(
        "--metric", choices=["euclidean", "cosine"],
        default="euclidean",
        help="Miara odległości DTW (domyślnie: euclidean)"
    )
    parser.add_argument(
        "--window", type=int, default=20,
        help="Okno Sakoe-Chiba w ramkach (domyślnie: 20)"
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Folder na wyniki JSON (domyślnie: results)"
    )

    args = parser.parse_args()

    evaluate(
        data_dir      = args.data,
        split         = args.split,
        test_rep      = args.test_rep,
        test_speakers = args.test_speakers,
        dtw_metric    = args.metric,
        dtw_window    = args.window,
        results_dir   = args.results_dir,
    )


if __name__ == "__main__":
    main()
