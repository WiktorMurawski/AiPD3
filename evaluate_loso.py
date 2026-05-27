"""
evaluate_loso.py

Ewaluacja leave-one-speaker-out (LOSO): każdy mówca raz jako testowy,
pozostali jako treningowi. Zbiera wyniki wszystkich rund i drukuje podsumowanie.

Uruchomienie:
    python evaluate_loso.py
    python evaluate_loso.py --data ../data/raw --metric euclidean --window 20
"""

import sys
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from dataset import scan_dataset, extract_features, split_by_speaker, build_templates, ALL_LABELS
from dtw import DTWClassifier


def evaluate_loso(
    data_dir:   str | Path,
    dtw_metric: str = "euclidean",
    dtw_window: int = 20,
    results_dir: str | Path = "results",
) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    # --- Wczytaj i wyekstrahuj cechy raz dla wszystkich nagrań ---
    print("Ekstrakcja cech MFCC (jednorazowo dla wszystkich nagrań)...")
    recs = scan_dataset(data_dir)
    recs = extract_features(recs, verbose=True)

    speakers = sorted({r.speaker for r in recs})
    n = len(speakers)
    print(f"\nMówcy ({n}): {speakers}")
    print(f"Protokół: leave-one-speaker-out ({n} rund)\n")
    print("=" * 60)

    # --- Wyniki per runda ---
    round_results = []

    for i, test_speaker in enumerate(speakers):
        train, test = split_by_speaker(recs, test_speakers=[test_speaker])
        templates   = build_templates(train)

        clf = DTWClassifier(metric=dtw_metric, window=dtw_window)
        clf.fit(templates)

        y_true, y_pred = [], []
        for rec in test:
            pred, _ = clf.predict(rec.features)
            y_true.append(rec.label)
            y_pred.append(pred)

        correct  = sum(t == p for t, p in zip(y_true, y_pred))
        accuracy = correct / len(y_true) if y_true else 0.0

        errors = [
            f"{rec.path.name}: {t} → {p}"
            for rec, t, p in zip(test, y_true, y_pred) if t != p
        ]

        round_results.append({
            "speaker":  test_speaker,
            "accuracy": round(accuracy, 4),
            "correct":  correct,
            "total":    len(y_true),
            "errors":   errors,
        })

        status = "OK" if accuracy == 1.0 else "X"
        print(f"  Runda {i+1:2d}/{n}  mówca {test_speaker:>4}:  "
              f"{accuracy*100:6.1f}%  ({correct}/{len(y_true)})  {status}")
        if errors:
            for err in errors:
                print(f"             {err}")

    # --- Podsumowanie ---
    mean_acc = sum(r["accuracy"] for r in round_results) / n
    total_correct = sum(r["correct"] for r in round_results)
    total_total   = sum(r["total"]   for r in round_results)

    print("=" * 60)
    print(f"\n  Średnia dokładność LOSO: {mean_acc*100:.1f}%")
    print(f"  Łącznie:                 {total_correct}/{total_total} "
          f"({total_correct/total_total*100:.1f}%)")

    worst = min(round_results, key=lambda r: r["accuracy"])
    best  = max(round_results, key=lambda r: r["accuracy"])
    print(f"  Najlepszy mówca:         {best['speaker']} "
          f"({best['accuracy']*100:.1f}%)")
    print(f"  Najgorszy mówca:         {worst['speaker']} "
          f"({worst['accuracy']*100:.1f}%)")

    # --- Zapis wyników ---
    output = {
        "protocol":    "leave-one-speaker-out",
        "dtw_metric":  dtw_metric,
        "dtw_window":  dtw_window,
        "n_speakers":  n,
        "mean_accuracy":       round(mean_acc, 4),
        "total_correct":       total_correct,
        "total":               total_total,
        "rounds":              round_results,
    }

    out_path = results_dir / "results_loso.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nWyniki zapisane do: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Ewaluacja LOSO")
    parser.add_argument("--data",       default="data/raw")
    parser.add_argument("--metric",     default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--window",     default=20, type=int)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    evaluate_loso(
        data_dir    = args.data,
        dtw_metric  = args.metric,
        dtw_window  = args.window,
        results_dir = args.results_dir,
    )


if __name__ == "__main__":
    main()