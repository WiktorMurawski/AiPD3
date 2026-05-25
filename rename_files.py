"""
rename_files.py
===============
Przemianowuje pliki WAV ze słownego zapisu cyfry na numeryczny.

Przykłady:
    zero_1.wav    → 0_1.wav
    jeden_2.wav   → 1_2.wav
    dziesięć_1.wav → 10_1.wav

Uruchomienie (podgląd bez zmian):
    python rename_files.py --data data/raw

Uruchomienie (faktyczna zmiana):
    python rename_files.py --data data/raw --apply
"""

import argparse
from pathlib import Path

WORD_TO_NUM = {
    "zero":     "0",
    "jeden":    "1",
    "dwa":      "2",
    "trzy":     "3",
    "cztery":   "4",
    "piec":     "5",
    "pięć":     "5",
    "szesc":    "6",
    "sześć":    "6",
    "siedem":   "7",
    "osiem":    "8",
    "dziewiec": "9",
    "dziewięć": "9",
    "dziesiec": "10",
    "dziesięć": "10",
}


def rename_all(data_dir: Path, apply: bool) -> None:
    to_rename = []

    for wav in sorted(data_dir.rglob("*.wav")):
        stem  = wav.stem   # np. "zero_1" lub "dziesięć_2"
        parts = stem.split("_")

        if len(parts) < 2:
            continue

        word = parts[0].lower()
        if word not in WORD_TO_NUM:
            continue  # już numeryczna albo nieznana – pomijamy

        new_stem = WORD_TO_NUM[word] + "_" + "_".join(parts[1:])
        new_path = wav.with_name(new_stem + ".wav")

        if new_path.exists():
            print(f"  [POMIŃ] Cel już istnieje: {new_path.name}")
            continue

        to_rename.append((wav, new_path))

    if not to_rename:
        print("Nie znaleziono plików do przemianowania.")
        return

    print(f"{'PODGLĄD' if not apply else 'ZMIANA'}: {len(to_rename)} plików\n")
    for src, dst in to_rename:
        rel_src = src.relative_to(data_dir)
        rel_dst = dst.relative_to(data_dir)
        print(f"  {rel_src}  →  {rel_dst.name}")
        if apply:
            src.rename(dst)

    if not apply:
        print("\nTo był tylko podgląd. Dodaj --apply żeby faktycznie zmienić nazwy.")
    else:
        print(f"\nPrzemianowano {len(to_rename)} plików.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default="data/raw", help="Folder z nagraniami")
    parser.add_argument("--apply", action="store_true", help="Faktycznie zmień nazwy")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"Błąd: folder '{data_dir}' nie istnieje.")
        return

    rename_all(data_dir, apply=args.apply)


if __name__ == "__main__":
    main()