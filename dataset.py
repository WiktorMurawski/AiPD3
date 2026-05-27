"""
dataset.py

Ładowanie i podział danych z nagrań o strukturze

Etykiety: liczba w nazwie pliku → polska nazwa cyfry.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass

from preprocessing import preprocess
from mfcc import extract_mfcc_with_deltas, extract_mfcc


DIGIT_LABELS = {
    0:  "zero",
    1:  "jeden",
    2:  "dwa",
    3:  "trzy",
    4:  "cztery",
    5:  "pięć",
    6:  "sześć",
    7:  "siedem",
    8:  "osiem",
    9:  "dziewięć",
    10: "dziesięć",
}

ALL_LABELS = [DIGIT_LABELS[i] for i in range(11)]


@dataclass
class Recording:
    path:       Path
    speaker:    str
    digit:      int
    label:      str
    rep:        int
    features:   np.ndarray | None = None


def scan_dataset(data_dir: str | Path) -> list[Recording]:
    """
    Przeszukuje data_dir i zwraca listę Recording dla każdego pliku WAV.

    Oczekiwana struktura: <data_dir>/<speaker>/<digit>_<rep>.wav
    Inne pliki są pomijane z ostrzeżeniem.
    """
    data_dir = Path(data_dir)
    recordings = []

    for speaker_dir in sorted(data_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker = speaker_dir.name

        for wav_path in sorted(speaker_dir.glob("*.wav")):
            stem = wav_path.stem
            parts = stem.split("_")

            if len(parts) < 2:
                print(f"  [SKIP] Nieznany format nazwy: {wav_path.name}")
                continue

            try:
                digit = int(parts[0])
                rep   = int(parts[1])
            except ValueError:
                print(f"  [SKIP] Nie można sparsować: {wav_path.name}")
                continue

            if digit not in DIGIT_LABELS:
                print(f"  [SKIP] Nieznana cyfra {digit}: {wav_path.name}")
                continue

            recordings.append(Recording(
                path    = wav_path,
                speaker = speaker,
                digit   = digit,
                label   = DIGIT_LABELS[digit],
                rep     = rep,
            ))

    recordings.sort(key=lambda r: (r.speaker, r.digit, r.rep))
    return recordings


def extract_features(
    recordings: list[Recording],
    frame_ms:   float = 25.0,
    step_ms:    float = 10.0,
    n_filters:  int   = 26,
    n_ceps:     int   = 13,
    verbose:    bool  = True,
) -> list[Recording]:
    """
    Uruchamia preprocessing + MFCC dla każdego nagrania i zapisuje wynik w polu Recording.features

    Zwraca nagrania z wypełnionym polem features.
    """
    valid = []
    errors = 0

    for i, rec in enumerate(recordings):
        if verbose and (i % 50 == 0 or i == len(recordings) - 1):
            print(f"  Ekstrakcja cech: {i+1}/{len(recordings)}...")

        try:
            frames, sr = preprocess(
                rec.path,
                frame_ms=frame_ms,
                step_ms=step_ms,
            )

            if len(frames) < 5:
                print(f"  [SKIP] Za krótki sygnał: {rec.path.name}")
                errors += 1
                continue

            # rec.features = extract_mfcc_with_deltas(
            #     frames, sr,
            #     n_filters=n_filters,
            #     n_ceps=n_ceps,
            # )
            rec.features = extract_mfcc(
                frames, sr,
                n_filters=n_filters,
                n_ceps=n_ceps,
            )
            valid.append(rec)

        except Exception as e:
            print(f"  [BŁĄD] {rec.path.name}: {e}")
            errors += 1

    if verbose:
        print(f"  OK: {len(valid)} nagrań  |  błędy: {errors}")

    return valid



def split_by_speaker(
    recordings: list[Recording],
    test_speakers: list[str],
) -> tuple[list[Recording], list[Recording]]:
    """
    Dzieli nagrania na zbiór treningowy i testowy według mówców.

    Strategia "speaker-independent": mówcy testowi nigdy nie pojawiają
    się w treningu.

    test_speakers - speakerzy przeznaczeni do testów
    """
    test_set  = set(test_speakers)
    train = [r for r in recordings if r.speaker not in test_set]
    test  = [r for r in recordings if r.speaker in test_set]
    return train, test


def split_by_repetition(
    recordings: list[Recording],
    test_rep: int = 2,
) -> tuple[list[Recording], list[Recording]]:
    """
    Dzieli nagrania według numeru powtórzenia.

    Wszyscy mówcy są obecni w obu zbiorach.

    test_rep - numer nagrania przeznaczonego do testu
    """
    train = [r for r in recordings if r.rep != test_rep]
    test  = [r for r in recordings if r.rep == test_rep]
    return train, test


def build_templates(
    train: list[Recording],
) -> dict[str, list[np.ndarray]]:
    """
    Buduje słownik wzorców dla DTWClassifier
    """
    templates: dict[str, list[np.ndarray]] = {label: [] for label in ALL_LABELS}

    for rec in train:
        if rec.features is not None:
            templates[rec.label].append(rec.features)

    for label in ALL_LABELS:
        n = len(templates[label])
        if n == 0:
            print(f"  [UWAGA] Brak wzorców dla klasy '{label}'!")

    return templates


def load_dataset(
    data_dir:      str | Path,
    split:         str        = "repetition",   # "repetition"/"speaker"
    test_rep:      int        = 2,
    test_speakers: list[str] | None = None,
    verbose:       bool       = True,
) -> tuple[dict[str, list[np.ndarray]], list[Recording]]:

    if verbose:
        print(f"[1/3] Skanowanie katalogu: {data_dir}")
    recs = scan_dataset(data_dir)
    if verbose:
        speakers = sorted({r.speaker for r in recs})
        print(f"  Znaleziono {len(recs)} nagrań, {len(speakers)} mówców: {speakers}")

    if verbose:
        print("[2/3] Ekstrakcja cech MFCC...")
    recs = extract_features(recs, verbose=verbose)

    if verbose:
        print(f"[3/3] Podział train/test (strategia: {split})...")
    if split == "repetition":
        train, test = split_by_repetition(recs, test_rep=test_rep)
    elif split == "speaker":
        if not test_speakers:
            raise ValueError("Dla split='speaker' podaj test_speakers.")
        train, test = split_by_speaker(recs, test_speakers)
    else:
        raise ValueError(f"Nieznana strategia: {split}")

    if verbose:
        print(f"  Trening: {len(train)} nagrań  |  Test: {len(test)} nagrań")

    templates = build_templates(train)
    return templates, test
