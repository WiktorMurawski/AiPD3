"""
preprocessing.py
================
Moduł wstępnego przetwarzania sygnału mowy przed ekstrakcją MFCC.

Kolejność operacji dla każdego nagrania:
    1. Wczytanie pliku WAV
    2. Konwersja do mono + normalizacja amplitudy
    3. Preemfaza     - wzmocnienie wysokich częstotliwości
    4. VAD           - wykrycie i wycięcie ciszy na początku/końcu
    5. Ramkowanie    - podział na nakładające się ramki
    6. Okienkowanie  - mnożenie ramek przez okno Hamminga
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """
    Wczytuje plik WAV i zwraca sygnał jako float32 w zakresie [-1, 1].

    Jeśli nagranie jest stereofoniczne, uśrednia kanały do mono.
    """
    signal, sample_rate = sf.read(path, dtype="float32")

    if signal.ndim == 2:
        signal = signal.mean(axis=1)

    return signal, sample_rate


def normalize(signal: np.ndarray) -> np.ndarray:
    """
    Normalizuje amplitudę sygnału do zakresu [-1, 1] względem wartości szczytowej.

    Zapobiega dominowaniu głośniejszych nagrań przy liczeniu odległości DTW.
    """
    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal
    return signal / peak

def pre_emphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    Filtr górnoprzepustowy pierwszego rzędu: y[n] = x[n] - coeff * x[n-1].
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def vad_trim(
    signal: np.ndarray,
    sample_rate: int,
    frame_ms: float = 20.0,
    energy_threshold: float = 0.02,
    pad_ms: float = 50.0,
) -> np.ndarray:
    """
    Usuwa ciszę z początku i końca nagrania na podstawie energii krótkoterminowej.
    """
    frame_len = int(sample_rate * frame_ms / 1000)
    pad_len   = int(sample_rate * pad_ms  / 1000)

    n_frames = len(signal) // frame_len
    frames   = signal[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms      = np.sqrt(np.mean(frames ** 2, axis=1))

    active = np.where(rms > energy_threshold)[0]

    if len(active) == 0:
        return signal

    start = max(0, active[0]  * frame_len - pad_len)
    end   = min(len(signal), (active[-1] + 1) * frame_len + pad_len)

    return signal[start:end]


def frame_signal(
    signal: np.ndarray,
    sample_rate: int,
    frame_ms: float = 25.0,
    step_ms: float  = 10.0,
) -> np.ndarray:
    """
    Dzieli sygnał na nakładające się ramki.
    """
    frame_len = int(sample_rate * frame_ms / 1000)
    step_len  = int(sample_rate * step_ms  / 1000)

    n_frames  = 1 + (len(signal) - frame_len) // step_len
    pad_len   = (n_frames - 1) * step_len + frame_len - len(signal)
    signal    = np.pad(signal, (0, max(0, pad_len)))

    indices = (
        np.arange(frame_len)[np.newaxis, :] + 
        np.arange(n_frames)[:, np.newaxis] * step_len
    )
    return signal[indices]


def apply_hamming(frames: np.ndarray) -> np.ndarray:
    """
    Mnoży każdą ramkę przez okno Hamminga.
    """
    frame_len = frames.shape[1]
    window    = np.hamming(frame_len)
    return frames * window


def preprocess(
    path: str | Path,
    frame_ms: float   = 25.0,
    step_ms: float    = 10.0,
    pre_emph: float   = 0.97,
    vad_threshold: float = 0.02,
    vad_pad_ms: float    = 50.0,
) -> tuple[np.ndarray, int]:
    """
    Kompletny preprocessing nagrania WAV.

    Zwraca gotowe ramki z oknem Hamminga oraz częstotliwość próbkowania.
    """
    signal, sr = load_wav(path)
    signal     = normalize(signal)
    signal     = pre_emphasis(signal, coeff=pre_emph)
    signal     = vad_trim(signal, sr,
                          energy_threshold=vad_threshold,
                          pad_ms=vad_pad_ms)
    frames     = frame_signal(signal, sr, frame_ms=frame_ms, step_ms=step_ms)
    frames     = apply_hamming(frames)

    return frames, sr
