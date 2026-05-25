"""
preprocessing.py
================
Moduł wstępnego przetwarzania sygnału mowy przed ekstrakcją MFCC.

Kolejność operacji dla każdego nagrania:
    1. Wczytanie pliku WAV
    2. Konwersja do mono + normalizacja amplitudy
    3. Pre-emphasis  – wzmocnienie wysokich częstotliwości
    4. VAD           – wykrycie i wycięcie ciszy na początku/końcu
    5. Ramkowanie    – podział na nakładające się ramki
    6. Okienkowanie  – mnożenie ramek przez okno Hamminga
"""

import numpy as np
import soundfile as sf
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Wczytywanie
# ---------------------------------------------------------------------------

def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """
    Wczytuje plik WAV i zwraca sygnał jako float32 w zakresie [-1, 1].

    Jeśli nagranie jest stereofoniczne, uśrednia kanały do mono.
    soundfile obsługuje 8/16/24/32-bit WAV bez dodatkowych zależności.

    Parameters
    ----------
    path : str lub Path
        Ścieżka do pliku .wav

    Returns
    -------
    signal : np.ndarray, shape (N,), dtype float32
        Próbki sygnału znormalizowane do [-1, 1].
    sample_rate : int
        Częstotliwość próbkowania w Hz (np. 16000, 44100).
    """
    signal, sample_rate = sf.read(path, dtype="float32")

    # Stereo → mono
    if signal.ndim == 2:
        signal = signal.mean(axis=1)

    return signal, sample_rate


# ---------------------------------------------------------------------------
# 2. Normalizacja amplitudy
# ---------------------------------------------------------------------------

def normalize(signal: np.ndarray) -> np.ndarray:
    """
    Normalizuje amplitudę sygnału do zakresu [-1, 1] względem wartości szczytowej.

    Zapobiega dominowaniu głośniejszych nagrań przy liczeniu odległości DTW.
    """
    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal  # cisza – nie dziel przez zero
    return signal / peak


# ---------------------------------------------------------------------------
# 3. Pre-emphasis
# ---------------------------------------------------------------------------

def pre_emphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    Filtr górnoprzepustowy pierwszego rzędu: y[n] = x[n] - coeff * x[n-1].

    Po co?
    ------
    Widmo mowy opada ~6 dB/oktawę dla wysokich częstotliwości.
    Pre-emphasis wyrównuje widmo, co poprawia numeryczną stabilność FFT
    i sprawia, że formantom wyższych rzędów nadajemy odpowiednie znaczenie.

    Parameters
    ----------
    coeff : float
        Współczynnik filtru, typowo 0.95–0.97.
    """
    # np.append zamiast pętli – szybkie wektorowe odejmowanie
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


# ---------------------------------------------------------------------------
# 4. Detekcja aktywności głosowej (VAD) – metoda energetyczna
# ---------------------------------------------------------------------------

def vad_trim(
    signal: np.ndarray,
    sample_rate: int,
    frame_ms: float = 20.0,
    energy_threshold: float = 0.02,
    pad_ms: float = 50.0,
) -> np.ndarray:
    """
    Usuwa ciszę z początku i końca nagrania na podstawie energii krótkoterminowej.

    Algorytm:
        1. Dzieli sygnał na krótkie ramki (frame_ms milisekund).
        2. Oblicza RMS (root mean square) każdej ramki.
        3. Znajduje pierwszą i ostatnią ramkę przekraczającą próg.
        4. Zwraca wycięty fragment z marginesem pad_ms po obu stronach.

    Parameters
    ----------
    energy_threshold : float
        Próg RMS względem znormalizowanego sygnału (zakres 0–1).
        Wartość 0.02 oznacza 2% amplitudy szczytowej.
    pad_ms : float
        Margines ciszy zachowany po obu stronach (ms) – żeby nie ucinać
        początku/końca słowa.
    """
    frame_len = int(sample_rate * frame_ms / 1000)
    pad_len   = int(sample_rate * pad_ms  / 1000)

    # Oblicz RMS dla każdej nieoverlappującej ramki
    n_frames = len(signal) // frame_len
    frames   = signal[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms      = np.sqrt(np.mean(frames ** 2, axis=1))

    # Znajdź ramki z energią powyżej progu
    active = np.where(rms > energy_threshold)[0]

    if len(active) == 0:
        # Brak aktywności – zwróć cały sygnał (edge case: bardzo cicha mowa)
        return signal

    # Przelicz indeksy ramek → próbki
    start = max(0, active[0]  * frame_len - pad_len)
    end   = min(len(signal), (active[-1] + 1) * frame_len + pad_len)

    return signal[start:end]


# ---------------------------------------------------------------------------
# 5. Ramkowanie (framing)
# ---------------------------------------------------------------------------

def frame_signal(
    signal: np.ndarray,
    sample_rate: int,
    frame_ms: float = 25.0,
    step_ms: float  = 10.0,
) -> np.ndarray:
    """
    Dzieli sygnał na nakładające się ramki.

    Dlaczego ramki?
    ---------------
    Sygnał mowy jest niestacjonarny – zmienia się w czasie.
    Zakładamy jednak, że w oknie ~25 ms jest lokalnie stacjonarny
    (quasi-stacjonarność), co pozwala stosować analizę spektralną.

    Parameters
    ----------
    frame_ms : float
        Długość ramki w milisekundach. Standard: 25 ms.
    step_ms : float
        Krok między ramkami w milisekundach. Standard: 10 ms.
        Nakładanie = frame_ms - step_ms = 15 ms (60% overlap).

    Returns
    -------
    frames : np.ndarray, shape (n_frames, frame_len)
        Macierz ramek; każdy wiersz to jedna ramka.
    """
    frame_len = int(sample_rate * frame_ms / 1000)
    step_len  = int(sample_rate * step_ms  / 1000)

    # Uzupełnij sygnał zerami, żeby ostatnia ramka była pełna
    n_frames  = 1 + (len(signal) - frame_len) // step_len
    pad_len   = (n_frames - 1) * step_len + frame_len - len(signal)
    signal    = np.pad(signal, (0, max(0, pad_len)))

    # Tworzenie ramek przez indeksowanie (bez pętli Python)
    indices = (
        np.arange(frame_len)[np.newaxis, :] +           # przesunięcie wewnątrz ramki
        np.arange(n_frames)[:, np.newaxis] * step_len   # przesunięcie ramki
    )
    return signal[indices]  # shape: (n_frames, frame_len)


# ---------------------------------------------------------------------------
# 6. Okienkowanie – okno Hamminga
# ---------------------------------------------------------------------------

def apply_hamming(frames: np.ndarray) -> np.ndarray:
    """
    Mnoży każdą ramkę przez okno Hamminga.

    Po co okno?
    -----------
    FFT zakłada, że sygnał jest okresowy. Na krawędziach ramki pojawia się
    nieciągłość → "wyciek widmowy" (spectral leakage). Okno Hamminga wygładza
    krawędzie do zera, redukując ten efekt.

    Wzór: w[n] = 0.54 - 0.46 * cos(2π·n / (N-1))

    Parameters
    ----------
    frames : np.ndarray, shape (n_frames, frame_len)

    Returns
    -------
    np.ndarray, shape (n_frames, frame_len)
    """
    frame_len = frames.shape[1]
    window    = np.hamming(frame_len)   # shape: (frame_len,)
    return frames * window              # broadcasting: każda ramka × to samo okno


# ---------------------------------------------------------------------------
# 7. Pełny pipeline – wygodna funkcja zbiorcza
# ---------------------------------------------------------------------------

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

    Parameters
    ----------
    path : str lub Path
        Ścieżka do pliku .wav

    Returns
    -------
    frames : np.ndarray, shape (n_frames, frame_len)
        Ramki gotowe do ekstrakcji MFCC.
    sample_rate : int
        Częstotliwość próbkowania.

    Example
    -------
    >>> frames, sr = preprocess("data/raw/speaker_1/jeden_01.wav")
    >>> print(frames.shape)   # np. (78, 400) dla 16kHz, 25ms ramki
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
