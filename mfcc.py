"""
mfcc.py

Implementacja ekstrakcji współczynników mel-cepstralnych (MFCC)

Pipeline dla każdej ramki:
    1. FFT
    2. Potęgowanie
    3. Filtrbank Mela
    4. Logarytm 
    5. DCT
    6. Delta + ΔΔ
"""

import numpy as np


def hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    """
    Konwertuje częstotliwości w Hz na skalę Mela.
    Wzór: mel = 2595 * log10(1 + hz / 700)
    """
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """
    Odwrotność hz_to_mel
    """
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank_old(
    n_filters: int,
    frame_len: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    if f_max is None:
        f_max = sample_rate / 2.0

    n_fft = frame_len // 2 + 1

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)

    hz_points  = mel_to_hz(mel_points)
    bin_points = np.floor((frame_len + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft - 1)

    filterbank = np.zeros((n_filters, n_fft))

    for m in range(n_filters):
        left   = bin_points[m]
        center = bin_points[m + 1]
        right  = bin_points[m + 2] 

        if center > left:
            for k in range(left, center + 1):
                filterbank[m, k] = (k - left) / (center - left)

        if right > center:
            for k in range(center, right + 1):
                filterbank[m, k] = (right - k) / (right - center)

    return filterbank


def mel_filterbank_fast(
    n_filters: int,
    frame_len: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """
    Wektorowa wersja mel_filterbank bez pętli
    """
    if f_max is None:
        f_max = sample_rate / 2.0

    n_fft = frame_len // 2 + 1

    mel_min    = hz_to_mel(f_min)
    mel_max    = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points  = mel_to_hz(mel_points)
    bin_points = np.floor((frame_len + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft - 1)

    k = np.arange(n_fft)[np.newaxis, :]

    left   = bin_points[:-2, np.newaxis]
    center = bin_points[1:-1, np.newaxis]
    right  = bin_points[2:,  np.newaxis]

    denom_up   = np.where(center > left,  center - left,  1)
    denom_down = np.where(right  > center, right - center, 1)

    up   = np.where((k >= left)   & (k <= center), (k - left)  / denom_up,   0.0)
    down = np.where((k > center)  & (k <= right),  (right - k) / denom_down, 0.0)

    return up + down


def dct_matrix(n_filters: int, n_ceps: int) -> np.ndarray:
    """
    Tworzy macierz DCT-II do wyodrębnienia n_ceps współczynników z n_filters wartości.

    Wzór (typ II, ortogonalna):
        DCT[k, n] = cos(pi/N * (n + 0.5) * k)   dla k = 0..K-1, n = 0..N-1

    n_filters - liczba wejść (wyjście filtrbanku)
    n_ceps - liczba wyjść (MFCC)
    """
    n = np.arange(n_filters)[np.newaxis, :]
    k = np.arange(n_ceps)[:, np.newaxis]
    return np.cos(np.pi / n_filters * (n + 0.5) * k)


def extract_mfcc(
    frames: np.ndarray,
    sample_rate: int,
    n_filters: int = 26,
    n_ceps: int    = 13,
    f_min: float   = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """
    Oblicza MFCC dla każdej ramki sygnału.

    Zwraca Macierz MFCC - każdy wiersz to wektor cech jednej ramki.
    """
    n_frames, frame_len = frames.shape
    n_fft = frame_len // 2 + 1  

    # FFT + periodogram
    # np.fft.rfft zwraca tylko dodatnie częstotliwości (n_fft punktów)
    spectrum  = np.fft.rfft(frames, n=frame_len)   # (n_frames, n_fft), zespolone
    power_spec = (1.0 / frame_len) * (np.abs(spectrum) ** 2)  # (n_frames, n_fft)

    # Filtrbank Mela
    fb = mel_filterbank_fast(n_filters, frame_len, sample_rate, f_min, f_max)
    filter_energies = power_spec @ fb.T
    filter_energies = np.where(filter_energies == 0, np.finfo(float).eps, filter_energies)

    # Logarytm
    log_energies = np.log(filter_energies) 

    # Krok 4: DCT
    dct = dct_matrix(n_filters, n_ceps) 
    mfcc = log_energies @ dct.T

    return mfcc


# ---------------------------------------------------------------------------
# 5. Delta i delta-delta (współczynniki różnicowe)
# ---------------------------------------------------------------------------

def compute_deltas(features: np.ndarray, N: int = 2) -> np.ndarray:
    """
    Oblicza współczynniki delta (pierwsza pochodna po czasie) metodą regresji.

    Dlaczego delta?
    ---------------
    Statyczne MFCC opisują "jak brzmi" dana ramka, ale nie "jak się zmienia".
    Delty dodają informację o dynamice mowy – tempie zmian formantów itp.
    Delta-delta (pochodna drugiego rzędu) opisuje przyspieszenie tych zmian.

    Wzór (ETSI / HTK standard):
        Δ[t] = Σ_{n=1}^{N}  n * (c[t+n] - c[t-n])
               ─────────────────────────────────────
                        2 * Σ_{n=1}^{N} n²

    Krawędzie sygnału są uzupełniane przez powielenie skrajnych ramek.

    Parameters
    ----------
    features : np.ndarray, shape (n_frames, n_features)
    N : int
        Kontekst (liczba ramek w przód i w tył). Standard: 2.

    Returns
    -------
    deltas : np.ndarray, shape (n_frames, n_features)
    """
    n_frames, n_features = features.shape
    deltas = np.zeros_like(features)
    denominator = 2.0 * sum(n ** 2 for n in range(1, N + 1))

    # Padding przez powielenie krawędziowych ramek
    padded = np.pad(features, ((N, N), (0, 0)), mode="edge")

    for t in range(n_frames):
        numerator = sum(
            n * (padded[t + N + n] - padded[t + N - n])
            for n in range(1, N + 1)
        )
        deltas[t] = numerator / denominator

    return deltas


def extract_mfcc_with_deltas(
    frames: np.ndarray,
    sample_rate: int,
    n_filters: int = 26,
    n_ceps: int    = 13,
    delta_n: int   = 2,
) -> np.ndarray:
    """
    Pełna ekstrakcja cech MFCC, Δ, ΔΔ dla każdej ramki
    """
    mfcc        = extract_mfcc(frames, sample_rate, n_filters, n_ceps)
    delta       = compute_deltas(mfcc, N=delta_n)
    delta_delta = compute_deltas(delta, N=delta_n)

    return np.concatenate([mfcc, delta, delta_delta], axis=1)
