"""
dtw.py
======
Implementacja Dynamic Time Warping (DTW) od zera.

DTW rozwiązuje kluczowy problem przy porównywaniu nagrań mowy:
    - słowo "jeden" wypowiedziane szybko i wolno ma różną liczbę ramek
    - proste porównanie ramka-do-ramki (euklidesowe) całkowicie zawodzi
    - DTW "rozciąga" oś czasu tak, żeby jak najlepiej dopasować dwa przebiegi

Zastosowanie tutaj:
    - Wzorzec  = uśrednione MFCC kilku nagrań danej cyfry
    - Zapytanie = MFCC nowego nagrania
    - Klasyfikacja: wybieramy cyfrę o minimalnym dystansie DTW
"""

import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Lokalne miary odległości
# ---------------------------------------------------------------------------

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Odległość euklidesowa między dwoma wektorami cech."""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Odległość kosinusowa ∈ [0, 2].

    Mierzy kąt między wektorami – uniezależnia od skali amplitudy.
    Dla MFCC bywa lepsza niż euklidesowa przy zmiennej głośności nagrań.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# 2. Macierz odległości lokalnych
# ---------------------------------------------------------------------------

def local_distance_matrix(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Oblicza macierz odległości lokalnych D[i, j] = dist(seq_a[i], seq_b[j]).

    Parameters
    ----------
    seq_a : np.ndarray, shape (T_a, n_features)
    seq_b : np.ndarray, shape (T_b, n_features)
    metric : 'euclidean' lub 'cosine'

    Returns
    -------
    D : np.ndarray, shape (T_a, T_b)
    """
    if metric == "euclidean":
        # Wektorowe liczenie odległości euklidesowej:
        # ||a - b||² = ||a||² + ||b||² - 2·a·bᵀ
        # Szybsze niż podwójna pętla Python
        sq_a = np.sum(seq_a ** 2, axis=1)[:, np.newaxis]   # (T_a, 1)
        sq_b = np.sum(seq_b ** 2, axis=1)[np.newaxis, :]   # (1, T_b)
        cross = seq_a @ seq_b.T                             # (T_a, T_b)
        dist_sq = sq_a + sq_b - 2 * cross
        # Numeryczne niedokładności mogą dać małe ujemne wartości
        return np.sqrt(np.maximum(dist_sq, 0))

    elif metric == "cosine":
        # Normalizacja wektorów, potem iloczyn skalarny
        norm_a = seq_a / (np.linalg.norm(seq_a, axis=1, keepdims=True) + 1e-10)
        norm_b = seq_b / (np.linalg.norm(seq_b, axis=1, keepdims=True) + 1e-10)
        return 1.0 - norm_a @ norm_b.T

    else:
        raise ValueError(f"Nieznana metryka: {metric}. Użyj 'euclidean' lub 'cosine'.")


# ---------------------------------------------------------------------------
# 3. Algorytm DTW – programowanie dynamiczne
# ---------------------------------------------------------------------------

def dtw(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    metric: str = "euclidean",
    window: int | None = None,
) -> tuple[float, np.ndarray]:
    """
    Oblicza dystans DTW między dwoma sekwencjami wektorów cech.

    Algorytm (programowanie dynamiczne):
    -------------------------------------
    1. D[i,j] = lokalna odległość między ramką i z seq_a a ramką j z seq_b
    2. C[i,j] = minimalna skumulowana odległość ścieżki kończącej się w (i,j)

    Rekurencja:
        C[i,j] = D[i,j] + min(C[i-1, j],    # wstawienie (rozciągnięcie seq_b)
                               C[i, j-1],    # usunięcie  (rozciągnięcie seq_a)
                               C[i-1, j-1])  # dopasowanie ramka-do-ramki

    Warunki brzegowe: C[0,0] = D[0,0],  C[-1,0] i C[0,-1] = ∞

    Parameters
    ----------
    seq_a, seq_b : np.ndarray
        Sekwencje wektorów cech; kształt (T, n_features).
    metric : str
        Miara odległości: 'euclidean' lub 'cosine'.
    window : int lub None
        Ograniczenie Sakoe-Chiba: dozwolone tylko komórki |i-j| ≤ window.
        Przyspiesza obliczenia i zapobiega zbyt "ekstremalnym" dopasowaniom.
        None = brak ograniczenia.

    Returns
    -------
    distance : float
        Znormalizowany dystans DTW (podzielony przez długość ścieżki).
    cost_matrix : np.ndarray, shape (T_a, T_b)
        Macierz skumulowanych kosztów (przydatna do wizualizacji).
    """
    T_a, T_b = len(seq_a), len(seq_b)

    # Macierz odległości lokalnych
    D = local_distance_matrix(seq_a, seq_b, metric=metric)

    # Macierz skumulowanych kosztów, inicjalnie ∞
    C = np.full((T_a, T_b), np.inf)
    C[0, 0] = D[0, 0]

    # Warunki brzegowe – pierwsza kolumna i pierwszy wiersz
    for i in range(1, T_a):
        if window is None or abs(i - 0) <= window:
            C[i, 0] = D[i, 0] + C[i - 1, 0]

    for j in range(1, T_b):
        if window is None or abs(0 - j) <= window:
            C[0, j] = D[0, j] + C[0, j - 1]

    # Wypełnianie macierzy (programowanie dynamiczne)
    for i in range(1, T_a):
        for j in range(1, T_b):
            # Opcjonalne ograniczenie okna Sakoe-Chiba
            if window is not None and abs(i - j) > window:
                continue  # C[i,j] pozostaje ∞ – ścieżka nie może przez nie przejść
            C[i, j] = D[i, j] + min(C[i - 1, j],
                                     C[i, j - 1],
                                     C[i - 1, j - 1])

    # Normalizacja przez długość optymalnej ścieżki (T_a + T_b)
    # Dzięki normalizacji dystans jest porównywalny między parami różnej długości
    distance = C[T_a - 1, T_b - 1] / (T_a + T_b)

    return distance, C


def dtw_fast(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    metric: str = "euclidean",
    window: int | None = None,
) -> tuple[float, np.ndarray]:
    """
    Zoptymalizowana wersja DTW – wylicza macierz D wektorowo, resztę pętlą.

    Uwaga: DTW nie daje się w pełni wektoryzować wiersz-po-wierszu, bo C[i,j]
    zależy od C[i, j-1] (lewy sąsiad w tym samym wierszu). Pętla po komórkach
    jest niezbędna; przyspieszenie pochodzi z wektorowego liczenia macierzy D.

    Identyczny wynik co dtw().
    """
    T_a, T_b = len(seq_a), len(seq_b)
    D = local_distance_matrix(seq_a, seq_b, metric=metric)

    C = np.full((T_a, T_b), np.inf)
    C[0, 0] = D[0, 0]

    for i in range(1, T_a):
        C[i, 0] = D[i, 0] + C[i - 1, 0]
    for j in range(1, T_b):
        C[0, j] = D[0, j] + C[0, j - 1]

    for i in range(1, T_a):
        j_start = max(1, i - window) if window is not None else 1
        j_end   = min(T_b, i + window + 1) if window is not None else T_b
        for j in range(j_start, j_end):
            C[i, j] = D[i, j] + min(C[i - 1, j],
                                     C[i, j - 1],
                                     C[i - 1, j - 1])

    distance = C[T_a - 1, T_b - 1] / (T_a + T_b)
    return distance, C


# ---------------------------------------------------------------------------
# 4. Klasyfikator DTW
# ---------------------------------------------------------------------------

class DTWClassifier:
    """
    Klasyfikator 1-NN oparty na dystansie DTW.

    Przechowuje wzorce (templates) – po jednym lub kilka na klasę.
    Nowe nagranie klasyfikuje przez znalezienie wzorca o minimalnym dystansie.

    Użycie:
    -------
        clf = DTWClassifier(metric="euclidean", window=20)
        clf.fit(templates)          # słownik: label -> lista macierzy MFCC
        label, dist = clf.predict(query_mfcc)
    """

    def __init__(
        self,
        metric: str = "euclidean",
        window: int | None = 20,
        use_fast: bool = True,
    ):
        """
        Parameters
        ----------
        metric : str
            Miara odległości: 'euclidean' lub 'cosine'.
        window : int lub None
            Ograniczenie okna Sakoe-Chiba.
            Wartość 20 ramek (~200 ms) to dobry kompromis.
        use_fast : bool
            Użyj dtw_fast() zamiast dtw() – szybsze dla długich sekwencji.
        """
        self.metric   = metric
        self.window   = window
        self.use_fast = use_fast
        self.templates: dict[str, list[np.ndarray]] = {}

    def fit(self, templates: dict[str, list[np.ndarray]]) -> None:
        """
        Zapisuje wzorce dla każdej klasy.

        Parameters
        ----------
        templates : dict
            Klucz: etykieta klasy (np. "zero", "jeden", ...).
            Wartość: lista macierzy MFCC, każda o kształcie (T_i, n_features).

        Przykład:
            templates = {
                "zero":  [mfcc_z1, mfcc_z2, mfcc_z3],
                "jeden": [mfcc_j1, mfcc_j2],
            }
        """
        self.templates = templates
        all_labels = list(templates.keys())
        print(f"Klasyfikator DTW gotowy: {len(all_labels)} klas, "
              f"metryka={self.metric}, okno={self.window}")

    def _dtw_fn(self, a, b):
        fn = dtw_fast if self.use_fast else dtw
        dist, _ = fn(a, b, metric=self.metric, window=self.window)
        return dist

    def predict(self, query: np.ndarray) -> tuple[str, float]:
        """
        Klasyfikuje nowe nagranie przez minimalizację dystansu DTW.

        Dla każdej klasy liczy DTW do wszystkich wzorców i bierze minimum
        (strategia "nearest neighbor").

        Parameters
        ----------
        query : np.ndarray, shape (T_q, n_features)
            Macierz MFCC nagrania do sklasyfikowania.

        Returns
        -------
        best_label : str
            Przewidziana klasa.
        best_dist : float
            Znormalizowany dystans DTW do wybranego wzorca.
        """
        best_label = None
        best_dist  = np.inf

        for label, template_list in self.templates.items():
            for template in template_list:
                dist = self._dtw_fn(query, template)
                if dist < best_dist:
                    best_dist  = dist
                    best_label = label

        return best_label, best_dist

    def predict_with_scores(self, query: np.ndarray) -> list[tuple[str, float]]:
        """
        Zwraca listę (klasa, dystans) posortowaną rosnąco po dystansie.

        Przydatne do analizy pomyłek i wizualizacji pewności klasyfikacji.
        """
        scores: dict[str, float] = {}

        for label, template_list in self.templates.items():
            # Bierzemy minimum ze wszystkich wzorców danej klasy
            min_dist = min(self._dtw_fn(query, t) for t in template_list)
            scores[label] = min_dist

        return sorted(scores.items(), key=lambda x: x[1])
