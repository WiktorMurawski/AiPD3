"""
dtw.py

Implementacja Dynamic Time Warping (DTW)
"""

import numpy as np
from pathlib import Path

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mierzy kąt między wektorami - uniezależnia od skali amplitudy.
    Dla MFCC bywa lepsza niż euklidesowa przy zmiennej głośności nagrań.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)

def local_distance_matrix(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Oblicza macierz odległości lokalnych D[i, j] = dist(seq_a[i], seq_b[j]).
    """
    if metric == "euclidean":
        sq_a = np.sum(seq_a ** 2, axis=1)[:, np.newaxis]
        sq_b = np.sum(seq_b ** 2, axis=1)[np.newaxis, :]
        cross = seq_a @ seq_b.T
        dist_sq = sq_a + sq_b - 2 * cross
        return np.sqrt(np.maximum(dist_sq, 0))

    elif metric == "cosine":
        norm_a = seq_a / (np.linalg.norm(seq_a, axis=1, keepdims=True) + 1e-10)
        norm_b = seq_b / (np.linalg.norm(seq_b, axis=1, keepdims=True) + 1e-10)
        return 1.0 - norm_a @ norm_b.T

    else:
        raise ValueError(f"Nieznana metryka: {metric}. Metryki: 'euclidean', 'cosine'.")


def dtw(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    metric: str = "euclidean",
    window: int | None = None,
) -> tuple[float, np.ndarray]:
    
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


class DTWClassifier:
    """
    Klasyfikator 1-NN oparty na dystansie DTW.

    Przechowuje wzorce (templates) - po jednym lub kilka na klasę.
    Nowe nagranie klasyfikuje przez znalezienie wzorca o minimalnym dystansie.
    """

    def __init__(
        self,
        metric: str = "euclidean",
        window: int | None = 20,
    ):
        """
        metric - metryka odległości ('euclidean' lub 'cosine')
        window - Ograniczenie okna Sakoe-Chiba.
        """
        self.metric   = metric
        self.window   = window
        self.templates: dict[str, list[np.ndarray]] = {}

    def fit(self, templates: dict[str, list[np.ndarray]]) -> None:
        """
        Zapisuje wzorce dla każdej klasy.
        """
        self.templates = templates
        all_labels = list(templates.keys())
        print(f"Klasyfikator DTW gotowy: {len(all_labels)} klas, "
              f"metryka={self.metric}, okno={self.window}")

    def _dtw_fn(self, a, b):
        dist, _ = dtw(a, b, metric=self.metric, window=self.window)
        return dist

    def predict(self, query: np.ndarray) -> tuple[str, float]:
        """
        Klasyfikuje nowe nagranie przez minimalizację dystansu DTW.

        Dla każdej klasy liczy DTW do wszystkich wzorców i bierze minimum (strategia "nearest neighbor").

        Zwraca przewidzianą klasę i dystans do wzorca.
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
        """
        scores: dict[str, float] = {}

        for label, template_list in self.templates.items():
            # Bierzemy minimum ze wszystkich wzorców danej klasy
            min_dist = min(self._dtw_fn(query, t) for t in template_list)
            scores[label] = min_dist

        return sorted(scores.items(), key=lambda x: x[1])
