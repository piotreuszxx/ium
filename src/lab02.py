import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
N_SPLITS = 5
EPOCHS = 100
LR = 1e-2
DEVICE = torch.device("cpu")


# Zadanie 3: prosta sieć w pełni połączona jako klasa dziedzicząca po nn.Module;
# 1–2 warstwy ukryte, funkcja aktywacji – ReLU
class IrisMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    n_classes: int,
    epochs: int = EPOCHS,
) -> IrisMLP:
    model = IrisMLP(n_features, n_classes).to(DEVICE)
    # Funkcja straty – CrossEntropyLoss; optymalizator – Adam
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # Zamień dane z zadania 1 na tensory (torch.Tensor)
    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)

    model.train()
    # Pętla treningowa obejmująca kilkadziesiąt epok:
    # forward, obliczenie straty, backward, optimizer.step()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
    return model


def mlp_accuracy(model: IrisMLP, X: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=DEVICE))
        pred = logits.argmax(dim=1).cpu().numpy()
    return accuracy_score(y, pred)


def pytorch_cv_scores(X: np.ndarray, y: np.ndarray) -> list[float]:
    # Scikit-learn nie obsłuży bezpośrednio modelu PyTorch – walidacja krzyżowa ręcznie
    # Podziel dane na 5 foldów indeksami z KFold
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    scores: list[float] = []

    for train_idx, test_idx in kf.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]
        # Dla każdego foldu utwórz nową instancję sieci (żeby nie trenować dalej tych samych wag)
        # Przeprowadź pełny trening jak w zadaniu 3
        model = train_mlp(X_tr, y_tr, n_features, n_classes)
        # Zapisz dokładność na foldzie testowym
        scores.append(mlp_accuracy(model, X_te, y_te))
    return scores


def main() -> None:
    # Zadanie 1. Wspólny zbiór danych
    # Wybierz jeden zbiór danych ze scikit-learn
    iris = load_iris()
    X, y = iris.data, iris.target
    # Podziel go funkcją train_test_split na zbiór treningowy i testowy
    # (proporcja 80/20, ustalony random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    # Oba modele trenują się i są oceniane na dokładnie tych samych danych

    # Zadanie 2. Model w scikit-learn
    # Zbuduj obiekt sklearn.pipeline.Pipeline łączący StandardScaler ze skalowaniem cech
    # i klasyfikator – LogisticRegression
    sklearn_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=RANDOM_SEED)),
        ]
    )

    # Zadanie 6: zmierz czas trenowania (moduł time)
    t0 = time.perf_counter()
    # Wytrenuj go metodą fit() na zbiorze treningowym z zadania 1
    sklearn_pipe.fit(X_train, y_train)
    sklearn_train_time = time.perf_counter() - t0

    # Sprawdź działanie modelu, wywołując predict() na zbiorze testowym
    y_pred_sk = sklearn_pipe.predict(X_test)
    # Wypisz wynik funkcji accuracy_score (punkt odniesienia do dalszego porównania)
    sklearn_test_acc = accuracy_score(y_test, y_pred_sk)
    sklearn_report = classification_report(y_test, y_pred_sk)

    # Zadanie 3. Model w PyTorch
    scaler_pt = StandardScaler()
    X_tr_s = scaler_pt.fit_transform(X_train)
    X_te_s = scaler_pt.transform(X_test)
    n_features, n_classes = X.shape[1], len(np.unique(y))

    t0 = time.perf_counter()
    mlp = train_mlp(X_tr_s, y_train, n_features, n_classes)
    pytorch_train_time = time.perf_counter() - t0

    # Po treningu sprawdź dokładność modelu na zbiorze testowym, analogicznie jak w zadaniu 2
    pytorch_test_acc = mlp_accuracy(mlp, X_te_s, y_test)

    # Zadanie 4. Walidacja krzyżowa w scikit-learn
    # Dla modelu (Pipeline'u) z zadania 2: KFold (shuffle=True, ustalony random_state)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    # Uruchom 5-krotną walidację krzyżową funkcją cross_val_score
    sk_cv = cross_val_score(
        sklearn_pipe, X_train, y_train, cv=kf, scoring="accuracy"
    )
    # Wypisz wynik dla każdego z 5 foldów
    print("sklearn CV (5 foldów):", sk_cv)
    # Policz i zapisz średnią wartość metryki oraz jej odchylenie standardowe
    print(f"  mean={sk_cv.mean():.4f}, std={sk_cv.std():.4f}")

    # Zadanie 5. Walidacja krzyżowa w PyTorch
    t0 = time.perf_counter()
    pt_cv = np.array(pytorch_cv_scores(X_train, y_train))
    pytorch_cv_time = time.perf_counter() - t0
    print("PyTorch CV (5 foldów):", pt_cv)
    # Policz średnią i odchylenie standardowe dokładności ze wszystkich 5 foldów
    print(f"  mean={pt_cv.mean():.4f}, std={pt_cv.std():.4f}")

    # Zadanie 6. Porównanie wyników – zapis zestawienia do pliku wyniki.txt
    lines = [
        f"test_size={TEST_SIZE}, random_state={RANDOM_SEED}",
        "",
        f"sklearn test accuracy: {sklearn_test_acc:.4f}",
        f"pytorch test accuracy: {pytorch_test_acc:.4f}",
        "",
        "| model    | fold 1   | fold 2   | fold 3   | fold 4   | fold 5   | mean     | std      |",
        f"| sklearn  | {sk_cv[0]:.4f}   | {sk_cv[1]:.4f}   | {sk_cv[2]:.4f}   | "
        f"{sk_cv[3]:.4f}   | {sk_cv[4]:.4f}   | {sk_cv.mean():.4f}   | {sk_cv.std():.4f}   |",
        f"| PyTorch  | {pt_cv[0]:.4f}   | {pt_cv[1]:.4f}   | {pt_cv[2]:.4f}   | "
        f"{pt_cv[3]:.4f}   | {pt_cv[4]:.4f}   | {pt_cv.mean():.4f}   | {pt_cv.std():.4f}   |",
        "",
        f"sklearn train time [s]: {sklearn_train_time:.4f}",
        f"pytorch train time [s]: {pytorch_train_time:.4f}",
        f"pytorch cv train time [s]: {pytorch_cv_time:.4f}",
        "",
        sklearn_report.strip(),
    ]
    out = ROOT / "wyniki.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nZapisano: {out}")
    print(f"sklearn test acc: {sklearn_test_acc:.4f}")
    print(f"PyTorch test acc: {pytorch_test_acc:.4f}")


if __name__ == "__main__":
    main()
