import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# zadanie 4 - parametry z konfiguracji w pliku .env
DATA_PATH = os.getenv("DATA_PATH", "data/iris.csv")
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "target")


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    # wczytanie csv do DataFrame
    if path is None:
        path = DATA_PATH
    file_path = resolve_path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku z danymi: {file_path}")
    return pd.read_csv(file_path)


def split_data(
    df: pd.DataFrame,
    test_size: float | None = None,
    seed: int | None = None,
    target_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # podział na zbiór treningowy i testowy. zwraca (X_train, X_test, y_train, y_test)
    if test_size is None:
        test_size = TEST_SIZE
    if seed is None:
        seed = RANDOM_SEED
    if target_column is None:
        target_column = TARGET_COLUMN

    if df.empty:
        raise ValueError("DataFrame jest pusty - nie da się podzielić danych.")
    if target_column not in df.columns:
        raise ValueError(f"Brak kolumny docelowej '{target_column}' w danych.")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    # to samo ziarno i stratify co w poprzednim laboratorium, przez co podział jest powtarzalny
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
