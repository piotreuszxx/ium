import pandas as pd
import pytest

from data_prep import load_data, split_data


def _sample(n: int = 150) -> pd.DataFrame:
    # z klasy, żeby stratify w split_data miało po czym dzielić
    labels = [i % 3 for i in range(n)]
    return pd.DataFrame(
        {
            "f1": [float(i) for i in range(n)],
            "f2": [float(i) * 0.5 for i in range(n)],
            "target": labels,
        }
    )


def test_split_sizes():
    df = _sample(150)
    _X_train, X_test, _y_train, y_test = split_data(
        df, test_size=0.2, seed=42, target_column="target"
    )
    assert len(X_test) == pytest.approx(len(df) * 0.2, abs=1)
    assert len(y_test) == len(X_test)


def test_split_covers_full_dataset():
    df = _sample(150)
    X_train, X_test, y_train, y_test = split_data(
        df, test_size=0.2, seed=42, target_column="target"
    )
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)


def test_split_empty_dataframe():
    empty = pd.DataFrame(columns=["f1", "f2", "target"])
    with pytest.raises(ValueError):
        split_data(empty, test_size=0.2, seed=42, target_column="target")


def test_split_invalid_test_size():
    df = _sample(150)
    with pytest.raises(ValueError):
        split_data(df, test_size=1.5, seed=42, target_column="target")


def test_load_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("data/brak_takiego_pliku.csv")
