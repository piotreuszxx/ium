import os
from pathlib import Path

import joblib
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import load_data, resolve_path, split_data

# zadanie 4 - parametry z konfiguracji w pliku .env
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_PATH = os.getenv("DATA_PATH", "data/iris.csv")
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
MODEL_TYPE = os.getenv("MODEL_TYPE", "logistic_regression")
MAX_ITER = int(os.getenv("MAX_ITER", "500"))
C = float(os.getenv("C", "1.0"))
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")


def build_pipeline() -> Pipeline:
    # zadanie 2 - Pipeline: skalowanie cech + klasyfikator, jak w poprzednim laboratorium.
    if MODEL_TYPE != "logistic_regression":
        raise ValueError(f"Nieobsługiwany MODEL_TYPE: {MODEL_TYPE}")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=MAX_ITER,
                    C=C,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def train() -> Path:
    df = load_data(DATA_PATH)
    X_train, _X_test, y_train, _y_test = split_data(df, TEST_SIZE, RANDOM_SEED)
    model = build_pipeline()
    model.fit(X_train, y_train)

    out = resolve_path(MODEL_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    # zapis modelu do models/ - evaluate.py wczyta ten sam plik, ale nie będzie już go trenować, tylko oceniać
    joblib.dump(model, out)
    return out


if __name__ == "__main__":
    saved = train()
    print(f"Zapisano model: {saved}")
