import os
from pathlib import Path

import joblib
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report

from data_prep import load_data, resolve_path, split_data

# zadanie 4 - te same ścieżki, podział i ziarno co przy treningu.
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_PATH = os.getenv("DATA_PATH", "data/iris.csv")
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
MODEL_TYPE = os.getenv("MODEL_TYPE", "logistic_regression")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
RESULTS_PATH = os.getenv("RESULTS_PATH", "wyniki.txt")


def evaluate() -> Path:
    model_path = resolve_path(MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Brak modelu ({model_path}). Najpierw uruchom: python src/train.py"
        )

    # model ktory wczytamy z dysku
    model = joblib.load(model_path)
    df = load_data(DATA_PATH)
    # to samo ziarno i test_size co w train.py, więc zbiór testowy jest ten sam.
    _X_train, X_test, _y_train, y_test = split_data(df, TEST_SIZE, RANDOM_SEED)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    lines = [
        f"model_type={MODEL_TYPE}",
        f"model_path={MODEL_PATH}",
        f"data_path={DATA_PATH}",
        f"test_size={TEST_SIZE}",
        f"random_seed={RANDOM_SEED}",
        "",
        f"accuracy_score: {acc:.4f}",
        "",
        report.strip(),
        "",
    ]
    out = resolve_path(RESULTS_PATH)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    saved = evaluate()
    print(f"Zapisano metryki: {saved}")
