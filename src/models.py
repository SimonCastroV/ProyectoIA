import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from joblib import dump

PROCESSED_PATH = Path("data/processed")
OUTPUT_PATH = Path("outputs/models")
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

# ------------------- MODELO LOGÍSTICO MULTICLASE -------------------
def train_logreg(train_path="data/processed/train_set.csv"):
    """
    Entrena un modelo de regresión logística multiclase (H/D/A).
    """
    df = pd.read_csv(train_path, encoding="utf-8", parse_dates=["DateTime"])

    # Variables de entrada (features)
    features = [
        "Diff_GF", "Diff_GA", "Diff_WinRate",
        "Home_GF_avg", "Away_GF_avg", "Home_WinRate", "Away_WinRate",
        "B365H", "B365D", "B365A"
    ]
    X = df[features].fillna(0)
    y = df["FTR"]

    # Escalado y entrenamiento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, multi_class="multinomial")
    model.fit(X_scaled, y)

    # Guardar modelo y escalador
    dump(model, OUTPUT_PATH / "logreg_model.joblib")
    dump(scaler, OUTPUT_PATH / "scaler.joblib")

    print(" Modelo LogReg entrenado y guardado.")
    return model, scaler


def evaluate_logreg(model, scaler, test_path="data/processed/test_set.csv"):
    """
    Evalúa el modelo logístico sobre el conjunto de test.
    """
    df = pd.read_csv(test_path, encoding="utf-8", parse_dates=["DateTime"])

    features = [
        "Diff_GF", "Diff_GA", "Diff_WinRate",
        "Home_GF_avg", "Away_GF_avg", "Home_WinRate", "Away_WinRate",
        "B365H", "B365D", "B365A"
    ]
    X = df[features].fillna(0)
    y_true = df["FTR"]

    X_scaled = scaler.transform(X)
    y_proba = model.predict_proba(X_scaled)
    y_pred = model.predict(X_scaled)

    # Métricas
    ll = log_loss(y_true, y_proba)
    bs = brier_score_loss(pd.get_dummies(y_true).values.flatten(), y_proba.flatten())
    acc = np.mean(y_pred == y_true)

    print(f" LogLoss: {ll:.4f} | Brier: {bs:.4f} | Accuracy: {acc:.3f}")
    df_pred = df.copy()
    df_pred[["P_H", "P_D", "P_A"]] = y_proba
    df_pred["Predicted"] = y_pred

    out_path = PROCESSED_PATH / "predictions_logreg.csv"
    df_pred.to_csv(out_path, index=False, encoding="utf-8")
    print(f" Predicciones guardadas en: {out_path}")

    return df_pred

# ------------------- OPCIONAL: MODELO POISSON -------------------
def poisson_probs(avg_home_goals, avg_away_goals, max_goals=10):
    """
    Calcula la distribución de probabilidades Poisson para goles esperados.
    Retorna matriz de P(H), P(D), P(A)
    """
    from scipy.stats import poisson

    home_goals = np.arange(0, max_goals + 1)
    away_goals = np.arange(0, max_goals + 1)
    probs = np.outer(poisson.pmf(home_goals, avg_home_goals),
                     poisson.pmf(away_goals, avg_away_goals))

    p_home = np.sum(np.tril(probs, -1))
    p_draw = np.sum(np.diag(probs))
    p_away = np.sum(np.triu(probs, 1))
    return p_home, p_draw, p_away


if __name__ == "__main__":
    model, scaler = train_logreg()
    evaluate_logreg(model, scaler)