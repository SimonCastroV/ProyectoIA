# src/calibrate.py
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load, dump
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

PROCESSED_PATH = Path("data/processed")
MODELS_PATH = Path("outputs/models")

# -------------------------------------------------------
# 🔧 Calibrar modelo (Platt / Isotónica)
# -------------------------------------------------------
def calibrate_model(model_name: str, method: str = "isotonic"):
    """
    Carga modelo entrenado y recalibra probabilidades.
    Guarda versión calibrada y métricas comparativas.
    """
    print(f"\n⚙️ Calibrando {model_name} con método={method}")

    # Cargar modelo entrenado
    model_path = MODELS_PATH / f"{model_name}_model.joblib"
    if not model_path.exists():
        print(f"⚠️ Modelo {model_name} no encontrado. Saltando...")
        return None
    model = load(model_path)

    # Cargar datos
    train = pd.read_csv(PROCESSED_PATH / "train_set.csv", encoding="utf-8", parse_dates=["DateTime"])
    test = pd.read_csv(PROCESSED_PATH / "test_set.csv", encoding="utf-8", parse_dates=["DateTime"])

    features = [
        "Diff_GF", "Diff_GA", "Diff_WinRate",
        "Home_GF_avg", "Away_GF_avg",
        "Home_WinRate", "Away_WinRate",
        "B365H", "B365D", "B365A"
    ]

    X_train = train[features].fillna(0)
    X_test = test[features].fillna(0)
    y_train = train["FTR"]
    y_test = test["FTR"]

    # Codificar etiquetas (para modelos como XGB / MLP)
    le = LabelEncoder()
    le.fit(["H", "D", "A"])
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    # Algunos modelos (como LogReg/RF) ya manejan texto directamente
    try:
        calibrated = CalibratedClassifierCV(model, cv=3, method=method)
        calibrated.fit(X_train, y_train)
        y_proba = calibrated.predict_proba(X_test)
        y_pred = calibrated.predict(X_test)
        labels_used = y_test
    except ValueError:
        # Caso XGBoost: usar etiquetas numéricas
        calibrated = CalibratedClassifierCV(model, cv=3, method=method)
        calibrated.fit(X_train, y_train_enc)
        y_proba = calibrated.predict_proba(X_test)
        y_pred = calibrated.predict(X_test)
        labels_used = y_test_enc

    # Evaluación (maneja ambos tipos)
    if labels_used is y_test_enc:
        y_pred_decoded = le.inverse_transform(y_pred)
        y_true_decoded = le.inverse_transform(y_test_enc)
    else:
        y_pred_decoded = y_pred
        y_true_decoded = y_test

    # Métricas
    ll = log_loss(y_true_decoded, y_proba)
    bs = brier_score_loss(pd.get_dummies(y_true_decoded).values.flatten(), y_proba.flatten())
    acc = accuracy_score(y_true_decoded, y_pred_decoded)

    print(f"📊 {model_name} calibrado -> LogLoss:{ll:.4f} | Brier:{bs:.4f} | Acc:{acc:.3f}")

    # Guardar modelo calibrado
    dump(calibrated, MODELS_PATH / f"{model_name}_calibrated.joblib")

    return {"model": model_name, "logloss": ll, "brier": bs, "acc": acc}


# -------------------------------------------------------
# 🚀 Comparar todos los modelos calibrados
# -------------------------------------------------------
def calibrate_all():
    models = ["logreg", "rf", "xgb", "mlp"]
    results = []
    for m in models:
        try:
            res = calibrate_model(m)
            if res:
                results.append(res)
        except Exception as e:
            print(f"⚠️ Error calibrando {m}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(PROCESSED_PATH / "metrics_calibrated.csv", index=False, encoding="utf-8")
    print("\n✅ Resultados finales guardados en metrics_calibrated.csv")
    return df


if __name__ == "__main__":
    calibrate_all()