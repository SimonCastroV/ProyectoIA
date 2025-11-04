import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

PROCESSED_PATH = Path("data/processed")
OUTPUT_PATH = Path("outputs/models")
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

def get_xy(path):
    """
    Carga el dataset desde CSV y devuelve X, y codificado numéricamente.
    """
    df = pd.read_csv(path, encoding="utf-8", parse_dates=["DateTime"])
    
    features = [
        "Diff_GF", "Diff_GA", "Diff_WinRate",
        "Home_GF_avg", "Away_GF_avg",
        "Home_WinRate", "Away_WinRate",
        "B365H", "B365D", "B365A"
    ]
    X = df[features].fillna(0)
    y = df["FTR"]

    # Codificar etiquetas 'H', 'D', 'A' como 0,1,2
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le


def evaluate_model(model, X_test, y_test, name, label_encoder):
    """
    Evalúa un modelo y calcula métricas estándar.
    """
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    logloss = log_loss(y_test, y_proba)
    brier = brier_score_loss(pd.get_dummies(y_test).values.flatten(), y_proba.flatten())
    acc = accuracy_score(y_test, y_pred)

    print(f"📊 {name} -> LogLoss:{logloss:.4f} | Brier:{brier:.4f} | Acc:{acc:.3f}")

    # Guardar predicciones
    df_out = pd.DataFrame({
        "True": label_encoder.inverse_transform(y_test),
        "Pred": label_encoder.inverse_transform(y_pred)
    })
    proba_df = pd.DataFrame(y_proba, columns=[f"P_{c}" for c in label_encoder.classes_])
    df_final = pd.concat([df_out, proba_df], axis=1)
    df_final.to_csv(PROCESSED_PATH / f"pred_{name.lower()}.csv", index=False, encoding="utf-8")

    return {"model": name, "logloss": logloss, "brier": brier, "accuracy": acc}

def train_advanced_models():
    X_train, y_train, le = get_xy(PROCESSED_PATH / "train_set.csv")
    X_test, y_test, _ = get_xy(PROCESSED_PATH / "test_set.csv")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    dump(rf, OUTPUT_PATH / "rf_model.joblib")
    results.append(evaluate_model(rf, X_test, y_test, "RandomForest", le))

    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        use_label_encoder=False
    )
    xgb.fit(X_train, y_train)
    dump(xgb, OUTPUT_PATH / "xgb_model.joblib")
    results.append(evaluate_model(xgb, X_test, y_test, "XGBoost", le))

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42)
    mlp.fit(X_train_s, y_train)
    dump(mlp, OUTPUT_PATH / "mlp_model.joblib")
    results.append(evaluate_model(mlp, X_test_s, y_test, "MLP", le))

    df_res = pd.DataFrame(results)
    df_res.to_csv(PROCESSED_PATH / "metrics_advanced_models.csv", index=False, encoding="utf-8")
    print("✅ Resultados guardados en metrics_advanced_models.csv")

    return df_res

if __name__ == "__main__":
    train_advanced_models()