# src/backtest.py
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path("data/processed")

# -------------------------------------------------------
# 📊 FUNCIONES AUXILIARES
# -------------------------------------------------------

def simulate_betting(df, model_name, stake=1.0, threshold=0.4):
    """
    Simula una estrategia de apuestas basada en las probabilidades del modelo.
    """
    df = df.copy()

    # Si hay varias columnas con el mismo nombre, quita duplicados
    df = df.loc[:, ~df.columns.duplicated()]

    bankroll = [100.0]
    profits = []

    for _, row in df.iterrows():
        probs = {"H": row["P_H"], "D": row["P_D"], "A": row["P_A"]}
        pred = max(probs, key=probs.get)
        p_pred = probs[pred]

        if p_pred >= threshold:
            odd = row[f"B365{pred}"]
            win = 1 if str(row["FTR"]).strip() == pred else 0
            profit = (odd - 1) * stake * win - stake * (1 - win)
        else:
            profit = 0.0

        bankroll.append(bankroll[-1] + profit)
        profits.append(profit)

    df["Profit"] = profits
    df["Capital"] = bankroll[1:]

    total_roi = (df["Capital"].iloc[-1] - 100) / 100
    returns = np.array(df["Profit"])
    sharpe = np.mean(returns) / (np.std(returns) + 1e-9)
    cummax = np.maximum.accumulate(df["Capital"])
    drawdown = np.min(df["Capital"] / cummax - 1)

    summary = {
        "Model": model_name,
        "ROI": total_roi,
        "Sharpe": sharpe,
        "MaxDrawdown": drawdown,
        "FinalCapital": df["Capital"].iloc[-1],
        "NumBets": np.sum(df["Profit"] != 0)
    }

    return summary, df


def run_backtests():
    models = {
        "logreg": "predictions_logreg.csv",
        "rf": "pred_randomforest.csv",
        "xgb": "pred_xgboost.csv",
        "mlp": "pred_mlp.csv"
    }
    results = []

    test_data = pd.read_csv(PROCESSED_PATH / "test_set.csv", encoding="utf-8")

    for model_name, file_name in models.items():
        file_path = PROCESSED_PATH / file_name
        if not file_path.exists():
            print(f"⚠️ Archivo de predicciones no encontrado: {file_name}")
            continue

        preds = pd.read_csv(file_path, encoding="utf-8")

        # unir por índice (sin perder orden)
        if len(preds) != len(test_data):
            print(f"⚠️ Tamaño diferente entre {file_name} y test_set.csv, ajustando por índice.")
        df = pd.concat([test_data.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

        # eliminar duplicados de columnas (como FTR)
        df = df.loc[:, ~df.columns.duplicated()]

        # verificar que tenga las columnas necesarias
        required_cols = ["B365H", "B365D", "B365A", "FTR", "P_H", "P_D", "P_A"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"⚠️ Columnas faltantes en {file_name}: {missing}")
            continue

        summary, df_bt = simulate_betting(df, model_name)
        results.append(summary)

        out_path = PROCESSED_PATH / f"backtest_{model_name}.csv"
        df_bt.to_csv(out_path, index=False, encoding="utf-8")
        print(f"✅ Backtest completado para {model_name} -> {out_path}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(PROCESSED_PATH / "backtest_summary.csv", index=False, encoding="utf-8")
    print("\n📈 Resultados finales guardados en backtest_summary.csv")
    print(df_results)
    return df_results


if __name__ == "__main__":
    run_backtests()