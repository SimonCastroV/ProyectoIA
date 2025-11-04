import pandas as pd
from pathlib import Path
import random

DATA_PATH = Path("data/processed")

# -------------------------------------------------------
# 🧠 CARGAR DATOS
# -------------------------------------------------------
def load_data():
    metrics = pd.read_csv(DATA_PATH / "metrics_calibrated.csv")
    backtest = pd.read_csv(DATA_PATH / "backtest_summary.csv")
    return metrics, backtest


# -------------------------------------------------------
# 💬 RESPUESTAS NATURALES
# -------------------------------------------------------
def describe_model_performance(backtest_df):
    best_roi = backtest_df.loc[backtest_df["ROI"].idxmax()]
    worst_roi = backtest_df.loc[backtest_df["ROI"].idxmin()]

    msg = (
        f"📈 El modelo más rentable fue **{best_roi['Model']}**, "
        f"con un ROI de {best_roi['ROI']*100:.2f}% y un Sharpe ratio de {best_roi['Sharpe']:.3f}.\n"
        f"💸 El menos rentable fue **{worst_roi['Model']}**, "
        f"con un ROI de {worst_roi['ROI']*100:.2f}%.\n"
        f"👉 En general, los modelos con mejores resultados fueron los basados en "
        f"aprendizaje profundo y ensambles (MLP y XGBoost)."
    )
    return msg


def get_model_stats(model_name, metrics_df, backtest_df):
    model_name = model_name.lower().strip()

    # Buscar nombre aproximado
    matches = [m for m in backtest_df["Model"].str.lower() if model_name in m]
    if not matches:
        return f"⚠️ No encontré información sobre el modelo '{model_name}'."
    match = matches[0]

    row_m = metrics_df[metrics_df["model"].str.lower() == match]
    row_b = backtest_df[backtest_df["Model"].str.lower() == match]

    if row_m.empty or row_b.empty:
        return f"⚠️ No se encontró información completa para el modelo '{model_name}'."

    m = row_m.iloc[0]
    b = row_b.iloc[0]

    return (
        f"📊 **{match.upper()}**\n"
        f"- LogLoss: {m['logloss']:.4f}\n"
        f"- Brier: {m['brier']:.4f}\n"
        f"- Accuracy: {m['acc']:.3f}\n"
        f"- ROI: {b['ROI']*100:.2f}%\n"
        f"- Sharpe: {b['Sharpe']:.3f}\n"
        f"- Max Drawdown: {b['MaxDrawdown']*100:.2f}%\n"
        f"- Apuestas simuladas: {int(b['NumBets'])}\n"
        f"- Capital final: {b['FinalCapital']:.2f} 💵"
    )


def random_tip():
    tips = [
        "Recuerda: un ROI positivo no siempre implica bajo riesgo. Mira también el Sharpe.",
        "Los modelos calibrados ofrecen probabilidades más realistas que las brutas.",
        "Puedes combinar estrategias entre MLP y XGBoost para diversificar riesgo.",
        "Evita apostar con probabilidad menor a 0.4: reduce la tasa de aciertos netos.",
    ]
    return f"💡 {random.choice(tips)}"


# -------------------------------------------------------
# 💬 BUCLE PRINCIPAL DE CHAT
# -------------------------------------------------------
def chat():
    print("🤖 Bienvenido al asistente de apuestas inteligentes (Premier League)\n")
    print("Escribe una pregunta (o 'salir' para terminar)\n")

    metrics, backtest = load_data()

    while True:
        user_input = input("Tú: ").strip().lower()
        if user_input in ["salir", "exit", "quit"]:
            print("👋 Hasta pronto. ¡Que tus predicciones sean ganadoras!")
            break

        if "mejor modelo" in user_input or "rentable" in user_input:
            print(describe_model_performance(backtest))
        elif "roi" in user_input:
            name = user_input.replace("roi", "").replace("del", "").replace("modelo", "").strip()
            print(get_model_stats(name, metrics, backtest))
        elif "xgboost" in user_input or "mlp" in user_input or "logreg" in user_input or "random" in user_input:
            for name in ["xgboost", "mlp", "logreg", "random"]:
                if name in user_input:
                    print(get_model_stats(name, metrics, backtest))
                    break
        else:
            print("🤔 No entiendo esa pregunta. Puedes preguntar cosas como:\n"
                  "- ¿Cuál modelo fue más rentable?\n"
                  "- Dame el ROI del MLP.\n"
                  "- Muéstrame las métricas del XGBoost.\n"
                  "- ¿Qué modelo tiene mejor Sharpe?\n")
        print(random_tip(), "\n")


if __name__ == "__main__":
    chat()