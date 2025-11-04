import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path("data/processed")

def simulate_market_odds(df: pd.DataFrame, vigorish: float = 0.05) -> pd.DataFrame:
    """
    Simula cuotas del mercado (odds) basadas en frecuencias históricas
    y añade probabilidades implícitas ajustadas por margen.
    """

    # 1️⃣ Calcular frecuencias empíricas de resultado
    p_home = np.mean(df["FTR"] == "H")
    p_draw = np.mean(df["FTR"] == "D")
    p_away = np.mean(df["FTR"] == "A")

    print(f"Frecuencias históricas -> H:{p_home:.2f}  D:{p_draw:.2f}  A:{p_away:.2f}")

    # 2️⃣ Simular cuotas tipo Bet365 (1 / probabilidad * margen)
    df["B365H"] = 1 / (p_home * (1 - vigorish))
    df["B365D"] = 1 / (p_draw * (1 - vigorish))
    df["B365A"] = 1 / (p_away * (1 - vigorish))

    # 3️⃣ Calcular probabilidades implícitas
    df["ImpProb_H"] = 1 / df["B365H"]
    df["ImpProb_D"] = 1 / df["B365D"]
    df["ImpProb_A"] = 1 / df["B365A"]

    # 4️⃣ Normalizar para que sumen ≈ 1
    total = df[["ImpProb_H", "ImpProb_D", "ImpProb_A"]].sum(axis=1)
    df["ImpProb_H"] /= total
    df["ImpProb_D"] /= total
    df["ImpProb_A"] /= total

    return df


def enrich_and_save(filename: str):
    """Carga dataset limpio, aplica simulación y guarda versión con cuotas."""
    path = PROCESSED_PATH / filename
    df = pd.read_csv(path, encoding="utf-8")

    df_enriched = simulate_market_odds(df)

    out_name = filename.replace("_clean", "_with_market")
    out_path = PROCESSED_PATH / out_name
    df_enriched.to_csv(out_path, index=False, encoding="utf-8")

    print(f"✅ Guardado: {out_path} ({len(df_enriched)} filas)")
    return df_enriched


if __name__ == "__main__":
    enrich_and_save("EPL_Set_clean.csv")
    enrich_and_save("EPL_Recent_clean.csv")