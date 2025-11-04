import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")

def load_dataset(filename: str) -> pd.DataFrame:
    """Lee un CSV con codificación flexible y convierte fechas."""
    path = RAW_PATH / filename
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="ISO-8859-1")

    # Normalizar nombres de columnas
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # Convertir fecha si existe columna DateTime o Date
    for col in ["DateTime", "Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            break

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza columnas y elimina registros incompletos."""
    columnas_minimas = ["Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    for col in columnas_minimas:
        if col not in df.columns:
            raise ValueError(f"Falta columna obligatoria: {col}")

    # Eliminar filas sin resultado o goles
    df = df.dropna(subset=["FTHG", "FTAG", "FTR"])
    df = df[df["FTR"].isin(["H", "D", "A"])]

    # Convertir tipos
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    # Normalizar nombres de equipos
    df["HomeTeam"] = df["HomeTeam"].str.strip().str.title()
    df["AwayTeam"] = df["AwayTeam"].str.strip().str.title()

    # Ordenar por fecha
    if "DateTime" in df.columns:
        df = df.sort_values("DateTime").reset_index(drop=True)

    return df


def save_processed(df: pd.DataFrame, name: str):
    """Guarda el dataset limpio en data/processed/."""
    PROCESSED_PATH.mkdir(exist_ok=True, parents=True)
    out_path = PROCESSED_PATH / f"{name}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f" Guardado: {out_path} ({len(df)} filas)")


if __name__ == "__main__":
    #  Cargar datasets
    df_hist = load_dataset("EPL_Set.csv")
    df_recent = load_dataset("epl_api_recent.csv")

    #  Limpiar
    df_hist_clean = clean_dataset(df_hist)
    df_recent_clean = clean_dataset(df_recent)

    #  Guardar
    save_processed(df_hist_clean, "EPL_Set_clean")
    save_processed(df_recent_clean, "EPL_Recent_clean")