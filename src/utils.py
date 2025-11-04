import pandas as pd
from pathlib import Path

PROCESSED_PATH = Path("data/processed")

def time_split(df: pd.DataFrame, date_split: str):
    """
    Divide el dataset en train/test usando una fecha de corte (walk-forward).
    """
    train = df[df["DateTime"] < date_split].copy()
    test = df[df["DateTime"] >= date_split].copy()

    print(f"📆 Fecha de corte: {date_split}")
    print(f"Train: {len(train)} partidos | Test: {len(test)} partidos")
    return train, test


def load_features_dataset(name: str = "EPL_Set_features.csv") -> pd.DataFrame:
    """
    Carga el dataset con features listo para modelar.
    """
    df = pd.read_csv(PROCESSED_PATH / name, encoding="utf-8", parse_dates=["DateTime"])
    print(f"✅ Dataset cargado: {name} ({len(df)} filas)")
    return df


if __name__ == "__main__":
    # Ejemplo rápido de uso
    df = load_features_dataset()
    # elegimos una fecha media (ajústala si deseas otro split)
    train, test = time_split(df, "2021-08-01")

    # Guardamos por conveniencia
    train.to_csv(PROCESSED_PATH / "train_set.csv", index=False, encoding="utf-8")
    test.to_csv(PROCESSED_PATH / "test_set.csv", index=False, encoding="utf-8")

    print("✅ train_set.csv y test_set.csv guardados en data/processed/")