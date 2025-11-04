import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path("data/processed")

def rolling_team_stats(df, team_col, goals_for_col, goals_against_col, result_col, n_matches=5):
    """
    Calcula estadísticas móviles (rolling window) por equipo.
    Retorna DataFrame con promedio de goles a favor/en contra y tasa de victorias.
    """
    stats = []
    teams = df[team_col].unique()

    for team in teams:
        team_df = df[df[team_col] == team].sort_values("DateTime").copy()
        # Goles a favor y en contra promedio en últimos N partidos
        team_df["GF_avg"] = team_df[goals_for_col].rolling(window=n_matches, min_periods=1).mean().shift(1)
        team_df["GA_avg"] = team_df[goals_against_col].rolling(window=n_matches, min_periods=1).mean().shift(1)

        # Porcentaje de victorias (1 si gana, 0 si no)
        team_df["WinRate"] = (team_df[result_col] == "W").astype(int)
        team_df["WinRate"] = team_df["WinRate"].rolling(window=n_matches, min_periods=1).mean().shift(1)

        stats.append(team_df)

    return pd.concat(stats, axis=0)


def make_features(df):
    """
    Genera variables agregadas por equipo (local y visitante).
    """
    #  Crear columna auxiliar de resultado relativo (desde perspectiva del equipo)
    df["HomeResult"] = np.where(df["FTR"] == "H", "W", np.where(df["FTR"] == "D", "D", "L"))
    df["AwayResult"] = np.where(df["FTR"] == "A", "W", np.where(df["FTR"] == "D", "D", "L"))

    #  Calcular stats para locales y visitantes
    home_stats = rolling_team_stats(df, "HomeTeam", "FTHG", "FTAG", "HomeResult")
    away_stats = rolling_team_stats(df, "AwayTeam", "FTAG", "FTHG", "AwayResult")

    #  Renombrar columnas
    home_stats = home_stats[["DateTime", "HomeTeam", "GF_avg", "GA_avg", "WinRate"]].rename(
        columns={"GF_avg": "Home_GF_avg", "GA_avg": "Home_GA_avg", "WinRate": "Home_WinRate"}
    )
    away_stats = away_stats[["DateTime", "AwayTeam", "GF_avg", "GA_avg", "WinRate"]].rename(
        columns={"GF_avg": "Away_GF_avg", "GA_avg": "Away_GA_avg", "WinRate": "Away_WinRate"}
    )

    #  Fusionar
    df_feat = df.merge(home_stats, on=["DateTime", "HomeTeam"], how="left")
    df_feat = df_feat.merge(away_stats, on=["DateTime", "AwayTeam"], how="left")

    #  Diferencias entre equipos
    df_feat["Diff_GF"] = df_feat["Home_GF_avg"] - df_feat["Away_GF_avg"]
    df_feat["Diff_GA"] = df_feat["Home_GA_avg"] - df_feat["Away_GA_avg"]
    df_feat["Diff_WinRate"] = df_feat["Home_WinRate"] - df_feat["Away_WinRate"]

    return df_feat


def enrich_features_and_save(filename: str):
    """Carga dataset con cuotas, genera features y guarda versión final."""
    df = pd.read_csv(PROCESSED_PATH / filename, encoding="utf-8", parse_dates=["DateTime"])
    df_feat = make_features(df)
    out_name = filename.replace("_with_market", "_features")
    df_feat.to_csv(PROCESSED_PATH / out_name, index=False, encoding="utf-8")
    print(f" Guardado: {out_name} ({len(df_feat)} filas)")
    return df_feat


if __name__ == "__main__":
    enrich_features_and_save("EPL_Set_with_market.csv")
    enrich_features_and_save("EPL_Recent_with_market.csv")