# src/ui/app.py
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load


# ----------------------------
# Rutas
# ----------------------------
DATA_PATH = Path("data/processed")
MODELS_PATH = Path("outputs/models")
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_CSV = DATA_DIR / "raw" / "EPL_Set.csv"  # Dataset histórico hasta 2021


# ----------------------------
# Helpers de limpieza / formateo
# ----------------------------
def clean_str(x: str) -> str:
    """Quita NBSP y espacios extra en strings."""
    return str(x).replace("\xa0", " ").strip()


def pct_str(x, clamp=True):
    """'52.3%' robusto ante None/NaN/inf; recorta a [0,1] si clamp=True."""
    try:
        xf = float(x)
    except Exception:
        return "–"
    if not math.isfinite(xf):
        return "–"
    if clamp:
        xf = min(max(xf, 0.0), 1.0)
    return f"{xf * 100:.1f}%"


def odds_str(p):
    """Cuota justa = 1/p. Si p=0 o inválido → '–'."""
    try:
        p = float(p)
        if p <= 0 or not math.isfinite(p):
            return "–"
        return f"{1.0 / p:.2f}"
    except Exception:
        return "–"


def read_csv_safely(path, parse_dates=None):
    """
    Intenta leer CSV probando varios encodings y limpia NBSP.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, parse_dates=parse_dates)
            break
        except UnicodeDecodeError as e:
            last_err = e
    else:
        raw = Path(path).read_bytes().replace(b"\xa0", b" ")
        df = pd.read_csv(io.StringIO(raw.decode("utf-8", errors="ignore")), parse_dates=parse_dates)

    # Normalizar strings
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
    return df


# Funciones adicionales 
def get_model_feature_order(model, default_feats):
    """
    Si el modelo/pipeline tiene feature_names_in_, úsalo para ordenar columnas.
    Si no, retorna default_feats.
    """
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)
    # XGBoost sklearn moderno a veces guarda en el booster:
    try:
        booster = getattr(model, "get_booster", lambda: None)()
        if booster is not None and getattr(booster, "feature_names", None):
            return list(booster.feature_names)
    except Exception:
        pass
    return list(default_feats)


def infer_label_mapping(model, df_all, feats):
    """
    Infiero un mapeo de labels del modelo -> {'H','D','A'} comparando predicciones
    con la verdad terreno FTR. Devuelve (mapping, classes).
    """
    # Necesitamos FTR en df_all
    if "FTR" not in df_all.columns:
        classes = list(getattr(model, "classes_", []))
        return {}, classes

    sample = df_all.dropna(subset=feats).copy()
    if sample.empty:
        classes = list(getattr(model, "classes_", []))
        return {}, classes

    if len(sample) > 10000:
        sample = sample.tail(10000)

    Xf = ensure_features(sample, df_all, feats).fillna(0.0)

    try:
        y_pred = model.predict(Xf)
    except Exception:
        classes = list(getattr(model, "classes_", []))
        return {}, classes

    classes = list(getattr(model, "classes_", []))
    if set(classes) >= {"H", "D", "A"}:
        return {c: c for c in classes}, classes

    # Co-ocurrencias
    counts = {}
    y_true = sample["FTR"].astype(str).values
    for pred, true in zip(y_pred, y_true):
        if true not in {"H", "D", "A"}:
            continue
        counts.setdefault(pred, {"H": 0, "D": 0, "A": 0})
        counts[pred][true] += 1

    triples = []
    for mlab, d in counts.items():
        for t in ["H", "D", "A"]:
            triples.append((mlab, t, d.get(t, 0)))
    triples.sort(key=lambda x: x[2], reverse=True)

    mapping, targets_used = {}, set()
    for mlab, t, c in triples:
        if mlab in mapping or t in targets_used:
            continue
        mapping[mlab] = t
        targets_used.add(t)
        if len(mapping) == 3:
            break

    for mlab in classes:
        if mlab not in mapping:
            for t in ["H", "D", "A"]:
                if t not in targets_used:
                    mapping[mlab] = t
                    targets_used.add(t)
                    break

    return mapping, classes



# ----------------------------
# Carga de equipos desde histórico (sin API)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_teams_from_csv(csv_path: Path):
    if not csv_path.exists():
        st.error(f"No encontré el dataset en {csv_path}")
        return []
    df = read_csv_safely(csv_path)
    if ("HomeTeam" not in df.columns) or ("AwayTeam" not in df.columns):
        st.error("El CSV no contiene columnas HomeTeam/AwayTeam")
        return []
    teams = pd.Index(df["HomeTeam"].dropna().unique()).union(pd.Index(df["AwayTeam"].dropna().unique()))
    teams = sorted(clean_str(t) for t in teams if isinstance(t, str) and t.strip())
    return teams


# ----------------------------
# Datos de features procesados (para construir X)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    # Datasets con features (histórico + reciente)
    df_hist = read_csv_safely(DATA_PATH / "EPL_Set_features.csv", parse_dates=["DateTime"])
    df_recent = read_csv_safely(DATA_PATH / "EPL_Recent_features.csv", parse_dates=["DateTime"])
    df_all = pd.concat([df_hist, df_recent], ignore_index=True)

    # Asegurar limpieza de nombres de equipos
    for col in ("HomeTeam", "AwayTeam"):
        if col in df_all.columns:
            df_all[col] = df_all[col].apply(clean_str)

    return df_all


@st.cache_data(show_spinner=False)
def league_baselines(df_all: pd.DataFrame):
    p_home = (df_all["FTR"] == "H").mean() if "FTR" in df_all.columns else 0.45
    p_draw = (df_all["FTR"] == "D").mean() if "FTR" in df_all.columns else 0.25
    p_away = (df_all["FTR"] == "A").mean() if "FTR" in df_all.columns else 0.30
    # Evitar ceros
    eps = 1e-6
    p_home = max(p_home, eps)
    p_draw = max(p_draw, eps)
    p_away = max(p_away, eps)
    vig = 0.05
    odds = {
        "H": 1.0 / (p_home * (1 - vig)),
        "D": 1.0 / (p_draw * (1 - vig)),
        "A": 1.0 / (p_away * (1 - vig)),
    }
    return {"p_home": p_home, "p_draw": p_draw, "p_away": p_away, "odds": odds}


# ----------------------------
# Carga del modelo XGBoost
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_xgb_model():
    """
    Prioriza xgb_calibrated.joblib. Si no existe, usa xgb_model.joblib.
    """
    cal = MODELS_PATH / "xgb_calibrated.joblib"
    raw = MODELS_PATH / "xgb_model.joblib"
    if cal.exists():
        return load(cal), "xgb_calibrated"
    if raw.exists():
        return load(raw), "xgb"
    raise FileNotFoundError("No encontré ni xgb_calibrated.joblib ni xgb_model.joblib en outputs/models/")


# ----------------------------
# Construcción de features para un partido HOME vs AWAY
# ----------------------------
def diff_nan_to_mean(df, col, val):
    if col not in df.columns:
        # Si el df no tiene la columna, usa el valor o 0.0
        return float(val) if pd.notna(val) else 0.0
    if pd.isna(val):
        series = df[col].dropna()
        return float(series.mean()) if not series.empty else 0.0
    return float(val)


def build_features_for_pair(df_all: pd.DataFrame, home: str, away: str):
    # Filas recientes por rol
    home_last = df_all[df_all["HomeTeam"] == home].sort_values("DateTime").tail(1)
    away_last = df_all[df_all["AwayTeam"] == away].sort_values("DateTime").tail(1)

    # HOME stats
    if home_last.empty:
        any_home = df_all[(df_all["HomeTeam"] == home) | (df_all["AwayTeam"] == home)].sort_values("DateTime").tail(1)
        if not any_home.empty:
            if any_home.iloc[0]["HomeTeam"] == home:
                home_stats = {
                    "Home_GF_avg": any_home.iloc[0].get("Home_GF_avg", np.nan),
                    "Home_GA_avg": any_home.iloc[0].get("Home_GA_avg", np.nan),
                    "Home_WinRate": any_home.iloc[0].get("Home_WinRate", np.nan),
                }
            else:
                home_stats = {
                    "Home_GF_avg": any_home.iloc[0].get("Away_GF_avg", np.nan),
                    "Home_GA_avg": any_home.iloc[0].get("Away_GA_avg", np.nan),
                    "Home_WinRate": any_home.iloc[0].get("Away_WinRate", np.nan),
                }
        else:
            home_stats = {"Home_GF_avg": np.nan, "Home_GA_avg": np.nan, "Home_WinRate": np.nan}
    else:
        home_stats = {
            "Home_GF_avg": home_last.iloc[0].get("Home_GF_avg", np.nan),
            "Home_GA_avg": home_last.iloc[0].get("Home_GA_avg", np.nan),
            "Home_WinRate": home_last.iloc[0].get("Home_WinRate", np.nan),
        }

    # AWAY stats
    if away_last.empty:
        any_away = df_all[(df_all["HomeTeam"] == away) | (df_all["AwayTeam"] == away)].sort_values("DateTime").tail(1)
        if not any_away.empty:
            if any_away.iloc[0]["AwayTeam"] == away:
                away_stats = {
                    "Away_GF_avg": any_away.iloc[0].get("Away_GF_avg", np.nan),
                    "Away_GA_avg": any_away.iloc[0].get("Away_GA_avg", np.nan),
                    "Away_WinRate": any_away.iloc[0].get("Away_WinRate", np.nan),
                }
            else:
                away_stats = {
                    "Away_GF_avg": any_away.iloc[0].get("Home_GF_avg", np.nan),
                    "Away_GA_avg": any_away.iloc[0].get("Home_GA_avg", np.nan),
                    "Away_WinRate": any_away.iloc[0].get("Home_WinRate", np.nan),
                }
        else:
            away_stats = {"Away_GF_avg": np.nan, "Away_GA_avg": np.nan, "Away_WinRate": np.nan}
    else:
        away_stats = {
            "Away_GF_avg": away_last.iloc[0].get("Away_GF_avg", np.nan),
            "Away_GA_avg": away_last.iloc[0].get("Away_GA_avg", np.nan),
            "Away_WinRate": away_last.iloc[0].get("Away_WinRate", np.nan),
        }

    # Diferencias
    diff_gf = (home_stats["Home_GF_avg"] - away_stats["Away_GF_avg"])
    diff_ga = (home_stats["Home_GA_avg"] - away_stats["Away_GA_avg"])
    diff_wr = (home_stats["Home_WinRate"] - away_stats["Away_WinRate"])

    # Odds del último H2H (si existe); si no, baseline de liga
    h2h = df_all[(df_all["HomeTeam"] == home) & (df_all["AwayTeam"] == away)].sort_values("DateTime").tail(1)
    if not h2h.empty:
        oddsH = float(h2h.iloc[0].get("B365H", np.nan))
        oddsD = float(h2h.iloc[0].get("B365D", np.nan))
        oddsA = float(h2h.iloc[0].get("B365A", np.nan))
    else:
        lbs = league_baselines(df_all)
        oddsH, oddsD, oddsA = lbs["odds"]["H"], lbs["odds"]["D"], lbs["odds"]["A"]

    row = {
        "HomeTeam": home,
        "AwayTeam": away,
        "Home_GF_avg": diff_nan_to_mean(df_all, "Home_GF_avg", home_stats["Home_GF_avg"]),
        "Away_GF_avg": diff_nan_to_mean(df_all, "Away_GF_avg", away_stats["Away_GF_avg"]),
        "Home_GA_avg": diff_nan_to_mean(df_all, "Home_GA_avg", home_stats["Home_GA_avg"]),
        "Away_GA_avg": diff_nan_to_mean(df_all, "Away_GA_avg", away_stats["Away_GA_avg"]),
        "Home_WinRate": diff_nan_to_mean(df_all, "Home_WinRate", home_stats["Home_WinRate"]),
        "Away_WinRate": diff_nan_to_mean(df_all, "Away_WinRate", away_stats["Away_WinRate"]),
        "Diff_GF": diff_nan_to_mean(df_all, "Diff_GF", diff_gf),
        "Diff_GA": diff_nan_to_mean(df_all, "Diff_GA", diff_ga),
        "Diff_WinRate": diff_nan_to_mean(df_all, "Diff_WinRate", diff_wr),
        "B365H": oddsH,
        "B365D": oddsD,
        "B365A": oddsA,
    }
    X = pd.DataFrame([row])
    return X


# ----------------------------
# Predicción (usa XGBoost cargado)
# ----------------------------
FEATS = [
    "Diff_GF",
    "Diff_GA",
    "Diff_WinRate",
    "Home_GF_avg",
    "Away_GF_avg",
    "Home_WinRate",
    "Away_WinRate",
    "B365H",
    "B365D",
    "B365A",
]


def ensure_features(X: pd.DataFrame, df_all: pd.DataFrame, feats) -> pd.DataFrame:
    """Garantiza que X tenga todas las columnas requeridas; si falta alguna, rellena con media (o 0)."""
    X = X.copy()
    for f in feats:
        if f not in X.columns:
            if f in df_all.columns:
                series = df_all[f].dropna()
                X[f] = float(series.mean()) if not series.empty else 0.0
            else:
                X[f] = 0.0
    return X[feats].astype(float)



def predict_match(model, X, df_all):
    # 1) Orden de features exacto
    feats_model = get_model_feature_order(model, FEATS)
    Xf = ensure_features(X, df_all, feats_model).fillna(0.0)

    # 2) Probabilidades crudas del modelo
    probs = model.predict_proba(Xf)
    classes = list(getattr(model, "classes_", []))

    # 3) Si ya son etiquetas H/D/A, mapear directo
    if set(classes) >= {"H","D","A"}:
        proba_map = {cls: probs[0, i] for i, cls in enumerate(classes)}
        pH, pD, pA = proba_map.get("H",0.0), proba_map.get("D",0.0), proba_map.get("A",0.0)
        return pH, pD, pA, classes, probs[0]

    # 4) Si no, inferir mapeo con el dataset
    mapping, classes_full = infer_label_mapping(model, df_all, feats_model)
    # classes_full debería coincidir con 'classes'
    proba_map_generic = {cls: probs[0, i] for i, cls in enumerate(classes)}

    # Reubicar a H/D/A
    pH = pD = pA = 0.0
    for mlab, p in proba_map_generic.items():
        target = mapping.get(mlab)
        if target == "H":
            pH = p
        elif target == "D":
            pD = p
        elif target == "A":
            pA = p

    # Normalizar por si queda algún hueco
    s = pH + pD + pA
    if s > 0:
        pH, pD, pA = pH/s, pD/s, pA/s

    return pH, pD, pA, classes, probs[0]


# ----------------------------
# UI simple: 2 selects + botón (sin chat)
# ----------------------------
def main():
    st.set_page_config(page_title="PremierAI — Predicción XGBoost", page_icon="⚽", layout="centered")
    st.title("⚽ Predicción de Ganador — Premier League (XGBoost)")
    st.caption("Usa tu dataset histórico (hasta 2021) y el modelo XGBoost guardado en outputs/models/. Sin APIs.")

    # Equipos desde histórico
    teams = load_teams_from_csv(RAW_CSV)
    if not teams:
        st.stop()

    # Datos procesados (para features)
    df_all = load_data()

    # Modelo XGB
    try:
        model, model_name = load_xgb_model()
    except Exception as e:
        st.error(str(e))
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Equipo Local", options=teams, index=0, key="home_team")
    with col2:
        away = st.selectbox("Equipo Visitante", options=teams, index=1, key="away_team")

    st.caption("Nota: el veredicto final solo puede ser Local o Visitante (el empate no se elige como 'ganador').")

    if st.button("Predecir ganador", type="primary"):
        if home == away:
            st.warning("El local y el visitante no pueden ser el mismo equipo.")
            st.stop()

        # Construir features y predecir
        X = build_features_for_pair(df_all, home, away)

        #  AQUÍ ESTÁ EL CAMBIO: ahora recibimos también 'classes' y 'raw'
        pH, pD, pA, classes, raw = predict_match(model, X, df_all)

        # Mostrar probabilidades
        st.subheader(f"{home} vs {away}")
        st.markdown(f"**Probabilidades del modelo {model_name.upper()}:**")
        st.markdown(
            f"-  Local (H): **{pct_str(pH)}**  \n"
            f"-  Empate (D): **{pct_str(pD)}**  \n"
            f"-  Visitante (A): **{pct_str(pA)}**"
        )

        # Cuotas justas (1/p)
        st.markdown("**Cuotas justas (1/p):**")
        st.markdown(f"- H: **{odds_str(pH)}** · D: **{odds_str(pD)}** · A: **{odds_str(pA)}**")

        # Cuotas de mercado usadas (si se encontraron en el último H2H, o baseline)
        oddsH = float(X.iloc[0].get("B365H", np.nan))
        oddsD = float(X.iloc[0].get("B365D", np.nan))
        oddsA = float(X.iloc[0].get("B365A", np.nan))
        if not (math.isfinite(oddsH) and oddsH > 0) or not (math.isfinite(oddsD) and oddsD > 0) or not (
            math.isfinite(oddsA) and oddsA > 0
        ):
            lbs = league_baselines(df_all)
            oddsH, oddsD, oddsA = lbs["odds"]["H"], lbs["odds"]["D"], lbs["odds"]["A"]

        st.markdown("**Cuotas usadas (sim/mercado):**")
        st.markdown(f"- H: **{oddsH:.2f}** · D: **{oddsD:.2f}** · A: **{oddsA:.2f}**")

        #  Expander de depuración (opcional)
        with st.expander("Detalles técnicos (debug)"):
            st.write("classes_ del modelo:", classes)
            st.write("fila predict_proba cruda:", [float(x) for x in np.ravel(raw)])

        # Veredicto: solo Local o Visitante (ignora empates)
        pick = "H" if pH >= pA else "A"
        label = {"H": "Local", "A": "Visitante"}[pick]
        emoji = {"H": "", "A": ""}[pick]
        st.success(f" **Gana:** {emoji} **{label}**")  

if __name__ == "__main__":
    main()
