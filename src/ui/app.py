# src/ui/app.py
import re
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
from joblib import load

DATA_PATH = Path("data/processed")
MODELS_PATH = Path("outputs/models")

# ----------------------------
# Utilidades de datos
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    # Datasets con features (histórico + reciente)
    df_hist = pd.read_csv(DATA_PATH / "EPL_Set_features.csv", encoding="utf-8", parse_dates=["DateTime"])
    df_recent = pd.read_csv(DATA_PATH / "EPL_Recent_features.csv", encoding="utf-8", parse_dates=["DateTime"])
    df_all = pd.concat([df_hist, df_recent], ignore_index=True)
    # Backtests y métricas (para responder preguntas generales)
    backtest = pd.read_csv(DATA_PATH / "backtest_summary.csv", encoding="utf-8")
    metrics = pd.read_csv(DATA_PATH / "metrics_calibrated.csv", encoding="utf-8")
    # Lista de equipos conocidos
    teams = sorted(set(df_all["HomeTeam"].unique()) | set(df_all["AwayTeam"].unique()))
    return df_all, backtest, metrics, teams

@st.cache_data(show_spinner=False)
def league_baselines(df_all: pd.DataFrame):
    # Frecuencias empíricas (para odds/priors si falta info específica)
    p_home = (df_all["FTR"] == "H").mean()
    p_draw = (df_all["FTR"] == "D").mean()
    p_away = (df_all["FTR"] == "A").mean()
    # Cuotas simuladas coherentes con un vigorish ~5%
    vig = 0.05
    odds = {
        "H": 1.0 / (p_home * (1 - vig)),
        "D": 1.0 / (p_draw * (1 - vig)),
        "A": 1.0 / (p_away * (1 - vig))
    }
    return {"p_home": p_home, "p_draw": p_draw, "p_away": p_away, "odds": odds}

# ----------------------------
# Modelos calibrados
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    for name in ["logreg", "rf", "xgb"]:
        path = MODELS_PATH / f"{name}_calibrated.joblib"
        if path.exists():
            models[name] = load(path)
    return models

# ----------------------------
# Extracción de intención simple
# ----------------------------
def extract_teams(user_text: str, team_list):
    text = user_text.lower()
    # Emparejar por substrings (robusto a may/minus y espacios)
    found = []
    for t in team_list:
        t_norm = t.lower()
        if re.search(rf"\b{re.escape(t_norm)}\b", text):
            found.append(t)
    # Si hay duplicados por “city”, “united”, filtrar por match exacto
    found = list(dict.fromkeys(found))
    if len(found) >= 2:
        # El orden en el texto define local vs visitante si es posible
        first = min(found, key=lambda t: text.index(t.lower()))
        others = [t for t in found if t != first]
        second = min(others, key=lambda t: text.index(t.lower()))
        return first, second
    return None, None

# ----------------------------
# Construcción de features para un partido HOME vs AWAY
# ----------------------------
def build_features_for_pair(df_all: pd.DataFrame, home: str, away: str):
    # 1) intentar usar la info más reciente de cada equipo en su rol
    #    (último partido del HOME jugando de local, y del AWAY jugando de visitante)
    home_last = df_all[df_all["HomeTeam"] == home].sort_values("DateTime").tail(1)
    away_last = df_all[df_all["AwayTeam"] == away].sort_values("DateTime").tail(1)

    # Si falta alguno, toma el más reciente partido del equipo (cualquier rol) para aproximar
    if home_last.empty:
        any_home = df_all[(df_all["HomeTeam"] == home) | (df_all["AwayTeam"] == home)].sort_values("DateTime").tail(1)
        # mapear columnas según rol en esa fila aproximada
        if not any_home.empty:
            # si jugó como local en esa fila
            if any_home.iloc[0]["HomeTeam"] == home:
                home_stats = {
                    "Home_GF_avg": any_home.iloc[0].get("Home_GF_avg", np.nan),
                    "Home_GA_avg": any_home.iloc[0].get("Home_GA_avg", np.nan),
                    "Home_WinRate": any_home.iloc[0].get("Home_WinRate", np.nan),
                }
            else:
                # si jugó como visitante, aproximar usando métricas 'Away_' como si fueran base del home
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

    # 2) diferencias
    diff_gf = (home_stats["Home_GF_avg"] - away_stats["Away_GF_avg"])
    diff_ga = (home_stats["Home_GA_avg"] - away_stats["Away_GA_avg"])
    diff_wr = (home_stats["Home_WinRate"] - away_stats["Away_WinRate"])

    # 3) odds: intentamos tomar de su último cara-a-cara con ese orden (si existe)
    h2h = df_all[(df_all["HomeTeam"] == home) & (df_all["AwayTeam"] == away)].sort_values("DateTime").tail(1)
    if not h2h.empty:
        oddsH = float(h2h.iloc[0].get("B365H", np.nan))
        oddsD = float(h2h.iloc[0].get("B365D", np.nan))
        oddsA = float(h2h.iloc[0].get("B365A", np.nan))
    else:
        # fallback: medias de liga (simuladas)
        lbs = league_baselines(df_all)
        oddsH, oddsD, oddsA = lbs["odds"]["H"], lbs["odds"]["D"], lbs["odds"]["A"]

    row = {
        "HomeTeam": home, "AwayTeam": away,
        "Home_GF_avg": diff_nan_to_mean(df_all, "Home_GF_avg", home_stats["Home_GF_avg"]),
        "Away_GF_avg": diff_nan_to_mean(df_all, "Away_GF_avg", away_stats["Away_GF_avg"]),
        "Home_GA_avg": diff_nan_to_mean(df_all, "Home_GA_avg", home_stats["Home_GA_avg"]),
        "Away_GA_avg": diff_nan_to_mean(df_all, "Away_GA_avg", away_stats["Away_GA_avg"]),
        "Home_WinRate": diff_nan_to_mean(df_all, "Home_WinRate", home_stats["Home_WinRate"]),
        "Away_WinRate": diff_nan_to_mean(df_all, "Away_WinRate", away_stats["Away_WinRate"]),
        "Diff_GF": diff_nan_to_mean(df_all, "Diff_GF", diff_gf),
        "Diff_GA": diff_nan_to_mean(df_all, "Diff_GA", diff_ga),
        "Diff_WinRate": diff_nan_to_mean(df_all, "Diff_WinRate", diff_wr),
        "B365H": oddsH, "B365D": oddsD, "B365A": oddsA
    }
    X = pd.DataFrame([row])
    return X

def diff_nan_to_mean(df, col, val):
    if pd.isna(val):
        return float(df[col].dropna().mean())
    return float(val)

# ----------------------------
# Predicción + recomendación
# ----------------------------
def predict_match(model, X):
    # Los modelos calibrados esperan features "crudos" (sin escalar)
    probs = model.predict_proba(X[[
        "Diff_GF","Diff_GA","Diff_WinRate",
        "Home_GF_avg","Away_GF_avg","Home_WinRate","Away_WinRate",
        "B365H","B365D","B365A"
    ]].fillna(0))
    # scikit ordena las columnas por alfabeto de clases: asume ['A','D','H'] o ['D','H','A'] → resolvemos via classes_
    classes = list(model.classes_) if hasattr(model, "classes_") else ["H","D","A"]
    # mapear
    proba_map = {cls: probs[0, i] for i, cls in enumerate(classes)}
    # asegurar orden H,D,A
    pH, pD, pA = proba_map.get("H",0.0), proba_map.get("D",0.0), proba_map.get("A",0.0)
    return pH, pD, pA

def best_bet(pH,pD,pA, oddsH,oddsD,oddsA, threshold=0.40, kelly_frac=0.25, bankroll=100.0):
    # EV = p*(odds-1) - (1-p)
    cand = {
        "H": (pH*(oddsH-1) - (1-pH), oddsH),
        "D": (pD*(oddsD-1) - (1-pD), oddsD),
        "A": (pA*(oddsA-1) - (1-pA), oddsA),
    }
    # escoger la mayor probabilidad
    probs = {"H": pH, "D": pD, "A": pA}
    pick = max(probs, key=probs.get)
    p_pick = probs[pick]
    ev, odd = cand[pick]

    # si no supera threshold, abstener
    if p_pick < threshold:
        return {"action": "abstener", "pick": pick, "stake": 0.0, "kelly": 0.0, "ev": ev}

    # Kelly fraccional
    k = max(0.0, (p_pick*odd - 1.0) / (odd - 1.0))
    stake = bankroll * k * kelly_frac
    return {"action": "apostar", "pick": pick, "stake": float(stake), "kelly": float(k), "ev": float(ev)}

# ----------------------------
# UI (Chat)
# ----------------------------
def main():
    st.set_page_config(page_title="BetMind – Chat Premier League", page_icon="⚽", layout="centered")
    st.title("🤖 BetMind · Chat de IA para apuestas (Premier League)")
    st.caption("Basado en tus modelos calibrados y datasets locales · sin APIs externas")

    df_all, backtest, metrics, teams = load_data()
    models = load_models()
    lbs = league_baselines(df_all)

    with st.sidebar:
        st.header("⚙️ Configuración")
        model_name = st.selectbox("Modelo predictivo", ["xgb","rf","logreg"], index=0,
                                  help="Usa los modelos calibrados que generaste.")
        threshold = st.slider("Umbral mínimo de probabilidad para apostar", 0.30, 0.70, 0.40, 0.01)
        kelly_frac = st.slider("Fracción de Kelly (gestión de riesgo)", 0.05, 0.50, 0.25, 0.05)
        bankroll = st.number_input("Bankroll de referencia (unidades)", min_value=10.0, value=100.0, step=10.0)

        st.divider()
        st.subheader("🏆 Resumen de backtesting")
        best_row = backtest.loc[backtest["ROI"].idxmax()]
        st.write(f"**Mejor ROI:** {best_row['Model']} · {best_row['ROI']*100:.1f}% · Sharpe {best_row['Sharpe']:.3f}")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role":"assistant",
            "content":"Hola, soy tu asistente de apuestas de Premier League. Pregúntame por un partido, ej.: **¿Quién gana Liverpool vs Arsenal?**"})

    # Mostrar historial
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_text = st.chat_input("Escribe tu pregunta...")
    if user_text:
        st.session_state.messages.append({"role":"user","content":user_text})
        with st.chat_message("user"): st.markdown(user_text)

        # Intento: detectar equipos
        home, away = extract_teams(user_text, teams)

        if (home is None) or (away is None):
            reply = ("No pude identificar a ambos equipos en tu mensaje. "
                     "Escribe algo como: **Quién gana Liverpool vs Arsenal**.")
        else:
            # Cargar modelo
            if model_name not in models:
                reply = f"No encontré el modelo calibrado **{model_name}** en outputs/models."
            else:
                model = models[model_name]
                X = build_features_for_pair(df_all, home, away)
                pH,pD,pA = predict_match(model, X)

                oddsH,oddsD,oddsA = float(X.iloc[0]["B365H"]), float(X.iloc[0]["B365D"]), float(X.iloc[0]["B365A"])
                rec = best_bet(pH,pD,pA, oddsH,oddsD,oddsA, threshold, kelly_frac, bankroll)

                reply = (
                    f"**{home} vs {away}**  \n"
                    f"Probabilidades del modelo **{model_name.upper()}**:  \n"
                    f"- 🏠 Local (H): **{pH*100:.1f}%**  \n"
                    f"- 🤝 Empate (D): **{pD*100:.1f}%**  \n"
                    f"- 🧳 Visitante (A): **{pA*100:.1f}%**  \n\n"
                    f"Cuotas usadas (sim/mercado):  H {oddsH:.2f} · D {oddsD:.2f} · A {oddsA:.2f}  \n\n"
                )

                if rec["action"] == "abstener":
                    reply += (f"**Recomendación:** 🙅 **Abstenerse** (la mayor prob. no supera el umbral {int(threshold*100)}%).  \n"
                              f"EV estimado para la mejor opción (‘{rec['pick']}’): **{rec['ev']:.3f}**.")
                else:
                    pretty = {"H":"Local","D":"Empate","A":"Visitante"}[rec["pick"]]
                    reply += (f"**Recomendación:** ✅ **Apostar a {pretty}**  \n"
                              f"• *Kelly* = {rec['kelly']:.3f} → stake sugerido (fracc. {kelly_frac:.2f}): **{rec['stake']:.2f}** unidades  \n"
                              f"• EV estimado: **{rec['ev']:.3f}**  \n"
                              f"• Umbral aplicado: **{int(threshold*100)}%**")

        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role":"assistant","content":reply})

if __name__ == "__main__":
    main()
