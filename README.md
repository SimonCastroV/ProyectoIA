# ProyectoIA — PremierAI (UI simple con XGBoost)

Predicción de **ganador Local/Visitante** para la Premier League usando un **modelo XGBoost** ya entrenado.  
Interfaz con **Streamlit**: 2 desplegables (Local y Visitante) + botón **“Predecir ganador”**.  
Sin dependencias de API: la UI usa tu **dataset histórico hasta 2021** y calcula **probabilidades H/D/A**, **cuotas justas (1/p)** y muestra el **veredicto (solo Local o Visitante)**.

---

## Requisitos

- **Python 3.10+** (recomendado 3.10–3.12)
- Paquetes (en `requirements.txt`):
  ```txt
  streamlit
  pandas
  numpy
  scikit-learn
  xgboost
  joblib
  ```

---

## Instalación y ejecución

```bash
# 1) Clonar
git clone <tu-repo.git> ProyectoIA
cd ProyectoIA

# 2) Crear entorno
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

# 3) Instalar dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4) Ejecutar la UI
streamlit run src/ui/app.py
```

Abre el navegador si no se abre solo (Streamlit mostrará la URL local).

---

## Uso

1. Selecciona **Equipo Local** y **Equipo Visitante** desde los desplegables (datos tomados de `data/raw/EPL_Set.csv`).  
2. Pulsa **“Predecir ganador”**.  
3. Verás:  
   - **Probabilidades** H/D/A (alineadas a las clases reales del modelo)  
   - **Cuotas justas** (1/p)  
   - **Cuotas usadas (sim/mercado)**: del último H2H si existe; si no, se usa un **baseline de liga**  
   - **Veredicto**: “Gana: Local” o “Gana: Visitante”

---

## Datos

- **Equipos**: provienen de `data/raw/EPL_Set.csv` (histórico hasta 2021).  
- **Features**: `data/processed/EPL_Set_features.csv` y `EPL_Recent_features.csv`.  
- Si falta un **cara-a-cara** con cuotas `B365H/B365D/B365A`, la UI aplica **baseline** de liga con `vig` ≈ 5%.


---

## Modelos

- La UI intentará cargar **`outputs/models/xgb_calibrated.joblib`**.  
- Si no existe, usará **`outputs/models/xgb_model.joblib`**.  
- Asegúrate de que el modelo sea **multiclase (H/D/A)**.  
- La UI detecta automáticamente si `classes_` son números (`0/1/2`) y **mapea** a `H/D/A` por co‑ocurrencias con la columna `FTR`.


---


## Troubleshooting

**1) `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa0 ...`**  
Tus CSV no están en UTF‑8. La UI ya usa un **lector robusto** que prueba `utf-8`, `utf-8-sig`, `cp1252` y `latin1`, y limpia `\xa0`. No necesitas tocar nada.

**2) `UnhashableParamError ... in 'infer_label_mapping'`**  
Ocurre si se cachea una función con el objeto `model`. En la UI actual **no** se cachea esa función.  
Si lo cambiaste, evita decorarla con `@st.cache_data` **o** renombra el arg a `_model`.

**3) Probabilidades 0% en todo**  
Generalmente es un **desalineamiento de clases** (p.ej. `classes_ = [0,1,2]`).  
La UI ya **infere el mapeo** hacia `H/D/A`. Abre el expander **“Detalles técnicos (debug)”** para ver `classes_` y `predict_proba` crudo.

**4) Falta alguna columna**  
La función `ensure_features` rellena columnas faltantes con media/0 y ordena según `feature_names_in_` si está disponible.  
Si tu modelo fue entrenado con columnas muy distintas, alinéalo o vuelve a entrenar.

**5) Limpiar caché**  
En el menú de Streamlit usa **“Clear cache”** y luego **“Rerun”**, o en código:
```python
st.cache_data.clear()
st.cache_resource.clear()
```


---

## Git — nueva rama y push

```bash
# crear rama desde main
git fetch origin
git checkout main && git pull --ff-only
git checkout -b feature/xgb-simple-ui

# commit de la UI nueva y cambios
git add src/ui/app.py
git commit -m "UI: modo predicción simple con XGBoost"

# subir rama
git push -u origin feature/xgb-simple-ui
```

