"""
Experimento 13 · Efecto compresión (descriptivo).

Pregunta: ¿qué anuncia una compresión del rango del USD/CLP?
  H1 (magnitud): después de una compresión, ¿el dólar se mueve más?
  H2 (dirección): cuando una compresión se resuelve con una ruptura,
                  ¿la dirección de la ruptura es más confiable?

No toca el modelo final ni sus parámetros: es evidencia para el informe.
Se corrió después de evaluar el holdout; el holdout se muestra solo como control.

Medidas (todas con datos hasta t):
  C_t  = (máximo de 20 días - mínimo de 20 días) / ATR de 100 días.
         Bajo = comprimido.
  ATR  = promedio simple del rango verdadero.
  Ruptura de rango: cierre sobre el máximo (bajo el mínimo) de los 20 días
         previos; se cuenta solo el primer día de cada ruptura.
  Comprimida: C_{t-1} en el 30% inferior de la ventana.

Salida: tablas en consola, experimentos/exp13_compresion.csv y 3 filas en
experimentos/registro_experimentos.csv.

Uso: F5 en Spyder o `python experimentos/13_compresion.py` desde la raíz del repo.
La raíz se toma de la variable de entorno QUANTCLP si existe.
"""
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(os.environ.get("QUANTCLP", Path(__file__).resolve().parents[1]))
BASE = RAIZ / "data" / "base_usdclp.xlsx"
REGISTRO = RAIZ / "experimentos" / "registro_experimentos.csv"
SALIDA = RAIZ / "experimentos" / "exp13_compresion.csv"
if not BASE.exists():
    raise FileNotFoundError(f"No encuentro la base en {BASE}. Debe estar en data/ dentro del repo.")

VENTANAS = {
    "2005-2018": ("2005-01-01", "2018-12-31"),
    "Seleccion": ("2023-01-01", "2025-04-04"),
    "Holdout (control)": ("2025-04-07", "2026-09-21"),  # deja 10 días para el retorno futuro
}
H = 10           # días hacia adelante para magnitud
N_RANGO = 20     # días del rango
N_ATR_LARGO = 100
Q_COMP = 0.30    # umbral de compresión

# ---------------------------------------------------------------- datos
d = pd.read_excel(BASE, sheet_name="datos", index_col=0)
d.index = pd.to_datetime(d.index)
cols = {"USDCLP Curncy|PX_OPEN": "o", "USDCLP Curncy|PX_HIGH": "h",
        "USDCLP Curncy|PX_LOW": "l", "USDCLP Curncy|PX_LAST": "c"}
px = d[list(cols)].rename(columns=cols).dropna()
px = px[px.index.dayofweek < 5]
repetido = (px["h"] == px["l"]) & (px["c"] == px["c"].shift())
px = px[~repetido].loc["2004-01-01":]

prev = px["c"].shift()
tr = pd.concat([px["h"] - px["l"], (px["h"] - prev).abs(),
                (px["l"] - prev).abs()], axis=1).max(axis=1)
atr_largo = tr.rolling(N_ATR_LARGO).mean()
comp = (px["h"].rolling(N_RANGO).max() - px["l"].rolling(N_RANGO).min()) / atr_largo

# magnitud futura: ATR de los próximos H días relativo al ATR largo de hoy
atr_fut = tr[::-1].rolling(H).mean()[::-1].shift(-1)
mag = pd.DataFrame({
    "comp": comp,
    "atr_fut": atr_fut / atr_largo,
    "mov_abs_pb": (px["c"].shift(-H) / px["c"] - 1).abs() * 1e4,
}).dropna()

# rupturas de rango
techo = px["h"].rolling(N_RANGO).max().shift(1)
piso = px["l"].rolling(N_RANGO).min().shift(1)
lado = pd.Series(0, index=px.index)
lado[px["c"] > techo] = 1
lado[px["c"] < piso] = -1
primero = (lado != 0) & (lado.shift() != lado)
rup = pd.DataFrame({
    "lado": lado, "comp_prev": comp.shift(1),
    "r5": px["c"].shift(-5) / px["c"] - 1,
    "r10": px["c"].shift(-10) / px["c"] - 1,
})[primero].dropna()

# ---------------------------------------------------------------- análisis
filas, registro = [], []
for nombre, (a, b) in VENTANAS.items():
    m = mag.loc[a:b].copy()
    m["quintil"] = pd.qcut(m["comp"], 5, labels=["Q1 comprimido", "Q2", "Q3", "Q4", "Q5 amplio"])
    t1 = m.groupby("quintil", observed=True).agg(
        dias=("comp", "size"), compresion=("comp", "mean"),
        atr_futuro=("atr_fut", "mean"), mov_10d_pb=("mov_abs_pb", "mean"))
    rho = stats.spearmanr(m["comp"], m["atr_fut"]).statistic

    r = rup.loc[a:b].copy()
    umbral = m["comp"].quantile(Q_COMP)
    r["grupo"] = np.where(r["comp_prev"] <= umbral, "comprimida", "resto")
    r["acierto5"] = np.sign(r["r5"]) == r["lado"]
    r["acierto10"] = np.sign(r["r10"]) == r["lado"]
    r["ret10_pb"] = r["lado"] * r["r10"] * 1e4
    t2 = r.groupby("grupo").agg(rupturas=("lado", "size"), acierto_5d=("acierto5", "mean"),
                                acierto_10d=("acierto10", "mean"), ret_10d_pb=("ret10_pb", "mean"))

    # diferencia de aciertos a 5 días (z de dos proporciones, unilateral)
    g1, g2 = r[r["grupo"] == "comprimida"], r[r["grupo"] == "resto"]
    n1, n2 = len(g1), len(g2)
    p1, p2 = g1["acierto5"].mean(), g2["acierto5"].mean()
    pp = (g1["acierto5"].sum() + g2["acierto5"].sum()) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2)) if n1 and n2 else np.nan
    p_val = 1 - stats.norm.cdf(z)

    print("=" * 78)
    print(f"{nombre}  ·  {a} a {b}")
    print("H1 · magnitud por quintil de compresión")
    print(t1.round(2).to_string())
    print(f"   Spearman(compresión, ATR futuro) = {rho:.3f}")
    print(f"H2 · rupturas de rango de {N_RANGO} días (umbral C = {umbral:.2f})")
    print(t2.round(3).to_string())
    print(f"   acierto 5d comprimida − resto = {p1 - p2:+.3f}  ·  z = {z:.2f}  ·  p = {p_val:.3f}")
    print("   (p sin corregir por solapamiento de rupturas cercanas)")

    t1 = t1.reset_index().assign(ventana=nombre, tabla="H1")
    t2 = t2.reset_index().assign(ventana=nombre, tabla="H2")
    filas += [t1, t2]
    registro.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experimento": f"13_compresion_{nombre.split()[0]}",
        "desde": a, "hasta": b, "elegida": False,
    })

pd.concat(filas, ignore_index=True).to_csv(SALIDA, index=False)
reg = pd.read_csv(REGISTRO)
reg = pd.concat([reg, pd.DataFrame(registro)], ignore_index=True)
reg.to_csv(REGISTRO, index=False)
print("=" * 78)
print(f"Guardado: {SALIDA} · {len(registro)} filas en {REGISTRO.name}")
