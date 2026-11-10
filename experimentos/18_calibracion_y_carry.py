"""
18_calibracion_y_carry.py — Tres chequeos del modelo congelado. No modifica nada.

  A. CALIBRACIÓN de P(sube) en el holdout: ¿la probabilidad que publica
     estrategia.py es mejor que la frecuencia histórica de alzas?
     Brier score, log loss, tabla de confiabilidad y test HAC sobre la
     diferencia de Brier (tipo Diebold-Mariano).
  B. CARRY ALTERNATIVO: neto del holdout con tres definiciones de carry, porque
     las bases dicen que el laboratorio usa su propia regla:
        1) TPM − Fed funds (la del informe)
        2) puntos NDF 1M (CHN1M), la vía transable del carry según el diccionario
        3) rendimiento implícito NDF 1M (CHNI1M) − Fed funds
  C. ATRASO DE EJECUCIÓN: neto del holdout con la posición aplicada 0 a 3 días
     tarde (para fijar la cifra exacta de la Tabla 18 del informe).

REQUISITO: correr antes estrategia.py (este script lee salidas/historial.csv).
Uso: F5 en Spyder.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------
RAIZ = Path(os.environ.get("QUANTCLP", Path(__file__).resolve().parents[1]))
BASE = RAIZ / "data" / "base_usdclp.xlsx"
HIST = RAIZ / "salidas" / "historial.csv"
SALIDA = RAIZ / "experimentos" / "exp18_calibracion_carry.csv"

SELECCION = ("2023-01-01", "2025-04-04")
HOLDOUT_DESDE = "2025-04-07"
COSTO_CLP = 0.50
POS_COMPRAR = 3.0
DIAS_NDF_1M = 30          # plazo del NDF 1M en días corridos (aprox.)

if not HIST.exists():
    raise FileNotFoundError(f"Falta {HIST}. Corre primero estrategia.py.")
sys.path.insert(0, str(RAIZ / "src"))
import triangulos as tg  # noqa: E402

# ----------------------------------------------------------------------
# DATOS: historial de posiciones + precios + tasas (mismo tratamiento que estrategia.py)
# ----------------------------------------------------------------------
clp = tg.cargar_usdclp(BASE)
F = clp.index
C = clp["c"].to_numpy(float)

h = pd.read_csv(HIST, parse_dates=["fecha"]).set_index("fecha")
kk = F.get_indexer(h.index)
if (kk < 0).any():
    raise ValueError("Hay fechas del historial que no están en la serie del USD/CLP.")
S0 = C[kk]
ret = C[kk + 1] / S0 - 1
if not np.allclose(ret, h["ret"].to_numpy(), atol=1e-12):
    raise ValueError("Los retornos no calzan con historial.csv: revisa que sea la misma base.")
dias = (F[kk + 1] - F[kk]).days.to_numpy()
pos = h["pos"].to_numpy(float)

COLS = {"CHOVCHOV Index|PX_LAST": "tpm", "FEDL01 Index|PX_LAST": "ff",
        "CHN1M Curncy|PX_LAST": "pts", "CHNI1M Curncy|PX_LAST": "imp"}
tasas = (pd.read_excel(BASE, sheet_name="datos", usecols=["fecha", *COLS])
           .rename(columns=COLS)
           .assign(fecha=lambda x: pd.to_datetime(x["fecha"]))
           .set_index("fecha").sort_index())
# Fed funds y NDF cierran después de Santiago: se usan con un día de rezago.
for c in ("ff", "pts", "imp"):
    tasas[c] = tasas[c].shift(1)
tasas = tasas.ffill().reindex(F, method="ffill").iloc[kk]

CARRY = {
    "TPM − Fed funds (informe)": -((tasas["tpm"] - tasas["ff"]).to_numpy() / 100 / 360) * dias,
    "Puntos NDF 1M": -(tasas["pts"].to_numpy() / S0) / DIAS_NDF_1M * dias,
    "NDF implícito 1M − Fed funds": -((tasas["imp"] - tasas["ff"]).to_numpy() / 100 / 360) * dias,
}

sel = (h.index >= pd.Timestamp(SELECCION[0])) & (h.index <= pd.Timestamp(SELECCION[1]))
ho = h.index >= pd.Timestamp(HOLDOUT_DESDE)


def neto(p, carry_u, costo=COSTO_CLP):
    dp = np.abs(np.diff(p, prepend=0.0))
    return p * ret + p * carry_u - dp * costo / S0


# Control: debe reproducir el +11,73 % del informe
control = neto(pos, CARRY["TPM − Fed funds (informe)"])[ho].sum() * 100
print(f"Control: neto holdout con la contabilidad del informe = {control:+.2f}% (informe: +11,73%)")

filas = []

# ----------------------------------------------------------------------
# A. CALIBRACIÓN
# ----------------------------------------------------------------------
y = (ret > 0).astype(float)
acierto = {n: float(np.mean(np.sign(pos[m]) == np.sign(ret[m])))
           for n, m in {1: sel & (np.abs(pos) == 1), 3: sel & (np.abs(pos) == 3)}.items()}
p_clim = float(y[sel].mean())                      # frecuencia de alzas en selección
P = np.where(pos == 0, p_clim,
             np.where(pos > 0, [acierto.get(abs(int(v)), p_clim) for v in pos],
                      [1 - acierto.get(abs(int(v)), 1 - p_clim) for v in pos]))


def brier(p):
    return (p - y) ** 2


def logloss(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def test_hac(d):
    """H1: media(d) > 0. Varianza Newey-West con rezago de Andrews simple."""
    d = np.asarray(d, float)
    n = len(d)
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    u = d - d.mean()
    v = u @ u / n
    for j in range(1, L + 1):
        v += 2 * (1 - j / (L + 1)) * (u[j:] @ u[:-j]) / n
    t = d.mean() / np.sqrt(v / n) if v > 0 else np.nan
    return t, 1 - norm.cdf(t)


print("\n" + "=" * 78)
print("A. CALIBRACIÓN DE P(sube) · holdout")
print("=" * 78)
print(f"Probabilidades fijadas en selección: ±1 → {acierto[1]:.3f} · ±3 → {acierto[3]:.3f} · "
      f"0 → {p_clim:.3f} (frecuencia de alzas)")

refs = {"Modelo (estrategia.py)": P, "Frecuencia de alzas (constante)": np.full(len(y), p_clim),
        "Moneda (0,5)": np.full(len(y), 0.5)}
tabA = pd.DataFrame({n: {"Brier": brier(p)[ho].mean(), "Log loss": logloss(p)[ho].mean()}
                     for n, p in refs.items()}).T
tabA["Brier skill vs. frecuencia %"] = (1 - tabA["Brier"] / tabA.loc["Frecuencia de alzas (constante)", "Brier"]) * 100
print(tabA.round(5).to_string())

for nombre, p_ref in [("frecuencia de alzas", refs["Frecuencia de alzas (constante)"]),
                      ("moneda 0,5", refs["Moneda (0,5)"])]:
    t, pv = test_hac((brier(p_ref) - brier(P))[ho])
    print(f"Test HAC, Brier del modelo < Brier de {nombre}: t = {t:.2f}, p = {pv:.3f}")
    filas.append({"bloque": "calibracion", "prueba": f"Brier vs {nombre}", "valor": t, "p": pv})

print("\nConfiabilidad (holdout): probabilidad publicada vs. frecuencia observada de alzas")
conf = (pd.DataFrame({"P publicada": P, "sube": y, "pos": pos})[ho]
          .assign(grupo=lambda x: np.select([x["pos"] == 3, x["pos"] == 1, x["pos"] == 0,
                                             x["pos"] == -1, x["pos"] == -3],
                                            ["+3", "+1", "0", "−1", "−3"], "?"))
          .groupby("grupo")
          .agg(dias=("sube", "size"), P_publicada=("P publicada", "mean"),
               frec_observada=("sube", "mean")))
conf = conf.reindex([g for g in ["+3", "+1", "0", "−1", "−3"] if g in conf.index])
print(conf.round(3).to_string())

for n, r in tabA.iterrows():
    filas.append({"bloque": "calibracion", "prueba": f"Brier {n}", "valor": r["Brier"], "p": np.nan})

# ----------------------------------------------------------------------
# B. CARRY ALTERNATIVO
# ----------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"B. NETO DEL HOLDOUT SEGÚN LA DEFINICIÓN DE CARRY (costo {COSTO_CLP} CLP)")
print("=" * 78)
ESTR = {"Triángulos": pos, "Comprar siempre (+3)": np.full(len(pos), POS_COMPRAR),
        "Siempre +1": np.ones(len(pos)), "No operar": np.zeros(len(pos))}
tabB = pd.DataFrame({cn: {en: neto(ep, cu)[ho].sum() * 100 for en, ep in ESTR.items()}
                     for cn, cu in CARRY.items()})
dif = {cn: (tasas["tpm"] - tasas["ff"]).to_numpy()[ho].mean() if cn.startswith("TPM") else
       (tasas["pts"].to_numpy() / S0 * 360 / DIAS_NDF_1M * 100)[ho].mean() if cn.startswith("Puntos") else
       (tasas["imp"] - tasas["ff"]).to_numpy()[ho].mean() for cn in CARRY}
tabB.loc["Diferencial promedio (pp anual)"] = pd.Series(dif)
print(tabB.round(2).to_string())
for cn in CARRY:
    for en in ESTR:
        filas.append({"bloque": "carry", "prueba": f"{en} | {cn}", "valor": tabB.loc[en, cn], "p": np.nan})

# ----------------------------------------------------------------------
# C. ATRASO DE EJECUCIÓN
# ----------------------------------------------------------------------
print("\n" + "=" * 78)
print("C. NETO CON LA POSICIÓN APLICADA L DÍAS TARDE (carry TPM − Fed funds)")
print("=" * 78)
cu = CARRY["TPM − Fed funds (informe)"]
tabC = {}
for L in range(4):
    pL = np.concatenate([np.zeros(L), pos[:len(pos) - L]]) if L else pos
    tabC[f"{L} días"] = {"Selección %": neto(pL, cu)[sel].sum() * 100,
                         "Holdout %": neto(pL, cu)[ho].sum() * 100}
    filas.append({"bloque": "atraso", "prueba": f"{L} dias holdout",
                  "valor": tabC[f'{L} días']["Holdout %"], "p": np.nan})
print(pd.DataFrame(tabC).round(2).to_string())

pd.DataFrame(filas).to_csv(SALIDA, index=False)
print(f"\nResultados guardados en {SALIDA}")
