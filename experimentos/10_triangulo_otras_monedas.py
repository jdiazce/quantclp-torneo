"""
EXP 10 — Prueba de robustez fuera del activo: el triángulo RA-SB-BC-OA,
CONGELADO (mismos parámetros, sin tocar nada), aplicado a otras monedas.

Si la lógica captura algo real del comportamiento del precio, debería funcionar
al menos en monedas parecidas al peso (emergentes). Si solo funciona en USD/CLP,
lo más probable es que sea suerte o un rasgo particular de ese mercado.

DETALLES
  · Solo el USD/CLP tiene OHLC en la base. Para el resto se reconstruye la vela
    con cierres: apertura = cierre anterior, máximo/mínimo = el mayor/menor de
    los dos. Para medir cuánto cambia el modelo con esa aproximación, se corre
    también el USD/CLP "solo cierres".
  · Período: 2023-01-01 – fin de la base (mismo rango del test + holdout).
  · Posición larga = comprar el ticker tal como se cotiza (USDBRL: comprar dólar).
  · Sin carry (la base no trae tasas overnight de cada país): resultado bruto
    menos costos. Costo por unidad de cambio de posición en puntos base.

Correr con F5 en Spyder. Requiere src/triangulos.py.
"""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(os.environ.get("QUANTCLP", Path(__file__).resolve().parents[1]))
importlib.invalidate_caches()
sys.path.insert(0, str(RAIZ / "src"))
import triangulos as tg  # noqa: E402

BASE = next(p for p in (RAIZ / "data" / "base_usdclp.xlsx",
                        RAIZ / "base_usdclp.xlsx") if p.exists())
OUT = RAIZ / "experimentos"
OUT.mkdir(parents=True, exist_ok=True)

CFG = {"ruptura": "A", "salida": "B", "base": "C"}       # RA-SB-BC-OA, congelado
DESDE = "2023-01-01"
HIST_DESDE = "2020-01-01"            # historia para que el detector tenga contexto
COSTO_BPS = 5.0                      # ≈ 0,5 CLP con el dólar a ~930
COSTOS_SENS_BPS = [0.0, 5.0, 10.0]
N_BOOT, BLOQUE = 5000, 10

MONEDAS = {"USDCLP (solo cierres)": "USDCLP Curncy", "USDBRL": "USDBRL Curncy",
           "USDMXN": "USDMXN Curncy", "USDCOP": "USDCOP Curncy",
           "USDPEN": "USDPEN Curncy", "USDZAR": "USDZAR Curncy",
           "USDCNH": "USDCNH Curncy", "USDJPY": "USDJPY Curncy",
           "EURUSD": "EURUSD Curncy", "DXY": "DXY Curncy"}


def ohlc_desde_cierres(c):
    c = c.dropna()
    c = c[c.index.dayofweek < 5]
    c = c[c.diff() != 0]                       # feriados con precio repetido
    o = c.shift(1)
    df = pd.DataFrame({"o": o, "h": np.maximum(o, c), "l": np.minimum(o, c), "c": c})
    return df.dropna()


def p_boot(x, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    nuevo = rng.random((N_BOOT, n)) < 1 / BLOQUE
    nuevo[:, 0] = True
    t = np.arange(n)
    ult = np.maximum.accumulate(np.where(nuevo, t, 0), axis=1)
    ii = (np.take_along_axis(rng.integers(0, n, (N_BOOT, n)), ult, axis=1) + t - ult) % n
    return float(np.mean((x - x.mean())[ii].mean(axis=1) >= x.mean()))


def evaluar(df, nombre):
    df = df.loc[HIST_DESDE:]
    det = tg.Detector(df)
    F = df.index
    idx = np.where(F >= pd.Timestamp(DESDE))[0][:-1]
    pos = np.array([tg.senal(det, det.detectar(k), k, CFG)[0] for k in idx], float)
    ret = det.C[idx + 1] / det.C[idx] - 1
    fe = F[idx]

    def neto(p, bps=COSTO_BPS):
        dp = np.abs(np.diff(p, prepend=0.0))
        return p * ret - dp * bps / 1e4

    nt, uno = neto(pos), neto(np.ones(len(idx)))
    mens = pd.Series(nt, index=fe).resample("ME").sum()
    sd = nt.std()
    eq = np.cumsum(nt)
    fila = {"días": len(idx),
            "activo_%": (det.C[idx[-1] + 1] / det.C[idx[0]] - 1) * 100,
            "bruto_%": (pos * ret).sum() * 100,
            "neto_%": nt.sum() * 100,
            "sharpe": nt.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
            "maxDD_%": (eq - np.maximum.accumulate(eq)).min() * 100,
            "%días_pos": (pos != 0).mean() * 100,
            "largo_bruto_%": (pos * ret)[pos > 0].sum() * 100,
            "corto_bruto_%": (pos * ret)[pos < 0].sum() * 100,
            "%meses>0": (mens > 0).mean() * 100,
            "+1_neto_%": uno.sum() * 100,
            "p(neto≤0)": p_boot(nt),
            "p(≤ +1)": p_boot(nt - uno)}
    for b in COSTOS_SENS_BPS:
        fila[f"neto_{b:.0f}bps_%"] = neto(pos, b).sum() * 100
    print(f"  {nombre:<24} listo")
    return fila


datos = pd.read_excel(BASE, sheet_name="datos")
datos["fecha"] = pd.to_datetime(datos["fecha"])
datos = datos.set_index("fecha").sort_index()

print(f"Triángulo RA-SB-BC-OA congelado · {DESDE} – fin de la base · costo {COSTO_BPS} pb\n")
# caché por moneda: si se corta, retoma donde quedó
CACHE = RAIZ / "data" / "cache" / "exp10_filas.pkl"
CACHE.parent.mkdir(parents=True, exist_ok=True)
res = pd.read_pickle(CACHE) if CACHE.exists() else {}
fuentes = {"USDCLP (OHLC real)": tg.cargar_usdclp(BASE)[["o", "h", "l", "c"]]}
fuentes |= {n: ohlc_desde_cierres(datos[f"{tk}|PX_LAST"]) for n, tk in MONEDAS.items()}
for nombre, df in fuentes.items():
    if nombre not in res:
        res[nombre] = evaluar(df, nombre)
        pd.to_pickle(res, CACHE)

tab = pd.DataFrame(res).T
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
cols = ["activo_%", "bruto_%", "neto_%", "sharpe", "maxDD_%", "%días_pos",
        "largo_bruto_%", "corto_bruto_%", "%meses>0", "+1_neto_%", "p(neto≤0)", "p(≤ +1)"]
print("\nRESULTADOS (sin carry)")
print(tab[cols].to_string(float_format=lambda x: f"{x:.2f}"))

otras = tab.drop(index=["USDCLP (OHLC real)", "USDCLP (solo cierres)"])
print(f"\nResumen otras monedas ({len(otras)}): neto > 0 en {(otras['neto_%'] > 0).sum()} · "
      f"le gana a +1 en {(otras['neto_%'] > otras['+1_neto_%']).sum()} · "
      f"Sharpe promedio {otras['sharpe'].mean():.2f} · "
      f"p(neto≤0) < 0,05 en {(otras['p(neto≤0)'] < 0.05).sum()}")

print("\nSENSIBILIDAD A COSTOS · neto %")
print(tab[[f"neto_{b:.0f}bps_%" for b in COSTOS_SENS_BPS]].round(2).to_string())
tab.to_csv(OUT / "exp10_otras_monedas.csv")
