"""
EXP 04d — Mini análisis: triángulo RA-SB-BC-OA (congelado) en meses alcistas
vs bajistas del dólar. Descriptivo, no cambia nada del modelo.

Tramos: 2023 – mar 2025 (usado para elegir, contaminado) y holdout (abr 2025 –
oct 2026). Mes alcista = el USD/CLP subió en el mes (cierre a cierre).
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
CFG = {"ruptura": "A", "salida": "B", "base": "C"}
COSTO = 0.50

clp = tg.cargar_usdclp(BASE)
HOLDOUT_DESDE = clp.index.max() - pd.DateOffset(months=18)
det = tg.Detector(clp)
F = clp.index
ex = (pd.read_excel(BASE, sheet_name="datos",
                    usecols=["fecha", "CHOVCHOV Index|PX_LAST", "FEDL01 Index|PX_LAST"])
        .set_axis(["fecha", "tpm", "ff"], axis=1)
        .assign(fecha=lambda x: pd.to_datetime(x["fecha"]))
        .set_index("fecha").sort_index())
ex["ff"] = ex["ff"].shift(1)
ex = ex.ffill().reindex(F, method="ffill")

idx = np.where(F >= pd.Timestamp("2023-01-01"))[0][:-1]
print(f"Calculando señales para {len(idx)} días...")
pos = np.array([tg.senal(det, det.detectar(k), k, CFG)[0] for k in idx], float)

S0, S1 = det.C[idx], det.C[idx + 1]
ret = S1 / S0 - 1
dias = (F[idx + 1] - F[idx]).days.to_numpy()
carry_u = -((ex["tpm"] - ex["ff"]).to_numpy()[idx] / 100 / 360) * dias
fe = F[idx]


def neto(p):
    dp = np.abs(np.diff(p, prepend=0.0))
    return p * ret + p * carry_u - dp * COSTO / S0


d = pd.DataFrame({"tri": neto(pos), "uno": neto(np.ones(len(idx))),
                  "dolar": np.log1p(ret), "largo": pos > 0, "corto": pos < 0,
                  "fuera": pos == 0}, index=fe)
m = d.resample("ME").agg({"tri": "sum", "uno": "sum", "dolar": "sum",
                          "largo": "mean", "corto": "mean", "fuera": "mean"})
m["dolar"] = np.expm1(m["dolar"])
m["tramo"] = np.where(m.index >= HOLDOUT_DESDE, "Holdout", "2023–mar 2025")
m["mes"] = np.where(m["dolar"] > 0, "Alcista", "Bajista")

g = m.groupby(["tramo", "mes"], sort=False)
tab = pd.DataFrame({
    "meses": g.size(),
    "dólar prom %": g["dolar"].mean() * 100,
    "tri neto total %": g["tri"].sum() * 100,
    "tri neto prom %": g["tri"].mean() * 100,
    "tri % meses>0": g["tri"].apply(lambda x: (x > 0).mean() * 100),
    "+1 neto total %": g["uno"].sum() * 100,
    "% días largo": g["largo"].mean() * 100,
    "% días corto": g["corto"].mean() * 100,
    "% días fuera": g["fuera"].mean() * 100})
pd.set_option("display.width", 200)
print("\nTRIÁNGULO RA-SB-BC-OA · meses alcistas vs bajistas del dólar · costo 0,5 CLP")
print(tab.round(2).to_string())

# captura: qué fracción de la subida/bajada capta
print("\nDetalle mensual")
print((m[["tramo", "mes", "dolar", "tri", "uno"]]
       .assign(dolar=lambda x: x["dolar"] * 100, tri=lambda x: x["tri"] * 100,
               uno=lambda x: x["uno"] * 100)
       .round(2).to_string()))
