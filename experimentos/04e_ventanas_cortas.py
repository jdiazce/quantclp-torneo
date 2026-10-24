"""
EXP 04e — Triángulo RA-SB-BC-OA (congelado) en ventanas cortas, como el
monitoreo en vivo (13 nov – 1 dic ≈ 13 días hábiles).

Para cada día de inicio posible se mide el resultado de los N días hábiles
siguientes (ventanas móviles) y se compara contra siempre +1, siempre +3 y no
operar. Ventanas: 13 días (el monitoreo) y 21 días (un mes).
Además: tasa de acierto en los días con posición.

Tramos: 2023 – mar 2025 (usado para elegir) y holdout (abr 2025 – oct 2026).
Costo 0,5 CLP, carry TPM − Fed funds.
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
VENTANAS = [13, 21]

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

S0 = det.C[idx]
ret = det.C[idx + 1] / S0 - 1
dias = (F[idx + 1] - F[idx]).days.to_numpy()
carry_u = -((ex["tpm"] - ex["ff"]).to_numpy()[idx] / 100 / 360) * dias
fe = F[idx]


def neto_diario(p):
    dp = np.abs(np.diff(p, prepend=0.0))
    return p * ret + p * carry_u - dp * COSTO / S0


nets = {"Triángulo": neto_diario(pos), "Siempre +1": neto_diario(np.ones(len(idx))),
        "Siempre +3": neto_diario(np.full(len(idx), 3.0))}
TRAMOS = {"2023 – mar 2025": fe < HOLDOUT_DESDE, "Holdout abr 2025 – oct 2026": fe >= HOLDOUT_DESDE,
          "Todo 2023 – 2026": np.ones(len(idx), bool)}

# acierto en días con posición
print("\nACIERTO EN DÍAS CON POSICIÓN (signo de la posición = signo del movimiento)")
for t, m in TRAMOS.items():
    act = m & (pos != 0)
    ac = np.mean(np.sign(pos[act]) == np.sign(ret[act])) * 100
    print(f"  {t:<28} {ac:5.1f}%  ({act.sum()} días con posición de {m.sum()})")

pd.set_option("display.width", 200)
for N in VENTANAS:
    filas = {}
    for t, m in TRAMOS.items():
        ii = np.where(m)[0]
        inicios = ii[: len(ii) - N + 1]
        if len(inicios) == 0:
            continue
        sumar = lambda x: np.array([x[i:i + N].sum() for i in inicios]) * 100
        tri, uno, tres = sumar(nets["Triángulo"]), sumar(nets["Siempre +1"]), sumar(nets["Siempre +3"])
        activo = np.array([(pos[i:i + N] != 0).any() for i in inicios])
        filas[t] = {"ventanas": len(inicios),
                    "% tri > 0": (tri > 0).mean() * 100,
                    "% tri = 0 (sin operar)": (~activo).mean() * 100,
                    "% tri < 0": ((tri < 0) & activo).mean() * 100,
                    "% le gana a +1": (tri > uno).mean() * 100,
                    "% le gana a +3": (tri > tres).mean() * 100,
                    "% le gana a no operar": (tri > 0).mean() * 100,
                    "tri mediana %": np.median(tri),
                    "tri peor %": tri.min(), "tri mejor %": tri.max(),
                    "+1 mediana %": np.median(uno), "+1 peor %": uno.min()}
    print(f"\nVENTANAS MÓVILES DE {N} DÍAS HÁBILES · costo {COSTO} CLP")
    print(pd.DataFrame(filas).T.round(2).to_string())
