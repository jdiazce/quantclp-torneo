"""
EXP 04b — Mini pausa: triángulo RA-SB-BC-OA (el "ganador" del exp 04) en la
PARTE 1 del holdout (inicio del holdout – 31 dic 2025), sin tocar nada.

  · La parte 2 del holdout (2026) sigue bloqueada: no se carga.
  · Esta mirada queda en el registro. Después de verla, la parte 1 deja de ser
    100% limpia para cualquier decisión sobre triángulos.
  · Carry con la definición de las bases: TPM − Fed funds (rezago 1).

Correr con F5 en Spyder. Requiere src/triangulos.py.
"""

import importlib
import os
import sys
from datetime import datetime
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
HOLDOUT_MESES = 18
HOLDOUT_2_DESDE = pd.Timestamp("2026-01-01")
CFG = {"ruptura": "A", "salida": "B", "base": "C"}
COSTO = 0.50
COSTOS_SENS = [0.0, 0.25, 0.50, 1.00, 2.00]
N_BOOT, BLOQUE = 5000, 10

clp = tg.cargar_usdclp(BASE)
HOLDOUT_DESDE = clp.index.max() - pd.DateOffset(months=HOLDOUT_MESES)
clp = clp[clp.index < HOLDOUT_2_DESDE]                 # 2026 nunca entra
det = tg.Detector(clp)
F = clp.index

ex = (pd.read_excel(BASE, sheet_name="datos",
                    usecols=["fecha", "CHOVCHOV Index|PX_LAST", "FEDL01 Index|PX_LAST"])
        .set_axis(["fecha", "tpm", "ff"], axis=1)
        .assign(fecha=lambda x: pd.to_datetime(x["fecha"]))
        .set_index("fecha").sort_index())
ex = ex[ex.index < HOLDOUT_2_DESDE]
ex["ff"] = ex["ff"].shift(1)
ex = ex.ffill().reindex(F, method="ffill")

idx = np.where(F >= HOLDOUT_DESDE)[0][:-1]
print(f"Holdout parte 1: {F[idx[0]].date()} – {F[idx[-1] + 1].date()} ({len(idx)} días)")
print("Detectando triángulos...")
pos = np.array([tg.senal(det, det.detectar(k), k, CFG)[0] for k in idx], float)

S0, S1 = det.C[idx], det.C[idx + 1]
ret = S1 / S0 - 1
dias = (F[idx + 1] - F[idx]).days.to_numpy()
carry_u = -((ex["tpm"] - ex["ff"]).to_numpy()[idx] / 100 / 360) * dias
fechas = F[idx]


def metricas(p, costo=COSTO):
    dp = np.abs(np.diff(p, prepend=0.0))
    br, ca, cs = p * ret, p * carry_u, -dp * costo / S0
    nt = br + ca + cs
    mens = pd.Series(nt, index=fechas).resample("ME").sum()
    eq = np.cumsum(nt)
    sd = nt.std()
    return {"bruto_%": br.sum() * 100, "carry_%": ca.sum() * 100,
            "costos_%": cs.sum() * 100, "neto_%": nt.sum() * 100,
            "sharpe": nt.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
            "maxDD_%": (eq - np.maximum.accumulate(eq)).min() * 100,
            "rotación": dp.sum(), "%días_pos": (p != 0).mean() * 100,
            "%días_±3": (np.abs(p) == 3).mean() * 100,
            "%meses>0": (mens > 0).mean() * 100}, nt


def p_boot(x, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    nuevo = rng.random((N_BOOT, n)) < 1 / BLOQUE
    nuevo[:, 0] = True
    t = np.arange(n)
    ult = np.maximum.accumulate(np.where(nuevo, t, 0), axis=1)
    ii = (np.take_along_axis(rng.integers(0, n, (N_BOOT, n)), ult, axis=1) + t - ult) % n
    return float(np.mean((x - x.mean())[ii].mean(axis=1) >= x.mean()))


estr = {"Triángulo RA-SB-BC-OA": pos, "Siempre +1": np.ones(len(idx)),
        "Siempre +3": np.full(len(idx), 3.0), "No operar": np.zeros(len(idx))}
tab = pd.DataFrame({n: metricas(p)[0] for n, p in estr.items()}).T
pd.set_option("display.width", 200)
print("\nHOLDOUT PARTE 1 · costo 0,5 CLP · carry TPM − Fed funds")
print(tab.round(2).to_string())

nt = metricas(pos)[1]
ref1 = metricas(np.ones(len(idx)))[1]
print(f"\nBootstrap: p(neto ≤ 0) = {p_boot(nt):.3f} · p(no le gana a +1) = {p_boot(nt - ref1):.3f}")

sens = pd.Series({f"{c:.2f}": metricas(pos, c)[0]["neto_%"] for c in COSTOS_SENS})
print("\nSensibilidad a costos (neto %):", sens.round(2).to_dict())

mens = pd.DataFrame({n: pd.Series(metricas(p)[1], index=fechas).resample("ME").sum() * 100
                     for n, p in estr.items() if n != "No operar"})
print("\nNeto mensual (%)")
print(mens.round(2).to_string())

reg_path = OUT / "registro_experimentos.csv"
fila = pd.DataFrame([{"timestamp": datetime.now().isoformat(timespec="seconds"),
                      "experimento": "04b_triangulo_RA-SB-BC-OA_holdout1",
                      "desde": str(F[idx[0]].date()), "hasta": str(F[idx[-1] + 1].date()),
                      "costo": COSTO, "neto_%": round(tab.loc["Triángulo RA-SB-BC-OA", "neto_%"], 3),
                      "sharpe": round(tab.loc["Triángulo RA-SB-BC-OA", "sharpe"], 3),
                      "pct_meses_pos": round(tab.loc["Triángulo RA-SB-BC-OA", "%meses>0"], 1),
                      "elegida": True}])
if reg_path.exists():
    fila = pd.concat([pd.read_csv(reg_path), fila], ignore_index=True)
fila.to_csv(reg_path, index=False)
print(f"\nRegistro: {len(fila)} intentos acumulados (holdout parte 1 abierto una vez).")
