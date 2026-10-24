"""
EXP 04c — EVALUACIÓN FINAL del triángulo RA-SB-BC-OA (congelado) en el
holdout: parte 1 (abr–dic 2025), parte 2 (2026) y holdout completo.

  · Desde aquí el triángulo NO se modifica: cualquier cambio invalida este
    resultado. Lo único limpio que queda después es el monitoreo en vivo.
  · Incluye la descomposición largo/corto para saber de dónde sale el resultado.
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
TRAMOS = {"Parte 1 (abr–dic 2025)": (None, HOLDOUT_2_DESDE),
          "Parte 2 (2026)": (HOLDOUT_2_DESDE, None),
          "Holdout completo": (None, None)}
CFG = {"ruptura": "A", "salida": "B", "base": "C"}
COSTO = 0.50
COSTOS_SENS = [0.0, 0.25, 0.50, 1.00, 2.00]
N_BOOT, BLOQUE = 5000, 10

clp = tg.cargar_usdclp(BASE)
HOLDOUT_DESDE = clp.index.max() - pd.DateOffset(months=HOLDOUT_MESES)
det = tg.Detector(clp)
F = clp.index

ex = (pd.read_excel(BASE, sheet_name="datos",
                    usecols=["fecha", "CHOVCHOV Index|PX_LAST", "FEDL01 Index|PX_LAST"])
        .set_axis(["fecha", "tpm", "ff"], axis=1)
        .assign(fecha=lambda x: pd.to_datetime(x["fecha"]))
        .set_index("fecha").sort_index())
ex["ff"] = ex["ff"].shift(1)
ex = ex.ffill().reindex(F, method="ffill")

idx = np.where(F >= HOLDOUT_DESDE)[0][:-1]
print(f"Holdout: {F[idx[0]].date()} – {F[idx[-1] + 1].date()} ({len(idx)} días)")
print("Detectando triángulos...")
pos = np.array([tg.senal(det, det.detectar(k), k, CFG)[0] for k in idx], float)

S0_all, S1_all = det.C[idx], det.C[idx + 1]
ret_all = S1_all / S0_all - 1
dias_all = (F[idx + 1] - F[idx]).days.to_numpy()
carry_all = -((ex["tpm"] - ex["ff"]).to_numpy()[idx] / 100 / 360) * dias_all
fechas_all = F[idx]


def p_boot(x, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    nuevo = rng.random((N_BOOT, n)) < 1 / BLOQUE
    nuevo[:, 0] = True
    t = np.arange(n)
    ult = np.maximum.accumulate(np.where(nuevo, t, 0), axis=1)
    ii = (np.take_along_axis(rng.integers(0, n, (N_BOOT, n)), ult, axis=1) + t - ult) % n
    return float(np.mean((x - x.mean())[ii].mean(axis=1) >= x.mean()))


def metricas(p, m, costo=COSTO):
    p = p[m]
    S0, ret, ca_u, fe = S0_all[m], ret_all[m], carry_all[m], fechas_all[m]
    dp = np.abs(np.diff(p, prepend=0.0))
    br, ca, cs = p * ret, p * ca_u, -dp * costo / S0
    nt = br + ca + cs
    mens = pd.Series(nt, index=fe).resample("ME").sum()
    eq = np.cumsum(nt)
    sd = nt.std()
    return {"bruto_%": br.sum() * 100, "carry_%": ca.sum() * 100,
            "costos_%": cs.sum() * 100, "neto_%": nt.sum() * 100,
            "sharpe": nt.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
            "maxDD_%": (eq - np.maximum.accumulate(eq)).min() * 100,
            "rotación": dp.sum(), "%días_pos": (p != 0).mean() * 100,
            "%días_±3": (np.abs(p) == 3).mean() * 100,
            "%meses>0": (mens > 0).mean() * 100}, nt


pd.set_option("display.width", 200)
estr = {"Triángulo RA-SB-BC-OA": pos, "Siempre +1": np.ones(len(idx)),
        "Siempre +3": np.full(len(idx), 3.0), "No operar": np.zeros(len(idx))}
reg_filas = []
for nombre, (a_, b_) in TRAMOS.items():
    m = np.ones(len(idx), bool)
    if a_ is not None:
        m &= fechas_all >= a_
    if b_ is not None:
        m &= fechas_all < b_
    tab = pd.DataFrame({n: metricas(p, m)[0] for n, p in estr.items()}).T
    nt = metricas(pos, m)[1]
    ref1 = metricas(np.ones(len(idx)), m)[1]
    print("\n" + "=" * 110)
    print(f"{nombre} · {fechas_all[m][0].date()} – {fechas_all[m][-1].date()} "
          f"({m.sum()} días) · costo 0,5 CLP · carry TPM − Fed funds")
    print("=" * 110)
    print(tab.round(2).to_string())
    print(f"Bootstrap: p(neto ≤ 0) = {p_boot(nt):.3f} · p(no le gana a +1) = {p_boot(nt - ref1):.3f}")
    sens = {f"{c:.2f}": round(metricas(pos, m, c)[0]["neto_%"], 2) for c in COSTOS_SENS}
    print("Sensibilidad a costos (neto %):", sens)
    pm = pos[m]
    bruto = pm * ret_all[m]
    print(f"Variación del dólar en el tramo: {(S1_all[m][-1] / S0_all[m][0] - 1) * 100:+.1f}% · "
          f"bruto largo {bruto[pm > 0].sum() * 100:+.2f}% ({(pm > 0).sum()} días) · "
          f"bruto corto {bruto[pm < 0].sum() * 100:+.2f}% ({(pm < 0).sum()} días)")
    reg_filas.append({"timestamp": datetime.now().isoformat(timespec="seconds"),
                      "experimento": f"04c_triangulo_final_{nombre}",
                      "desde": str(fechas_all[m][0].date()), "hasta": str(fechas_all[m][-1].date()),
                      "costo": COSTO, "neto_%": round(tab.loc["Triángulo RA-SB-BC-OA", "neto_%"], 3),
                      "sharpe": round(tab.loc["Triángulo RA-SB-BC-OA", "sharpe"], 3),
                      "pct_meses_pos": round(tab.loc["Triángulo RA-SB-BC-OA", "%meses>0"], 1),
                      "elegida": True})

mens = pd.DataFrame({n: pd.Series(metricas(p, np.ones(len(idx), bool))[1], index=fechas_all)
                     .resample("ME").sum() * 100 for n, p in estr.items() if n != "No operar"})
print("\nNeto mensual, holdout completo (%)")
print(mens.round(2).to_string())

reg_path = OUT / "registro_experimentos.csv"
reg = pd.DataFrame(reg_filas)
if reg_path.exists():
    reg = pd.concat([pd.read_csv(reg_path), reg], ignore_index=True)
reg.to_csv(reg_path, index=False)
print(f"\nRegistro: {len(reg)} intentos acumulados. Holdout completo abierto: triángulo congelado.")
