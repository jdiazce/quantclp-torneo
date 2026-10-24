"""
EXP 04 — Juego: mejor variante vs. peor variante invertida, solo desde 2023.

ADVERTENCIA: la selección y la evaluación usan el MISMO período (2023 -> inicio
del holdout). El resultado es in-sample por construcción y no sirve como
evidencia. Las 24 variantes y la inversión quedan anotadas en el registro.

Requiere src/triangulos.py y la caché de detecciones del exp 03
(si no existe, la crea; tarda ~1 min solo para 2023+).
Correr con F5 en Spyder.
"""

import importlib
import os
import pickle
import sys
from datetime import datetime
from itertools import product
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
CACHE = RAIZ / "data" / "cache" / "detecciones.pkl"

DESDE = "2023-01-01"
HOLDOUT_MESES = 18
COSTO = 0.50
COSTOS_SENS = [0.0, 0.25, 0.50, 1.00, 2.00]
MIN_HOLD = 3
N_BOOT, BLOQUE = 5000, 10

df = tg.cargar_usdclp(BASE)
HOLDOUT_DESDE = df.index.max() - pd.DateOffset(months=HOLDOUT_MESES)
df = df[df.index < HOLDOUT_DESDE]
det = tg.Detector(df)
F = df.index
idx = np.where(F >= pd.Timestamp(DESDE))[0][:-1]
print(f"Período: {F[idx[0]].date()} -> {F[idx[-1] + 1].date()} | "
      f"holdout desde {HOLDOUT_DESDE.date()} bloqueado")

# detecciones: reutiliza la caché del exp 03 si cubre estos días
tris = {}
if CACHE.exists():
    with open(CACHE, "rb") as fh:
        tris = pickle.load(fh)["tris"]
faltan = [k for k in idx if k not in tris]
if faltan:
    print(f"Detectando {len(faltan)} días...")
    for k in faltan:
        tris[k] = det.detectar(k)

S0, S1 = det.C[idx], det.C[idx + 1]
ret = S1 / S0 - 1
dias = (F[idx + 1] - F[idx]).days.to_numpy()
carry_u = -(df["pts1m"].to_numpy()[idx] / S0) * dias / 30


def min_hold(pos, n):
    out = pos.copy()
    ult = 0
    for i in range(1, len(out)):
        if pos[i] != out[i - 1] and i - ult >= n:
            ult = i
        else:
            out[i] = out[i - 1]
    return out


def pnl(pos, costo=COSTO):
    dpos = np.abs(np.diff(pos, prepend=0.0))
    bruto, carry, cst = pos * ret, pos * carry_u, -dpos * costo / S0
    return bruto, carry, cst, bruto + carry + cst, dpos


def metricas(pos, costo=COSTO):
    br, ca, cs, nt, dp = pnl(pos, costo)
    mens = pd.Series(nt, index=F[idx]).resample("ME").sum()
    act = pd.Series(pos != 0, index=F[idx]).resample("ME").sum()
    eq = np.cumsum(nt)
    sd = nt.std()
    return {"bruto_%": br.sum() * 100, "carry_%": ca.sum() * 100,
            "costos_%": cs.sum() * 100, "neto_%": nt.sum() * 100,
            "sharpe": nt.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
            "maxDD_%": (eq - np.maximum.accumulate(eq)).min() * 100,
            "rotación": dp.sum(), "%dias_pos": (pos != 0).mean() * 100,
            "%meses>0": (mens > 0).mean() * 100,
            "%meses_act>0": (mens[act > 0] > 0).mean() * 100}


variantes = {}
for r_, s_, b_ in product("AB", "AB", "ABC"):
    cfg = {"ruptura": r_, "salida": s_, "base": b_}
    pos = np.array([tg.senal(det, tris[k], k, cfg)[0] for k in idx], float)
    variantes[f"R{r_}-S{s_}-B{b_}-OA"] = pos
    variantes[f"R{r_}-S{s_}-B{b_}-OB"] = min_hold(pos, MIN_HOLD)

tab = pd.DataFrame({n: metricas(p) for n, p in variantes.items()}).T \
        .sort_values("neto_%", ascending=False)
pd.set_option("display.width", 200)
print("\n24 variantes · 2023+ · costo 0,5 CLP · ordenadas por neto")
print(tab[["neto_%", "sharpe", "maxDD_%", "rotación", "%meses>0"]].round(2).to_string())

mejor, peor = tab.index[0], tab.index[-1]
duelo = {f"Mejor: {mejor}": variantes[mejor],
         f"Peor invertida: −({peor})": -variantes[peor],
         f"Peor original: {peor}": variantes[peor],
         "Siempre +1": np.ones(len(idx)),
         "No operar": np.zeros(len(idx))}
res = pd.DataFrame({n: metricas(p) for n, p in duelo.items()}).T
print("\n" + "=" * 90)
print("DUELO (in-sample: elegido y evaluado en el mismo período)")
print("=" * 90)
print(res.round(2).to_string())

sens = pd.DataFrame({f"{c:.2f}": {n: metricas(p, c)["neto_%"] for n, p in duelo.items()}
                     for c in COSTOS_SENS})
print("\nSensibilidad a costos · neto % (CLP por USD)")
print(sens.round(2).to_string())


def p_boot(x, B=N_BOOT, bloque=BLOQUE, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    nuevo = rng.random((B, n)) < 1 / bloque
    nuevo[:, 0] = True
    t = np.arange(n)
    ult = np.maximum.accumulate(np.where(nuevo, t, 0), axis=1)
    ii = (np.take_along_axis(rng.integers(0, n, (B, n)), ult, axis=1) + t - ult) % n
    m = (x - x.mean())[ii].mean(axis=1)
    return float(np.mean(m >= x.mean()))


ref1 = pnl(np.ones(len(idx)))[3]
print("\nBootstrap (sin ajustar por selección, por eso es optimista):")
for n in list(duelo)[:2]:
    nt = pnl(duelo[n])[3]
    print(f"  {n:<40} p vs 0 = {p_boot(nt):.3f} | p vs +1 = {p_boot(nt - ref1):.3f}")

reg_path = OUT / "registro_experimentos.csv"
filas = [{"timestamp": datetime.now().isoformat(timespec="seconds"),
          "experimento": f"04_{n}", "desde": DESDE, "hasta": str(HOLDOUT_DESDE.date()),
          "costo": COSTO, "neto_%": round(res.loc[n, "neto_%"], 3),
          "sharpe": round(res.loc[n, "sharpe"], 3),
          "pct_meses_pos": round(res.loc[n, "%meses>0"], 1),
          "elegida": True} for n in list(duelo)[:2]]
reg = pd.DataFrame(filas)
if reg_path.exists():
    reg = pd.concat([pd.read_csv(reg_path), reg], ignore_index=True)
reg.to_csv(reg_path, index=False)
print(f"\nRegistro: {len(reg)} intentos acumulados.")
