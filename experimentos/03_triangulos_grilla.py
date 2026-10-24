"""
EXP 03 — Grilla paralela de variantes de la estrategia de triángulos.

VARIANTES (2 × 2 × 3 × 2 = 24)
  ruptura   A: 3 cierres fuera + 0,45 ATR       B: 1 cierre a >= 1 ATR
  salida    A: vence a los 20 días               B: objetivo = altura proyectada,
                                                    stop = cierre 1 ATR adentro
  base      A: a <= 0,6 ATR de la recta          B: + no pasado el 75% al ápice
            C: + vela de rechazo
  rotación  A: libre                             B: mínimo 3 días por posición
Comunes a todas: apoyo = toca la recta rota y vuelve a cerrar afuera; si cierra
adentro tiene 2 días para salir de nuevo o la ruptura muere. Ruptura bajista
simétrica (−1 / −3).

PERÍODOS
  TRAIN  2005-01-01 – 2018-12-31   se elige aquí
  VALID  2019-01-01 – 2022-12-31   se mira UNA vez, para la variante elegida
  VISTO  2023-01-01 – holdout      ya contaminado (exp 02), solo informativo
  HOLDOUT últimos 18 meses          no se carga

REGLA DE SELECCIÓN (fijada antes de correr)
  Mayor Sharpe neto en TRAIN con costo 0,5 CLP. Significancia de la grilla
  completa con White Reality Check (bootstrap estacionario).

La detección es lenta (~10 min la primera vez) y se guarda en
data/cache/detecciones.pkl. Las corridas siguientes tardan segundos.
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
CACHE.parent.mkdir(parents=True, exist_ok=True)

HOLDOUT_MESES = 18
DESDE = "2005-01-01"
PERIODOS = {"TRAIN": ("2005-01-01", "2018-12-31"),
            "VALID": ("2019-01-01", "2022-12-31"),
            "VISTO": ("2023-01-01", "2099-12-31")}
COSTO = 0.50
COSTOS_SENS = [0.0, 0.25, 0.50, 1.00, 2.00]
MIN_HOLD = 3
N_BOOT, BLOQUE = 2000, 10

# ----------------------------------------------------------------------
# DATOS (sin holdout)
# ----------------------------------------------------------------------
df = tg.cargar_usdclp(BASE)
HOLDOUT_DESDE = df.index.max() - pd.DateOffset(months=HOLDOUT_MESES)
df = df[df.index < HOLDOUT_DESDE]
det = tg.Detector(df)
F = df.index
print(f"Datos hasta {F.max().date()} (holdout desde {HOLDOUT_DESDE.date()} bloqueado)")

idx = np.where(F >= pd.Timestamp(DESDE))[0][:-1]   # el último no tiene t+1

# ----------------------------------------------------------------------
# DETECCIÓN (con caché)
# ----------------------------------------------------------------------
firma = (str(F[idx[0]].date()), str(F[idx[-1]].date()), len(F),
         tg.TOL_ATR, tg.SPAN_MIN, tg.SPAN_MAX, tg.MAX_TRIANGULOS)
tris = None
if CACHE.exists():
    with open(CACHE, "rb") as fh:
        c = pickle.load(fh)
    if c.get("firma") == firma:
        tris = c["tris"]
        print("Detecciones cargadas desde caché.")
if tris is None:
    print(f"Detectando triángulos en {len(idx)} días (una sola vez)...")
    tris = {}
    for n, k in enumerate(idx, 1):
        tris[k] = det.detectar(k)
        if n % 250 == 0:
            print(f"  {n}/{len(idx)}  {F[k].date()}")
    with open(CACHE, "wb") as fh:
        pickle.dump({"firma": firma, "tris": tris}, fh)

# ----------------------------------------------------------------------
# SEÑALES POR VARIANTE
# ----------------------------------------------------------------------
S0, S1 = det.C[idx], det.C[idx + 1]
ret = S1 / S0 - 1
dias = (F[idx + 1] - F[idx]).days.to_numpy()
carry_u = -(df["pts1m"].to_numpy()[idx] / S0) * dias / 30   # 1 unidad larga


def min_hold(pos, n):
    """Una posición nueva se mantiene al menos n días antes de cambiar."""
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
    bruto = pos * ret
    carry = pos * carry_u
    cst = -dpos * costo / S0
    return bruto, carry, cst, bruto + carry + cst, dpos


variantes = {}
for r_, s_, b_ in product("AB", "AB", "ABC"):
    cfg = {"ruptura": r_, "salida": s_, "base": b_}
    res = [tg.senal(det, tris[k], k, cfg) for k in idx]
    pos = np.array([x[0] for x in res], float)
    casos = np.array([x[1] for x in res])
    tipos = np.array([x[2] for x in res])
    variantes[f"R{r_}-S{s_}-B{b_}-OA"] = (pos, casos, tipos)
    variantes[f"R{r_}-S{s_}-B{b_}-OB"] = (min_hold(pos, MIN_HOLD), casos, tipos)
print(f"{len(variantes)} variantes calculadas.\n")

bench = {"Siempre +1": np.ones(len(idx)), "Siempre +3": np.full(len(idx), 3.0),
         "No operar": np.zeros(len(idx))}

mask = {p: (F[idx] >= pd.Timestamp(a)) & (F[idx] <= pd.Timestamp(b))
        for p, (a, b) in PERIODOS.items()}


def metricas(pos, m, costo=COSTO):
    br, ca, cs, nt, dp = (x[m] for x in pnl(pos, costo))
    fe = F[idx][m]
    mens = pd.Series(nt, index=fe).resample("ME").sum()
    act = pd.Series(pos[m] != 0, index=fe).resample("ME").sum()
    sd = nt.std()
    eq = np.cumsum(nt)
    return {"bruto_%": br.sum() * 100, "carry_%": ca.sum() * 100,
            "costos_%": cs.sum() * 100, "neto_%": nt.sum() * 100,
            "neto_anual_%": nt.mean() * 252 * 100,
            "sharpe": nt.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
            "maxDD_%": (eq - np.maximum.accumulate(eq)).min() * 100,
            "rot_anual": dp.sum() / m.sum() * 252,
            "%dias_pos": (pos[m] != 0).mean() * 100,
            "%dias_±3": (np.abs(pos[m]) == 3).mean() * 100,
            "%meses>0": (mens > 0).mean() * 100,
            "%meses_act>0": (mens[act > 0] > 0).mean() * 100}


pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
cols_show = ["neto_%", "neto_anual_%", "sharpe", "maxDD_%", "rot_anual",
             "%dias_pos", "%dias_±3", "%meses>0", "%meses_act>0"]

tab_train = pd.DataFrame({n: metricas(v[0], mask["TRAIN"]) for n, v in variantes.items()}).T
tab_train = tab_train.sort_values("sharpe", ascending=False)
bt_train = pd.DataFrame({n: metricas(p, mask["TRAIN"]) for n, p in bench.items()}).T

print("=" * 100)
print("TRAIN 2005–2018 · 24 variantes · costo 0,5 CLP · ordenadas por Sharpe neto")
print("=" * 100)
print(tab_train[cols_show].round(2).to_string())
print("\nReferencias TRAIN:")
print(bt_train[cols_show].round(2).to_string())


# ----------------------------------------------------------------------
# WHITE REALITY CHECK (todas las variantes, TRAIN)
# ----------------------------------------------------------------------
def indices_estacionarios(n, B, bloque, seed=0):
    rng = np.random.default_rng(seed)
    nuevo = rng.random((B, n)) < 1 / bloque
    nuevo[:, 0] = True
    inicio = rng.integers(0, n, (B, n))
    t = np.arange(n)
    ult = np.maximum.accumulate(np.where(nuevo, t, 0), axis=1)
    base = np.take_along_axis(inicio, ult, axis=1)
    return (base + (t - ult)) % n


def reality_check(netos, ref, B=N_BOOT, bloque=BLOQUE):
    """p-valor de H0: ninguna variante le gana a ref (White 2000)."""
    D = netos - ref[None, :]                     # variantes × días
    n = D.shape[1]
    dbar = D.mean(axis=1)
    V = np.sqrt(n) * dbar.max()
    ii = indices_estacionarios(n, B, bloque)
    Vb = np.empty(B)
    for b in range(B):
        Vb[b] = (np.sqrt(n) * (D[:, ii[b]].mean(axis=1) - dbar)).max()
    return float(np.mean(Vb >= V))


m = mask["TRAIN"]
netos = np.vstack([pnl(v[0])[3][m] for v in variantes.values()])
p_rc0 = reality_check(netos, np.zeros(m.sum()))
p_rc1 = reality_check(netos, pnl(bench["Siempre +1"])[3][m])
print(f"\nWhite Reality Check (24 variantes, TRAIN): "
      f"p vs No operar = {p_rc0:.3f} | p vs Siempre +1 = {p_rc1:.3f}")

# ----------------------------------------------------------------------
# DESCOMPOSICIÓN POR CASO × TIPO (TRAIN): referencia y elegida
# ----------------------------------------------------------------------
ganadora = tab_train.index[0]
REF = "RA-SA-BA-OA"


def por_caso_tipo(nombre, m):
    pos, casos, tipos = variantes[nombre]
    nt = pnl(pos)[3]
    d = pd.DataFrame({"caso": casos[m], "tipo": tipos[m], "pos": pos[m], "neto": nt[m]})
    d = d[d["pos"] != 0]
    g = d.groupby(["caso", "tipo"])["neto"]
    return pd.DataFrame({"dias": g.size(), "neto_bps": g.sum() * 1e4,
                         "bps_dia": g.mean() * 1e4,
                         "acierto": g.apply(lambda x: (x > 0).mean())})


for nombre in dict.fromkeys([REF, ganadora]):
    print(f"\nPor caso × tipo · {nombre} · TRAIN")
    print(por_caso_tipo(nombre, mask["TRAIN"]).round(2).to_string())

# ----------------------------------------------------------------------
# VALIDACIÓN (una sola vez) y período visto
# ----------------------------------------------------------------------
print("\n" + "=" * 100)
print(f"VARIANTE ELEGIDA EN TRAIN: {ganadora}  (referencia: {REF})")
print("=" * 100)
cols_dec = ["bruto_%", "carry_%", "costos_%", "neto_%", "sharpe", "maxDD_%",
            "rot_anual", "%meses>0", "%meses_act>0"]
for per in ("TRAIN", "VALID", "VISTO"):
    filas = {n: metricas(variantes[n][0], mask[per]) for n in dict.fromkeys([ganadora, REF])}
    filas.update({n: metricas(p, mask[per]) for n, p in bench.items()})
    print(f"\n{per} {PERIODOS[per][0][:4]}–{F[idx][mask[per]].max().year}"
          + ("  (contaminado por exp 02)" if per == "VISTO" else ""))
    print(pd.DataFrame(filas).T[cols_dec].round(2).to_string())

mv = mask["VALID"]
nt_g = pnl(variantes[ganadora][0])[3][mv]
p_v0 = reality_check(nt_g[None, :], np.zeros(mv.sum()))
p_v1 = reality_check(nt_g[None, :], pnl(bench["Siempre +1"])[3][mv])
print(f"\nBootstrap VALID (elegida): p vs No operar = {p_v0:.3f} | "
      f"p vs Siempre +1 = {p_v1:.3f}")

sens = pd.DataFrame({f"{c:.2f}": {per: metricas(variantes[ganadora][0], mask[per], c)["neto_%"]
                                  for per in PERIODOS} for c in COSTOS_SENS})
print(f"\nSensibilidad a costos · {ganadora} · neto % (CLP por USD)")
print(sens.round(2).to_string())

# ----------------------------------------------------------------------
# REGISTRO
# ----------------------------------------------------------------------
tab_train.to_csv(OUT / "exp03_grilla_train.csv")
reg_path = OUT / "registro_experimentos.csv"
filas = []
for n in variantes:
    mt = tab_train.loc[n]
    filas.append({"timestamp": datetime.now().isoformat(timespec="seconds"),
                  "experimento": f"03_grilla_{n}",
                  "desde": PERIODOS["TRAIN"][0], "hasta": PERIODOS["TRAIN"][1],
                  "costo": COSTO, "neto_%": round(mt["neto_%"], 3),
                  "sharpe": round(mt["sharpe"], 3),
                  "pct_meses_pos": round(mt["%meses>0"], 1),
                  "p_reality_check_0": round(p_rc0, 4),
                  "p_reality_check_+1": round(p_rc1, 4),
                  "elegida": n == ganadora})
reg = pd.DataFrame(filas)
if reg_path.exists():
    reg = pd.concat([pd.read_csv(reg_path), reg], ignore_index=True)
reg.to_csv(reg_path, index=False)
print(f"\nRegistro: {len(reg)} intentos acumulados en {reg_path.name}.")
