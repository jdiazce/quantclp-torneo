"""
EXP 06 — Doble validación: probit ordenado (IPSA + monedas + cobre) y
triángulo RA-SB-BC-OA como segundo paso.

PASO 1 — PROBIT ORDENADO (EMV, src/probit_ordenado.py)
  Regresores al cierre de t: IPSA, cobre (HG1), USDBRL, USDMXN, USDCOP,
  USDPEN, USDZAR (retornos log diarios, estandarizados con el TRAIN).
  Objetivo: r(t→t+1) en 5 categorías (percentiles 10/30/70/90 del TRAIN).
  Posición: E[r] + carry contra umbrales c1 < c2 (0 / ±1 / ±3).

PASO 2 — COMBINACIÓN con el triángulo (p = probit, q = triángulo)
  A  validador : opera solo si coinciden en signo; tamaño = el menor. Si no, 0.
  B  reemplazo : si uno es 0, usa el otro; si coinciden, el de menor tamaño;
                 si se contradicen, 0 (la opción de menos riesgo).
  C  opuesto   : invierte ambas señales. Si alguna es 0, queda 0. Si coinciden,
                 va al revés con el menor tamaño. Si se contradicen:
                 C1 = sigue el opuesto del probit · C2 = el opuesto del triángulo.

PERÍODOS
  TRAIN  2010-01-01 – 2019-10-17
         · CALIB 2015-01 – 2019-10: walk-forward interno (reestima cada mes con
           todo lo anterior desde 2010). Aquí se fijan los umbrales del probit
           y se ELIGE la combinación ganadora (mayor Sharpe neto).
  TEST   2023-01-01 – inicio del holdout (modelo estimado con todo el TRAIN).
  HOLDOUT últimos 18 meses: no se carga.

OJO: el triángulo RA-SB-BC-OA se eligió mirando 2023+ (exp 04), así que el
TEST de las combinaciones que lo usan está contaminado. El CALIB no.

Correr con F5 en Spyder. Requiere src/triangulos.py, src/probit_ordenado.py y
la caché data/cache/detecciones.pkl del exp 03 (si falta, la completa).
"""

import importlib
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

RAIZ = Path(os.environ.get("QUANTCLP", Path(__file__).resolve().parents[1]))
importlib.invalidate_caches()
sys.path.insert(0, str(RAIZ / "src"))
import triangulos as tg                      # noqa: E402
from probit_ordenado import ProbitOrdenado    # noqa: E402

BASE = next(p for p in (RAIZ / "data" / "base_usdclp.xlsx",
                        RAIZ / "base_usdclp.xlsx") if p.exists())
OUT = RAIZ / "experimentos"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = RAIZ / "data" / "cache" / "detecciones.pkl"

TRAIN = ("2010-01-01", "2019-10-17")
CALIB_DESDE = "2015-01-01"
TEST_DESDE = "2023-01-01"
HOLDOUT_MESES = 18
CORTES_PCT = [10, 30, 70, 90]
COSTO = 0.50
COSTOS_SENS = [0.0, 0.25, 0.50, 1.00, 2.00]
N_BOOT, BLOQUE = 5000, 10
GRID_C1 = [50, 60, 70, 80, 90]
GRID_C2 = [90, 95, 97.5, 100]           # 100 = nunca ±3
CFG_TRI = {"ruptura": "A", "salida": "B", "base": "C"}     # RA-SB-BC-OA

SERIES = {"IPSA Index|PX_LAST": "ipsa", "HG1 Comdty|PX_LAST": "cobre",
          "USDBRL Curncy|PX_LAST": "brl", "USDMXN Curncy|PX_LAST": "mxn",
          "USDCOP Curncy|PX_LAST": "cop", "USDPEN Curncy|PX_LAST": "pen",
          "USDZAR Curncy|PX_LAST": "zar"}
EXOG = list(SERIES.values())

# ----------------------------------------------------------------------
# DATOS
# ----------------------------------------------------------------------
clp = tg.cargar_usdclp(BASE)
HOLDOUT_DESDE = clp.index.max() - pd.DateOffset(months=HOLDOUT_MESES)
clp = clp[clp.index < HOLDOUT_DESDE]
det = tg.Detector(clp)
F = clp.index

ex = (pd.read_excel(BASE, sheet_name="datos", usecols=["fecha", *SERIES])
        .rename(columns=SERIES)
        .assign(fecha=lambda x: pd.to_datetime(x["fecha"]))
        .set_index("fecha").sort_index())
# último cierre disponible en cada día del USD/CLP (mercado cerrado = retorno 0)
ex = ex[ex.index < HOLDOUT_DESDE].ffill().reindex(F, method="ffill")

d = np.log(ex).diff()
d["c"] = clp["c"]
d["ret_next"] = clp["c"].shift(-1) / clp["c"] - 1
dias = pd.Series(F, index=F).diff().shift(-1).dt.days
d["carry_u"] = -(clp["pts1m"] / clp["c"]) * dias / 30      # 1 unidad larga

# triángulo (causal) en los días que se usan
with open(CACHE, "rb") as fh:
    tris = pickle.load(fh)["tris"]
usar = ((F >= pd.Timestamp(CALIB_DESDE)) & (F <= pd.Timestamp(TRAIN[1]))) | \
       (F >= pd.Timestamp(TEST_DESDE))
q = np.full(len(F), np.nan)
for k in np.where(usar)[0][:-1]:
    if k not in tris:
        tris[k] = det.detectar(k)
    q[k] = tg.senal(det, tris[k], k, CFG_TRI)[0]
d["tri"] = q

d = d.iloc[:-1]
tr = d.loc[TRAIN[0]:TRAIN[1]].dropna(subset=EXOG + ["ret_next"])
te = d.loc[TEST_DESDE:].dropna(subset=EXOG + ["ret_next"])
print(f"TRAIN {tr.index.min().date()} – {tr.index.max().date()} ({len(tr)} días) · "
      f"CALIB desde {CALIB_DESDE} · TEST {te.index.min().date()} – "
      f"{te.index.max().date()} ({len(te)} días)")
print(f"Holdout bloqueado desde {HOLDOUT_DESDE.date()}\n")


# ----------------------------------------------------------------------
# PASO 1: PROBIT
# ----------------------------------------------------------------------
class Pipeline:
    def fit(self, D):
        X = D[EXOG].to_numpy(float)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1
        r = D["ret_next"].to_numpy()
        self.cortes = np.percentile(r, CORTES_PCT)
        y = np.digitize(r, self.cortes)
        self.media_cat = np.array([r[y == j].mean() for j in range(5)])
        self.m = ProbitOrdenado(5).fit((X - self.mu) / self.sd, y)
        return self

    def e_ret(self, D):
        X = (D[EXOG].to_numpy(float) - self.mu) / self.sd
        return self.m.predict_proba(X) @ self.media_cat


def posiciones(e_neto, c1, c2):
    a = np.abs(e_neto)
    return np.sign(e_neto) * np.where(a >= c2, 3, np.where(a >= c1, 1, 0))


def pnl(pos, D, costo=COSTO):
    dpos = np.abs(np.diff(pos, prepend=0.0))
    bruto = pos * D["ret_next"].to_numpy()
    carry = pos * D["carry_u"].to_numpy()
    cst = -dpos * costo / D["c"].to_numpy()
    return bruto, carry, cst, bruto + carry + cst, dpos


# walk-forward interno en CALIB
cal = tr.loc[CALIB_DESDE:].copy()
e_cal = pd.Series(np.nan, index=cal.index)
for _, bloque in cal.groupby(cal.index.to_period("M")):
    hist = tr.loc[:bloque.index[0] - pd.Timedelta(days=1)].iloc[:-1]   # embargo 1 día
    e_cal.loc[bloque.index] = Pipeline().fit(hist).e_ret(bloque)
e_neto_cal = e_cal.to_numpy() + cal["carry_u"].to_numpy()

a = np.abs(e_neto_cal)
mejor = (-np.inf,)
for p1 in GRID_C1:
    for p2 in GRID_C2:
        if p2 <= p1:
            continue
        c1 = np.percentile(a, p1)
        c2 = np.inf if p2 == 100 else np.percentile(a, p2)
        s = pnl(posiciones(e_neto_cal, c1, c2), cal)[3].sum()
        if s > mejor[0]:
            mejor = (s, c1, c2, p1, p2)
_, C1, C2, P1, P2 = mejor

modelo = Pipeline().fit(tr)
e_te = modelo.e_ret(te)
p_cal = posiciones(e_neto_cal, C1, C2)
p_te = posiciones(e_te + te["carry_u"].to_numpy(), C1, C2)
q_cal, q_te = cal["tri"].to_numpy(), te["tri"].to_numpy()

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
z = modelo.m.beta_ / modelo.m.se_beta_
print("PASO 1 · coeficientes del probit (TRAIN completo, estandarizados)")
print(pd.DataFrame({"coef": modelo.m.beta_, "z": z}, index=EXOG).round(3).to_string())
print(f"pseudo-R2 McFadden = {modelo.m.r2_mcfadden_:.4f} · "
      f"umbrales: percentiles {P1}/{P2} de |E_neto| en CALIB\n")


# ----------------------------------------------------------------------
# PASO 2: COMBINACIONES
# ----------------------------------------------------------------------
def comb_A(p, q):
    ok = (np.sign(p) == np.sign(q)) & (p != 0)
    return np.where(ok, np.sign(p) * np.minimum(np.abs(p), np.abs(q)), 0.0)


def comb_B(p, q):
    out = np.where(p == 0, q, np.where(q == 0, p, 0.0))
    igual = (np.sign(p) == np.sign(q)) & (p != 0)
    return np.where(igual, np.sign(p) * np.minimum(np.abs(p), np.abs(q)), out)


def comb_C(p, q, en_conflicto):
    out = -comb_A(p, q)                           # coinciden -> al revés
    conflicto = (p != 0) & (q != 0) & (np.sign(p) != np.sign(q))
    alt = -p if en_conflicto == "probit" else -q
    return np.where(conflicto, alt, out)          # si alguno es 0 -> 0


def todas(p, q):
    return {"Probit solo": p, "Triángulo solo": q,
            "A validador": comb_A(p, q), "B reemplazo": comb_B(p, q),
            "C1 opuesto (conflicto: −probit)": comb_C(p, q, "probit"),
            "C2 opuesto (conflicto: −triángulo)": comb_C(p, q, "tri")}


def metricas(pos, D, costo=COSTO):
    br, ca, cs, nt, dp = pnl(pos, D, costo)
    mens = pd.Series(nt, index=D.index).resample("ME").sum()
    act = pd.Series(pos != 0, index=D.index).resample("ME").sum()
    eq = np.cumsum(nt)
    sd = nt.std()
    return {"bruto_%": br.sum() * 100, "carry_%": ca.sum() * 100,
            "costos_%": cs.sum() * 100, "neto_%": nt.sum() * 100,
            "neto_anual_%": nt.mean() * 252 * 100,
            "sharpe": nt.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
            "maxDD_%": (eq - np.maximum.accumulate(eq)).min() * 100,
            "rot_anual": dp.sum() / len(pos) * 252,
            "%días_pos": (pos != 0).mean() * 100,
            "%días_±3": (np.abs(pos) == 3).mean() * 100,
            "%meses>0": (mens > 0).mean() * 100,
            "%meses_act>0": (mens[act > 0] > 0).mean() * 100 if (act > 0).any() else np.nan}


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


def diebold_mariano(e1, e2):
    dl = e1 ** 2 - e2 ** 2
    n = len(dl)
    dm = dl.mean()
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    g = [np.mean((dl[k:] - dm) * (dl[:n - k] - dm)) for k in range(L + 1)]
    v = g[0] + 2 * sum((1 - k / (L + 1)) * g[k] for k in range(1, L + 1))
    s = dm / np.sqrt(v / n)
    return s, norm.cdf(s)


bench = lambda n: {"Siempre +1": np.ones(n), "Siempre +3": np.full(n, 3.0),
                   "No operar": np.zeros(n)}
cols = ["bruto_%", "carry_%", "costos_%", "neto_%", "neto_anual_%", "sharpe", "maxDD_%",
        "rot_anual", "%días_pos", "%días_±3", "%meses>0", "%meses_act>0"]

res = {}
for per, (p, qq, D) in {"CALIB": (p_cal, q_cal, cal), "TEST": (p_te, q_te, te)}.items():
    estr = todas(p, qq) | bench(len(D))
    res[per] = pd.DataFrame({n: metricas(v, D) for n, v in estr.items()}).T
    print("=" * 130)
    print(f"{per} {D.index.min().date()} – {D.index.max().date()} · costo {COSTO} CLP"
          + ("  ·  combinaciones con triángulo CONTAMINADAS (elegido mirando este período)"
             if per == "TEST" else "  ·  pseudo fuera de muestra (walk-forward)"))
    print("=" * 130)
    print(res[per][cols].to_string(float_format=lambda x: f"{x:.2f}"))
    print()

nombres = list(todas(p_cal, q_cal))
ganadora = res["CALIB"].loc[nombres, "sharpe"].astype(float).idxmax()
print(f"COMBINACIÓN GANADORA EN CALIB (regla: mayor Sharpe neto): {ganadora}")
print(f"  CALIB neto {res['CALIB'].loc[ganadora, 'neto_%']:.2f}%  ·  "
      f"TEST neto {res['TEST'].loc[ganadora, 'neto_%']:.2f}%  ·  "
      f"Siempre +1 TEST {res['TEST'].loc['Siempre +1', 'neto_%']:.2f}%\n")

# significancia en TEST
r_te = te["ret_next"].to_numpy()
s0, p0 = diebold_mariano(r_te - e_te, r_te)
sm, pm = diebold_mariano(r_te - e_te, r_te - tr["ret_next"].mean())
print("SIGNIFICANCIA TEST")
print(f"  Diebold-Mariano del probit vs paseo aleatorio: {s0:+.2f} (p={p0:.3f}) · "
      f"vs media TRAIN: {sm:+.2f} (p={pm:.3f})   [negativo = modelo mejor]")
ref1 = pnl(np.ones(len(te)), te)[3]
for n, v in todas(p_te, q_te).items():
    nt = pnl(v, te)[3]
    print(f"  {n:<36} bootstrap p(neto ≤ 0) = {p_boot(nt):.3f} · "
          f"p(no le gana a +1) = {p_boot(nt - ref1):.3f}")

sens = pd.DataFrame({f"{c:.2f}": {n: metricas(v, te, c)["neto_%"]
                                  for n, v in todas(p_te, q_te).items()}
                     for c in COSTOS_SENS})
print("\nSENSIBILIDAD A COSTOS · TEST · neto % (CLP por USD)")
print(sens.round(2).to_string())

# ----------------------------------------------------------------------
# REGISTRO
# ----------------------------------------------------------------------
reg_path = OUT / "registro_experimentos.csv"
filas = [{"timestamp": datetime.now().isoformat(timespec="seconds"),
          "experimento": f"06_{n}", "desde": TEST_DESDE,
          "hasta": str(te.index.max().date()), "costo": COSTO,
          "neto_%": round(float(res["TEST"].loc[n, "neto_%"]), 3),
          "sharpe": round(float(res["TEST"].loc[n, "sharpe"]), 3),
          "pct_meses_pos": round(float(res["TEST"].loc[n, "%meses>0"]), 1),
          "elegida": n == ganadora} for n in nombres]
reg = pd.DataFrame(filas)
if reg_path.exists():
    reg = pd.concat([pd.read_csv(reg_path), reg], ignore_index=True)
reg.to_csv(reg_path, index=False)
print(f"\nRegistro: {len(reg)} intentos acumulados.")
