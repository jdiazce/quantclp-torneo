"""
EXP 05 — Probit ordenado (EMV) para la dirección del USD/CLP en t+1,
con y sin el estado del triángulo como regresor.

DATOS
  TRAIN  2015-01-01 – 2019-10-17   (antes del estallido social)
         · 2015-2016 estima, 2017-2019 calibra umbrales (walk-forward interno)
  TEST   2023-01-01 – inicio del holdout
  2019-10-18 – 2022-12-31 excluido (estallido, pandemia, proceso constituyente)
  HOLDOUT últimos 18 meses: no se carga.

VARIABLE A EXPLICAR
  r(t→t+1) en 5 categorías con cortes en los percentiles 10/30/70/90 del TRAIN.

REGRESORES (cierres de t; el diccionario permite usar los de la misma fecha)
  cobre (HG1), DXY, canasta LatAm (BRL, MXN, COP), VIX, diferencial de tasas
  implícito en el NDF 1M, momentum propio 5 y 20 días, estado del triángulo.

PROBABILIDAD → POSICIÓN
  E[r] = Σ P(j)·media_r(j) (medias del TRAIN). Se compara E[r] + carry contra
  umbrales c1 < c2 fijados en el bloque de calibración: |E_neto| < c1 → 0,
  c1 ≤ … < c2 → ±1, ≥ c2 → ±3.

MODELOS (7, todos al registro)
  M0 sin triángulo | M1 + triángulo RA-SB-BC-OA (elegido mirando 2023+: CONTAMINADO)
  M2 + triángulo RA-SA-BC-OA (elegido en 2005-2018: limpio)
  cada uno FIJO (estimado solo con TRAIN) y WF (reestima cada mes sumando
  los datos de TEST ya realizados) + M0 con regresores rezagados (robustez).

Correr con F5 en Spyder. Requiere src/triangulos.py y src/probit_ordenado.py.
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
import triangulos as tg              # noqa: E402
from probit_ordenado import ProbitOrdenado  # noqa: E402

BASE = next(p for p in (RAIZ / "data" / "base_usdclp.xlsx",
                        RAIZ / "base_usdclp.xlsx") if p.exists())
OUT = RAIZ / "experimentos"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = RAIZ / "data" / "cache" / "detecciones.pkl"

TRAIN = ("2015-01-01", "2019-10-17")
CALIB_DESDE = "2017-01-01"
TEST_DESDE = "2023-01-01"
HOLDOUT_MESES = 18
CORTES_PCT = [10, 30, 70, 90]
COSTO = 0.50
COSTOS_SENS = [0.0, 0.25, 0.50, 1.00, 2.00]
N_BOOT, BLOQUE = 5000, 10
GRID_C1 = [50, 60, 70, 80, 90]          # percentiles de |E_neto| en calibración
GRID_C2 = [90, 95, 97.5, 100]           # 100 = nunca ±3
EXOG = ["cobre", "dxy", "latam", "vix", "dif_tasas", "mom5", "mom20"]

# ----------------------------------------------------------------------
# DATOS
# ----------------------------------------------------------------------
series = {"HG1 Comdty|PX_LAST": "hg", "DXY Curncy|PX_LAST": "dxy",
          "USDBRL Curncy|PX_LAST": "brl", "USDMXN Curncy|PX_LAST": "mxn",
          "USDCOP Curncy|PX_LAST": "cop", "VIX Index|PX_LAST": "vix"}
ex = (pd.read_excel(BASE, sheet_name="datos", usecols=["fecha", *series])
        .rename(columns=series)
        .assign(fecha=lambda x: pd.to_datetime(x["fecha"]))
        .set_index("fecha").sort_index())

clp = tg.cargar_usdclp(BASE)
HOLDOUT_DESDE = clp.index.max() - pd.DateOffset(months=HOLDOUT_MESES)
clp = clp[clp.index < HOLDOUT_DESDE]
det = tg.Detector(clp)
F = clp.index

# exógenas al calendario del USD/CLP: último cierre disponible a esa fecha
ex = ex[ex.index < HOLDOUT_DESDE].ffill().reindex(F, method="ffill")

d = pd.DataFrame(index=F)
d["c"] = clp["c"]
d["cobre"] = np.log(ex["hg"]).diff()
d["dxy"] = np.log(ex["dxy"]).diff()
d["latam"] = np.log(ex[["brl", "mxn", "cop"]]).diff().mean(axis=1)
d["vix"] = np.log(ex["vix"]).diff()
d["dif_tasas"] = clp["pts1m"] / clp["c"] * 12 * 100          # % anual
d["mom5"] = np.log(d["c"]).diff(5)
d["mom20"] = np.log(d["c"]).diff(20)
d["ret_next"] = d["c"].shift(-1) / d["c"] - 1                 # objetivo
dias = pd.Series(F, index=F).diff().shift(-1).dt.days
d["carry_u"] = -(clp["pts1m"] / clp["c"]) * dias / 30          # 1 unidad larga

# triángulo: posición de la variante en t (causal)
with open(CACHE, "rb") as fh:
    tris = pickle.load(fh)["tris"]
VAR_TRI = {"tri_RA-SB-BC-OA": {"ruptura": "A", "salida": "B", "base": "C"},
           "tri_RA-SA-BC-OA": {"ruptura": "A", "salida": "A", "base": "C"}}
usar = ((F >= pd.Timestamp(TRAIN[0])) & (F <= pd.Timestamp(TRAIN[1]))) | \
       (F >= pd.Timestamp(TEST_DESDE))
k_usar = np.where(usar)[0][:-1]
for col, cfg in VAR_TRI.items():
    v = np.full(len(F), np.nan)
    for k in k_usar:
        if k not in tris:
            tris[k] = det.detectar(k)
        v[k] = tg.senal(det, tris[k], k, cfg)[0]
    d[col] = v

d = d.iloc[:-1]                                              # sin t+1
tr = d.loc[TRAIN[0]:TRAIN[1]].dropna(subset=EXOG + ["ret_next"])
te = d.loc[TEST_DESDE:].dropna(subset=EXOG + ["ret_next"])
print(f"TRAIN {tr.index.min().date()} – {tr.index.max().date()} ({len(tr)} días) | "
      f"TEST {te.index.min().date()} – {te.index.max().date()} ({len(te)} días)")
print(f"Holdout bloqueado desde {HOLDOUT_DESDE.date()}\n")


# ----------------------------------------------------------------------
# PIEZAS DEL MODELO
# ----------------------------------------------------------------------
class Pipeline:
    """Estandariza, categoriza y estima con un conjunto de entrenamiento."""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, D):
        X = D[self.cols].to_numpy(float)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1
        r = D["ret_next"].to_numpy()
        self.cortes = np.percentile(r, CORTES_PCT)
        y = np.digitize(r, self.cortes)
        self.media_cat = np.array([r[y == j].mean() for j in range(5)])
        self.m = ProbitOrdenado(5).fit((X - self.mu) / self.sd, y)
        return self

    def e_ret(self, D):
        X = (D[self.cols].to_numpy(float) - self.mu) / self.sd
        P = self.m.predict_proba(X)
        return P @ self.media_cat, P


def posiciones(e_neto, c1, c2):
    a = np.abs(e_neto)
    return np.sign(e_neto) * np.where(a >= c2, 3, np.where(a >= c1, 1, 0))


def pnl(pos, D, costo=COSTO):
    S0 = D["c"].to_numpy()
    dpos = np.abs(np.diff(pos, prepend=0.0))
    bruto = pos * D["ret_next"].to_numpy()
    carry = pos * D["carry_u"].to_numpy()
    cst = -dpos * costo / S0
    return bruto, carry, cst, bruto + carry + cst, dpos


def calibrar(cols):
    """Walk-forward interno en TRAIN: estima hasta el mes anterior, predice el mes."""
    cal = tr.loc[CALIB_DESDE:]
    e = pd.Series(np.nan, index=cal.index)
    for mes, bloque in cal.groupby(cal.index.to_period("M")):
        # embargo: el último dato de entrenamiento tiene su t+1 antes del mes
        hist = tr.loc[:bloque.index[0] - pd.Timedelta(days=1)].iloc[:-1]
        e.loc[bloque.index] = Pipeline(cols).fit(hist).e_ret(bloque)[0]
    e_neto = e.to_numpy() + cal["carry_u"].to_numpy()
    a = np.abs(e_neto)
    mejor = (-np.inf, None, None)
    for p1 in GRID_C1:
        for p2 in GRID_C2:
            if p2 <= p1:
                continue
            c1 = np.percentile(a, p1)
            c2 = np.inf if p2 == 100 else np.percentile(a, p2)
            nt = pnl(posiciones(e_neto, c1, c2), cal)[3]
            if nt.sum() > mejor[0]:
                mejor = (nt.sum(), c1, c2, p1, p2)
    return mejor[1:]


def predecir_test(cols, modo):
    if modo == "FIJO":
        return Pipeline(cols).fit(tr).e_ret(te)[0]
    e = pd.Series(np.nan, index=te.index)
    for mes, bloque in te.groupby(te.index.to_period("M")):
        hist = pd.concat([tr, te.loc[:bloque.index[0] - pd.Timedelta(days=1)].iloc[:-1]])
        e.loc[bloque.index] = Pipeline(cols).fit(hist).e_ret(bloque)[0]
    return e.to_numpy()


# ----------------------------------------------------------------------
# MÉTRICAS
# ----------------------------------------------------------------------
def metricas(pos, D, costo=COSTO):
    br, ca, cs, nt, dp = pnl(pos, D, costo)
    mens = pd.Series(nt, index=D.index).resample("ME").sum()
    act = pd.Series(pos != 0, index=D.index).resample("ME").sum()
    eq = np.cumsum(nt)
    sd = nt.std()
    return {"bruto_%": br.sum() * 100, "carry_%": ca.sum() * 100,
            "costos_%": cs.sum() * 100, "neto_%": nt.sum() * 100,
            "sharpe": nt.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
            "maxDD_%": (eq - np.maximum.accumulate(eq)).min() * 100,
            "rotación": dp.sum(), "%días_pos": (pos != 0).mean() * 100,
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


def diebold_mariano(e1, e2, h=1):
    """DM con varianza HAC (Newey-West). H1: modelo 1 mejor (pérdida menor)."""
    dl = e1 ** 2 - e2 ** 2
    n = len(dl)
    dm = dl.mean()
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    g = [np.mean((dl[k:] - dm) * (dl[:n - k] - dm)) for k in range(L + 1)]
    v = g[0] + 2 * sum((1 - k / (L + 1)) * g[k] for k in range(1, L + 1))
    stat = dm / np.sqrt(v / n)
    return stat, norm.cdf(stat)


# ----------------------------------------------------------------------
# CORRIDA
# ----------------------------------------------------------------------
specs = {"M0": EXOG, "M1": EXOG + ["tri_RA-SB-BC-OA"], "M2": EXOG + ["tri_RA-SA-BC-OA"]}
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

print("COEFICIENTES (estimados con todo el TRAIN, regresores estandarizados)")
coef = {}
for n, cols in specs.items():
    p = Pipeline(cols).fit(tr)
    z = p.m.beta_ / p.m.se_beta_
    coef[n] = pd.Series([f"{b:+.3f} ({zz:+.1f})" for b, zz in zip(p.m.beta_, z)], index=cols)
    coef[n]["pseudo-R2"] = f"{p.m.r2_mcfadden_:.4f}"
print(pd.DataFrame(coef).fillna("").to_string())
print("(entre paréntesis: z. |z| > 2 ≈ significativo al 5%)\n")

r_te = te["ret_next"].to_numpy()
media_tr = tr["ret_next"].mean()
filas, preds, poss = {}, {}, {}
for n, cols in specs.items():
    c1, c2, p1, p2 = calibrar(cols)
    for modo in ("FIJO", "WF"):
        e = predecir_test(cols, modo)
        e_neto = e + te["carry_u"].to_numpy()
        pos = posiciones(e_neto, c1, c2)
        nom = f"{n}-{modo}"
        preds[nom], poss[nom] = e, pos
        filas[nom] = metricas(pos, te) | {"umbral_pct": f"{p1}/{p2}",
                                          "acierto_dir": np.mean(np.sign(e) == np.sign(r_te)) * 100}

# robustez: regresores exógenos rezagados un día
te_lag, tr_lag = te.copy(), tr.copy()
for D in (te_lag, tr_lag):
    D[["cobre", "dxy", "latam", "vix"]] = d[["cobre", "dxy", "latam", "vix"]].shift(1).loc[D.index]
tr0, te0 = tr, te
tr, te = tr_lag.dropna(subset=EXOG), te_lag.dropna(subset=EXOG)
c1, c2, p1, p2 = calibrar(EXOG)
e = predecir_test(EXOG, "FIJO")
pos = posiciones(e + te["carry_u"].to_numpy(), c1, c2)
filas["M0-FIJO-rezago"] = metricas(pos, te) | {"umbral_pct": f"{p1}/{p2}",
                                               "acierto_dir": np.mean(np.sign(e) == np.sign(te["ret_next"])) * 100}
tr, te = tr0, te0

bench = {"Siempre +1": np.ones(len(te)), "Siempre +3": np.full(len(te), 3.0),
         "No operar": np.zeros(len(te))}
for n, p in bench.items():
    filas[n] = metricas(p, te)

tab = pd.DataFrame(filas).T
cols_show = ["bruto_%", "carry_%", "costos_%", "neto_%", "sharpe", "maxDD_%", "rotación",
             "%días_pos", "%días_±3", "%meses>0", "%meses_act>0", "acierto_dir", "umbral_pct"]
print("=" * 120)
print(f"TEST {TEST_DESDE[:4]} – {te.index.max().date()} · costo {COSTO} CLP · "
      "M1 usa un triángulo elegido mirando este período (contaminado)")
print("=" * 120)
print(tab[cols_show].to_string(float_format=lambda x: f"{x:.2f}"))

print("\nSIGNIFICANCIA (TEST)")
ref1 = pnl(np.ones(len(te)), te)[3]
sig = {}
for nom in preds:
    nt = pnl(poss[nom], te)[3]
    s0, p0 = diebold_mariano(r_te - preds[nom], r_te - 0.0)
    sm, pm = diebold_mariano(r_te - preds[nom], r_te - media_tr)
    sig[nom] = {"DM vs paseo aleatorio": f"{s0:+.2f} (p={p0:.3f})",
                "DM vs media TRAIN": f"{sm:+.2f} (p={pm:.3f})",
                "boot neto>0": f"{p_boot(nt):.3f}",
                "boot neto>+1": f"{p_boot(nt - ref1):.3f}"}
print(pd.DataFrame(sig).T.to_string())
print("DM: estadístico negativo = el modelo tiene menor error cuadrático.")

mejor = tab.loc[list(preds), "sharpe"].astype(float).idxmax()
sens = pd.DataFrame({f"{c:.2f}": {n: metricas(poss[n], te, c)["neto_%"] for n in preds}
                     for c in COSTOS_SENS})
print("\nSENSIBILIDAD A COSTOS · neto % (CLP por USD)")
print(sens.round(2).to_string())

# ----------------------------------------------------------------------
# REGISTRO
# ----------------------------------------------------------------------
reg_path = OUT / "registro_experimentos.csv"
filas_reg = [{"timestamp": datetime.now().isoformat(timespec="seconds"),
              "experimento": f"05_probit_{n}", "desde": TEST_DESDE,
              "hasta": str(te.index.max().date()), "costo": COSTO,
              "neto_%": round(float(tab.loc[n, "neto_%"]), 3),
              "sharpe": round(float(tab.loc[n, "sharpe"]), 3),
              "pct_meses_pos": round(float(tab.loc[n, "%meses>0"]), 1),
              "elegida": False}
             for n in list(preds) + ["M0-FIJO-rezago"]]
reg = pd.DataFrame(filas_reg)
if reg_path.exists():
    reg = pd.concat([pd.read_csv(reg_path), reg], ignore_index=True)
reg.to_csv(reg_path, index=False)
print(f"\nRegistro: {len(reg)} intentos acumulados.")
pd.DataFrame({**{f"e_{n}": preds[n] for n in preds},
              **{f"pos_{n}": poss[n] for n in poss}}, index=te.index) \
  .to_csv(OUT / "exp05_predicciones_test.csv")
