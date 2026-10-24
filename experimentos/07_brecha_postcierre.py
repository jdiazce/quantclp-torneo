"""
EXP 07 — Modelo de "brecha post-cierre" con probit ordenado (EMV).

IDEA ECONÓMICA
  El USD/CLP de la base cierra en Santiago (Regional BGN onshore). ZAR, MXN, DXY,
  S&P 500 y VIX cierran después, en Nueva York. Lo que pasa en el mundo entre el
  cierre de Santiago y el de Nueva York todavía no está en el precio del peso y se
  incorpora al día siguiente. El diccionario autoriza usar todos los cierres de la
  misma fecha ("usen los cierres tal como vienen").

REGRESORES (todos al cierre de t, retornos log estandarizados con el TRAIN)
  Post-cierre (Nueva York): ZAR, MXN, DXY, SPX, VIX (cambio log), cobre COMEX
  Misma sesión (región/Santiago): BRL, COP, IPSA y el propio retorno del CLP
  (el modelo aprende la brecha: cuánto se movió el mundo vs. cuánto el peso).

DOS ESPECIFICACIONES (no se elige entre ellas: S2 es la prueba de robustez)
  S1  según las reglas: cierres de t tal como vienen.
  S2  ejecutable en la vida real: las series de Nueva York entran rezagadas un día.
      Lo que sobreviva en S2 no depende del desfase horario.

CARRY (definición de las bases): TPM (CHOVCHOV, Santiago) − Fed funds (FEDL01,
  publicado el día siguiente → rezago 1). Largo paga, corto cobra. Base act/360.

POSICIÓN
  E[r] = Σ P(j)·media_r(j). Se opera con E[r] + carry contra c1 < c2.
  Histéresis opcional: una posición abierta se mantiene mientras E_neto conserve
  el signo y |E_neto| ≥ c1/2 (menos rotación). Umbrales e histéresis se fijan en
  CALIB (walk-forward 2015-2019), nunca mirando el TEST.

PERÍODOS
  TRAIN 2010-01-01 – 2019-10-17 (CALIB 2015-01 – 2019-10)
  TEST  2023-01-01 – inicio del holdout · HOLDOUT (18 meses) bloqueado.

SALIDAS: decomposición bruto/carry/costos/rotación, % meses ganadores, DM,
  bootstrap, calibración de P(sube) (Brier y tabla de confiabilidad), sensibilidad
  a costos y la señal del último día con su lectura de convicción.

Correr con F5 en Spyder. Requiere src/triangulos.py (carga de datos) y
src/probit_ordenado.py.
"""

import importlib
import os
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

TRAIN = ("2010-01-01", "2019-10-17")
CALIB_DESDE = "2015-01-01"
TEST_DESDE = "2023-01-01"
HOLDOUT_MESES = 18
CORTES_PCT = [10, 30, 70, 90]
COSTO = 0.50
COSTOS_SENS = [0.0, 0.25, 0.50, 1.00, 2.00]
N_BOOT, BLOQUE = 5000, 10
GRID_C1 = [70, 80, 90, 95, 97.5]        # percentiles de |E_neto| en CALIB
GRID_C2 = [95, 97.5, 99, 100]           # 100 = nunca ±3
GRID_HIST = [False, True]

NY = {"USDZAR Curncy|PX_LAST": "zar", "USDMXN Curncy|PX_LAST": "mxn",
      "DXY Curncy|PX_LAST": "dxy", "SPX Index|PX_LAST": "spx",
      "VIX Index|PX_LAST": "vix", "HG1 Comdty|PX_LAST": "cobre"}
LOCAL = {"USDBRL Curncy|PX_LAST": "brl", "USDCOP Curncy|PX_LAST": "cop",
         "IPSA Index|PX_LAST": "ipsa"}
TASAS = {"CHOVCHOV Index|PX_LAST": "tpm", "FEDL01 Index|PX_LAST": "ff"}
EXOG = list(NY.values()) + list(LOCAL.values()) + ["clp"]

# ----------------------------------------------------------------------
# DATOS
# ----------------------------------------------------------------------
clp = tg.cargar_usdclp(BASE)
HOLDOUT_DESDE = clp.index.max() - pd.DateOffset(months=HOLDOUT_MESES)
clp = clp[clp.index < HOLDOUT_DESDE]
F = clp.index

cols = {**NY, **LOCAL, **TASAS}
ex = (pd.read_excel(BASE, sheet_name="datos", usecols=["fecha", *cols])
        .rename(columns=cols)
        .assign(fecha=lambda x: pd.to_datetime(x["fecha"]))
        .set_index("fecha").sort_index())
ex = ex[ex.index < HOLDOUT_DESDE]
ex["ff"] = ex["ff"].shift(1)                  # Fed funds se publica al día siguiente
ex = ex.ffill().reindex(F, method="ffill")    # último dato disponible en cada día CLP

base = pd.DataFrame(index=F)
for c in list(NY.values()) + list(LOCAL.values()):
    base[c] = np.log(ex[c]).diff()
base["clp"] = np.log(clp["c"]).diff()
base["c"] = clp["c"]
base["ret_next"] = clp["c"].shift(-1) / clp["c"] - 1
dias = pd.Series(F, index=F).diff().shift(-1).dt.days
base["carry_u"] = -(ex["tpm"] - ex["ff"]) / 100 / 360 * dias   # 1 unidad larga
base = base.iloc[:-1]


def armar(spec):
    D = base.copy()
    if spec == "S2":
        D[list(NY.values())] = D[list(NY.values())].shift(1)
    D = D.dropna(subset=EXOG + ["ret_next", "carry_u"])
    return D.loc[TRAIN[0]:TRAIN[1]], D.loc[TEST_DESDE:]


# ----------------------------------------------------------------------
# MODELO
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
        # fracción de días al alza dentro de cada categoría (para P(sube))
        self.frac_up = np.array([(r[y == j] > 0).mean() for j in range(5)])
        self.m = ProbitOrdenado(5).fit((X - self.mu) / self.sd, y)
        return self

    def predecir(self, D):
        P = self.m.predict_proba((D[EXOG].to_numpy(float) - self.mu) / self.sd)
        return P @ self.media_cat, P @ self.frac_up


def posiciones(e_neto, c1, c2, hist):
    a = np.abs(e_neto)
    s = np.sign(e_neto)
    pos = s * np.where(a >= c2, 3, np.where(a >= c1, 1, 0))
    if hist:
        for i in range(1, len(pos)):
            if pos[i] == 0 and pos[i - 1] != 0 and s[i] == np.sign(pos[i - 1]) \
                    and a[i] >= c1 / 2:
                pos[i] = np.sign(pos[i - 1])           # mantiene ±1
    return pos


def pnl(pos, D, costo=COSTO):
    dpos = np.abs(np.diff(pos, prepend=0.0))
    bruto = pos * D["ret_next"].to_numpy()
    carry = pos * D["carry_u"].to_numpy()
    cst = -dpos * costo / D["c"].to_numpy()
    return bruto, carry, cst, bruto + carry + cst, dpos


def correr(spec):
    tr, te = armar(spec)
    cal = tr.loc[CALIB_DESDE:]
    e_cal = np.full(len(cal), np.nan)
    pu_cal = np.full(len(cal), np.nan)
    pos_idx = np.arange(len(cal))
    for _, ii in pd.Series(pos_idx, index=cal.index).groupby(cal.index.to_period("M")):
        bloque = cal.iloc[ii.to_numpy()]
        hist = tr.loc[:bloque.index[0] - pd.Timedelta(days=1)].iloc[:-1]   # embargo
        e_cal[ii.to_numpy()], pu_cal[ii.to_numpy()] = Pipeline().fit(hist).predecir(bloque)
    en_cal = e_cal + cal["carry_u"].to_numpy()
    a = np.abs(en_cal)
    mejor = (-np.inf,)
    for p1 in GRID_C1:
        for p2 in GRID_C2:
            if p2 <= p1:
                continue
            for h in GRID_HIST:
                c1 = np.percentile(a, p1)
                c2 = np.inf if p2 == 100 else np.percentile(a, p2)
                s = pnl(posiciones(en_cal, c1, c2, h), cal)[3].sum()
                if s > mejor[0]:
                    mejor = (s, c1, c2, p1, p2, h)
    _, c1, c2, p1, p2, h = mejor
    mod = Pipeline().fit(tr)
    e_te, pu_te = mod.predecir(te)
    return {"tr": tr, "cal": cal, "te": te, "mod": mod,
            "e_cal": e_cal, "pu_cal": pu_cal, "e_te": e_te, "pu_te": pu_te,
            "pos_cal": posiciones(en_cal, c1, c2, h),
            "pos_te": posiciones(e_te + te["carry_u"].to_numpy(), c1, c2, h),
            "c1": c1, "c2": c2, "umbral": f"p{p1}/p{p2}" + (" +hist" if h else "")}


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
    """H1: el modelo 1 tiene menor pérdida. Varianza HAC de Newey-West."""
    dl = e1 - e2
    n = len(dl)
    dm = dl.mean()
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    g = [np.mean((dl[k:] - dm) * (dl[:n - k] - dm)) for k in range(L + 1)]
    v = g[0] + 2 * sum((1 - k / (L + 1)) * g[k] for k in range(1, L + 1))
    s = dm / np.sqrt(v / n)
    return s, norm.cdf(s)


def confiabilidad(pu, r, n_bins=5):
    up = (r > 0).astype(float)
    t = pd.DataFrame({"P(sube)": pu, "sube": up})
    t["tramo"] = pd.qcut(t["P(sube)"], n_bins, duplicates="drop")
    return t.groupby("tramo", observed=True).agg(
        días=("sube", "size"), P_media=("P(sube)", "mean"), frecuencia_real=("sube", "mean"))


def conviccion(p_up, pos):
    if pos == 0:
        return "BAJA — el modelo no ve ventaja suficiente: fuera de mercado"
    lado = "comprar dólar" if pos > 0 else "vender dólar"
    nivel = "ALTA" if abs(pos) == 3 else "MEDIA"
    return f"{nivel} — {lado} (P(sube) = {p_up:.1%})"


# ----------------------------------------------------------------------
# CORRIDA
# ----------------------------------------------------------------------
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
R = {s: correr(s) for s in ("S1", "S2")}
tr, te = R["S1"]["tr"], R["S1"]["te"]
print(f"TRAIN {tr.index.min().date()} – {tr.index.max().date()} ({len(tr)} días) · "
      f"TEST {te.index.min().date()} – {te.index.max().date()} ({len(te)} días) · "
      f"holdout bloqueado desde {HOLDOUT_DESDE.date()}\n")

print("COEFICIENTES (TRAIN completo, regresores estandarizados) · z entre paréntesis")
coef = {s: pd.Series([f"{b:+.3f} ({b / se:+.1f})" for b, se in
                      zip(R[s]["mod"].m.beta_, R[s]["mod"].m.se_beta_)], index=EXOG)
        for s in R}
tabc = pd.DataFrame(coef)
tabc.loc["pseudo-R2"] = [f"{R[s]['mod'].m.r2_mcfadden_:.4f}" for s in R]
tabc.loc["umbrales (CALIB)"] = [R[s]["umbral"] for s in R]
print(tabc.to_string())
print("S1 = reglas del torneo · S2 = series de Nueva York rezagadas un día\n")

cols_show = ["bruto_%", "carry_%", "costos_%", "neto_%", "neto_anual_%", "sharpe",
             "maxDD_%", "rot_anual", "%días_pos", "%días_±3", "%meses>0", "%meses_act>0"]
for per in ("cal", "te"):
    D = R["S1"][per]
    filas = {f"Modelo {s}": metricas(R[s][f"pos_{per}"], R[s][per]) for s in R}
    filas |= {"Siempre +1": metricas(np.ones(len(D)), D),
              "Siempre +3": metricas(np.full(len(D), 3.0), D),
              "No operar": metricas(np.zeros(len(D)), D)}
    nombre = "CALIB 2015–2019 (walk-forward)" if per == "cal" else "TEST 2023 – holdout"
    print("=" * 130)
    print(f"{nombre} · costo {COSTO} CLP · carry overnight TPM − Fed funds")
    print("=" * 130)
    print(pd.DataFrame(filas).T[cols_show].to_string(float_format=lambda x: f"{x:.2f}"))
    print()

print("SIGNIFICANCIA (TEST)")
for s in R:
    D, e, pu, pos = R[s]["te"], R[s]["e_te"], R[s]["pu_te"], R[s]["pos_te"]
    r = D["ret_next"].to_numpy()
    up = (r > 0).astype(float)
    clima = R[s]["tr"]["ret_next"].gt(0).mean()
    dm1, p1 = diebold_mariano((r - e) ** 2, r ** 2)
    dm2, p2 = diebold_mariano((up - pu) ** 2, (up - clima) ** 2)
    nt = pnl(pos, D)[3]
    ref1 = pnl(np.ones(len(D)), D)[3]
    print(f"  {s}: DM error cuadrático vs paseo aleatorio {dm1:+.2f} (p={p1:.3f}) · "
          f"DM Brier vs climatología {dm2:+.2f} (p={p2:.3f}) · "
          f"bootstrap p(neto≤0) {p_boot(nt):.3f} · p(no le gana a +1) {p_boot(nt - ref1):.3f}")
print("  DM negativo = el modelo predice mejor. Brier: P(sube) vs frecuencia de alzas del TRAIN.\n")

print("CALIBRACIÓN DE P(sube) · TEST · S1")
print(confiabilidad(R["S1"]["pu_te"], R["S1"]["te"]["ret_next"].to_numpy())
      .round(3).to_string())

sens = pd.DataFrame({f"{c:.2f}": {f"Modelo {s}": metricas(R[s]["pos_te"], R[s]["te"], c)["neto_%"]
                                  for s in R} for c in COSTOS_SENS})
print("\nSENSIBILIDAD A COSTOS · TEST · neto % (CLP por USD)")
print(sens.round(2).to_string())

# señal del último día disponible (antes del holdout)
s = "S1"
f_ult = R[s]["te"].index[-1]
p_up, pos_ult = R[s]["pu_te"][-1], int(R[s]["pos_te"][-1])
print(f"\nSEÑAL DEL DÍA ({f_ult.date()}, modelo {s}): posición {pos_ult:+d} · "
      f"E[r] = {R[s]['e_te'][-1] * 1e4:+.1f} bps · convicción {conviccion(p_up, pos_ult)}")

# ----------------------------------------------------------------------
# REGISTRO
# ----------------------------------------------------------------------
reg_path = OUT / "registro_experimentos.csv"
filas = []
for s in R:
    m = metricas(R[s]["pos_te"], R[s]["te"])
    filas.append({"timestamp": datetime.now().isoformat(timespec="seconds"),
                  "experimento": f"07_brecha_{s}", "desde": TEST_DESDE,
                  "hasta": str(R[s]["te"].index.max().date()), "costo": COSTO,
                  "neto_%": round(m["neto_%"], 3), "sharpe": round(m["sharpe"], 3),
                  "pct_meses_pos": round(m["%meses>0"], 1), "elegida": s == "S1"})
reg = pd.DataFrame(filas)
if reg_path.exists():
    reg = pd.concat([pd.read_csv(reg_path), reg], ignore_index=True)
reg.to_csv(reg_path, index=False)
print(f"\nRegistro: {len(reg)} intentos acumulados.")
pd.DataFrame({f"{k}_{s}": R[s][k] for s in R for k in ("e_te", "pu_te", "pos_te")},
             index=te.index).to_csv(OUT / "exp07_predicciones_test.csv")
