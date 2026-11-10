"""
experimentos/17_ml_y_correlaciones.py — ¿Hay información que un modelo de ML pudiera explotar?

Dos partes, ambas SOLO con datos previos al holdout (2005–2018 y ventana de selección):

  A. Mapa de información: correlación de cada variable al cierre de t con el retorno del
     USD/CLP de t a t+1, con la banda de ±1,96/√n que tendría una variable sin información.
  B. Benchmark de ML: logit, bosque aleatorio y gradient boosting con las mismas variables,
     hiperparámetros FIJOS (sin búsqueda), walk-forward anual con ventana expansiva y
     embargo de 5 días. Posición ±1 según P(alza) > 0,5, y una versión con abstención si
     |P − 0,5| < 0,03. Se comparan con triángulos y siempre +1 en las mismas ventanas.

Es un experimento de DESCARTE: 3 modelos × 2 reglas = 6 intentos que se agregan al registro.
Se corrió después de abrir el holdout, pero el holdout no se usa.

Variables al cierre de t (la base indica que todos los cierres de una fecha están disponibles):
retorno del USD/CLP (1, 5 y 20 días), cobre, DXY, BRL, MXN, COP, S&P 500, VIX,
volatilidad emergente (JPMVXYEM), tasa del Tesoro a 10 años y diferencial TPM − Fed funds.

Salidas
  experimentos/exp17_correlaciones.csv
  experimentos/exp17_ml.csv
  informe/fig_correlaciones.png

Correr con F5 (~6 min: incluye los triángulos de 2010–2018 para comparar).
"""

import importlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(os.environ.get("QUANTCLP", Path(__file__).resolve().parents[1]))
if not (RAIZ / "estrategia.py").exists():
    raise FileNotFoundError(f"{RAIZ} no es la raíz del repo")
importlib.invalidate_caches()
sys.path.insert(0, str(RAIZ / "src"))
import triangulos as tg  # noqa: E402

BASE = RAIZ / "data" / "base_usdclp.xlsx"
OUT_COR = RAIZ / "experimentos" / "exp17_correlaciones.csv"
OUT_ML = RAIZ / "experimentos" / "exp17_ml.csv"
OUT_FIG = RAIZ / "informe" / "fig_correlaciones.png"
CACHE = RAIZ / "salidas" / "exp17_cache"                 # posiciones de triángulos (se puede borrar)
CACHE.mkdir(parents=True, exist_ok=True)

CFG = {"ruptura": "A", "salida": "B", "base": "C"}
VENTANAS = {"2010–2018": ("2010-01-01", "2018-12-31"),
            "Selección": ("2023-01-01", "2025-04-04")}
EXCLUIDA = ("2019-01-01", "2022-12-31")          # no se usa para entrenar ni evaluar
INICIO_ENTRENO = "2005-01-01"
EMBARGO = 5
UMBRALES = {"±1 siempre": 0.0, "con abstención": 0.03}
COSTO_CLP = 0.50
MOSTRAR = True

SERIES = {"Cobre": "HG1 Comdty|PX_LAST", "DXY": "DXY Curncy|PX_LAST",
          "USD/BRL": "USDBRL Curncy|PX_LAST", "USD/MXN": "USDMXN Curncy|PX_LAST",
          "USD/COP": "USDCOP Curncy|PX_LAST", "S&P 500": "SPX Index|PX_LAST",
          "VIX": "VIX Index|PX_LAST", "Vol. emergente": "JPMVXYEM Index|PX_LAST"}

# ----------------------------------------------------------------------
# DATOS: todo en el calendario del USD/CLP, con el último dato disponible al cierre de t
# ----------------------------------------------------------------------
clp = tg.cargar_usdclp(BASE)
F = clp.index
raw = (pd.read_excel(BASE, sheet_name="datos",
                     usecols=["fecha", *SERIES.values(), "USGG10YR Index|PX_LAST",
                              "CHOVCHOV Index|PX_LAST", "FEDL01 Index|PX_LAST"])
         .assign(fecha=lambda x: pd.to_datetime(x["fecha"])).set_index("fecha").sort_index())


def al_cierre(col):
    return raw[col].dropna().reindex(F, method="ffill")


lc = np.log(clp["c"])
X = pd.DataFrame(index=F)
X["USD/CLP 1d"] = lc.diff()
X["USD/CLP 5d"] = lc.diff(5)
X["USD/CLP 20d"] = lc.diff(20)
for nom, col in SERIES.items():
    X[nom] = np.log(al_cierre(col)).diff()
X["Tesoro 10A"] = al_cierre("USGG10YR Index|PX_LAST").diff()
ff = raw["FEDL01 Index|PX_LAST"].dropna().shift(1).reindex(F, method="ffill")
tpm = al_cierre("CHOVCHOV Index|PX_LAST")
X["TPM − FF"] = tpm - ff
y_ret = clp["c"].shift(-1) / clp["c"] - 1                     # retorno de t a t+1
carry = -((tpm - ff) / 100 / 360) * pd.Series(F, index=F).diff().shift(-1).dt.days
D = X.assign(ret=y_ret, carry=carry, S=clp["c"]).dropna()
D = D.loc[INICIO_ENTRENO:]
excl = (D.index >= pd.Timestamp(EXCLUIDA[0])) & (D.index <= pd.Timestamp(EXCLUIDA[1]))
D = D[~excl]
VARS = list(X.columns)


def en(v):
    a, b = VENTANAS[v]
    return (D.index >= pd.Timestamp(a)) & (D.index <= pd.Timestamp(b))


def contabilidad(p, sub):
    p = np.asarray(p, float)
    dp = np.abs(np.diff(p, prepend=0.0))
    return p * sub["ret"].to_numpy() + p * sub["carry"].to_numpy() - dp * COSTO_CLP / sub["S"].to_numpy()


# ----------------------------------------------------------------------
# A. MAPA DE INFORMACIÓN
# ----------------------------------------------------------------------
cor = []
for v in ("2005–2018", "Selección"):
    m = (D.index <= pd.Timestamp("2018-12-31")) if v == "2005–2018" else en(v)
    sub = D[m]
    banda = 1.96 / np.sqrt(len(sub))
    for x in VARS:
        r = sub[x].corr(sub["ret"])
        rs = sub[x].corr(sub["ret"], method="spearman")
        cor.append({"ventana": v, "variable": x, "pearson": r, "spearman": rs,
                    "banda_95": banda, "n": len(sub), "fuera_de_banda": abs(r) > banda})
cor = pd.DataFrame(cor)
cor.to_csv(OUT_COR, index=False)
print("A. Correlación con el retorno de t+1")
print(cor.pivot_table(index="variable", columns="ventana", values="pearson", sort=False)
         .round(3).to_string())
for v, g in cor.groupby("ventana", sort=False):
    print(f"  {v}: banda ±{g['banda_95'].iloc[0]:.3f} · fuera de banda: "
          f"{int(g['fuera_de_banda'].sum())} de {len(g)} · "
          f"R² máximo de una variable {(g['pearson'] ** 2).max() * 100:.2f} %")

# ----------------------------------------------------------------------
# B. ML WALK-FORWARD
# ----------------------------------------------------------------------
MODELOS = {
    "Logit": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
    "Bosque aleatorio": lambda: RandomForestClassifier(n_estimators=300, max_depth=4,
                                                       min_samples_leaf=50, random_state=0,
                                                       n_jobs=1),
    "Gradient boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                                max_iter=200, random_state=0),
}


def walk_forward(nombre, v):
    m_eval = en(v)
    años = sorted(set(D.index[m_eval].year))
    prob = pd.Series(np.nan, index=D.index[m_eval])
    for a in años:
        prueba = m_eval & (D.index.year == a)
        ini_prueba = D.index[prueba][0]
        entreno = np.where(D.index < ini_prueba)[0]
        entreno = entreno[:-EMBARGO]                            # embargo
        Xe, ye = D[VARS].iloc[entreno], (D["ret"].iloc[entreno] > 0).astype(int)
        mod = MODELOS[nombre]().fit(Xe, ye)
        prob[D.index[prueba]] = mod.predict_proba(D.loc[prueba, VARS])[:, 1]
    return prob


res = []
print("\nB. ML walk-forward (neto % con carry y costo 0,5 CLP)")
for v in VENTANAS:
    sub = D[en(v)]
    sube = (sub["ret"] > 0).astype(int).to_numpy()
    for nom in MODELOS:
        pr = walk_forward(nom, v).to_numpy()
        auc = roc_auc_score(sube, pr)
        for regla, tau in UMBRALES.items():
            pos = np.where(pr > 0.5 + tau, 1, np.where(pr < 0.5 - tau, -1, 0))
            n = contabilidad(pos, sub)
            act = pos != 0
            res.append({"ventana": v, "modelo": nom, "regla": regla, "auc": auc,
                        "acierto_%": (np.sign(pos[act]) == np.sign(sub["ret"].to_numpy()[act])).mean() * 100,
                        "pct_dias": act.mean() * 100, "neto_%": n.sum() * 100,
                        "sharpe": n.mean() / n.std() * np.sqrt(252) if n.std() > 0 else np.nan})
    # referencias en la misma ventana y el mismo calendario
    n1 = contabilidad(np.ones(len(sub)), sub)
    res.append({"ventana": v, "modelo": "Siempre +1", "regla": "—", "auc": np.nan,
                "acierto_%": (sub["ret"] > 0).mean() * 100, "pct_dias": 100,
                "neto_%": n1.sum() * 100, "sharpe": n1.mean() / n1.std() * np.sqrt(252)})
    ks = [F.get_loc(f) for f in sub.index]
    f_cache = CACHE / "posiciones_triangulos.pkl"
    memo = pickle.loads(f_cache.read_bytes()) if f_cache.exists() else {}
    faltan = [k for k in ks if k not in memo]
    if faltan:
        det = tg.Detector(clp)
        print(f"  triángulos en {v}: {len(faltan)} días por calcular...", flush=True)
        for k in faltan:
            memo[k] = float(tg.senal(det, det.detectar(k), k, CFG)[0])
        f_cache.write_bytes(pickle.dumps(memo))
    pt = np.array([memo[k] for k in ks], float)
    nt = contabilidad(pt, sub)
    act = pt != 0
    res.append({"ventana": v, "modelo": "Triángulos", "regla": "—", "auc": np.nan,
                "acierto_%": (np.sign(pt[act]) == np.sign(sub["ret"].to_numpy()[act])).mean() * 100,
                "pct_dias": act.mean() * 100, "neto_%": nt.sum() * 100,
                "sharpe": nt.mean() / nt.std() * np.sqrt(252)})
res = pd.DataFrame(res)
res.to_csv(OUT_ML, index=False)
print(res.round(3).to_string(index=False))

# ----------------------------------------------------------------------
# FIGURA: MAPA DE INFORMACIÓN
# ----------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, (v, g) in zip(axs, cor.groupby("ventana", sort=False)):
    yy = np.arange(len(g))
    b = g["banda_95"].iloc[0]
    ax.axvspan(-b, b, color="0.88", label="zona sin información (±1,96/√n)")
    col = np.where(g["fuera_de_banda"], "#c62828", "#1565c0")
    ax.barh(yy, g["pearson"], color=col)
    ax.axvline(0, color="0.4", lw=0.6)
    ax.set_title(f"{v} (n = {g['n'].iloc[0]})", fontsize=10, loc="left")
    ax.set_xlabel("Correlación con el retorno del USD/CLP de t a t+1")
axs[0].set_yticks(np.arange(len(VARS)))
axs[0].set_yticklabels(VARS, fontsize=9)
axs[0].invert_yaxis()
axs[0].legend(fontsize=8, loc="lower left")
lim = max(0.08, cor["pearson"].abs().max() * 1.15)
for ax in axs:
    ax.set_xlim(-lim, lim)
fig.suptitle("Mapa de información: correlaciones pequeñas (R² ≤ 2,4 %) e inestables entre ventanas",
             fontsize=11, x=0.01, ha="left")
plt.tight_layout()
fig.savefig(OUT_FIG, dpi=170, bbox_inches="tight")
if MOSTRAR:
    plt.show()
plt.close(fig)
print(f"\nGuardado: {OUT_COR.name} · {OUT_ML.name} · {OUT_FIG.name}")
