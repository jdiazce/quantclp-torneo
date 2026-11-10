"""
experimentos/16_robustez_ejecucion_azar.py — Cuatro preguntas que haría un jurado.

  A. ¿Y si ejecuto tarde?            Retraso de 1 a 3 días en aplicar la posición.
  B. ¿Es el timing o la exposición?  Placebos: misma secuencia de posiciones desplazada en el
                                     tiempo, y dirección al azar en cada operación.
  C. ¿Depende del bloque elegido?    Bootstrap estacionario con bloques de 5, 10 y 20 días.
  D. ¿Cuánto infla la búsqueda?      Sharpe deflactado (Bailey y López de Prado, 2014) de la
                                     ventana de selección con N = 24 y N = 56 intentos.
  E. ¿Cómo son las operaciones?      Número, % ganadoras, ganancia y pérdida media, duración.

No modifica el modelo ni elige nada. Usa la misma contabilidad que estrategia.py.

Salidas
  experimentos/exp16_robustez.csv       (A, B, C, D en formato largo)
  experimentos/exp16_operaciones.csv    (E, una fila por operación)
  informe/fig_placebo.png

Correr con F5 (~3 min).
"""

import importlib
import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

RAIZ = Path(os.environ.get("QUANTCLP", Path(__file__).resolve().parents[1]))
if not (RAIZ / "estrategia.py").exists():
    raise FileNotFoundError(f"{RAIZ} no es la raíz del repo")
importlib.invalidate_caches()
sys.path.insert(0, str(RAIZ / "src"))
import triangulos as tg  # noqa: E402

BASE = RAIZ / "data" / "base_usdclp.xlsx"
OUT_CSV = RAIZ / "experimentos" / "exp16_robustez.csv"
OUT_OPS = RAIZ / "experimentos" / "exp16_operaciones.csv"
OUT_FIG = RAIZ / "informe" / "fig_placebo.png"

CFG = {"ruptura": "A", "salida": "B", "base": "C"}
DESDE = "2023-01-01"
SELECCION = ("2023-01-01", "2025-04-04")
HOLDOUT = "2025-04-07"
COSTO_CLP = 0.50
N_BOOT, N_SIGNO, SEMILLA = 10_000, 10_000, 12
MOSTRAR = True

# ----------------------------------------------------------------------
# DATOS, SEÑALES Y CONTABILIDAD
# ----------------------------------------------------------------------
df = tg.cargar_usdclp(BASE)
det = tg.Detector(df)
F = df.index
idx = np.where(F >= pd.Timestamp(DESDE))[0]
print(f"Detectando triángulos en {len(idx)} días...")
TRIS = {int(k): det.detectar(k) for k in idx}
POS = np.array([tg.senal(det, TRIS[int(k)], k, CFG)[0] for k in idx], float)

tasas = (pd.read_excel(BASE, sheet_name="datos",
                       usecols=["fecha", "CHOVCHOV Index|PX_LAST", "FEDL01 Index|PX_LAST"])
           .set_axis(["fecha", "tpm", "ff"], axis=1)
           .assign(fecha=lambda x: pd.to_datetime(x["fecha"]))
           .set_index("fecha").sort_index())
tasas["ff"] = tasas["ff"].shift(1)
tasas = tasas.ffill().reindex(F, method="ffill")
kk = idx[:-1]
S0 = det.C[kk]
RET = det.C[kk + 1] / S0 - 1
CARRY = -((tasas["tpm"] - tasas["ff"]).to_numpy()[kk] / 100 / 360) * (F[kk + 1] - F[kk]).days.to_numpy()
FE = F[kk]
VENT = {"Selección": (FE >= pd.Timestamp(SELECCION[0])) & (FE <= pd.Timestamp(SELECCION[1])),
        "Holdout": FE >= pd.Timestamp(HOLDOUT)}
P = POS[:-1]


def neto(p, m=None):
    """Neto diario; con m, recalcula solo dentro de la ventana (posición previa = 0)."""
    p = np.asarray(p, float)
    if m is not None:
        r, c, s = RET[m], CARRY[m], S0[m]
    else:
        r, c, s = RET, CARRY, S0
    dp = np.abs(np.diff(p, prepend=0.0))
    return p * r + p * c - dp * COSTO_CLP / s


filas = []


def anotar(bloque, prueba, ventana, valor, p=np.nan, nota=""):
    filas.append({"bloque": bloque, "prueba": prueba, "ventana": ventana,
                  "valor": valor, "p": p, "nota": nota})


# ----------------------------------------------------------------------
# A. RETRASO DE EJECUCIÓN
# ----------------------------------------------------------------------
print("\nA. Retraso de ejecución (neto %)")
for lag in range(4):
    pl = np.r_[np.zeros(lag), P[:len(P) - lag]] if lag else P
    n = neto(pl)
    txt = []
    for v, m in VENT.items():
        anotar("A retraso", f"{lag} días", v, n[m].sum() * 100)
        txt.append(f"{v} {n[m].sum() * 100:+6.2f}")
    print(f"  {lag} días: " + " · ".join(txt))

# ----------------------------------------------------------------------
# B. PLACEBOS
# ----------------------------------------------------------------------
rng = np.random.default_rng(SEMILLA)
dist = {}
print("\nB. Placebos (p unilateral: share de placebos ≥ resultado real)")
for v, m in VENT.items():
    pv = P[m]
    real = neto(pv, m).sum() * 100
    circ = np.array([neto(np.roll(pv, s), m).sum() * 100 for s in range(1, len(pv))])
    ep = np.cumsum(np.r_[1, np.diff(pv) != 0])               # episodios de posición constante
    tr = np.cumsum(np.r_[1, np.diff(np.sign(pv)) != 0])       # operaciones (mismo signo)
    sig = np.array([neto(pv * rng.choice([-1, 1], size=tr.max() + 1)[tr], m).sum() * 100
                    for _ in range(N_SIGNO)])
    dist[v] = (real, circ, sig)
    anotar("B placebo", "desplazamiento circular", v, circ.mean(), (circ >= real).mean(),
           f"n={len(circ)}; real={real:.2f}")
    anotar("B placebo", "signo al azar por operación", v, sig.mean(), (sig >= real).mean(),
           f"n={N_SIGNO}; real={real:.2f}")
    print(f"  {v:<10} real {real:+6.2f} · circular: media {circ.mean():+6.2f}, "
          f"p = {(circ >= real).mean():.3f} · signo al azar: media {sig.mean():+6.2f}, "
          f"p = {(sig >= real).mean():.3f}")


# ----------------------------------------------------------------------
# C. BOOTSTRAP ESTACIONARIO CON DISTINTOS BLOQUES (holdout)
# ----------------------------------------------------------------------
def boot_estacionario(d, bloque, n_rep, rng):
    """p unilateral de H0: media(d) <= 0 (Politis y Romano, 1994), remuestreando d centrada."""
    T = len(d)
    dc = d - d.mean()
    q = 1 / bloque
    ind = np.empty((n_rep, T), int)
    ind[:, 0] = rng.integers(0, T, n_rep)
    nuevo = rng.random((n_rep, T)) < q
    salto = rng.integers(0, T, (n_rep, T))
    for t in range(1, T):
        ind[:, t] = np.where(nuevo[:, t], salto[:, t], (ind[:, t - 1] + 1) % T)
    medias = dc[ind].mean(1)
    return float(np.mean(medias >= d.mean()))


print("\nC. Bootstrap del neto en el holdout (p unilateral)")
m = VENT["Holdout"]
n_tri = neto(P[m], m)
REF = {"No operar": np.zeros(m.sum()), "Siempre +1": np.ones(m.sum()),
       "Comprar siempre (+3)": np.full(m.sum(), 3.0)}
for b in (5, 10, 20):
    txt = []
    for nom, pr in REF.items():
        p = boot_estacionario(n_tri - neto(pr, m), b, N_BOOT, np.random.default_rng(SEMILLA))
        anotar("C bootstrap", f"vs {nom}", "Holdout", b, p, "valor = largo medio del bloque")
        txt.append(f"vs {nom} {p:.3f}")
    print(f"  bloque {b:>2}: " + " · ".join(txt))


# ----------------------------------------------------------------------
# D. SHARPE DEFLACTADO
# ----------------------------------------------------------------------
def sharpe_deflactado(x, sr_trials_var, n_trials):
    """Bailey y López de Prado (2014). Sharpe diario (no anualizado)."""
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    em = 0.5772156649
    sr0 = np.sqrt(sr_trials_var) * ((1 - em) * stats.norm.ppf(1 - 1 / n_trials)
                                    + em * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    z = (sr - sr0) * np.sqrt(len(x) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf(z)), float(sr0 * np.sqrt(252)), float(sr * np.sqrt(252))


print("\nD. Sharpe deflactado (ventana de selección)")
ms = VENT["Selección"]
sr_var = []
for r, s, b in product("AB", "AB", "ABC"):                    # las 12 variantes con O = A
    cfg = {"ruptura": r, "salida": s, "base": b}
    pv = np.array([tg.senal(det, TRIS[int(k)], k, cfg)[0] for k in idx], float)[:-1]
    x = neto(pv[ms], ms)
    sr_var.append(x.mean() / x.std(ddof=1) if x.std() > 0 else 0.0)
V = float(np.var(sr_var, ddof=1))
x_sel = neto(P[ms], ms)
for N in (24, 56):
    dsr, sr0a, sra = sharpe_deflactado(x_sel, V, N)
    anotar("D Sharpe deflactado", f"N = {N}", "Selección", dsr, np.nan,
           f"Sharpe anual {sra:.2f}; umbral esperado por azar {sr0a:.2f}; "
           f"var. entre 12 variantes O=A")
    print(f"  N = {N}: Sharpe {sra:.2f} vs. máximo esperado por azar {sr0a:.2f} → "
          f"DSR = {dsr:.3f}")
x_hol = neto(P[VENT['Holdout']], VENT["Holdout"])
psr = sharpe_deflactado(x_hol, 0.0, 2)[0]       # con varianza 0 el umbral es 0: PSR
anotar("D Sharpe deflactado", "PSR (N = 1)", "Holdout", psr, np.nan,
       "probabilidad de que el Sharpe verdadero sea > 0")
print(f"  Holdout (un solo modelo): P(Sharpe verdadero > 0) = {psr:.3f}")

# ----------------------------------------------------------------------
# E. OPERACIONES
# ----------------------------------------------------------------------
n_all = neto(P)
ops = []
signo = np.sign(P)
i = 0
while i < len(P):
    if signo[i] == 0:
        i += 1
        continue
    j = i
    while j + 1 < len(P) and signo[j + 1] == signo[i]:
        j += 1
    pnl = n_all[i:j + 1].sum()
    if j + 1 < len(P) and P[j + 1] == 0:                       # costo de cierre
        pnl += n_all[j + 1]
    ops.append({"inicio": FE[i].date(), "fin": FE[j].date(), "lado": int(signo[i]),
                "max_pos": int(np.abs(P[i:j + 1]).max()), "dias": j - i + 1,
                "neto_pct": pnl * 100,
                "ventana": "Holdout" if FE[i] >= pd.Timestamp(HOLDOUT) else
                ("Selección" if FE[i] <= pd.Timestamp(SELECCION[1]) else "—")})
    i = j + 1
ops = pd.DataFrame(ops)
ops.to_csv(OUT_OPS, index=False)


def resumen_ops(o):
    g, p_ = o[o["neto_pct"] > 0], o[o["neto_pct"] <= 0]
    return pd.Series({
        "operaciones": len(o), "% ganadoras": len(g) / len(o) * 100,
        "ganancia media %": g["neto_pct"].mean(), "pérdida media %": p_["neto_pct"].mean(),
        "ganancia/pérdida": g["neto_pct"].mean() / -p_["neto_pct"].mean(),
        "factor de beneficio": g["neto_pct"].sum() / -p_["neto_pct"].sum(),
        "duración media (días)": o["dias"].mean(), "duración mediana": o["dias"].median(),
        "mejor %": o["neto_pct"].max(), "peor %": o["neto_pct"].min(),
        "largas": (o["lado"] > 0).sum(), "cortas": (o["lado"] < 0).sum()})


print("\nE. Operaciones (operación = días seguidos con posición del mismo signo)")
tab_ops = pd.DataFrame({v: resumen_ops(ops[ops["ventana"] == v]) for v in VENT}).T
print(tab_ops.round(2).to_string())
for v, r in tab_ops.iterrows():
    for c, val in r.items():
        anotar("E operaciones", c, v, val)

pd.DataFrame(filas).to_csv(OUT_CSV, index=False)

# ----------------------------------------------------------------------
# FIGURA: PLACEBOS
# ----------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, v in zip(axs, VENT):
    real, circ, sig = dist[v]
    ax.hist(sig, bins=60, color="0.75", density=True, label="dirección al azar por operación")
    ax.hist(circ, bins=40, color="#1565c0", alpha=0.45, density=True,
            label="misma secuencia, desplazada en el tiempo")
    ax.axvline(real, color="#1b5e20", lw=2, label=f"estrategia ({real:+.1f} %)".replace(".", ","))
    ax.set_title(f"{v}: p = {(circ >= real).mean():.3f} (desplazada) · "
                 f"{(sig >= real).mean():.3f} (al azar)".replace(".", ","), fontsize=10, loc="left")
    ax.set_xlabel("Resultado neto (%)")
    ax.set_yticks([])
axs[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Placebos: misma exposición, rotación y duración; distinto momento o dirección",
             fontsize=11, x=0.01, ha="left")
plt.tight_layout()
fig.savefig(OUT_FIG, dpi=170, bbox_inches="tight")
if MOSTRAR:
    plt.show()
plt.close(fig)
print(f"\nGuardado: {OUT_CSV.name} · {OUT_OPS.name} · {OUT_FIG.name}")
