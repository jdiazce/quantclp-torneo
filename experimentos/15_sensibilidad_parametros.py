"""
experimentos/15_sensibilidad_parametros.py — ¿El resultado depende de un valor exacto?

Mueve cada parámetro del modelo congelado un 20 % hacia abajo y hacia arriba, de a uno,
manteniendo el resto fijo, y mide el neto en la ventana de selección y en el holdout.
No elige nada: el modelo sigue siendo el congelado (protocolo, punto 8).

Los parámetros se cambian en memoria (tg.X = valor); src/triangulos.py no se modifica.
La contención (85 %) se mueve ±10 puntos porque +20 % superaría el 100 %.

Salidas
  experimentos/exp15_sensibilidad.csv
  informe/fig_sensibilidad.png
  salidas/exp15_cache/   posiciones de cada variante (para no recalcular; se puede borrar)

Correr con F5. La primera vez tarda ~25 min (redetecta triángulos en 26 variantes);
después lee el caché.
"""

import importlib
import functools
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAIZ = Path(os.environ.get("QUANTCLP", Path(__file__).resolve().parents[1]))
if not (RAIZ / "estrategia.py").exists():
    raise FileNotFoundError(f"{RAIZ} no es la raíz del repo")
importlib.invalidate_caches()
sys.path.insert(0, str(RAIZ / "src"))
import triangulos as tg  # noqa: E402

BASE = RAIZ / "data" / "base_usdclp.xlsx"
CACHE = RAIZ / "salidas" / "exp15_cache"
CACHE.mkdir(parents=True, exist_ok=True)
OUT_CSV = RAIZ / "experimentos" / "exp15_sensibilidad.csv"
OUT_FIG = RAIZ / "informe" / "fig_sensibilidad.png"

CFG = {"ruptura": "A", "salida": "B", "base": "C"}          # RA-SB-BC-OA (congelado)
DESDE = "2023-01-01"
SELECCION = ("2023-01-01", "2025-04-04")
HOLDOUT = "2025-04-07"
COSTO_CLP = 0.50
MOSTRAR = True

# (etiqueta, parámetro en triangulos.py, bajo, alto, ¿cambia la detección?)
VARIANTES = [
    ("Tolerancia de ruptura y toque (0,45 ATR)", "TOL_ATR", 0.36, 0.54, True),
    ("Convergencia mínima (1,3×)", "CONTRACCION_MIN", 1.04, 1.56, True),
    ("Contención mínima (85 %)", "CONTENCION_MIN", 0.75, 0.95, True),
    ("Duración mínima (25 días)", "SPAN_MIN", 20, 30, True),
    ("Duración máxima (220 días)", "SPAN_MAX", 176, 264, True),
    ("Altura mínima (1,5 ATR)", "ALTURA_MIN_ATR", 1.2, 1.8, True),
    ("Altura máxima (8 ATR)", "ALTURA_MAX_ATR", 6.4, 9.6, True),
    ("Ventana del pivote (3 días)", "W_PIVOTE", 2, 4, True),
    ("Toques mínimos (5)", "TOQUES_TOTAL", 4, 6, True),
    ("Ápice máximo (1,5× largo)", "APICE_MAX", 1.2, 1.8, True),
    ("Mecha máxima (1 ATR)", "MECHA_MAX_ATR", 0.8, 1.2, True),
    ("ATR (14 días)", "ATR_N", 11, 17, True),
    ("Agrupación de pivotes (0,5 ATR)", "CLUSTER_ATR", 0.4, 0.6, True),
    ("Apoyo en el borde (0,6 ATR)", "APOYO_MAX_ATR", 0.48, 0.72, False),
    ("Cierres para confirmar (3)", "N_CONFIRMA", 2, 4, False),
    ("Plazo máximo (60 días)", "VIGENCIA_B_MAX", 48, 72, False),
    ("Stop (1 ATR adentro)", "STOP_B_ATR", 0.8, 1.2, False),
    ("Días para volver a salir (2)", "DIAS_VOLVER", 1, 3, False),
    ("Mantención del ±3 (3 días)", "HOLD_3", 2, 4, False),
]

# ----------------------------------------------------------------------
# DATOS Y CONTABILIDAD (idéntica a estrategia.py)
# ----------------------------------------------------------------------
df = tg.cargar_usdclp(BASE)
F = df.index
idx = np.where(F >= pd.Timestamp(DESDE))[0]
tasas = (pd.read_excel(BASE, sheet_name="datos",
                       usecols=["fecha", "CHOVCHOV Index|PX_LAST", "FEDL01 Index|PX_LAST"])
           .set_axis(["fecha", "tpm", "ff"], axis=1)
           .assign(fecha=lambda x: pd.to_datetime(x["fecha"]))
           .set_index("fecha").sort_index())
tasas["ff"] = tasas["ff"].shift(1)
tasas = tasas.ffill().reindex(F, method="ffill")
Cc = df["c"].to_numpy(float)
kk = idx[:-1]
RET = Cc[kk + 1] / Cc[kk] - 1
DIAS = (F[kk + 1] - F[kk]).days.to_numpy()
CARRY = -((tasas["tpm"] - tasas["ff"]).to_numpy()[kk] / 100 / 360) * DIAS
FE = F[kk]
M_SEL = (FE >= pd.Timestamp(SELECCION[0])) & (FE <= pd.Timestamp(SELECCION[1]))
M_HOL = FE >= pd.Timestamp(HOLDOUT)


def metricas(pos):
    p = np.asarray(pos, float)[:-1]
    dp = np.abs(np.diff(p, prepend=0.0))
    neto = p * RET + p * CARRY - dp * COSTO_CLP / Cc[kk]
    out = {}
    for nom, m in (("sel", M_SEL), ("hol", M_HOL)):
        x = neto[m]
        out[f"neto_{nom}"] = x.sum() * 100
        out[f"sharpe_{nom}"] = x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
        out[f"pct_dias_{nom}"] = (p[m] != 0).mean() * 100
        out[f"rotacion_{nom}"] = dp[m].sum()
    return out


DEFAULTS = {p: getattr(tg, p) for _, p, *_ in VARIANTES if p != "ATR_N"}
DEFAULTS["ATR_N"] = tg.ATR_N


def correr(param=None, valor=None, tris_base=None):
    """Posiciones 2023→ con un parámetro cambiado (None = modelo congelado)."""
    for p, v in DEFAULTS.items():
        setattr(tg, p, v)
    if param is not None:
        setattr(tg, param, valor)
    det = tg.Detector(df)
    if param == "ATR_N":                     # n y w son argumentos por defecto: se pasan
        det.ATR = tg.atr_serie(det.H, det.L, det.C, n=valor)
    if param == "W_PIVOTE":
        det._pivotes = functools.partial(det._pivotes, w=valor)
    if tris_base is not None:
        tris = tris_base
    else:
        tris = {int(k): det.detectar(k) for k in idx}
    pos = [int(tg.senal(det, tris[int(k)], k, CFG)[0]) for k in idx]
    for p, v in DEFAULTS.items():
        setattr(tg, p, v)
    return pos, tris


def cache(nombre, fn):
    f = CACHE / f"{nombre}.json"
    if f.exists():
        return json.loads(f.read_text())
    pos = fn()
    f.write_text(json.dumps(pos))
    return pos


# ----------------------------------------------------------------------
# CORRIDAS
# ----------------------------------------------------------------------
print("Modelo congelado...")
f_tris = CACHE / "base_triangulos.pkl"
if f_tris.exists():
    tris_base = pickle.loads(f_tris.read_bytes())
    pos_base, _ = correr(tris_base=tris_base)
else:
    pos_base, tris_base = correr()
    f_tris.write_bytes(pickle.dumps(tris_base))
(CACHE / "base.json").write_text(json.dumps(pos_base))
m0 = metricas(pos_base)
print(f"  selección {m0['neto_sel']:+.2f} %  ·  holdout {m0['neto_hol']:+.2f} %")

filas = [{"parametro": "—", "etiqueta": "Modelo congelado", "lado": "base", "valor": np.nan,
          **m0}]
for etq, par, bajo, alto, geo in VARIANTES:
    for lado, v in (("bajo", bajo), ("alto", alto)):
        nombre = f"{par}_{v}"
        print(f"  {etq:<45} {lado:<4} = {v}", end=" ... ", flush=True)
        pos = cache(nombre, lambda: correr(par, v, None if geo else tris_base)[0])
        m = metricas(pos)
        print(f"sel {m['neto_sel']:+6.2f}  hol {m['neto_hol']:+6.2f}")
        filas.append({"parametro": par, "etiqueta": etq, "lado": lado, "valor": v, **m})

res = pd.DataFrame(filas)
res.to_csv(OUT_CSV, index=False)

# ----------------------------------------------------------------------
# RESUMEN
# ----------------------------------------------------------------------
var = res[res["lado"] != "base"]
UNO_HOL = metricas(np.ones(len(idx)))["neto_hol"]
UNO_SEL = metricas(np.ones(len(idx)))["neto_sel"]
print("\n" + "=" * 78)
print(f"SENSIBILIDAD · {len(var)} variantes (19 parámetros × 2)")
print("=" * 78)
for nom, col, uno in (("Selección", "neto_sel", UNO_SEL), ("Holdout", "neto_hol", UNO_HOL)):
    x = var[col]
    print(f"{nom:<10} base {m0[col]:+6.2f} · mediana {x.median():+6.2f} · rango "
          f"[{x.min():+.2f}, {x.max():+.2f}] · positivas {(x > 0).mean():.0%} · "
          f"> siempre +1 ({uno:+.2f}) {(x > uno).mean():.0%}")
tabla = (var.pivot_table(index="etiqueta", columns="lado", values=["neto_sel", "neto_hol"],
                         sort=False).round(2))
print("\n" + tabla.to_string())

# ----------------------------------------------------------------------
# FIGURA
# ----------------------------------------------------------------------
orden = [e for e, *_ in VARIANTES]
fig, axs = plt.subplots(1, 2, figsize=(13, 7.5), sharey=True)
for ax, col, tit, uno in ((axs[0], "neto_sel", "Ventana de selección (2023 – abr 2025)", UNO_SEL),
                          (axs[1], "neto_hol", "Holdout (abr 2025 – oct 2026)", UNO_HOL)):
    for i, e in enumerate(orden):
        b = var[(var["etiqueta"] == e) & (var["lado"] == "bajo")][col].iloc[0]
        a = var[(var["etiqueta"] == e) & (var["lado"] == "alto")][col].iloc[0]
        ax.plot([b, a], [i, i], color="0.7", lw=2, zorder=1)
        ax.scatter([b], [i], color="#1565c0", s=28, zorder=2, label="−20 %" if i == 0 else None)
        ax.scatter([a], [i], color="#c62828", s=28, zorder=2, label="+20 %" if i == 0 else None)
    ax.axvline(m0[col], color="#1b5e20", lw=1.5, label="modelo congelado")
    ax.axvline(uno, color="0.4", lw=1, ls="--", label="siempre +1")
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_title(f"{tit}\nmodelo congelado {m0[col]:+.1f} % · siempre +1 {uno:+.1f} %",
                 fontsize=10, loc="left")
    ax.set_xlabel("Resultado neto acumulado (%)")
axs[0].set_yticks(range(len(orden)))
axs[0].set_yticklabels(orden, fontsize=8)
axs[0].invert_yaxis()
h_, l_ = axs[1].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Sensibilidad: cada parámetro movido ±20 % con el resto fijo", fontsize=12,
             x=0.01, ha="left")
plt.tight_layout()
fig.savefig(OUT_FIG, dpi=170, bbox_inches="tight")
if MOSTRAR:
    plt.show()
plt.close(fig)
print(f"\nGuardado: {OUT_CSV.name} · {OUT_FIG.name}")
