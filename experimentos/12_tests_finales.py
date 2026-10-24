"""
EXP 12 — Tests exigidos por las bases para el triángulo RA-SB-BC-OA (congelado):

  1. Rentabilidad neta de carry y costos, reportada ANTES y DESPUÉS de costos.
  2. Diebold-Mariano sobre la calidad de la predicción, contra las tres referencias.
  3. Bootstrap no paramétrico (estacionario, por bloques) sobre el resultado,
     contra las tres referencias.

CÓMO SE CONVIERTE LA POSICIÓN EN PREDICCIÓN (para el Diebold-Mariano)
  Pérdida direccional: 1 si la dirección predicha falla, 0 si acierta, 0,5 si la
    predicción es 0 (abstenerse equivale a lanzar una moneda).
      · Triángulo: dirección = signo de la posición.
      · Comprar siempre / siempre +1: predicen siempre alza.
      · No operar: predice 0 todos los días (pérdida 0,5 constante).
  Pérdida cuadrática: predicción puntual = posición × k, con k = retorno absoluto
    medio de 2023 – mar 2025 (fijado ANTES del holdout). Referencias: +1 → k,
    +3 → 3k, no operar → 0 (paseo aleatorio).
  DM con varianza HAC (Newey-West) y corrección de Harvey-Leybourne-Newbold.
  Estadístico negativo = el triángulo predice mejor.

REFERENCIAS
  "Comprar siempre" se toma como +3 (exposición máxima) y "siempre +1" como +1.
  Si el laboratorio define "comprar siempre" distinto, cambiar POS_COMPRAR.

Tramos: holdout completo, sus dos partes y 2023 – mar 2025 (usado para elegir,
solo como referencia). Costo 0,5 CLP; carry TPM − Fed funds (rezago 1).
Correr con F5 en Spyder. Requiere src/triangulos.py.
"""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

RAIZ = Path(os.environ.get("QUANTCLP", Path(__file__).resolve().parents[1]))
if not (RAIZ / "src" / "triangulos.py").exists():
    raise FileNotFoundError(f"No encuentro {RAIZ / 'src' / 'triangulos.py'}. "
                            "Copia triangulos.py a la carpeta src/ del proyecto.")
importlib.invalidate_caches()
sys.path.insert(0, str(RAIZ / "src"))
import triangulos as tg  # noqa: E402

BASE = next(p for p in (RAIZ / "data" / "base_usdclp.xlsx",
                        RAIZ / "base_usdclp.xlsx") if p.exists())
OUT = RAIZ / "experimentos"
OUT.mkdir(parents=True, exist_ok=True)
CFG = {"ruptura": "A", "salida": "B", "base": "C"}
COSTO = 0.50
COSTOS_SENS = [0.0, 0.25, 0.50, 1.00, 2.00]
POS_COMPRAR = 3.0
N_BOOT, BLOQUE = 10000, 10

clp = tg.cargar_usdclp(BASE)
HOLDOUT_DESDE = clp.index.max() - pd.DateOffset(months=18)
PARTE2 = pd.Timestamp("2026-01-01")
det = tg.Detector(clp)
F = clp.index
ex = (pd.read_excel(BASE, sheet_name="datos",
                    usecols=["fecha", "CHOVCHOV Index|PX_LAST", "FEDL01 Index|PX_LAST"])
        .set_axis(["fecha", "tpm", "ff"], axis=1)
        .assign(fecha=lambda x: pd.to_datetime(x["fecha"]))
        .set_index("fecha").sort_index())
ex["ff"] = ex["ff"].shift(1)
ex = ex.ffill().reindex(F, method="ffill")

idx = np.where(F >= pd.Timestamp("2023-01-01"))[0][:-1]
print(f"Calculando señales para {len(idx)} días...")
pos = np.array([tg.senal(det, det.detectar(k), k, CFG)[0] for k in idx], float)

S0 = det.C[idx]
ret = det.C[idx + 1] / S0 - 1
dias = (F[idx + 1] - F[idx]).days.to_numpy()
carry_u = -((ex["tpm"] - ex["ff"]).to_numpy()[idx] / 100 / 360) * dias
fe = F[idx]

ESTR = {"Triángulo": pos, "Comprar siempre (+3)": np.full(len(idx), POS_COMPRAR),
        "Siempre +1": np.ones(len(idx)), "No operar": np.zeros(len(idx))}
REFS = [n for n in ESTR if n != "Triángulo"]
TRAMOS = {"Holdout completo": fe >= HOLDOUT_DESDE,
          "Holdout parte 1 (abr–dic 2025)": (fe >= HOLDOUT_DESDE) & (fe < PARTE2),
          "Holdout parte 2 (2026)": fe >= PARTE2,
          "2023 – mar 2025 (usado para elegir)": fe < HOLDOUT_DESDE}
K = np.abs(ret[fe < HOLDOUT_DESDE]).mean()           # fijado antes del holdout


def componentes(p, costo=COSTO):
    dp = np.abs(np.diff(p, prepend=0.0))
    return p * ret, p * carry_u, -dp * costo / S0


def p_boot(x, seed=0):
    """p-valor unilateral de media(x) <= 0, bootstrap estacionario."""
    rng = np.random.default_rng(seed)
    n = len(x)
    nuevo = rng.random((N_BOOT, n)) < 1 / BLOQUE
    nuevo[:, 0] = True
    tt = np.arange(n)
    ult = np.maximum.accumulate(np.where(nuevo, tt, 0), axis=1)
    ii = (np.take_along_axis(rng.integers(0, n, (N_BOOT, n)), ult, axis=1) + tt - ult) % n
    return float(np.mean((x - x.mean())[ii].mean(axis=1) >= x.mean()))


def diebold_mariano(l1, l2, h=1):
    """H1: el modelo 1 tiene menor pérdida. HAC + corrección HLN. Devuelve (DM, p)."""
    d = l1 - l2
    n = len(d)
    if np.allclose(d, d[0]):
        return np.nan, np.nan
    dm = d.mean()
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    g = [np.mean((d[k:] - dm) * (d[:n - k] - dm)) for k in range(L + 1)]
    v = g[0] + 2 * sum((1 - k / (L + 1)) * g[k] for k in range(1, L + 1))
    s = dm / np.sqrt(v / n)
    s *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)          # HLN
    return s, t_dist.cdf(s, df=n - 1)


def perdida_dir(p, r):
    s = np.sign(p)
    return np.where(s == 0, 0.5, (s != np.sign(r)).astype(float))


pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
filas_reg = []
for tramo, m in TRAMOS.items():
    r = ret[m]
    print("\n" + "=" * 115)
    print(f"{tramo} · {fe[m][0].date()} – {fe[m][-1].date()} ({m.sum()} días) · "
          f"costo {COSTO} CLP · carry TPM − Fed funds")
    print("=" * 115)

    # 1) rentabilidad antes y después de costos
    res = {}
    for n, p in ESTR.items():
        br, ca, cs = (x[m] for x in componentes(p))
        nt = br + ca + cs
        eq = np.cumsum(nt)
        res[n] = {"bruto_%": br.sum() * 100, "carry_%": ca.sum() * 100,
                  "antes de costos_%": (br + ca).sum() * 100, "costos_%": cs.sum() * 100,
                  "después de costos_%": nt.sum() * 100,
                  "sharpe": nt.mean() / nt.std() * np.sqrt(252) if nt.std() > 0 else np.nan,
                  "maxDD_%": (eq - np.maximum.accumulate(eq)).min() * 100,
                  "acierto_días_activos_%": (np.mean(np.sign(p[m][p[m] != 0]) ==
                                                     np.sign(r[p[m] != 0])) * 100
                                             if (p[m] != 0).any() else np.nan)}
    print("RENTABILIDAD")
    print(pd.DataFrame(res).T.round(2).to_string())

    # 2) Diebold-Mariano y 3) bootstrap
    nt_tri = sum(x[m] for x in componentes(pos))
    pr_tri = pos[m] * K
    tests = {}
    for ref in REFS:
        pref = ESTR[ref][m]
        dm_d, p_d = diebold_mariano(perdida_dir(pos[m], r), perdida_dir(pref, r))
        dm_c, p_c = diebold_mariano((r - pr_tri) ** 2, (r - pref * K) ** 2)
        nt_ref = sum(x[m] for x in componentes(ESTR[ref]))
        tests[ref] = {"DM direccional": dm_d, "p DM dir.": p_d,
                      "DM cuadrático": dm_c, "p DM cuad.": p_c,
                      "dif. neta_%": (nt_tri - nt_ref).sum() * 100,
                      "p bootstrap": p_boot(nt_tri - nt_ref)}
    tests["(solo el triángulo)"] = {"dif. neta_%": nt_tri.sum() * 100,
                                    "p bootstrap": p_boot(nt_tri)}
    print("\nTESTS: triángulo vs cada referencia (H1: el triángulo es mejor; p < 0,05 = significativo)")
    print(pd.DataFrame(tests).T.round(3).to_string())

    sens = {f"{c:.2f}": sum(x[m] for x in componentes(pos, c)).sum() * 100 for c in COSTOS_SENS}
    print("Sensibilidad a costos, triángulo (neto %, CLP por USD):",
          {k: round(v, 2) for k, v in sens.items()})
    filas_reg.append(pd.DataFrame(tests).T.assign(tramo=tramo))

pd.concat(filas_reg).to_csv(OUT / "exp12_tests_finales.csv")
print(f"\nGuardado en {OUT / 'exp12_tests_finales.csv'}")
print("Nota: el triángulo se eligió entre 24 variantes con datos hasta mar 2025;")
print("el holdout no participó en la elección, por lo que sus p-valores no requieren ajuste.")
