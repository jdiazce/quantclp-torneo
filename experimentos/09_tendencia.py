"""
EXP 09 — Chequeo rápido: tendencia como filtro de régimen (versión sistemática de
"el dólar viene cayendo, me pongo corto"). Posición = signo del retorno de L días.
Descartado: el signo del resultado cambia según el período.
Correr con F5 en Spyder.
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
COSTO = 0.50
PERIODOS = {"2010-2019": ("2010-01-01", "2019-10-17"), "2015-2019": ("2015-01-01", "2019-10-17"),
            "2023-mar2025": ("2023-01-01", "2025-04-04")}

clp = tg.cargar_usdclp(BASE)
H = clp.index.max() - pd.DateOffset(months=18)
clp = clp[clp.index < H]
ex = (pd.read_excel(BASE, sheet_name="datos",
                    usecols=["fecha", "CHOVCHOV Index|PX_LAST", "FEDL01 Index|PX_LAST"])
        .set_axis(["fecha", "tpm", "ff"], axis=1)
        .assign(fecha=lambda x: pd.to_datetime(x["fecha"])).set_index("fecha").sort_index())
ex = ex[ex.index < H]
ex["ff"] = ex["ff"].shift(1)
ex = ex.ffill().reindex(clp.index, method="ffill")
c = clp["c"]
r = c.shift(-1) / c - 1
dias = pd.Series(clp.index, index=clp.index).diff().shift(-1).dt.days
car = -(ex["tpm"] - ex["ff"]) / 100 / 360 * dias


def correr(pos, a, b):
    m = (pos.index >= a) & (pos.index <= b)
    p = pos[m]
    dp = p.diff().abs().fillna(p.abs())
    nt = p * r[m] + p * car[m] - dp * COSTO / c[m]
    return nt.sum() * 100, nt.mean() / nt.std() * np.sqrt(252)


filas, reg = {}, []
estr = {f"Tendencia {L}d": np.sign(np.log(c).diff(L)) for L in (20, 60, 120, 250)}
estr["Siempre +1"] = pd.Series(1.0, index=c.index)
for n, p in estr.items():
    fila = {}
    for per, (a, b) in PERIODOS.items():
        nt, sh = correr(p, a, b)
        fila[f"neto {per} %"], fila[f"sharpe {per}"] = nt, sh
    filas[n] = fila
    if n.startswith("Tendencia"):
        reg.append({"timestamp": datetime.now().isoformat(timespec="seconds"),
                    "experimento": f"09_{n.replace(' ', '_')}", "desde": "2023-01-01",
                    "hasta": "2025-04-04", "costo": COSTO,
                    "neto_%": round(fila["neto 2023-mar2025 %"], 3),
                    "sharpe": round(fila["sharpe 2023-mar2025"], 3), "elegida": False})
print(pd.DataFrame(filas).T.round(2).to_string())

reg_path = OUT / "registro_experimentos.csv"
reg = pd.DataFrame(reg)
if reg_path.exists():
    prev = pd.read_csv(reg_path)
    reg = pd.concat([prev, reg[~reg["experimento"].isin(prev["experimento"])]], ignore_index=True)
reg.to_csv(reg_path, index=False)
print(f"Registro: {len(reg)} intentos.")
