"""
estrategia.py — Estrategia automatizada del torneo FEN–Itaú "¿Sube o baja el dólar mañana?"

Modelo: detector de triángulos sobre el OHLC diario del USD/CLP (variante
RA-SB-BC-OA, congelada). Toda la lógica está en src/triangulos.py.

ENTREGA TRES COSAS
  1. SEÑAL DEL DÍA: posición para mañana (−3, −1, 0, +1, +3), caso detectado,
     tipo de triángulo y lectura de convicción.
  2. CURVA DE RESULTADO: neto acumulado (carry y costos incluidos) contra comprar
     siempre, no operar y siempre +1. Se guarda en salidas/curva.png.
  3. LECTURA DE CONVICCIÓN: qué pasó en los días con exposición máxima (±3), con
     exposición media (±1) y en los días de abstención (0).

REGLAS DE POSICIÓN (al cierre de t, se aplica al retorno de t a t+1)
  +3  rompió el techo, volvió a apoyar en él y retomó la subida (retest alcista)
  +1  ruptura alcista vigente, o precio en la base del triángulo con vela de rechazo
  −1  ruptura bajista vigente, o precio en el techo con vela de rechazo
  −3  rompió el piso, el piso actuó de resistencia y retomó la caída
   0  sin patrón vigente, o patrones en conflicto
  Ruptura = 3 cierres seguidos fuera de la recta + 0,45 ATR. Salida = objetivo
  (altura del triángulo proyectada), stop (cierre 1 ATR adentro) o 60 días.

RESULTADO DIARIO = posición × variación del USD/CLP + carry − costos
  Carry: TPM (CHOVCHOV) − Fed funds (FEDL01, rezago 1 día), base act/360.
  Largo en dólar paga el diferencial; corto lo recibe.
  Costos: COSTO_CLP pesos por dólar por cada unidad de cambio de posición.
  Supuesto: "comprar siempre" = +3 todos los días.

USO
  Spyder: abrir y correr con F5. Para usar una base actualizada, cambiar BASE.
  Terminal: python estrategia.py [ruta_a_la_base.xlsx]
"""

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------
RAIZ = Path(os.environ.get("QUANTCLP", Path(__file__).resolve().parent))
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].endswith(".xlsx") \
    else next((p for p in (RAIZ / "data" / "base_usdclp.xlsx",
                           RAIZ / "base_usdclp.xlsx") if p.exists()), None)
SALIDAS = RAIZ / "salidas"
SALIDAS.mkdir(parents=True, exist_ok=True)

CURVA_DESDE = "2023-01-01"        # inicio de la curva
HOLDOUT_DESDE = "2025-04-07"      # tramo fuera de muestra (para marcarlo en la curva)
COSTO_CLP = 0.50
POS_COMPRAR = 3.0
CFG = {"ruptura": "A", "salida": "B", "base": "C"}   # RA-SB-BC-OA (congelado)
MOSTRAR_GRAFICO = True

if BASE is None or not BASE.exists():
    raise FileNotFoundError("No encuentro base_usdclp.xlsx en data/ ni en la raíz del proyecto.")
if not (RAIZ / "src" / "triangulos.py").exists():
    raise FileNotFoundError(f"Falta {RAIZ / 'src' / 'triangulos.py'}.")
importlib.invalidate_caches()
sys.path.insert(0, str(RAIZ / "src"))
import triangulos as tg  # noqa: E402

DESCRIPCION = {
    "RETEST_ARRIBA": "rompió el techo, volvió a apoyar en él y retomó la subida",
    "RUPTURA_ARRIBA": "ruptura alcista confirmada y vigente",
    "BASE": "precio en la base del triángulo, con vela de rechazo alcista",
    "TECHO": "precio en el techo del triángulo, con vela de rechazo bajista",
    "RUPTURA_ABAJO": "ruptura bajista confirmada y vigente",
    "RETEST_ABAJO": "rompió el piso, el piso actuó de resistencia y retomó la caída",
    "NADA": "no hay triángulo con señal vigente",
    "CONFLICTO": "hay triángulos con señales opuestas: se abstiene",
}
TIPOS = {"ASC": "ascendente", "DESC": "descendente", "SIM": "simétrico", "": "—"}

# ----------------------------------------------------------------------
# DATOS
# ----------------------------------------------------------------------
clp = tg.cargar_usdclp(BASE)
det = tg.Detector(clp)
F = clp.index
tasas = (pd.read_excel(BASE, sheet_name="datos",
                       usecols=["fecha", "CHOVCHOV Index|PX_LAST", "FEDL01 Index|PX_LAST"])
           .set_axis(["fecha", "tpm", "ff"], axis=1)
           .assign(fecha=lambda x: pd.to_datetime(x["fecha"]))
           .set_index("fecha").sort_index())
tasas["ff"] = tasas["ff"].shift(1)                 # Fed funds se publica al día siguiente
tasas = tasas.ffill().reindex(F, method="ffill")

# ----------------------------------------------------------------------
# SEÑALES HISTÓRICAS (causales: en cada día solo se usan datos hasta ese día)
# ----------------------------------------------------------------------
idx = np.where(F >= pd.Timestamp(CURVA_DESDE))[0]
print(f"Base: {BASE.name} · último dato {F[-1].date()} · calculando {len(idx)} señales...")
reg = []
for k in idx:
    tris = det.detectar(k)
    p, caso, tipo = tg.senal(det, tris, k, CFG)
    reg.append({"fecha": F[k], "k": k, "pos": p, "caso": caso, "tipo": tipo, "tris": tris})
sen = pd.DataFrame(reg).set_index("fecha")

# ----------------------------------------------------------------------
# 1) SEÑAL DEL DÍA
# ----------------------------------------------------------------------
hoy = sen.iloc[-1]
k = int(hoy["k"])
pos = int(hoy["pos"])
if pos == 0:
    conv = "SIN SEÑAL — fuera de mercado"
elif abs(pos) == 3:
    conv = "ALTA — patrón confirmado con retest (exposición máxima)"
else:
    conv = "MEDIA — patrón vigente sin retest (exposición moderada)"
direccion = {1: "SUBE (comprar dólar)", -1: "BAJA (vender dólar)", 0: "SIN DIRECCIÓN"}[int(np.sign(pos))]

print("\n" + "=" * 78)
print(f"SEÑAL DEL DÍA · cierre {F[k].date()} · para la jornada siguiente")
print("=" * 78)
print(f"  USD/CLP cierre     : {det.C[k]:.2f}")
print(f"  Dirección esperada : {direccion}")
print(f"  Posición           : {pos:+d}")
print(f"  Caso               : {hoy['caso']} — {DESCRIPCION.get(hoy['caso'], '')}")
print(f"  Triángulo          : {TIPOS.get(hoy['tipo'], hoy['tipo'])}")
print(f"  Convicción         : {conv}")
if hoy["tris"]:
    print("  Triángulos detectados hoy:")
    for t in hoy["tris"]:
        p_t, caso_t = tg.estado(det, t, k, CFG)
        techo, base_ = tg.recta(t["U"], k), tg.recta(t["L"], k)
        lineas = (f"techo {techo:7.2f} · base {base_:7.2f}" if k <= t["apice"]
                  else "ya pasó el ápice")
        print(f"    · {TIPOS[t['tipo']]:<12} desde {F[t['ini']].date()} · {lineas} · "
              f"estado {caso_t} ({p_t:+d})")
else:
    print("  No hay triángulos formados en la ventana de análisis.")

with open(SALIDAS / "senal_del_dia.json", "w", encoding="utf-8") as fh:
    json.dump({"fecha_cierre": str(F[k].date()), "usdclp": round(float(det.C[k]), 2),
               "posicion": pos, "direccion": direccion, "caso": hoy["caso"],
               "tipo": hoy["tipo"], "conviccion": conv}, fh, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 2) CURVA DE RESULTADO
# ----------------------------------------------------------------------
h = sen.iloc[:-1]                                   # el último día aún no tiene t+1
kk = h["k"].to_numpy()
S0 = det.C[kk]
ret = det.C[kk + 1] / S0 - 1
dias = (F[kk + 1] - F[kk]).days.to_numpy()
carry_u = -((tasas["tpm"] - tasas["ff"]).to_numpy()[kk] / 100 / 360) * dias


def resultado(p):
    dp = np.abs(np.diff(p, prepend=0.0))
    return pd.DataFrame({"bruto": p * ret, "carry": p * carry_u,
                         "costos": -dp * COSTO_CLP / S0}, index=h.index)


ESTR = {"Triángulos": h["pos"].to_numpy(float),
        "Comprar siempre (+3)": np.full(len(h), POS_COMPRAR),
        "Siempre +1": np.ones(len(h)), "No operar": np.zeros(len(h))}
RES = {n: resultado(p) for n, p in ESTR.items()}
neto = pd.DataFrame({n: r.sum(axis=1) for n, r in RES.items()})

fig, ax = plt.subplots(figsize=(12, 5.5))
estilos = {"Triángulos": dict(lw=2.4, color="#1b5e20"),
           "Comprar siempre (+3)": dict(lw=1.2, color="#c62828", alpha=0.8),
           "Siempre +1": dict(lw=1.2, color="#1565c0", alpha=0.8),
           "No operar": dict(lw=1.0, color="0.5", ls="--")}
for n in neto:
    ax.plot(neto.index, neto[n].cumsum() * 100, label=n, **estilos[n])
ax.axvline(pd.Timestamp(HOLDOUT_DESDE), color="0.3", lw=1, ls=":")
ax.text(pd.Timestamp(HOLDOUT_DESDE), ax.get_ylim()[1] * 0.95, "  fuera de muestra →",
        fontsize=9, color="0.3", va="top")
ax.axhline(0, color="0.6", lw=0.6)
ax.set_ylabel("Resultado neto acumulado (% del nocional por unidad)")
ax.set_title(f"USD/CLP · triángulos vs. referencias · carry TPM − Fed funds · costo {COSTO_CLP} CLP")
ax.legend(loc="upper left")
plt.tight_layout()
fig.savefig(SALIDAS / "curva.png", dpi=150)
if MOSTRAR_GRAFICO:
    plt.show()
plt.close(fig)

tab = pd.DataFrame({n: {"bruto_%": r["bruto"].sum() * 100, "carry_%": r["carry"].sum() * 100,
                        "costos_%": r["costos"].sum() * 100,
                        "neto_%": r.sum(axis=1).sum() * 100}
                    for n, r in RES.items()}).T
fuera = neto.index >= pd.Timestamp(HOLDOUT_DESDE)
tab["neto fuera de muestra_%"] = neto[fuera].sum() * 100
print("\n" + "=" * 78)
print(f"CURVA DE RESULTADO · {h.index[0].date()} – {h.index[-1].date()} "
      f"(fuera de muestra desde {HOLDOUT_DESDE}) · guardada en salidas/curva.png")
print("=" * 78)
print(tab.round(2).to_string())

# ----------------------------------------------------------------------
# 3) LECTURA DE CONVICCIÓN
# ----------------------------------------------------------------------
d = pd.DataFrame({"pos": h["pos"].to_numpy(), "caso": h["caso"].to_numpy(),
                  "ret": ret, "neto": neto["Triángulos"].to_numpy(),
                  "uno": neto["Siempre +1"].to_numpy()}, index=h.index)
d["nivel"] = np.select([d["pos"].abs() == 3, d["pos"].abs() == 1], ["±3 (máxima)", "±1 (media)"],
                       "0 (abstención)")


def lectura(x, total):
    activo = x["pos"] != 0
    return pd.Series({
        "días": len(x),
        "% de los días": len(x) / total * 100,
        "acierto %": (np.sign(x["pos"]) == np.sign(x["ret"]))[activo].mean() * 100
        if activo.any() else np.nan,
        "neto total %": x["neto"].sum() * 100,
        "neto prom. (pb/día)": x["neto"].mean() * 1e4,
        "dólar prom. (pb/día)": x["ret"].mean() * 1e4,
        "|mov. dólar| prom. (pb)": x["ret"].abs().mean() * 1e4,
        "siempre +1 en esos días %": x["uno"].sum() * 100})


print("\n" + "=" * 78)
print("LECTURA DE CONVICCIÓN · por nivel de exposición")
print("=" * 78)
for nombre, m in {"Todo el período": np.ones(len(d), bool),
                  "Fuera de muestra": d.index >= pd.Timestamp(HOLDOUT_DESDE)}.items():
    print(f"\n{nombre}")
    print(d[m].groupby("nivel").apply(lectura, total=m.sum()).round(2).to_string())

print("\nPor caso (todo el período)")
print(d[d["pos"] != 0].groupby("caso").apply(lectura, total=len(d))
       .drop(columns=["% de los días"]).round(2).to_string())

ab = d[d["pos"] == 0]
print(f"\nAbstención: en los {len(ab)} días en 0, siempre +1 hizo {ab['uno'].sum() * 100:+.2f}%. "
      f"Es lo que el modelo evitó (si es negativo) o dejó pasar (si es positivo).")

d.drop(columns=["uno"]).assign(neto_acum=d["neto"].cumsum()).to_csv(SALIDAS / "historial.csv")
print(f"\nArchivos en {SALIDAS}: senal_del_dia.json · curva.png · historial.csv")
