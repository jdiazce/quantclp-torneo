"""
estrategia.py — Estrategia automatizada del torneo FEN–Itaú "¿Sube o baja el dólar mañana?"

Modelo: detector de triángulos sobre el OHLC diario del USD/CLP (variante
RA-SB-BC-OA, congelada). Toda la lógica está en src/triangulos.py; este script no
cambia ningún parámetro.

ENTREGA TRES COSAS (las tres que piden las bases)
  1. SEÑAL DEL DÍA: dirección esperada, probabilidad estimada de alza y posición
     para mañana (−3, −1, 0, +1, +3), con el caso y el triángulo que la generan.
  2. CURVA DE RESULTADO: neto acumulado (base 100, carry y costos incluidos) contra
     comprar siempre, no operar y siempre +1, con un panel de exposición diaria.
       salidas/curva.png          desde 2023, con el holdout sombreado
       salidas/curva_holdout.png  solo el holdout, partiendo en 100
  3. LECTURA DE CONVICCIÓN: qué pasó en los días de exposición máxima (±3), media
     (±1) y de abstención (0).

REGLAS DE POSICIÓN (al cierre de t, se aplica al retorno de t a t+1)
  +3  rompió el techo, volvió a apoyar en él y retomó la subida (retest alcista)
  +1  ruptura alcista vigente, o precio en la base con vela de rechazo
  −1  ruptura bajista vigente, o precio en el techo con vela de rechazo
  −3  rompió el piso, el piso actuó de resistencia y retomó la caída
   0  sin patrón vigente, ruptura en duda o patrones en conflicto

RESULTADO DIARIO = posición × variación del USD/CLP − carry − costos
  Carry: TPM (CHOVCHOV) − Fed funds (FEDL01, rezago 1 día), act/360, días corridos.
  Largo en dólar paga el diferencial; corto lo recibe.
  Costos: COSTO_CLP pesos por dólar por cada unidad de cambio de posición.
  "Comprar siempre" = +3 todos los días.

USO
  Spyder: abrir y correr con F5 (lee data/base_usdclp.xlsx).
  Terminal: python estrategia.py [ruta_a_la_base]
  La base puede venir en .xlsx (hoja "datos", formato ancho) o en .parquet / .csv
  (formato largo: fecha, ticker, campo, valor). Para .parquet: uv pip install pyarrow

MONITOREO EN VIVO (13 nov – 1 dic 2026)
  Cada día, con la base actualizada: correr el script. La señal queda en
  salidas/senal_del_dia.json y se agrega a salidas/bitacora_senales.csv; el tramo
  "En vivo" aparece solo en la tabla por tramo y en la curva.
"""

import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------
RAIZ = Path(os.environ.get("QUANTCLP", Path(__file__).resolve().parent))
FORMATOS = (".xlsx", ".parquet", ".csv")
_arg = next((a for a in sys.argv[1:] if a.lower().endswith(FORMATOS)), None)
BASE = Path(_arg) if _arg else next(
    (p for p in (RAIZ / "data" / f"base_usdclp{e}" for e in FORMATOS) if p.exists()), None)
SALIDAS = RAIZ / "salidas"
SALIDAS.mkdir(parents=True, exist_ok=True)

CURVA_DESDE = "2023-01-01"
SELECCION = ("2023-01-01", "2025-04-04")       # ventana de selección (probabilidades)
HOLDOUT = ("2025-04-07", "2026-10-05")         # fuera de muestra
EN_VIVO_DESDE = "2026-11-13"                   # monitoreo del torneo
COSTO_CLP = 0.50
POS_COMPRAR = 3.0
CFG = {"ruptura": "A", "salida": "B", "base": "C"}     # RA-SB-BC-OA (congelado)
MOSTRAR_GRAFICO = True

if BASE is None or not BASE.exists():
    raise FileNotFoundError("No encuentro la base en data/ (base_usdclp.xlsx, .parquet o .csv).")
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
COLUMNAS = ["USDCLP Curncy|PX_OPEN", "USDCLP Curncy|PX_HIGH", "USDCLP Curncy|PX_LOW",
            "USDCLP Curncy|PX_LAST", "CHN1M Curncy|PX_LAST",
            "CHOVCHOV Index|PX_LAST", "FEDL01 Index|PX_LAST"]


# ----------------------------------------------------------------------
# LECTURA DE LA BASE (xlsx ancho, o parquet/csv largo)
# ----------------------------------------------------------------------
def base_en_xlsx(base: Path) -> Path:
    """Devuelve un .xlsx con la hoja 'datos' que lee triangulos.cargar_usdclp.
    Si la base viene en formato largo, la pasa a ancho con las columnas necesarias,
    para que la limpieza del USD/CLP sea exactamente la del modelo congelado."""
    if base.suffix.lower() == ".xlsx":
        return base
    if base.suffix.lower() == ".parquet":
        try:
            largo = pd.read_parquet(base)
        except ImportError as e:
            raise ImportError("Para leer .parquet: uv pip install pyarrow (en la Terminal) "
                              "y reiniciar el kernel.") from e
    else:
        largo = pd.read_csv(base)
    largo["col"] = largo["ticker"].astype(str) + "|" + largo["campo"].astype(str)
    ancha = (largo[largo["col"].isin(COLUMNAS)]
             .pivot_table(index="fecha", columns="col", values="valor", aggfunc="last")
             .reindex(columns=COLUMNAS).reset_index())
    ancha["fecha"] = pd.to_datetime(ancha["fecha"])
    destino = SALIDAS / "_base_convertida.xlsx"
    ancha.to_excel(destino, sheet_name="datos", index=False)
    return destino


XLSX = base_en_xlsx(BASE)
clp = tg.cargar_usdclp(XLSX)
det = tg.Detector(clp)
F = clp.index
tasas = (pd.read_excel(XLSX, sheet_name="datos",
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
print(f"Base: {BASE.name} · último dato {F[-1].date()} · calculando {len(idx)} señales "
      f"(tarda ~1 min)...")
reg = []
for k in idx:
    tris = det.detectar(k)
    p, caso, tipo = tg.senal(det, tris, k, CFG)
    reg.append({"fecha": F[k], "k": k, "pos": p, "caso": caso, "tipo": tipo, "tris": tris})
sen = pd.DataFrame(reg).set_index("fecha")

# ----------------------------------------------------------------------
# PROBABILIDAD EMPÍRICA: acierto por nivel de exposición en la ventana de
# selección (solo datos previos al holdout). P(sube) = acierto si la posición es
# positiva, 1 − acierto si es negativa, frecuencia de alzas si es 0.
# ----------------------------------------------------------------------
_h = sen.iloc[:-1]
_k = _h["k"].to_numpy()
_r = det.C[_k + 1] / det.C[_k] - 1
_sel = (_h.index >= pd.Timestamp(SELECCION[0])) & (_h.index <= pd.Timestamp(SELECCION[1]))
_p = _h["pos"].to_numpy()
ACIERTO = {n: float(np.mean(np.sign(_p[m]) == np.sign(_r[m])))
           for n, m in {1: _sel & (np.abs(_p) == 1), 3: _sel & (np.abs(_p) == 3)}.items()}
P_ALZA = float(np.mean(_r[_sel] > 0))


def prob_sube(p):
    if p == 0:
        return P_ALZA
    a = ACIERTO[abs(int(p))]
    return a if p > 0 else 1 - a


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
print(f"  P(sube) estimada   : {prob_sube(pos):.1%}  (acierto histórico del nivel "
      f"en {SELECCION[0][:4]}–{SELECCION[1][:7]})")
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

senal = {"fecha_cierre": str(F[k].date()), "usdclp": round(float(det.C[k]), 2),
         "posicion": pos, "direccion": direccion, "caso": hoy["caso"],
         "tipo": hoy["tipo"], "prob_sube": round(prob_sube(pos), 3), "conviccion": conv}
with open(SALIDAS / "senal_del_dia.json", "w", encoding="utf-8") as fh:
    json.dump(senal, fh, ensure_ascii=False, indent=2)

# bitácora: una fila por fecha de cierre (si se corre dos veces el mismo día, se reemplaza)
f_bit = SALIDAS / "bitacora_senales.csv"
fila = pd.DataFrame([{**senal, "corrido_el": datetime.now().isoformat(timespec="seconds")}])
if f_bit.exists():
    bit = pd.read_csv(f_bit, dtype={"fecha_cierre": str})
    bit = pd.concat([bit[bit["fecha_cierre"] != senal["fecha_cierre"]], fila])
else:
    bit = fila
bit.sort_values("fecha_cierre").to_csv(f_bit, index=False)

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
RES = {n: resultado(p) for n, p in ESTR.items()}     # entrada en ene 2023
neto = pd.DataFrame({n: r.sum(axis=1) for n, r in RES.items()})

ESTILOS = {"Triángulos": dict(lw=2.4, color="#1b5e20"),
           "Comprar siempre (+3)": dict(lw=1.2, color="#c62828", alpha=0.85),
           "Siempre +1": dict(lw=1.2, color="#1565c0", alpha=0.85),
           "No operar": dict(lw=1.0, color="0.5", ls="--")}
FECHA_FIN = F[kk + 1]                               # el resultado de t se realiza en t+1


def curva(desde, archivo, titulo, sombrear=None):
    m = h.index >= pd.Timestamp(desde)
    x = np.r_[h.index[m][:1], FECHA_FIN[m]]
    fig, (ax, axp) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3.2, 1]})
    for n in neto:
        y = 100 * (1 + np.r_[0, neto.loc[m, n].cumsum()])
        ax.plot(x, y, label=f"{n} ({y[-1] - 100:+.1f})".replace(".", ","), **ESTILOS[n])
    for (a, b, etq) in (sombrear or []):
        for eje in (ax, axp):
            eje.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="0.9", lw=0, zorder=0)
        ax.text(pd.Timestamp(a), 0.97, f" {etq}", transform=ax.get_xaxis_transform(),
                fontsize=9, color="0.35", va="top")
    ax.axhline(100, color="0.6", lw=0.6)
    ax.set_ylabel("Resultado acumulado (base 100)")
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=11)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=9, frameon=False)
    ax.grid(alpha=0.25)
    p = h.loc[m, "pos"].to_numpy(float)
    axp.bar(h.index[m], p, width=1.0,
            color=np.where(p > 0, "#1b5e20", np.where(p < 0, "#c62828", "0.6")))
    axp.set_yticks([-3, -1, 0, 1, 3])
    axp.set_ylim(-3.5, 3.5)
    axp.set_ylabel("Exposición")
    axp.axhline(0, color="0.5", lw=0.5)
    plt.tight_layout()
    fig.savefig(SALIDAS / archivo, dpi=170, bbox_inches="tight")
    if MOSTRAR_GRAFICO:
        plt.show()
    plt.close(fig)


fin_datos = str(h.index[-1].date())
sombras = [(HOLDOUT[0], min(HOLDOUT[1], fin_datos), "holdout (fuera de muestra)")]
if fin_datos >= EN_VIVO_DESDE:
    sombras.append((EN_VIVO_DESDE, fin_datos, "en vivo"))
curva(CURVA_DESDE, "curva.png",
      "USD/CLP · triángulos vs. referencias · neto de carry (TPM − Fed funds) y costos "
      f"({str(COSTO_CLP).replace('.', ',')} CLP)", sombras)
curva(HOLDOUT[0], "curva_holdout.png",
      "Holdout: desde el 7 abr 2025, base 100 · neto de carry y costos")

# tabla por tramo
TRAMOS = {"Selección (ene 2023 – abr 2025)": SELECCION,
          "Holdout · abr–dic 2025": (HOLDOUT[0], "2025-12-31"),
          "Holdout · ene–oct 2026": ("2026-01-01", HOLDOUT[1]),
          "Holdout completo": HOLDOUT,
          "Posterior a la base": ("2026-10-06", "2026-11-12"),
          "En vivo (13 nov – 1 dic)": (EN_VIVO_DESDE, "2026-12-31")}


def resumen(x):
    acum = x.cumsum()
    return {"días": len(x), "neto_%": x.sum() * 100,
            "sharpe": x.mean() / x.std(ddof=0) * np.sqrt(252) if x.std() > 0 else np.nan,
            "caida_max_%": (acum - acum.cummax().clip(lower=0)).min() * 100}


filas = []
for nom, (a, b) in TRAMOS.items():
    m = (h.index >= pd.Timestamp(a)) & (h.index <= pd.Timestamp(b))
    if not m.any():
        continue
    for n in neto:
        r = resumen(neto.loc[m, n])
        r.update(tramo=nom, estrategia=n,
                 pct_dias_con_posicion=(ESTR[n][m] != 0).mean() * 100)
        filas.append(r)
tramos = pd.DataFrame(filas)[["tramo", "estrategia", "días", "neto_%", "sharpe",
                              "caida_max_%", "pct_dias_con_posicion"]]
tramos.to_csv(SALIDAS / "tabla_tramos.csv", index=False)

mh = (h.index >= pd.Timestamp(HOLDOUT[0])) & (h.index <= pd.Timestamp(HOLDOUT[1]))
tab = pd.DataFrame({n: {"bruto_%": r.loc[mh, "bruto"].sum() * 100,
                        "carry_%": r.loc[mh, "carry"].sum() * 100,
                        "costos_%": r.loc[mh, "costos"].sum() * 100,
                        "neto_%": r.loc[mh].sum(axis=1).sum() * 100}
                    for n, r in RES.items()}).T
print("\n" + "=" * 78)
print("CURVA DE RESULTADO · salidas/curva.png y salidas/curva_holdout.png")
print("=" * 78)
print(f"Descomposición en el holdout ({HOLDOUT[0]} – {HOLDOUT[1]}):")
print(tab.round(2).to_string())
print("\nNeto por tramo, % (también en salidas/tabla_tramos.csv):")
print(tramos.pivot_table(index="tramo", columns="estrategia", values="neto_%", sort=False)
            [list(neto.columns)].round(2).to_string())
t_tri = tramos[tramos["estrategia"] == "Triángulos"].set_index("tramo")
print("\nTriángulos por tramo:")
print(t_tri.drop(columns="estrategia").round(2).to_string())

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
for nombre, m in {"Todo el período": np.ones(len(d), bool), "Holdout": mh}.items():
    print(f"\n{nombre}")
    print(d[m].groupby("nivel").apply(lectura, total=m.sum()).round(2).to_string())
    act = (d["pos"] != 0) & m
    acierta = np.sign(d.loc[act, "pos"]) == np.sign(d.loc[act, "ret"])
    print(f"  Acierta {int(acierta.sum())} de {int(act.sum())} días con posición "
          f"({acierta.mean():.1%}).")

print("\nPor caso (todo el período)")
print(d[d["pos"] != 0].groupby("caso").apply(lectura, total=len(d))
       .drop(columns=["% de los días"]).round(2).to_string())

ab = d[mh & (d["pos"] == 0)]
print(f"\nAbstención en el holdout: en los {len(ab)} días en 0, siempre +1 hizo "
      f"{ab['uno'].sum() * 100:+.2f}%. Es lo que el modelo evitó (si es negativo) o dejó "
      f"pasar (si es positivo).")

d.drop(columns=["uno"]).assign(neto_acum=d["neto"].cumsum()).to_csv(SALIDAS / "historial.csv")
print(f"\nArchivos en {SALIDAS}: senal_del_dia.json · bitacora_senales.csv · curva.png · "
      f"curva_holdout.png · tabla_tramos.csv · historial.csv")
