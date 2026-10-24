"""
EXP 02 — Backtest de triángulos (lógica del screener v14b) sobre USD/CLP.

REGLA DE POSICIÓN (al cierre de t, se aplica al retorno t -> t+1)
  +3  rompió hacia arriba, volvió a apoyar en el techo roto y retoma la subida
  +1  rompió hacia arriba (ruptura vigente)
  +1  está en la base del triángulo (soporte)
  -1  está en el techo del triángulo (resistencia)
  -3  rompió hacia abajo, el piso roto actuó de resistencia y retoma la caída
   0  todo lo demás (incluye ruptura bajista sin pullback: no definida aún)

PROTOCOLO
  - Período operado: OPERAR_DESDE -> HOLDOUT_DESDE (exclusivo).
  - Holdout bloqueado: los últimos 18 meses de la base no se tocan.
  - Detección día a día usando SOLO datos <= t (pivotes se confirman con W barras).
  - PnL = posición × retorno spot + carry (NDF 1M) − costos por rotación.
  - Cada corrida se anota en experimentos/registro_experimentos.csv.

Correr con F5 en Spyder. Requiere: numpy, pandas, matplotlib, openpyxl.
"""

import importlib
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# RUTAS
# ----------------------------------------------------------------------
RAIZ = Path(os.environ.get("QUANTCLP", Path(__file__).resolve().parents[1]))
BASE = next(p for p in (RAIZ / "data" / "base_usdclp.xlsx",
                        RAIZ / "base_usdclp.xlsx") if p.exists())
OUT = RAIZ / "experimentos"
OUT.mkdir(parents=True, exist_ok=True)
FIGS = OUT / "figs_exp02"
FIGS.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# PARÁMETROS — fijados ANTES de mirar resultados
# ----------------------------------------------------------------------
OPERAR_DESDE = "2023-01-01"
HOLDOUT_MESES = 18                 # se calcula desde la última fecha de la base
VENTANA_HIST = 450                 # barras de historia para detectar en cada t

COSTO_BASE = 0.50                  # CLP por USD por unidad de cambio de posición
COSTOS_SENS = [0.0, 0.25, 0.50, 1.00, 2.00]

# Geometría (idéntica al screener v14b)
CLUSTER_BARRAS, CLUSTER_ATR = 5, 0.50
W_PIVOTE, MAX_PIVOTES = 3, 18
SPAN_MIN, SPAN_MAX = 25, 220
TOL_ATR = 0.45
TOQUES_MIN_LADO, TOQUES_TOTAL = 2, 5
CONTRACCION_MIN, CONTENCION_MIN = 1.30, 0.85
PLANO_ATR = 0.010
ALTURA_MIN_ATR, ALTURA_MAX_ATR = 1.5, 8.0
DEDUPE_ATR = 0.80
APICE_MIN, APICE_MAX = 0.10, 1.50
N_CONFIRMA, ATR_N = 3, 14
MECHA_MAX_ATR = 1.00
APOYO_MAX_ATR = 0.60
MAX_TRIANGULOS = 3

# Nuevos (estados post-ruptura)
VIGENCIA_RUPTURA = 20              # barras que dura una ruptura como señal
RETEST_MAX = 15                    # barras máx. entre ruptura y apoyo
HOLD_3 = 3                         # barras que se mantiene el ±3 tras el rebote
SEPARACION_ATR = 1.0               # v2: antes del retest el precio debe alejarse
                                   #     >= 1 ATR de la línea rota (v1 no lo exigía
                                   #     y marcaba "retest" al día siguiente)
VERSION = "v2"

# ----------------------------------------------------------------------
# DATOS
# ----------------------------------------------------------------------
cols = {"USDCLP Curncy|PX_OPEN": "o", "USDCLP Curncy|PX_HIGH": "h",
        "USDCLP Curncy|PX_LOW": "l", "USDCLP Curncy|PX_LAST": "c",
        "CHN1M Curncy|PX_LAST": "pts1m"}
raw = pd.read_excel(BASE, sheet_name="datos", usecols=["fecha", *cols])
df = (raw.rename(columns=cols)
         .assign(fecha=lambda x: pd.to_datetime(x["fecha"]))
         .set_index("fecha").sort_index())
df = df.dropna(subset=["o", "h", "l", "c"])
# fines de semana fuera (la base trae calendario corrido)
df = df[df.index.dayofweek < 5]
# feriados con cotización rancia: rango cero y cierre igual al anterior
df["pts1m"] = df["pts1m"].ffill()
df = df.loc["2004-01-01":]          # OHLC plano antes de 2004
rancio = (df["h"] == df["l"]) & (df["c"].diff() == 0)
df = df[~rancio]

fin_base = df.index.max()
HOLDOUT_DESDE = fin_base - pd.DateOffset(months=HOLDOUT_MESES)
print(f"Base hasta {fin_base.date()} | holdout bloqueado desde {HOLDOUT_DESDE.date()}")
print(f"Período operado: {OPERAR_DESDE} -> {HOLDOUT_DESDE.date()} (excl.)")
print(f"Feriados rancios eliminados: {int(rancio.sum())}\n")

df = df[df.index < HOLDOUT_DESDE]   # el holdout no entra ni a la memoria
H, L, C, O = (df[k].to_numpy(float) for k in ("h", "l", "c", "o"))
FECHAS = df.index


def atr_serie(h, l, c, n=ATR_N):
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).rolling(n, min_periods=1).mean().to_numpy()


ATR = atr_serie(H, L, C)            # causal: solo usa barras <= i


# ----------------------------------------------------------------------
# DETECCIÓN (geometría del screener, sin filtros de entrada long-only)
# ----------------------------------------------------------------------
def recta(Ln, x):
    return Ln["p1"] + Ln["m"] * (x - Ln["i1"])


def agrupar(piv, px, atr_s, buscar_max):
    if not piv:
        return piv
    grupos, act = [], [piv[0]]
    for i in piv[1:]:
        ref = act[-1]
        if i - ref <= CLUSTER_BARRAS and abs(px[i] - px[ref]) < atr_s[i] * CLUSTER_ATR:
            act.append(i)
        else:
            grupos.append(act)
            act = [i]
    grupos.append(act)
    f = max if buscar_max else min
    return [f(g, key=lambda i: px[i]) for g in grupos]


def es_mecha(i, px, c, atr_s, es_minimo):
    a, b = max(0, i - 2), min(len(c), i + 3)
    if es_minimo:
        return (c[a:b].min() - px[i]) > atr_s[i] * MECHA_MAX_ATR
    return (px[i] - c[a:b].max()) > atr_s[i] * MECHA_MAX_ATR


def pivotes(h, l, c, atr_s, w=W_PIVOTE):
    altos = [i for i in range(w, len(h) - w)
             if h[i] == h[i - w:i + w + 1].max()
             and not es_mecha(i, h, c, atr_s, False)]
    bajos = [i for i in range(w, len(l) - w)
             if l[i] == l[i - w:i + w + 1].min()
             and not es_mecha(i, l, c, atr_s, True)]
    return agrupar(altos, h, atr_s, True), agrupar(bajos, l, atr_s, False)


def detectar(k):
    """Triángulos convergentes visibles al cierre de k (índices globales)."""
    a0g = max(0, k + 1 - VENTANA_HIST)
    hv, lv, cv, av = H[a0g:k + 1], L[a0g:k + 1], C[a0g:k + 1], ATR[a0g:k + 1]
    p_alto, p_bajo = pivotes(hv, lv, cv, av)

    def rectas(piv, px):
        pv, out = piv[-MAX_PIVOTES:], []
        for a in range(len(pv)):
            for b in range(a + 1, len(pv)):
                i1, i2 = pv[a], pv[b]
                if SPAN_MIN <= i2 - i1 <= SPAN_MAX:
                    out.append({"i1": i1, "p1": px[i1], "i2": i2, "p2": px[i2],
                                "m": (px[i2] - px[i1]) / (i2 - i1)})
        return out

    cand = []
    for U in rectas(p_alto, hv):
        for Lo in rectas(p_bajo, lv):
            ini, fin = min(U["i1"], Lo["i1"]), max(U["i2"], Lo["i2"])
            if not (SPAN_MIN <= fin - ini <= SPAN_MAX):
                continue
            h0 = recta(U, ini) - recta(Lo, ini)
            h1 = recta(U, fin) - recta(Lo, fin)
            if h0 <= 0 or h1 <= 0 or h0 / h1 < CONTRACCION_MIN:
                continue
            atr_med = float(np.mean(av[ini:fin + 1]))
            if atr_med <= 0:
                continue
            su, si = U["m"] / atr_med, Lo["m"] / atr_med
            if su > PLANO_ATR or si < -PLANO_ATR:
                continue
            if abs(su) < PLANO_ATR and abs(si) < PLANO_ATR:
                continue
            if not (ALTURA_MIN_ATR <= h0 / atr_med <= ALTURA_MAX_ATR):
                continue
            xs = np.arange(ini, fin + 1)
            tol = av[ini:fin + 1] * TOL_ATR
            dentro = np.mean((cv[ini:fin + 1] <= recta(U, xs) + tol) &
                             (cv[ini:fin + 1] >= recta(Lo, xs) - tol))
            if dentro < CONTENCION_MIN:
                continue
            t_sup = [i for i in p_alto if ini <= i <= fin
                     and abs(hv[i] - recta(U, i)) < av[i] * TOL_ATR]
            t_inf = [i for i in p_bajo if ini <= i <= fin
                     and abs(lv[i] - recta(Lo, i)) < av[i] * TOL_ATR]
            if len(t_sup) < TOQUES_MIN_LADO or len(t_inf) < TOQUES_MIN_LADO:
                continue
            T = len(t_sup) + len(t_inf)
            if T < TOQUES_TOTAL:
                continue
            dm = U["m"] - Lo["m"]
            if dm == 0:
                continue
            apice = ini + (recta(Lo, ini) - recta(U, ini)) / dm
            largo = fin - ini
            if not (fin + APICE_MIN * largo <= apice <= fin + APICE_MAX * largo):
                continue
            # pasar a índices globales
            Ug = {**U, "i1": U["i1"] + a0g, "i2": U["i2"] + a0g}
            Lg = {**Lo, "i1": Lo["i1"] + a0g, "i2": Lo["i2"] + a0g}
            cand.append({"U": Ug, "L": Lg, "ini": ini + a0g, "fin": fin + a0g,
                         "apice": apice + a0g, "toques": T,
                         "score": T / largo * 100 * min(h0 / h1, 3.0)})

    cand.sort(key=lambda x: -x["score"])
    aref = float(np.median(av))
    final = []
    for t in cand:
        dup = any(all(abs(recta(t[q], x) - recta(f[q], x)) <= aref * DEDUPE_ATR
                      for x in ((min(t["ini"], f["ini"]) + max(t["fin"], f["fin"])) / 2,
                                max(t["fin"], f["fin"]))
                      for q in ("U", "L"))
                  for f in final)
        if not dup:
            final.append(t)
            if len(final) >= MAX_TRIANGULOS:
                break
    return final


# ----------------------------------------------------------------------
# ESTADO DE UN TRIÁNGULO AL CIERRE DE k
# ----------------------------------------------------------------------
def estado(t, k):
    U, Lo, fin, apice = t["U"], t["L"], t["fin"], t["apice"]
    arr = aba = 0
    bc, dir_ = None, 0
    for i in range(fin + 1, k + 1):
        tl = ATR[i] * TOL_ATR
        if C[i] > recta(U, i) + tl:
            arr, aba = arr + 1, 0
        elif C[i] < recta(Lo, i) - tl:
            aba, arr = aba + 1, 0
        else:
            arr = aba = 0
        if arr >= N_CONFIRMA or aba >= N_CONFIRMA:
            bc, dir_ = i, (1 if arr else -1)   # bc = día en que se confirma
            break

    if bc is None:                                   # sigue adentro
        if k > apice:
            return 0, "DENTRO", {}
        a = ATR[k]
        d_base = (C[k] - recta(Lo, k)) / a
        d_techo = (recta(U, k) - C[k]) / a
        if 0 <= d_base <= APOYO_MAX_ATR:
            return 1, "BASE", {}
        if 0 <= d_techo <= APOYO_MAX_ATR:
            return -1, "TECHO", {}
        return 0, "DENTRO", {}

    if bc - (N_CONFIRMA - 1) > apice or k - bc > VIGENCIA_RUPTURA:
        return 0, "VENCIDA", {}

    linea = U if dir_ == 1 else Lo
    # invalidación: vuelve a cerrar al otro lado de la línea rota
    for i in range(bc + 1, k + 1):
        tl = ATR[i] * TOL_ATR
        if (dir_ == 1 and C[i] < recta(linea, i) - tl) or \
           (dir_ == -1 and C[i] > recta(linea, i) + tl):
            return 0, "FALLIDA", {}

    # retest: el precio vuelve a tocar la línea rota
    r, separado = None, False
    for i in range(bc, min(k, bc + RETEST_MAX) + 1):
        dist = dir_ * (C[i] - recta(linea, i)) / ATR[i]
        if not separado:
            separado = dist >= SEPARACION_ATR
            continue
        lim = ATR[i] * APOYO_MAX_ATR
        if (dir_ == 1 and L[i] <= recta(linea, i) + lim) or \
           (dir_ == -1 and H[i] >= recta(linea, i) - lim):
            r = i
            break

    info = {"bc": bc, "r": r}
    if r is not None:
        # rebote: primer cierre que retoma la dirección de la ruptura
        j = None
        for i in range(r + 1, k + 1):
            if (dir_ == 1 and C[i] > C[i - 1] and C[i] > C[r]) or \
               (dir_ == -1 and C[i] < C[i - 1] and C[i] < C[r]):
                j = i
                break
        if j is not None and j <= k < j + HOLD_3:
            info["j"] = j
            return (3, "RETEST_ARRIBA", info) if dir_ == 1 \
                else (-3, "PULLBACK_ABAJO", info)

    return (1, "RUPTURA_ARRIBA", info) if dir_ == 1 else (0, "RUPTURA_ABAJO", info)


def senal(k):
    tris = detectar(k)
    est = [(estado(t, k), t) for t in tris]
    activos = [(e, t) for e, t in est if e[0] != 0]
    if not activos:
        return 0, "NADA", None, None
    mx = max(abs(e[0]) for e, _ in activos)
    top = [(e, t) for e, t in activos if abs(e[0]) == mx]
    if len({np.sign(e[0]) for e, _ in top}) > 1:
        return 0, "CONFLICTO", None, None
    (pos, caso, info), t = top[0]              # ya vienen ordenados por score
    return pos, caso, t, info


# ----------------------------------------------------------------------
# BACKTEST
# ----------------------------------------------------------------------
idx_op = np.where(FECHAS >= pd.Timestamp(OPERAR_DESDE))[0]
idx_op = idx_op[:-1]                       # el último día no tiene t+1 en muestra
print(f"Calculando señales para {len(idx_op)} días...")
reg, ejemplos = [], {}
for n, k in enumerate(idx_op, 1):
    if n % 100 == 0:
        print(f"  {n}/{len(idx_op)}")
    pos, caso, t, info = senal(k)
    reg.append({"fecha": FECHAS[k], "k": k, "pos": pos, "caso": caso})
    if t is not None:
        ejemplos.setdefault(caso, [])
        prev = reg[-2]["caso"] if len(reg) > 1 else None
        if prev != caso:                      # primer día de cada episodio
            ejemplos[caso].append((k, t, info))

bt = pd.DataFrame(reg).set_index("fecha")
k_arr = bt["k"].to_numpy()
S0, S1 = C[k_arr], C[k_arr + 1]
dias = (FECHAS[k_arr + 1] - FECHAS[k_arr]).days.to_numpy()
pts = df["pts1m"].to_numpy()[k_arr]

bt["ret"] = S1 / S0 - 1
bt["carry_u"] = -(pts / S0) * dias / 30          # carry de 1 unidad larga
bt["dpos"] = bt["pos"].diff().abs().fillna(bt["pos"].abs())


def pnl(pos, costo):
    dpos = pos.diff().abs().fillna(pos.abs())
    bruto = pos * bt["ret"]
    carry = pos * bt["carry_u"]
    cst = -dpos * costo / S0
    return pd.DataFrame({"bruto": bruto, "carry": carry, "costos": cst,
                         "neto": bruto + carry + cst, "rot": dpos})


estr = {"Triángulos": bt["pos"].astype(float),
        "Siempre +1": pd.Series(1.0, index=bt.index),
        "Comprar siempre (+3)": pd.Series(3.0, index=bt.index),
        "No operar": pd.Series(0.0, index=bt.index)}
res = {n: pnl(p, COSTO_BASE) for n, p in estr.items()}


# ----------------------------------------------------------------------
# MÉTRICAS
# ----------------------------------------------------------------------
def bootstrap_p(x, n_boot=5000, bloque=10, seed=0):
    """Bootstrap estacionario: p-valor unilateral de media <= 0."""
    x = np.asarray(x)
    n = len(x)
    rng = np.random.default_rng(seed)
    xc = x - x.mean()
    medias = np.empty(n_boot)
    for b in range(n_boot):
        ii = np.empty(n, dtype=int)
        ii[0] = rng.integers(n)
        for t in range(1, n):
            ii[t] = rng.integers(n) if rng.random() < 1 / bloque else (ii[t - 1] + 1) % n
        medias[b] = xc[ii].mean()
    return float(np.mean(medias >= x.mean()))


filas = []
for n, r in res.items():
    nt = r["neto"]
    sd = nt.std()
    dd = (nt.cumsum() - nt.cumsum().cummax()).min()
    filas.append({"estrategia": n,
                  "bruto_%": r["bruto"].sum() * 100,
                  "carry_%": r["carry"].sum() * 100,
                  "costos_%": r["costos"].sum() * 100,
                  "neto_%": nt.sum() * 100,
                  "rotación": r["rot"].sum(),
                  "sharpe": nt.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
                  "maxDD_%": dd * 100})
tab = pd.DataFrame(filas).set_index("estrategia")

# Mensual
mens = pd.DataFrame({n: r["neto"].resample("ME").sum() for n, r in res.items()})
act = bt["pos"].ne(0).resample("ME").sum()
tri = mens["Triángulos"]
resumen_mes = {
    "meses": len(mens),
    "meses con posición": int((act > 0).sum()),
    "% meses neto > 0": (tri > 0).mean() * 100,
    "% meses activos neto > 0": (tri[act > 0] > 0).mean() * 100,
    "% meses le gana a +1": (tri > mens["Siempre +1"]).mean() * 100,
    "% meses le gana a +3": (tri > mens["Comprar siempre (+3)"]).mean() * 100,
}

# Por caso
bt["neto"] = res["Triángulos"]["neto"]
por_caso = (bt.groupby("caso")
              .agg(dias=("pos", "size"), pos=("pos", "first"),
                   neto_bps=("neto", lambda x: x.sum() * 1e4),
                   bps_dia=("neto", lambda x: x.mean() * 1e4),
                   acierto=("neto", lambda x: (x[x != 0] > 0).mean()))
              .sort_values("pos"))

pd.set_option("display.width", 140)
print("\n=== RESULTADO (costo base %.2f CLP) ===" % COSTO_BASE)
print(tab.round(2).to_string())
print("\n=== MENSUAL ===")
for kk, v in resumen_mes.items():
    print(f"  {kk:<28} {v:.1f}" if isinstance(v, float) else f"  {kk:<28} {v}")
print("\n=== POR CASO ===")
print(por_caso.round(2).to_string())

sens = pd.DataFrame({f"{c:.2f}": {n: pnl(p, c)["neto"].sum() * 100
                                  for n, p in estr.items()} for c in COSTOS_SENS})
print("\n=== SENSIBILIDAD A COSTOS (neto %, CLP por USD) ===")
print(sens.round(2).to_string())

p_abs = bootstrap_p(res["Triángulos"]["neto"])
p_vs1 = bootstrap_p(res["Triángulos"]["neto"] - res["Siempre +1"]["neto"])
print(f"\nBootstrap estacionario (bloque 10): p(media<=0) = {p_abs:.3f} | "
      f"p(no le gana a +1) = {p_vs1:.3f}")
print("Distribución de posiciones:", bt["pos"].value_counts().sort_index().to_dict())

# ----------------------------------------------------------------------
# GRÁFICOS
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
for n, r in res.items():
    ax.plot(r["neto"].cumsum() * 100, label=n, lw=1.8 if n == "Triángulos" else 1)
ax.axhline(0, color="0.5", lw=0.6)
ax.set_title(f"Neto acumulado (% sobre 1 unidad) · costo {COSTO_BASE} CLP")
ax.legend()
plt.tight_layout()
fig.savefig(FIGS / "curva.png", dpi=130)
plt.show()

fig, ax = plt.subplots(figsize=(12, 4))
x = np.arange(len(mens))
ax.bar(x - 0.2, mens["Triángulos"] * 100, 0.4, label="Triángulos")
ax.bar(x + 0.2, mens["Siempre +1"] * 100, 0.4, label="Siempre +1", alpha=0.6)
ax.set_xticks(x, [d.strftime("%y-%m") for d in mens.index], rotation=90, fontsize=8)
ax.axhline(0, color="0.4", lw=0.6)
ax.set_title("Neto mensual (%)")
ax.legend()
plt.tight_layout()
fig.savefig(FIGS / "mensual.png", dpi=130)
plt.show()

COLOR_CASO = {"RETEST_ARRIBA": "darkgreen", "RUPTURA_ARRIBA": "green",
              "BASE": "teal", "TECHO": "darkorange", "PULLBACK_ABAJO": "red"}
def dibujar(caso, k, t, info, n_ej):
    a, b = max(0, t["ini"] - 30), min(len(C) - 1, k + 20)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(a, b + 1):
        col = "#26a69a" if C[i] >= O[i] else "#ef5350"
        alpha = 1.0 if i <= k else 0.35          # lo posterior a la señal, tenue
        ax.plot([i, i], [L[i], H[i]], color=col, lw=0.7, alpha=alpha)
        ax.plot([i, i], [O[i], C[i]], color=col, lw=2.2, alpha=alpha)
    xs = np.arange(t["ini"], min(b, int(t["apice"]) + 5) + 1)
    ax.plot(xs, recta(t["U"], xs), color="#1565c0", lw=1.4, label="techo")
    ax.plot(xs, recta(t["L"], xs), color="#ef6c00", lw=1.4, label="base")
    ax.axvspan(t["ini"], t["fin"], color="0.85", alpha=0.35)
    ax.axvline(k, color=COLOR_CASO[caso], lw=1.5, ls="--", label="señal (cierre t)")
    for etq, m, cl in (("bc", "^", "k"), ("r", "o", "purple"), ("j", "*", "gold")):
        if info.get(etq) is not None:
            ii = info[etq]
            ax.scatter(ii, C[ii], marker=m, s=90, color=cl, zorder=5,
                       label={"bc": "ruptura confirmada", "r": "apoyo/retest",
                              "j": "retoma"}[etq])
    tk = np.linspace(a, b, 8).astype(int)
    ax.set_xticks(tk, [FECHAS[i].strftime("%Y-%m-%d") for i in tk], fontsize=8)
    pos = int(bt.loc[FECHAS[k], "pos"])
    ax.set_title(f"{caso} · posición {pos:+d} · {FECHAS[k].date()}  "
                 f"(barras tenues = después de la señal)")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    fig.savefig(FIGS / f"ejemplo_{caso}_{n_ej + 1}.png", dpi=130)
    plt.show()



for caso in ["RETEST_ARRIBA", "RUPTURA_ARRIBA", "BASE", "TECHO", "PULLBACK_ABAJO"]:
    eps = ejemplos.get(caso, [])
    print(f"{caso}: {len(eps)} episodios")
    if not eps:
        continue
    for n_ej, (k, t, info) in enumerate(eps[:1] + eps[-1:] if len(eps) > 1 else eps):
        dibujar(caso, k, t, info, n_ej)


# ----------------------------------------------------------------------
# REGISTRO DE EXPERIMENTOS
# ----------------------------------------------------------------------
reg_path = OUT / "registro_experimentos.csv"
fila = {"timestamp": datetime.now().isoformat(timespec="seconds"),
        "experimento": f"02_triangulos_{VERSION}", "separacion_atr": SEPARACION_ATR, "desde": OPERAR_DESDE,
        "hasta": str(HOLDOUT_DESDE.date()), "costo": COSTO_BASE,
        "vigencia": VIGENCIA_RUPTURA, "retest_max": RETEST_MAX, "hold3": HOLD_3,
        "apoyo_atr": APOYO_MAX_ATR, "tol_atr": TOL_ATR,
        "neto_%": round(tab.loc["Triángulos", "neto_%"], 3),
        "sharpe": round(tab.loc["Triángulos", "sharpe"], 3),
        "pct_meses_pos": round(resumen_mes["% meses neto > 0"], 1),
        "p_boot": round(p_abs, 4), "p_vs_+1": round(p_vs1, 4)}
reg_df = pd.DataFrame([fila])
if reg_path.exists():                       # tolera columnas nuevas entre versiones
    reg_df = pd.concat([pd.read_csv(reg_path), reg_df], ignore_index=True)
reg_df.to_csv(reg_path, index=False)
n_prev = len(reg_df)
print(f"\nRegistrado en {reg_path.name} (intento n° {n_prev} en el registro).")
bt.to_csv(OUT / "exp02_senales.csv")
