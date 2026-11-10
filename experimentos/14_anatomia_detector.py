"""
experimentos/14_anatomia_detector.py — Cómo encuentra el detector un triángulo y qué descarta.

DESCRIPTIVO. No modifica el modelo ni elige nada: reproduce paso a paso la cadena de
filtros de triangulos.Detector.detectar y VERIFICA que el resultado sea idéntico al
del detector congelado (si no lo es, el script se detiene).

Salidas
  informe/fig_detector_pasos.png     pivotes → rectas candidatas → triángulo → señales
  informe/fig_detector_galeria.png   qué capta (casos reales) y qué rechaza (casi-triángulos)
  experimentos/exp14_embudo_detector.csv   candidatos que sobreviven a cada filtro

Correr con F5 desde Spyder. Tarda unos minutos: recalcula el detector día a día y verifica cada día contra el congelado.
"""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RAIZ = Path(os.environ.get("QUANTCLP", Path(__file__).resolve().parents[1]))
if not (RAIZ / "estrategia.py").exists():
    raise FileNotFoundError(f"{RAIZ} no es la raíz del repo")
importlib.invalidate_caches()
sys.path.insert(0, str(RAIZ / "src"))
import triangulos as tg  # noqa: E402

BASE = RAIZ / "data" / "base_usdclp.xlsx"
OUT_FIG = RAIZ / "informe"
OUT_CSV = RAIZ / "experimentos" / "exp14_embudo_detector.csv"
OUT_FIG.mkdir(exist_ok=True)

CFG = {"ruptura": "A", "salida": "B", "base": "C"}      # RA-SB-BC-OA (congelado)
EMBUDO_DESDE = "2023-01-01"                             # días para el embudo
DIA_EJEMPLO = "2025-06-19"                              # primer día de la ruptura del ejemplo
GALERIA_CADA = 10                                       # cada cuántos días buscar casi-triángulos
MOSTRAR = True

C_TECHO, C_BASE, C_PRECIO = "#c62828", "#2e7d32", "0.25"
MESES = "ene feb mar abr may jun jul ago sep oct nov dic".split()

# ----------------------------------------------------------------------
# DATOS
# ----------------------------------------------------------------------
df = tg.cargar_usdclp(BASE)
det = tg.Detector(df)
F = df.index
H, L, C, O, A = det.H, det.L, det.C, det.O, det.ATR


def kdia(fecha):
    return int(np.searchsorted(F, pd.Timestamp(fecha)))


# ----------------------------------------------------------------------
# 1) CADENA DE FILTROS INSTRUMENTADA (misma lógica y orden que Detector.detectar)
# ----------------------------------------------------------------------
ETAPAS = ["pares techo–base", "duración 25–220 días", "convergencia ≥ 1,3×",
          "forma (no canal, no rango)", "altura 1,5–8 ATR", "contención ≥ 85 %",
          "toques ≥ 2 por lado", "toques ≥ 5 en total", "ápice adelante ≤ 1,5× largo",
          "sin duplicados, máx. 3"]


def rectas(piv, px):
    pv, out = piv[-tg.MAX_PIVOTES:], []
    for x in range(len(pv)):
        for y in range(x + 1, len(pv)):
            i1, i2 = pv[x], pv[y]
            if tg.SPAN_MIN <= i2 - i1 <= tg.SPAN_MAX:
                out.append({"i1": i1, "p1": px[i1], "m": (px[i2] - px[i1]) / (i2 - i1), "i2": i2})
    return out


def evaluar(U, Lo, hv, lv, cv, av, p_alto, p_bajo, completo=False):
    """Devuelve (etapa en que muere o None, métricas). Con completo=True evalúa todos
    los filtros aunque falle uno (para encontrar casi-triángulos)."""
    ini, fin = min(U["i1"], Lo["i1"]), max(U["i2"], Lo["i2"])
    ok, met = {}, {"ini": ini, "fin": fin}
    ok[1] = tg.SPAN_MIN <= fin - ini <= tg.SPAN_MAX
    if not ok[1] and not completo:
        return 1, met
    h0 = tg.recta(U, ini) - tg.recta(Lo, ini)
    h1 = tg.recta(U, fin) - tg.recta(Lo, fin)
    ok[2] = h0 > 0 and h1 > 0 and h0 / h1 >= tg.CONTRACCION_MIN
    met.update(h0=h0, h1=h1)
    if not ok[2] and not completo:
        return 2, met
    atr_med = float(np.mean(av[ini:fin + 1]))
    if atr_med <= 0:
        return 2, met
    su, si = U["m"] / atr_med, Lo["m"] / atr_med
    plano_s, plano_i = abs(su) < tg.PLANO_ATR, abs(si) < tg.PLANO_ATR
    ok[3] = not (su > tg.PLANO_ATR or si < -tg.PLANO_ATR) and not (plano_s and plano_i)
    met.update(atr_med=atr_med, rango=plano_s and plano_i, pend=abs(su) + abs(si))
    if not ok[3] and not completo:
        return 3, met
    ok[4] = h0 > 0 and tg.ALTURA_MIN_ATR <= h0 / atr_med <= tg.ALTURA_MAX_ATR
    if not ok[4] and not completo:
        return 4, met
    xs = np.arange(ini, fin + 1)
    tol = av[ini:fin + 1] * tg.TOL_ATR
    dentro = np.mean((cv[ini:fin + 1] <= tg.recta(U, xs) + tol) &
                     (cv[ini:fin + 1] >= tg.recta(Lo, xs) - tol))
    ok[5] = dentro >= tg.CONTENCION_MIN
    met["dentro"] = dentro
    if not ok[5] and not completo:
        return 5, met
    ts = [i for i in p_alto if ini <= i <= fin and abs(hv[i] - tg.recta(U, i)) < av[i] * tg.TOL_ATR]
    ti = [i for i in p_bajo if ini <= i <= fin and abs(lv[i] - tg.recta(Lo, i)) < av[i] * tg.TOL_ATR]
    met.update(toq_s=ts, toq_i=ti)
    ok[6] = len(ts) >= tg.TOQUES_MIN_LADO and len(ti) >= tg.TOQUES_MIN_LADO
    if not ok[6] and not completo:
        return 6, met
    ok[7] = len(ts) + len(ti) >= tg.TOQUES_TOTAL
    if not ok[7] and not completo:
        return 7, met
    dm = U["m"] - Lo["m"]
    if dm == 0:
        ok[8] = False
    else:
        apice = ini + (tg.recta(Lo, ini) - tg.recta(U, ini)) / dm
        largo = fin - ini
        ok[8] = fin + tg.APICE_MIN * largo <= apice <= fin + tg.APICE_MAX * largo
        met["apice"] = apice
    met["ok"] = ok
    if completo:
        fallas = [e for e, v in ok.items() if not v]
        return (fallas[0] if fallas else None), met
    return (None if ok[8] else 8), met


def traza(k, completo=False):
    """Recorre la cadena de filtros del día k. Devuelve conteo por etapa, triángulos
    finales y (si completo) los candidatos con métricas."""
    a0g = max(0, k + 1 - tg.VENTANA_HIST)
    hv, lv, cv, av = H[a0g:k + 1], L[a0g:k + 1], C[a0g:k + 1], A[a0g:k + 1]
    p_alto, p_bajo = det._pivotes(hv, lv, cv, av)
    RU, RL = rectas(p_alto, hv), rectas(p_bajo, lv)
    vivos = np.zeros(len(ETAPAS), int)
    vivos[0] = len(RU) * len(RL)
    cand, todos = [], []
    for U in RU:
        for Lo in RL:
            muere, met = evaluar(U, Lo, hv, lv, cv, av, p_alto, p_bajo, completo)
            etapa = muere if muere is not None else 9
            vivos[1:etapa] += 1
            if completo:
                todos.append((muere, U, Lo, met, a0g))
            if muere is None:
                ini, fin = met["ini"], met["fin"]
                T = len(met["toq_s"]) + len(met["toq_i"])
                Ug = {**U, "i1": U["i1"] + a0g, "i2": U["i2"] + a0g}
                Lg = {**Lo, "i1": Lo["i1"] + a0g, "i2": Lo["i2"] + a0g}
                cand.append({"U": Ug, "L": Lg, "ini": ini + a0g, "fin": fin + a0g,
                             "apice": met["apice"] + a0g, "h0": met["h0"],
                             "tipo": tg.clasificar(U["m"], Lo["m"], met["atr_med"]),
                             "score": T / (fin - ini) * 100 * min(met["h0"] / met["h1"], 3.0)})
    cand.sort(key=lambda x: -x["score"])
    aref = float(np.median(av))
    final = []
    for t in cand:
        dup = any(all(abs(tg.recta(t[q], x) - tg.recta(f[q], x)) <= aref * tg.DEDUPE_ATR
                      for x in ((min(t["ini"], f["ini"]) + max(t["fin"], f["fin"])) / 2,
                                max(t["fin"], f["fin"]))
                      for q in ("U", "L"))
                  for f in final)
        if not dup:
            final.append(t)
            if len(final) >= tg.MAX_TRIANGULOS:
                break
    vivos[9] = len(final)
    return vivos, final, todos


# ----------------------------------------------------------------------
# 2) EMBUDO: candidatos que sobreviven a cada filtro, promedio por día
# ----------------------------------------------------------------------
dias = np.where(F >= pd.Timestamp(EMBUDO_DESDE))[0]
print(f"Embudo del detector: {len(dias)} días desde {EMBUDO_DESDE}...")
conteos, n_tri, senales = [], [], []
for k in dias:
    vivos, final, _ = traza(k)
    oficial = det.detectar(k)
    if [(t["ini"], t["fin"], round(t["apice"], 6)) for t in final] != \
       [(t["ini"], t["fin"], round(t["apice"], 6)) for t in oficial]:
        raise RuntimeError(f"La traza no coincide con el detector el {F[k].date()}")
    conteos.append(vivos)
    n_tri.append(len(final))
    senales.append(tg.senal(det, oficial, k, CFG)[0])
conteos = np.array(conteos)
emb = pd.DataFrame({"etapa": ETAPAS,
                    "candidatos promedio por día": conteos.mean(0),
                    "% del total": conteos.mean(0) / conteos[:, 0].mean() * 100,
                    "% días con ≥1": (conteos > 0).mean(0) * 100})
senales = np.array(senales)
extra = pd.DataFrame({"etapa": ["con señal ≠ 0 (posición)", "  de ellos ±3"],
                      "candidatos promedio por día": [np.nan, np.nan],
                      "% del total": [np.nan, np.nan],
                      "% días con ≥1": [(senales != 0).mean() * 100,
                                        (np.abs(senales) == 3).mean() * 100]})
emb = pd.concat([emb, extra], ignore_index=True)
emb.to_csv(OUT_CSV, index=False)
print("Verificación: la traza coincide con el detector congelado en todos los días.")
print(emb.round(3).to_string(index=False))


# ----------------------------------------------------------------------
# UTILIDADES DE GRÁFICO
# ----------------------------------------------------------------------
def velas(ax, a, b, alpha=1.0):
    x = np.arange(a, b + 1)
    ax.vlines(x, L[a:b + 1], H[a:b + 1], color=C_PRECIO, lw=0.7, alpha=alpha)
    ax.hlines(C[a:b + 1], x, x + 0.4, color=C_PRECIO, lw=0.7, alpha=alpha)
    ax.hlines(O[a:b + 1], x - 0.4, x, color=C_PRECIO, lw=0.7, alpha=alpha)


def eje_fechas(ax, a, b, n=5):
    t = np.linspace(a, b, n).astype(int)
    ax.set_xticks(t)
    ax.set_xticklabels([f"{MESES[F[i].month - 1]}-{F[i].year % 100:02d}" for i in t], fontsize=8)
    ax.set_xlim(a - 1, b + 1)


def linea(ax, Ln, x0, x1, **kw):
    xs = np.array([x0, x1])
    ax.plot(xs, tg.recta(Ln, xs), **kw)


def triangulo(ax, t, hasta, alpha=1.0, banda=False):
    xe = min(hasta, t["apice"])
    linea(ax, t["U"], t["ini"], xe, color=C_TECHO, lw=1.8, alpha=alpha)
    linea(ax, t["L"], t["ini"], xe, color=C_BASE, lw=1.8, alpha=alpha)
    if banda:
        xs = np.arange(t["ini"], int(xe) + 1)
        for Ln, c in ((t["U"], C_TECHO), (t["L"], C_BASE)):
            ax.fill_between(xs, tg.recta(Ln, xs) - A[xs] * tg.TOL_ATR,
                            tg.recta(Ln, xs) + A[xs] * tg.TOL_ATR, color=c, alpha=0.12, lw=0)


def pivotes_detalle(k):
    """Pivotes crudos, mechas descartadas y pivotes absorbidos al agruparse."""
    a0g = max(0, k + 1 - tg.VENTANA_HIST)
    hv, lv, cv, av = H[a0g:k + 1], L[a0g:k + 1], C[a0g:k + 1], A[a0g:k + 1]
    w = tg.W_PIVOTE
    crudo_a = [i for i in range(w, len(hv) - w) if hv[i] == hv[i - w:i + w + 1].max()]
    crudo_b = [i for i in range(w, len(lv) - w) if lv[i] == lv[i - w:i + w + 1].min()]
    mecha_a = [i for i in crudo_a if det._mecha(i, hv, cv, av, False)]
    mecha_b = [i for i in crudo_b if det._mecha(i, lv, cv, av, True)]
    fa, fb = det._pivotes(hv, lv, cv, av)
    agr_a = [i for i in crudo_a if i not in mecha_a and i not in fa]
    agr_b = [i for i in crudo_b if i not in mecha_b and i not in fb]
    g = lambda v: np.array(v, int) + a0g
    return dict(fa=g(fa), fb=g(fb), ma=g(mecha_a), mb=g(mecha_b), aa=g(agr_a), ab=g(agr_b))


# ----------------------------------------------------------------------
# 3) FIGURA: LOS PASOS DEL DETECTOR (caso real)
# ----------------------------------------------------------------------
kE = kdia(DIA_EJEMPLO)
_, finalE, _ = traza(kE)
pos_E, caso_E, _ = tg.senal(det, finalE, kE, CFG)
tE = next(t for t in finalE if tg.estado(det, t, kE, CFG)[0] == pos_E)
a, b = tE["ini"] - 15, kE
pv = pivotes_detalle(kE)
dentro_ventana = lambda v: v[(v >= a) & (v <= b)]

fig = plt.figure(figsize=(14, 9.5))
gs = GridSpec(3, 3, height_ratios=[3, 3.2, 1], hspace=0.45, wspace=0.18)
axA, axB, axC = (fig.add_subplot(gs[0, j]) for j in range(3))
axD = fig.add_subplot(gs[1, :])
axP = fig.add_subplot(gs[2, :], sharex=axD)

# (a) pivotes
velas(axA, a, b)
for v, mk, c, lab in ((pv["fa"], "v", C_TECHO, "pivote de máximo"),
                      (pv["fb"], "^", C_BASE, "pivote de mínimo")):
    v = dentro_ventana(v)
    px = H[v] + 0.3 * A[v] if mk == "v" else L[v] - 0.3 * A[v]
    axA.scatter(v, px, marker=mk, color=c, s=30, zorder=3, label=lab)
for v, px in ((dentro_ventana(pv["ma"]), H), (dentro_ventana(pv["mb"]), L)):
    axA.scatter(v, px[v], marker="x", color="k", s=35, zorder=3,
                label="mecha descartada" if len(v) else None)
for v, px in ((dentro_ventana(pv["aa"]), H), (dentro_ventana(pv["ab"]), L)):
    axA.scatter(v, px[v], facecolors="none", edgecolors="0.4", s=30, zorder=3,
                label="absorbido al agrupar" if len(v) else None)
axA.axvspan(b - tg.W_PIVOTE + 0.5, b + 1, color="0.85", lw=0)
axA.text(b - 1.5, axA.get_ylim()[0], "sin\nconfirmar", fontsize=7, ha="right", va="bottom",
         color="0.35")
axA.set_title("(a) Pivotes confirmados\n(extremo de ±3 días; se confirma 3 días después)", fontsize=10, loc="left")
axA.legend(fontsize=7, loc="upper right", framealpha=0.9)
eje_fechas(axA, a, b, 4)

# (b) rectas candidatas
velas(axB, a, b, alpha=0.35)
a0g = max(0, kE + 1 - tg.VENTANA_HIST)
RU = rectas(list(pv["fa"] - a0g), H[a0g:kE + 1])
RL = rectas(list(pv["fb"] - a0g), L[a0g:kE + 1])
for R, c in ((RU, C_TECHO), (RL, C_BASE)):
    for r in R:
        rg = {**r, "i1": r["i1"] + a0g, "i2": r["i2"] + a0g}
        if rg["i1"] >= a:
            linea(axB, rg, rg["i1"], b, color=c, lw=0.6, alpha=0.35)
axB.set_ylim(axA.get_ylim())
axB.set_title(f"(b) Rectas candidatas ({len(RU)} techos × {len(RL)} bases\n"
              f"= {len(RU) * len(RL):,} pares en la ventana de 450 días)".replace(",", "."),
              fontsize=10, loc="left")
eje_fechas(axB, a, b, 4)

# (c) el triángulo aceptado
velas(axC, a, b)
triangulo(axC, tE, b, banda=True)
_, met = evaluar({**tE["U"], "i1": tE["U"]["i1"] - a0g, "i2": tE["U"]["i2"] - a0g},
                 {**tE["L"], "i1": tE["L"]["i1"] - a0g, "i2": tE["L"]["i2"] - a0g},
                 H[a0g:kE + 1], L[a0g:kE + 1], C[a0g:kE + 1], A[a0g:kE + 1],
                 list(pv["fa"] - a0g), list(pv["fb"] - a0g))
ts, ti = np.array(met["toq_s"]) + a0g, np.array(met["toq_i"]) + a0g
axC.scatter(ts, H[ts], s=60, facecolors="none", edgecolors=C_TECHO, lw=1.5, zorder=4)
axC.scatter(ti, L[ti], s=60, facecolors="none", edgecolors=C_BASE, lw=1.5, zorder=4)
largo = tE["fin"] - tE["ini"]
txt = (f"{tg.clasificar(tE['U']['m'], tE['L']['m'], met['atr_med'])} · {largo} días\n"
       f"altura {met['h0'] / met['atr_med']:.1f} ATR · convergencia {met['h0'] / met['h1']:.1f}×\n"
       f"contención {met['dentro']:.0%} · toques {len(ts)}+{len(ti)}")
txt = txt.replace("ASC", "ascendente").replace("DESC", "descendente").replace("SIM", "simétrico")
axC.text(0.02, 0.03, txt, transform=axC.transAxes, fontsize=8, va="bottom",
         bbox=dict(fc="white", ec="0.7", lw=0.5))
axC.set_ylim(axA.get_ylim())
axC.set_title("(c) Triángulo aceptado\n(pasa todos los filtros; banda = ±0,45 ATR)", fontsize=10, loc="left")
eje_fechas(axC, a, b, 4)

# (d) qué hace la estrategia después
b2 = min(len(F) - 1, kE + 70)
hist = []
for k in range(tE["ini"], b2 + 1):
    p, caso, _ = tg.senal(det, det.detectar(k), k, CFG)
    hist.append((k, p, caso))
hist = pd.DataFrame(hist, columns=["k", "p", "caso"]).set_index("k")
velas(axD, a, b2)
triangulo(axD, tE, b2)
xs = np.arange(tE["fin"], int(min(tE["apice"], b2)) + 1)
axD.plot(xs, tg.recta(tE["U"], xs), color=C_TECHO, lw=0.8, ls="--")
axD.plot(xs, tg.recta(tE["L"], xs), color=C_BASE, lw=0.8, ls="--")
d = 1 if pos_E > 0 else -1
lin = tE["U"] if d == 1 else tE["L"]
obj = tg.recta(lin, kE) + d * tE["h0"]
axD.hlines(obj, kE, b2, color="0.3", lw=1, ls=":")
axD.text(b2, obj, " objetivo (altura proyectada)", fontsize=8, va="bottom", ha="right")
xs2 = np.arange(kE, min(kE + 15, b2) + 1)
axD.plot(xs2, tg.recta(lin, xs2) - d * A[xs2] * tg.STOP_B_ATR, color="0.3", lw=1, ls="-.")
axD.text(xs2[-1], tg.recta(lin, xs2[-1]) - d * A[xs2[-1]], " stop: cierre 1 ATR\n adentro de la recta",
         fontsize=8, va="top", ha="left")
lo_, hi_ = L[a:b2 + 1].min(), max(H[a:b2 + 1].max(), obj)
axD.set_ylim(lo_ - 2.5 * A[kE], hi_ + 4.5 * A[kE])
marcas = {"BASE": ("rebote en base +1", C_BASE), "RUPTURA_ARRIBA": ("ruptura +1", "#1565c0"),
          "RETEST_ARRIBA": ("retest +3", "#6a1b9a"), "TECHO": ("rechazo en techo −1", C_TECHO),
          "RUPTURA_ABAJO": ("ruptura −1", "#1565c0"), "RETEST_ABAJO": ("retest −3", "#6a1b9a"),
          "CONFLICTO": ("conflicto → 0", "0.4")}
ya, nivel = set(), 0
for k, r in hist.iterrows():
    if r["caso"] in marcas and r["caso"] not in ya and (k == hist.index[0] or
                                                      hist.loc[k - 1, "caso"] != r["caso"]):
        lab, c = marcas[r["caso"]]
        y = H[k] + (1.5 + 1.6 * (nivel % 3)) * A[k]
        nivel += 1
        axD.annotate(lab, (k, H[k]), (k, y), fontsize=8, color=c, ha="center",
                     arrowprops=dict(arrowstyle="->", color=c, lw=0.8),
                     bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))
        ya.add(r["caso"])
axD.axvline(kE, color="0.5", lw=0.6, ls=":")
axD.set_title(f"(d) Lo que hace la estrategia: cada día se recalcula todo con datos hasta ese "
              f"día (señal del {F[kE].day} {MESES[F[kE].month - 1]} {F[kE].year}: {caso_E}, "
              f"{pos_E:+d})", fontsize=10, loc="left")
axP.step(hist.index, hist["p"], where="post", color="k", lw=1.2)
axP.fill_between(hist.index, 0, hist["p"], step="post", color="#1565c0", alpha=0.25)
axP.set_yticks([-3, -1, 0, 1, 3])
axP.set_ylim(-3.5, 3.5)
axP.set_ylabel("posición", fontsize=8)
axP.axhline(0, color="0.6", lw=0.5)
eje_fechas(axD, a, b2, 8)
plt.setp(axD.get_xticklabels(), visible=False)
eje_fechas(axP, a, b2, 8)
fig.suptitle("Cómo encuentra el detector un triángulo · USD/CLP, caso real del holdout",
             fontsize=12, x=0.01, ha="left")
fig.savefig(OUT_FIG / "fig_detector_pasos.png", dpi=170, bbox_inches="tight")
if MOSTRAR:
    plt.show()
plt.close(fig)


# ----------------------------------------------------------------------
# 4) FIGURA: GALERÍA — qué capta (casos reales) y qué rechaza (casi-triángulos)
# ----------------------------------------------------------------------
def primer_dia(caso, desde, hasta):
    for k in range(kdia(desde), kdia(hasta) + 1):
        tris = det.detectar(k)
        p, c, _ = tg.senal(det, tris, k, CFG)
        if c == caso:
            return k, tris, p
    raise ValueError(f"No encontré {caso} entre {desde} y {hasta}")


def buscar_fallida():
    """Primer día en que una ruptura con posición pasa a cero por ruptura fallida."""
    prev = None
    for k in dias:
        tris = det.detectar(k)
        p, c, _ = tg.senal(det, tris, k, CFG)
        if prev is not None and prev[1].startswith("RUPTURA") and p == 0:
            for t in tris:
                if tg.estado(det, t, k, CFG)[1] in ("EN_DUDA", "FALLIDA"):
                    return k, t, prev[0]
        prev = (k, c)
    return None


CAPTA = []
k1, tr1, _ = primer_dia("RETEST_ABAJO", "2024-04-01", "2024-06-01")
CAPTA.append(("Ruptura bajista + retest → −3", k1,
              [t for t in tr1 if tg.estado(det, t, k1, CFG)[1] == "RETEST_ABAJO"][:1]))
k2, tr2, _ = primer_dia("BASE", "2026-05-20", "2026-06-05")
CAPTA.append(("Rebote en la base con vela de rechazo → +1", k2,
              [t for t in tr2 if tg.estado(det, t, k2, CFG)[1] == "BASE"][:1]))
fall = buscar_fallida()
if fall:
    k3, t3, _ = fall
    CAPTA.append(("Ruptura fallida: vuelve adentro → 0", k3, [t3]))
k4, tr4, _ = primer_dia("CONFLICTO", "2025-06-25", "2025-07-10")
CAPTA.append(("Dos triángulos con señales opuestas → 0", k4,
              [t for t in tr4 if tg.estado(det, t, k4, CFG)[0] != 0]))

# casi-triángulos: cumplen todos los filtros salvo uno
RECHAZA = {2: "Canal: el ancho casi no se reduce", 3: "Rango: techo y base planos",
           5: "Se escapa: < 85 % de cierres dentro", 7: "Pocos toques: menos de 5"}
mejor = {}
for k in dias[::GALERIA_CADA]:
    _, _, todos = traza(k, completo=True)
    for muere, U, Lo, met, a0 in todos:
        if muere not in RECHAZA:
            continue
        fallas = [e for e, v in met["ok"].items() if not v] if "ok" in met else [muere]
        # un canal (rectas paralelas) también tiene el ápice lejos: se acepta {2, 8}
        permitido = [[2], [2, 8]] if muere == 2 else [[muere]]
        if fallas not in permitido or (muere == 3 and not met.get("rango")):
            continue
        if muere == 2 and not (met["h0"] > 0 and met["h1"] > 0):     # canal, no rectas cruzadas
            continue
        T = len(met.get("toq_s", [])) + len(met.get("toq_i", []))
        clave = (T, -met.get("pend", 0)) if muere == 3 else (T, met["fin"] - met["ini"])
        if muere not in mejor or clave > mejor[muere][0]:
            Ug = {**U, "i1": U["i1"] + a0, "i2": U["i2"] + a0}
            Lg = {**Lo, "i1": Lo["i1"] + a0, "i2": Lo["i2"] + a0}
            mejor[muere] = (clave, k, Ug, Lg, met["ini"] + a0, met["fin"] + a0,
                            [i + a0 for i in met.get("toq_s", [])],
                            [i + a0 for i in met.get("toq_i", [])])

n = max(len(CAPTA), len(RECHAZA))
fig, axs = plt.subplots(2, n, figsize=(4.2 * n, 7.5))
for j, (tit, k, ts_) in enumerate(CAPTA):
    ax = axs[0, j]
    a_ = min(t["ini"] for t in ts_) - 5
    b_ = min(len(F) - 1, k + 15)
    velas(ax, a_, b_)
    for t in ts_:
        triangulo(ax, t, b_)
    ax.axvline(k, color="#1565c0", lw=1, ls=":")
    ax.set_title(tit, fontsize=9, loc="left")
    eje_fechas(ax, a_, b_, 3)
axs[0, 0].set_ylabel("LO QUE CAPTA", fontsize=10, fontweight="bold")
for j, (e, tit) in enumerate(RECHAZA.items()):
    ax = axs[1, j]
    if e not in mejor:
        ax.set_visible(False)
        continue
    _, k, U, Lo, ini, fin, ts_, ti_ = mejor[e]
    a_, b_ = ini - 5, min(fin + 10, len(F) - 1)
    velas(ax, a_, b_)
    linea(ax, U, ini, fin + 5, color=C_TECHO, lw=1.6)
    linea(ax, Lo, ini, fin + 5, color=C_BASE, lw=1.6)
    ax.scatter(ts_, H[ts_], s=45, facecolors="none", edgecolors=C_TECHO, lw=1.3, zorder=4)
    ax.scatter(ti_, L[ti_], s=45, facecolors="none", edgecolors=C_BASE, lw=1.3, zorder=4)
    ax.set_title("✗ " + tit, fontsize=9, loc="left")
    eje_fechas(ax, a_, b_, 3)
axs[1, 0].set_ylabel("LO QUE RECHAZA", fontsize=10, fontweight="bold")
fig.suptitle("Arriba: situaciones reales en que la estrategia actúa (línea punteada = día de la "
             "señal). Abajo: candidatos que cumplen todos los filtros salvo uno.",
             fontsize=10, x=0.01, ha="left")
plt.tight_layout()
fig.savefig(OUT_FIG / "fig_detector_galeria.png", dpi=170, bbox_inches="tight")
if MOSTRAR:
    plt.show()
plt.close(fig)
print(f"\nFiguras en {OUT_FIG}: fig_detector_pasos.png · fig_detector_galeria.png")
print(f"Tabla en {OUT_CSV}")
