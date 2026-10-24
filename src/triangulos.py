"""
src/triangulos.py — Detección de triángulos (geometría del screener v14b) y
estados de posición, todo causal: en el día k solo se usan barras <= k.

Uso:
    from triangulos import cargar_usdclp, Detector, estado, CFG_BASE
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# GEOMETRÍA (idéntica al screener v14b)
# ----------------------------------------------------------------------
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
ATR_N = 14
MECHA_MAX_ATR = 1.00
APOYO_MAX_ATR = 0.60
MAX_TRIANGULOS = 3
VENTANA_HIST = 450


# ----------------------------------------------------------------------
# DATOS
# ----------------------------------------------------------------------
def cargar_usdclp(base: Path) -> pd.DataFrame:
    cols = {"USDCLP Curncy|PX_OPEN": "o", "USDCLP Curncy|PX_HIGH": "h",
            "USDCLP Curncy|PX_LOW": "l", "USDCLP Curncy|PX_LAST": "c",
            "CHN1M Curncy|PX_LAST": "pts1m"}
    raw = pd.read_excel(base, sheet_name="datos", usecols=["fecha", *cols])
    df = (raw.rename(columns=cols)
             .assign(fecha=lambda x: pd.to_datetime(x["fecha"]))
             .set_index("fecha").sort_index())
    df = df.dropna(subset=["o", "h", "l", "c"])
    df = df[df.index.dayofweek < 5]                 # calendario corrido
    df["pts1m"] = df["pts1m"].ffill()
    df = df.loc["2004-01-01":]                      # OHLC plano antes de 2004
    rancio = (df["h"] == df["l"]) & (df["c"].diff() == 0)
    return df[~rancio]


def atr_serie(h, l, c, n=ATR_N):
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).rolling(n, min_periods=1).mean().to_numpy()


def recta(Ln, x):
    return Ln["p1"] + Ln["m"] * (x - Ln["i1"])


def clasificar(m_sup, m_inf, atr_med):
    su, si = m_sup / atr_med, m_inf / atr_med
    if abs(su) < PLANO_ATR and si > 0:
        return "ASC"
    if abs(si) < PLANO_ATR and su < 0:
        return "DESC"
    return "SIM"


# ----------------------------------------------------------------------
# DETECCIÓN
# ----------------------------------------------------------------------
class Detector:
    def __init__(self, df):
        self.H, self.L, self.C, self.O = (df[k].to_numpy(float)
                                          for k in ("h", "l", "c", "o"))
        self.ATR = atr_serie(self.H, self.L, self.C)

    @staticmethod
    def _agrupar(piv, px, atr_s, buscar_max):
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

    @staticmethod
    def _mecha(i, px, c, atr_s, es_min):
        a, b = max(0, i - 2), min(len(c), i + 3)
        if es_min:
            return (c[a:b].min() - px[i]) > atr_s[i] * MECHA_MAX_ATR
        return (px[i] - c[a:b].max()) > atr_s[i] * MECHA_MAX_ATR

    def _pivotes(self, h, l, c, a, w=W_PIVOTE):
        altos = [i for i in range(w, len(h) - w)
                 if h[i] == h[i - w:i + w + 1].max()
                 and not self._mecha(i, h, c, a, False)]
        bajos = [i for i in range(w, len(l) - w)
                 if l[i] == l[i - w:i + w + 1].min()
                 and not self._mecha(i, l, c, a, True)]
        return (self._agrupar(altos, h, a, True),
                self._agrupar(bajos, l, a, False))

    def detectar(self, k):
        a0g = max(0, k + 1 - VENTANA_HIST)
        hv, lv = self.H[a0g:k + 1], self.L[a0g:k + 1]
        cv, av = self.C[a0g:k + 1], self.ATR[a0g:k + 1]
        p_alto, p_bajo = self._pivotes(hv, lv, cv, av)

        def rectas(piv, px):
            pv, out = piv[-MAX_PIVOTES:], []
            for x in range(len(pv)):
                for y in range(x + 1, len(pv)):
                    i1, i2 = pv[x], pv[y]
                    if SPAN_MIN <= i2 - i1 <= SPAN_MAX:
                        out.append({"i1": i1, "p1": px[i1], "m": (px[i2] - px[i1]) / (i2 - i1),
                                    "i2": i2})
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
                t_sup = sum(1 for i in p_alto if ini <= i <= fin
                            and abs(hv[i] - recta(U, i)) < av[i] * TOL_ATR)
                t_inf = sum(1 for i in p_bajo if ini <= i <= fin
                            and abs(lv[i] - recta(Lo, i)) < av[i] * TOL_ATR)
                if t_sup < TOQUES_MIN_LADO or t_inf < TOQUES_MIN_LADO:
                    continue
                T = t_sup + t_inf
                if T < TOQUES_TOTAL:
                    continue
                dm = U["m"] - Lo["m"]
                if dm == 0:
                    continue
                apice = ini + (recta(Lo, ini) - recta(U, ini)) / dm
                largo = fin - ini
                if not (fin + APICE_MIN * largo <= apice <= fin + APICE_MAX * largo):
                    continue
                Ug = {**U, "i1": U["i1"] + a0g, "i2": U["i2"] + a0g}
                Lg = {**Lo, "i1": Lo["i1"] + a0g, "i2": Lo["i2"] + a0g}
                cand.append({"U": Ug, "L": Lg, "ini": ini + a0g, "fin": fin + a0g,
                             "apice": apice + a0g, "h0": h0,
                             "tipo": clasificar(U["m"], Lo["m"], atr_med),
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
# ESTADOS
# ----------------------------------------------------------------------
# cfg:
#   ruptura  "A" = 3 cierres seguidos fuera + 0,45 ATR  |  "B" = 1 cierre a >= 1 ATR
#   salida   "A" = vence a los 20 días                   |  "B" = objetivo (altura
#                proyectada) o stop (cierre 1 ATR adentro), máx. 60 días
#   base     "A" = a <= 0,6 ATR de la recta              |  "B" = A + no pasado el
#                75% del camino al ápice                 |  "C" = A + vela de rechazo
CFG_BASE = {"ruptura": "A", "salida": "A", "base": "A"}

N_CONFIRMA = 3
RUPT_B_ATR = 1.0
VIGENCIA_A = 20
VIGENCIA_B_MAX = 60
STOP_B_ATR = 1.0
DIAS_VOLVER = 2          # si vuelve a cerrar adentro, tiene 2 días para salir otra vez
HOLD_3 = 3
AVANCE_MAX = 0.75


def estado(det, t, k, cfg):
    """(posición, caso) del triángulo t al cierre de k."""
    H, L, C, O, A = det.H, det.L, det.C, det.O, det.ATR
    U, Lo, fin, apice = t["U"], t["L"], t["fin"], t["apice"]

    # 1) ¿rompió? -------------------------------------------------------
    bc, d = None, 0
    arr = aba = 0
    for i in range(fin + 1, k + 1):
        if cfg["ruptura"] == "A":
            tl = A[i] * TOL_ATR
            if C[i] > recta(U, i) + tl:
                arr, aba = arr + 1, 0
            elif C[i] < recta(Lo, i) - tl:
                aba, arr = aba + 1, 0
            else:
                arr = aba = 0
            if arr >= N_CONFIRMA or aba >= N_CONFIRMA:
                bc, d, ini_r = i, (1 if arr else -1), i - N_CONFIRMA + 1
                break
        else:
            if C[i] >= recta(U, i) + A[i] * RUPT_B_ATR:
                bc, d, ini_r = i, 1, i
                break
            if C[i] <= recta(Lo, i) - A[i] * RUPT_B_ATR:
                bc, d, ini_r = i, -1, i
                break

    # 2) sin ruptura: base / techo -----------------------------------------
    if bc is None:
        if k > apice:
            return 0, "DENTRO"
        a = A[k]
        d_base = (C[k] - recta(Lo, k)) / a
        d_techo = (recta(U, k) - C[k]) / a
        en_base = 0 <= d_base <= APOYO_MAX_ATR
        en_techo = 0 <= d_techo <= APOYO_MAX_ATR
        if cfg["base"] == "B":
            avance = (k - t["ini"]) / max(apice - t["ini"], 1)
            if avance > AVANCE_MAX:
                return 0, "DENTRO"
        if cfg["base"] == "C":
            mitad = (H[k] + L[k]) / 2
            en_base = en_base and C[k] > O[k] and C[k] >= mitad
            en_techo = en_techo and C[k] < O[k] and C[k] <= mitad
        if en_base:
            return 1, "BASE"
        if en_techo:
            return -1, "TECHO"
        return 0, "DENTRO"

    if ini_r > apice:
        return 0, "VENCIDA"

    # 3) post-ruptura ------------------------------------------------------
    linea = U if d == 1 else Lo
    lado = lambda i: d * (C[i] - recta(linea, i))      # > 0: afuera (bien)
    objetivo = recta(linea, bc) + d * t["h0"]
    vig = VIGENCIA_A if cfg["salida"] == "A" else VIGENCIA_B_MAX
    if k - bc > vig:
        return 0, "VENCIDA"

    adentro_desde = None
    toque = None
    retoma = None
    for i in range(bc + 1, k + 1):
        if cfg["salida"] == "B":
            extremo = H[i] if d == 1 else L[i]
            if d * (extremo - objetivo) >= 0:
                return 0, "OBJETIVO"
            if lado(i) < -A[i] * STOP_B_ATR:
                return 0, "STOP"
        # regla de los 2 días: si cierra adentro, debe volver a salir
        if lado(i) < 0:
            if adentro_desde is None:
                adentro_desde = i
            if i - adentro_desde >= DIAS_VOLVER:
                return 0, "FALLIDA"
        else:
            adentro_desde = None
        # apoyo: toca la línea rota
        if toque is None:
            ext = L[i] if d == 1 else H[i]
            if d * (ext - recta(linea, i)) <= A[i] * APOYO_MAX_ATR:
                toque = i
        # rechazo: tras el toque, cierra afuera
        if toque is not None and retoma is None and lado(i) > 0:
            retoma = i

    if adentro_desde is not None:                       # en duda: afuera de mercado
        return 0, "EN_DUDA"
    sufijo = "ARRIBA" if d == 1 else "ABAJO"
    if retoma is not None and k < retoma + HOLD_3:
        return 3 * d, f"RETEST_{sufijo}"
    return d, f"RUPTURA_{sufijo}"


def senal(det, tris, k, cfg):
    activos = []
    for t in tris:
        p, caso = estado(det, t, k, cfg)
        if p != 0:
            activos.append((p, caso, t["tipo"]))
    if not activos:
        return 0, "NADA", ""
    mx = max(abs(p) for p, _, _ in activos)
    top = [x for x in activos if abs(x[0]) == mx]
    if len({np.sign(p) for p, _, _ in top}) > 1:
        return 0, "CONFLICTO", ""
    return top[0]                                       # tris ya vienen por score
