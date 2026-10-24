# ¿Sube o baja el dólar mañana? — Torneo FEN–Banco Itaú

Estrategia sistemática para predecir la dirección del USD/CLP del día siguiente
y convertirla en una posición ∈ {−3, −1, 0, +1, +3}.

**Modelo:** detector de triángulos sobre el OHLC diario del USD/CLP (variante
RA-SB-BC-OA). Entra en rupturas confirmadas y en retests, y se abstiene (0)
cuando no hay patrón vigente.

## Resultado fuera de muestra (abr 2025 – oct 2026, 377 días)

| Estrategia | Neto (carry y costos incluidos) | Sharpe | MaxDD |
|---|---|---|---|
| **Triángulos** | **+11,7%** | **1,00** | −6,0% |
| Comprar siempre (+3) | −4,9% | −0,09 | −48,5% |
| Siempre +1 | −1,6% | −0,09 | −16,2% |
| No operar | 0,0% | — | 0,0% |

- Diebold-Mariano direccional: supera a las tres referencias (p = 0,022 – 0,045).
- Bootstrap estacionario del neto: p = 0,053 contra no operar; 0,118 contra siempre +1.
- Robusto a costos: +6,6% con 2 CLP por unidad de cambio de posición.

## Cómo correrlo

1. Python 3.11+ e instalar dependencias: `pip install -r requirements.txt`
2. Copiar la base del torneo a `data/base_usdclp.xlsx` (no se incluye: datos Bloomberg).
3. Ejecutar `python estrategia.py` (o F5 en Spyder). Para una base actualizada:
   `python estrategia.py ruta/a/la/base.xlsx`

Genera en `salidas/`:
- `senal_del_dia.json` — posición, dirección, caso y convicción para la jornada siguiente.
- `curva.png` — resultado neto acumulado contra las tres referencias.
- `historial.csv` — posición, caso y resultado de cada día.

En consola imprime además la lectura de convicción (días en ±3, ±1 y 0).

## Estructura

```
estrategia.py          estrategia automatizada (señal del día, curva, convicción)
src/triangulos.py      carga de datos, detector de triángulos y reglas de posición
src/probit_ordenado.py probit ordenado por máxima verosimilitud (experimentos 05–08)
experimentos/          todos los experimentos, en orden, y el registro de intentos
informe/               informe final
data/                  (vacío) aquí va la base del torneo
```

## Supuestos

- Resultado diario = posición × variación del USD/CLP + carry − costos.
- Carry: TPM (CHOVCHOV) − Fed funds (FEDL01, rezago de 1 día por publicación), act/360.
  Largo en dólar paga el diferencial, corto lo recibe.
- Costos: 0,5 CLP por dólar por unidad de cambio de posición; sensibilidad 0 – 2 CLP.
- "Comprar siempre" se interpreta como +3 todos los días.
- Solo se usa información disponible al cierre de t. Feriados con precio repetido
  y fines de semana se eliminan; OHLC desde 2004.

## Protocolo

- Holdout: últimos 18 meses de la base, no usados para elegir el modelo.
- La variante se eligió entre 24 con datos hasta marzo de 2025 (experimentos 03–04)
  y se evaluó una sola vez en el holdout (04b–04c).
- `experimentos/registro_experimentos.csv` lista todos los intentos (53), incluidos
  los descartados.
