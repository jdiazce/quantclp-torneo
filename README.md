# quantclp-torneo · Triángulos y compresión en el USD/CLP

Estrategia para el Torneo FEN–Banco Itaú **"¿Sube o baja el dólar mañana?"**
Juan Pablo Díaz Cerda · Rodrigo Gutierrez Albornoz · Facultad de Economía y Negocios, Universidad de Chile

La estrategia detecta triángulos convergentes en el precio diario del USD/CLP y toma posición
(−3, −1, 0, +1, +3) solo ante rupturas confirmadas, retests y rebotes en los bordes del patrón.
El resto del tiempo se abstiene. Fuera de muestra (7 abr 2025 – 5 oct 2026) obtiene **+11,7% neto
de carry y costos**, con Sharpe 1,00 y caída máxima de −6,0%.

## Resultado fuera de muestra (7 abr 2025 – 5 oct 2026, 377 días)

| Estrategia | Neto (carry y costos incluidos) | Sharpe | Caída máx. |
|---|---|---|---|
| **Triángulos** | **+11,7%** | **1,00** | −6,0% |
| Comprar siempre (+3) | −4,9% | −0,09 | −48,5% |
| Siempre +1 | −1,6% | −0,09 | −16,2% |
| No operar | 0,0% | — | 0,0% |

- Diebold-Mariano direccional: supera a las tres referencias (p = 0,022 – 0,045).
- Bootstrap estacionario del neto: p = 0,053 contra no operar; 0,118 contra siempre +1.
- Robusto a costos: +6,6% con 2 CLP por unidad de cambio de posición; el neto llega a cero recién con ~3,9 CLP.
- Robusto al carry: +12,1% si el carry se mide con puntos NDF 1M en vez de TPM − Fed funds.

---

## Cómo ejecutarlo, paso a paso

Requisitos: Python 3.11 o superior.

### 1. Clonar el repositorio
```bash
git clone https://github.com/jdiazce/quantclp-torneo.git
cd quantclp-torneo
```

### 2. Instalar dependencias
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Poner la base de datos
La base del torneo no se incluye en el repositorio porque son datos de Bloomberg. Hay que copiarla en:
```
data/base_usdclp.xlsx
```
Es el archivo entregado en el kick off, con las hojas `datos` y `diccionario`.

### 4. Ejecutar
**Señal del día, curva y lectura de convicción** (unos minutos; también funciona con F5 en Spyder):
```bash
python estrategia.py
```
Con una base actualizada en otra ruta:
```bash
python estrategia.py ruta/a/la/base.xlsx
```

**Backtest completo con estadísticas y gráficos:**
```bash
jupyter lab backtest.ipynb
```
Luego, *Run All*. El notebook incluye las estadísticas de trading (neto, volatilidad, Sharpe,
caída máxima, % de aciertos y número de operaciones), los gráficos, los supuestos y el mecanismo de
ejecución. El notebook guardado en el repositorio ya trae los resultados visibles.

**Experimentos** (opcional): cada script de `experimentos/` se corre por separado, por ejemplo
`python experimentos/15_sensibilidad_parametros.py`. Algunos requieren haber corrido antes
`estrategia.py`.

### Salidas de `estrategia.py` (carpeta `salidas/`)
| Archivo | Contenido |
|---|---|
| `senal_del_dia.json` | Posición para la jornada siguiente, dirección, caso, tipo de triángulo, P(sube) y convicción |
| `curva.png` | Neto acumulado vs. comprar siempre, siempre +1 y no operar, desde 2023 |
| `historial.csv` | Posición, caso, retorno y neto de cada día |

En consola imprime además el resultado por tramo y la lectura de convicción (días en ±3, ±1 y 0).

---

## Mecanismo de ejecución (para el monitor del laboratorio)

- **Al cierre, close-to-close.** La señal del día *t* se calcula después del cierre de Santiago, con el
  OHLC de `USDCLP Curncy` hasta *t*. La posición se toma al cierre de *t* y se mantiene hasta el cierre
  de *t+1*. El retorno es `PX_LAST(t+1) / PX_LAST(t) − 1`.
- **No se usa** la apertura como precio de ejecución ni el dólar observado.
- **Datos que necesita cada día:** `PX_OPEN`, `PX_HIGH`, `PX_LOW` y `PX_LAST` de `USDCLP Curncy`. La
  señal usa los cuatro campos, no solo el cierre. `CHOVCHOV Index` (TPM) y `FEDL01 Index` (Fed funds)
  solo se usan para contabilizar el carry.
- **Rutina diaria:** agregar la fila del día a la base → `python estrategia.py base.xlsx` → leer
  `posicion` en `salidas/senal_del_dia.json`.
- **Feriados:** los días con máximo = mínimo y cierre igual al anterior se tratan como mercado cerrado
  y se omiten.
- **Parámetros congelados:** nada se modifica durante el monitoreo.

### Qué gatilla cada posición
| Posición | Caso | Gatillo al cierre de *t* |
|---|---|---|
| +3 | `RETEST_ARRIBA` | Ruptura alcista previa; el mínimo vuelve a ≤ 0,6 ATR del techo roto y el precio cierra otra vez afuera. Dura 3 días |
| +1 | `RUPTURA_ARRIBA` | 3 cierres seguidos sobre el techo por más de 0,45 ATR, mientras siga vigente |
| +1 | `BASE` | Cierre a ≤ 0,6 ATR sobre la base, con cierre > apertura y en la mitad superior del rango del día |
| −1 | `TECHO` | Cierre a ≤ 0,6 ATR bajo el techo, con cierre < apertura y en la mitad inferior del rango |
| −1 | `RUPTURA_ABAJO` | 3 cierres seguidos bajo la base por más de 0,45 ATR |
| −3 | `RETEST_ABAJO` | Simétrico al retest alcista. Dura 3 días |
| 0 | — | Sin triángulo vigente, ápice vencido, ruptura fallida o señales opuestas |

Una ruptura se cierra con la primera de estas condiciones: alcanza el objetivo (altura del triángulo),
cierra de vuelta dentro del triángulo, cierra más de 1 ATR adentro de la recta rota o cumple 60 días hábiles.

---

## Modelo y supuestos

- **Un solo modelo:** detector algorítmico de triángulos sobre el OHLC del USD/CLP
  (`src/triangulos.py`), variante congelada RA-SB-BC-OA. No hay combinación de modelos: el probit
  ordenado, la señal post-cierre, las reglas de tendencia y los modelos de aprendizaje automático se
  probaron y se descartaron antes del holdout (`experimentos/registro_experimentos.csv`).
- **Probabilidad de alza:** frecuencia de acierto de cada nivel de exposición en la ventana de
  selección (51,2% para ±1, 57,1% para ±3, 50,9% en abstención).
- **Carry:** TPM − Fed funds (rezagada un día), act/360 por días corridos. Largo paga, corto recibe.
- **Costos:** 0,5 CLP por dólar por unidad de cambio de posición (≈ 5 pb); sensibilidad de 0 a 2 CLP.
- **Referencias:** comprar siempre = +3 todos los días; siempre +1; no operar.
- **Ventanas:** selección ene 2023 – 4 abr 2025; holdout 7 abr 2025 – 5 oct 2026.
- **Datos:** solo información disponible al cierre de *t*. Se eliminan fines de semana y feriados con
  precio repetido; el OHLC se usa desde 2004.

## Protocolo

- **Holdout bloqueado:** los últimos 18 meses de la base no se usaron para elegir ni ajustar nada.
- **Selección:** la variante se eligió entre 24 con la ventana de selección (experimentos 03–04) y se
  evaluó una sola vez en el holdout, en dos aperturas registradas (04b–04c).
- **Registro:** `experimentos/registro_experimentos.csv` lista todos los intentos, incluidos los descartados.

El detalle está en el notebook (`backtest.ipynb`) y en el informe (`informe/`).

---

## Estructura
```
estrategia.py        Estrategia automatizada: señal del día, curva y convicción
backtest.ipynb       Backtest con estadísticas de trading y gráficos
src/triangulos.py    Carga de datos, detector y reglas de posición (modelo congelado)
src/probit_ordenado.py  Probit ordenado por máxima verosimilitud (experimentos 05–08, descartado)
experimentos/        Scripts numerados de cada prueba (02 a 18) y registro de experimentos
informe/             Informe en LaTeX y sus figuras
data/                Aquí va base_usdclp.xlsx (no incluida)
salidas/             Se genera al ejecutar estrategia.py
```
