# portfolio_lab

Sistema modular de análisis y construcción de portafolios de inversión en Python,
con enfoque cuantitativo y estructura diseñada para escalar de forma ordenada.

---

## Objetivo

Construir un pipeline reproducible que cubra el ciclo completo:
selección de activos → descarga de precios → limpieza y alineación →
cálculo de rendimientos → métricas de riesgo y retorno → optimización de pesos.

---

## Alcance — Fase 1

| Dimensión | Decisión |
|---|---|
| Instrumentos | Renta variable (acciones y ETFs) |
| Portafolios | Long-only |
| Optimización | Mínima varianza |
| Fuente de datos | Yahoo Finance (`yfinance`) |
| Tipo de precio | `Adj Close` (ajustado por splits y dividendos) |
| Frecuencia | Diaria |

**Fuera del alcance actual:** deuda, derivados, short selling, optimización robusta,
max-Sharpe, costos de transacción, GARCH, Monte Carlo complejo, bases de datos SQL,
dashboards interactivos.

---

## Estructura del repositorio

```
portfolio_lab/
│
├── config/
│   ├── settings.yaml        # Parámetros globales (fechas, price_field, etc.)
│   └── assets.yaml          # Lista de tickers a analizar
│
├── data/
│   ├── raw/                 # CSVs descargados sin modificar (gitignored)
│   ├── processed/           # Series limpias y alineadas (gitignored)
│   └── metadata/            # Información auxiliar
│
├── notebooks/               # Exploración visual (sin lógica principal)
│   ├── 01_data_exploration.ipynb
│   ├── 02_portfolio_analysis.ipynb
│   └── 03_risk_analysis.ipynb
│
├── outputs/
│   ├── tables/              # Tablas generadas (gitignored)
│   ├── figures/             # Gráficas generadas (gitignored)
│   └── reports/             # Reportes exportados (gitignored)
│
├── scripts/                 # Puntos de entrada del pipeline
│   ├── run_download.py
│   ├── run_prepare_data.py
│   ├── run_portfolio.py     # Fase 2
│   └── run_risk_report.py   # Fase 2
│
└── src/                     # Toda la lógica del proyecto
    ├── data/                # Descarga, carga, limpieza, validación
    ├── analytics/           # Rendimientos, estadísticas, covarianza
    ├── portfolio/           # Construcción, métricas, optimización
    ├── risk/                # VaR, TVaR, escenarios
    ├── reporting/           # Tablas, gráficas, exportación
    └── utils/               # Logger, configuración, paths
```

---

## Flujo del pipeline

```
[config/assets.yaml]
[config/settings.yaml]
        │
        ▼
run_download.py
  → src/data/downloader.py
  → data/raw/<ticker>.csv  (un archivo por activo)
        │
        ▼
run_prepare_data.py
  → src/data/loader.py       (leer raw)
  → src/data/validator.py    (validar cobertura y faltantes)
  → src/data/cleaner.py      (limpiar y alinear — inner join, sin imputación)
  → src/analytics/returns.py (rendimientos simples)
  → data/processed/prices_aligned.csv
  → data/processed/returns.csv
        │
        ▼
run_portfolio.py  [Fase 2]
  → src/analytics/statistics.py  (medias, volatilidades)
  → src/analytics/covariance.py  (matriz de covarianza)
  → src/portfolio/optimization.py (mínima varianza)
  → outputs/tables/portfolio_weights.csv
        │
        ▼
run_risk_report.py  [Fase 2]
  → src/risk/var.py   (VaR histórico y paramétrico)
  → src/risk/tvar.py  (TVaR histórico)
  → outputs/tables/risk_metrics.csv
  → outputs/figures/
```

---

## Principios metodológicos

1. Todos los activos usan la misma fuente y el mismo tipo de precio.
2. El análisis se construye sobre un **periodo común comparable** (inner join).
3. No se imputan precios faltantes — se eliminan.
4. Los datos raw se mantienen intactos en `data/raw/`.
5. Los notebooks no contienen lógica — solo consumen funciones de `src/`.
6. Cada archivo tiene una responsabilidad clara y única.

---

## Instalación

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

Versión de Python recomendada: **3.11+**

---

## Ejecución

Todos los comandos deben ejecutarse desde el directorio `portfolio_lab/`:

```bash
# Paso 1: Descargar precios históricos
python scripts/run_download.py

# Paso 2: Limpiar, validar, alinear y calcular rendimientos
python scripts/run_prepare_data.py

# Paso 3: Optimizar portafolio  [Fase 2]
python scripts/run_portfolio.py

# Paso 4: Reporte de riesgo  [Fase 2]
python scripts/run_risk_report.py
```

### Configuración rápida

Edita `config/assets.yaml` para cambiar los tickers:

```yaml
tickers:
  - AAPL
  - MSFT
  - SPY
```

Edita `config/settings.yaml` para ajustar el periodo o nivel de confianza:

```yaml
start_date: "2019-01-01"
end_date: "2024-12-31"
price_field: "Adj Close"
confidence_level: 0.95
```

---

## Próximos pasos (Fase 2)

- [ ] `analytics/statistics.py` — medias y volatilidades anualizadas
- [ ] `analytics/covariance.py` — matriz de covarianza muestral
- [ ] `portfolio/optimization.py` — mínima varianza con `scipy.optimize`
- [ ] `portfolio/metrics.py` — retorno, varianza y volatilidad del portafolio
- [ ] `risk/var.py` — VaR histórico y paramétrico
- [ ] `risk/tvar.py` — TVaR (Expected Shortfall)
- [ ] `reporting/` — tablas resumen y gráficas con matplotlib
- [ ] `run_portfolio.py` y `run_risk_report.py` — scripts completos
