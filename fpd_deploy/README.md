# FPD Risk API + Mesa de Riesgo (Dashboard)

Despliegue del modelo de predicción de **First Payment Default (FPD)** desarrollado
en la etapa de modelado del curso de Ciencia de Datos Aplicada — ITBA.

## ⚠️ Corrección metodológica importante

Tras la devolución del profesor, se detectó que la variable `Factoring` era
**data leakage**: el 93% de los préstamos en estado `DELAYED` tenían
`Factoring = Yes`, lo cual indica que ese campo se actualiza *después* de
que el préstamo entra en mora — no es un dato disponible al originar el crédito.

**Se re-entrenó el modelo sin esa variable.** El AUC-ROC bajó de 0.9356 a
**0.7218** — una caída esperable y correcta: gran parte del poder predictivo
anterior provenía de hacer "trampa" con información del resultado.

## Estructura del proyecto

```
fpd_deploy/
├── app/
│   ├── main.py            # Capa de servicio (API REST - FastAPI)
│   ├── model_service.py   # Capa de lógica del modelo (feature eng. + predicción)
│   └── schemas.py         # Contratos de entrada/salida (Pydantic)
├── models/
│   ├── modelo_fpd.joblib  # Modelo XGBoost entrenado (sin leakage)
│   ├── features.joblib    # Lista de columnas esperadas por el modelo
│   └── metrics.json       # Métricas de validación
├── dashboard.py            # Interfaz Streamlit — "Mesa de Riesgo"
├── plantilla_creditos.csv  # Plantilla de ejemplo para carga de CSV propio
├── sample_portfolio.csv    # Muestra de 500 préstamos para la demo de ejemplo
├── requirements.txt
└── README.md
```

## Qué incluye el dashboard

1. **Evaluar préstamo individual** — formulario con los datos de un crédito,
   devuelve probabilidad de FPD, banda de riesgo y recomendación.
2. **Cargar base de créditos (CSV)** — el usuario sube su propio archivo con
   varios créditos, el sistema los evalúa a todos y permite descargar los
   resultados.
3. **Demo con portfolio de ejemplo** — para probar el sistema sin tener un
   CSV propio a mano.

## Cómo correr el proyecto

### Opción A — Desplegado en la nube (recomendado para compartir con el equipo)

Ver la sección **"Deploy en Streamlit Community Cloud"** más abajo. El resultado
es un link público (`https://tuapp.streamlit.app`) que cualquiera puede abrir
sin instalar nada.

### Opción B — Local

**1. Instalar dependencias**

```bash
pip install -r requirements.txt
```

**2. Levantar el dashboard (interfaz)**

```bash
streamlit run dashboard.py
```

Se abre en http://localhost:8501

**3. (Opcional) Levantar también la API REST por separado**

Para mostrar en la demo que el modelo también puede consumirse como servicio
independiente (lo pedido explícitamente por la consigna):

```bash
uvicorn app.main:app --reload --port 8000
```

Documentación interactiva (Swagger UI): http://localhost:8000/docs

Endpoints disponibles:
- `GET /health` — chequeo de salud del servicio
- `GET /model-info` — métricas del modelo en producción
- `POST /predict` — predicción de FPD para un préstamo nuevo

> El dashboard NO depende de que la API esté corriendo — usa el modelo
> directamente vía `app/model_service.py`. La API es una segunda forma de
> exponer el mismo modelo, pensada para integraciones (por ejemplo, que un
> sistema externo de la fintech le pegue a `/predict`).

## Deploy en Streamlit Community Cloud (gratis)

1. Subí esta carpeta a un repositorio de **GitHub** (puede ser privado).
2. Entrá a https://share.streamlit.io con tu cuenta de GitHub.
3. Click en **"New app"** → elegí el repo → branch `main` → archivo principal: `dashboard.py`.
4. Click en **"Deploy"**. Tarda 1-3 minutos.
5. Te da un link público (`https://tuapp.streamlit.app`) — compartilo con tus compañeros.

Cualquiera de los integrantes del equipo puede hacer este deploy: alcanza con
que el repo esté en GitHub (puede ser de la cuenta de cualquiera del grupo) y
conectarlo desde su propia cuenta de Streamlit Cloud.

**Importante:** cada vez que hagan `git push` con cambios, la app se redespliega
sola — no hace falta repetir el proceso de deploy.

## Ejemplo de uso de la API (cURL)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gmv_usd": 120.0,
    "loan_amount_usd": 90.0,
    "down_payment_usd": 30.0,
    "first_installment_usd": 30.0,
    "second_installment_usd": 30.0,
    "third_installment_usd": 30.0,
    "credit_score": 410,
    "product": "SPLIT_3",
    "product_vertical": "Tecnología",
    "order_type": "First order"
  }'
```

Respuesta:
```json
{
  "fpd_probability": 0.2803,
  "fpd_prediction": 0,
  "risk_band": "BAJO",
  "top_drivers": ["loan_to_gmv_ratio", "down_payment_ratio", "Product_SPLIT_3"]
}
```

## Formato del CSV para carga masiva

Columnas requeridas (nombres exactos):

| Columna | Tipo | Ejemplo |
|---|---|---|
| gmv_usd | número | 120.0 |
| loan_amount_usd | número | 90.0 |
| down_payment_usd | número | 30.0 |
| first_installment_usd | número | 30.0 |
| second_installment_usd | número (0 si no aplica) | 30.0 |
| third_installment_usd | número (0 si no aplica) | 30.0 |
| credit_score | número | 410 |
| product | texto | SPLIT_1, SPLIT_3, SPLIT_6, SPLIT_9 o SPLIT_12 |
| product_vertical | texto | Moda, Supermercados, Hogar, Farmacias, Tecnología o Salud |
| order_type | texto | First order o Follow up order |

Hay una plantilla descargable directamente desde el dashboard (pestaña "Cargar
base de créditos").

## Notas de diseño

- **Separación de capas**: `model_service.py` no sabe nada de HTTP/Streamlit;
  `main.py` (API) y `dashboard.py` (interfaz) son dos "consumidores" distintos
  de la misma lógica de modelo. Esto permite testear la lógica del modelo de
  forma aislada (como se hizo durante el desarrollo).
- **Validación de entrada**: tanto la API (Pydantic) como la carga de CSV en
  el dashboard validan los datos antes de predecir, y muestran errores claros
  en vez de fallar silenciosamente.
- **Umbral de decisión calibrado**: no se usa el 0.5 por defecto. Se calibró
  en ~0.49 sobre la curva precision-recall del conjunto de validación, dado
  que el `scale_pos_weight` usado en el entrenamiento desplaza la distribución
  de probabilidades del modelo.
