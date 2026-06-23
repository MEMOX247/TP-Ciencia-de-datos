"""
Capa de servicio — API REST que expone el modelo de predicción de FPD.

Ejecutar localmente:
    uvicorn app.main:app --reload --port 8000

Documentación interactiva (Swagger UI):
    http://localhost:8000/docs
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model_service import fpd_service
from app.schemas import FPDResponse, LoanRequest, ModelInfoResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fpd_api")

app = FastAPI(
    title="FPD Risk API",
    description=(
        "Servicio de predicción de First Payment Default (FPD) para préstamos BNPL. "
        "Expone el modelo XGBoost entrenado en la etapa de modelado, sin variables "
        "de leakage (ver /model-info para el detalle de métricas)."
    ),
    version="2.0.0",
)

# Habilitado para que el dashboard (Streamlit, en otro puerto) pueda consumir la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Salud"])
def root():
    return {"status": "ok", "service": "FPD Risk API", "version": "2.0.0"}


@app.get("/health", tags=["Salud"])
def health():
    """Chequeo simple de que el modelo está cargado y listo."""
    return {"status": "healthy", "model_loaded": fpd_service.model is not None}


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Modelo"])
def model_info():
    """Métricas de validación del modelo actualmente en producción."""
    m = fpd_service.metrics
    return ModelInfoResponse(
        auc_roc_test=m["auc_roc_test"],
        recall_fpd=m["recall_fpd"],
        precision_fpd=m["precision_fpd"],
        f1_fpd=m["f1_fpd"],
        note=(
            "Modelo re-entrenado sin la variable 'Factoring', identificada como data "
            "leakage: el 93% de los préstamos en mora tenían Factoring=Yes, lo cual "
            "indica que el campo se actualiza post-originación."
        ),
    )


@app.post("/predict", response_model=FPDResponse, tags=["Predicción"])
def predict(loan: LoanRequest):
    """
    Recibe los datos de un préstamo al momento de originación y devuelve
    la probabilidad estimada de First Payment Default (FPD).
    """
    try:
        result = fpd_service.predict(loan.model_dump())
        return FPDResponse(**result)
    except Exception as exc:  # manejo de errores explícito, pedido por la consigna
        logger.exception("Error al generar la predicción")
        raise HTTPException(
            status_code=400, detail=f"No se pudo generar la predicción: {exc}"
        ) from exc
