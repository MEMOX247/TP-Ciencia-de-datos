"""Esquemas de entrada y salida del servicio (contrato de la API)."""

from typing import Literal

from pydantic import BaseModel, Field


class LoanRequest(BaseModel):
    """Datos disponibles al momento de originar el préstamo (sin leakage)."""

    gmv_usd: float = Field(..., gt=0, description="Valor total de la compra (GMV) en USD")
    loan_amount_usd: float = Field(..., gt=0, description="Monto financiado en USD")
    down_payment_usd: float = Field(..., ge=0, description="Pago inicial / enganche en USD")
    first_installment_usd: float = Field(..., gt=0, description="Monto de la primera cuota en USD")
    second_installment_usd: float | None = Field(0.0, ge=0, description="Monto de la 2da cuota (0 si no aplica)")
    third_installment_usd: float | None = Field(0.0, ge=0, description="Monto de la 3ra cuota (0 si no aplica)")
    credit_score: float = Field(..., description="Score crediticio del usuario")
    product: Literal["SPLIT_1", "SPLIT_3", "SPLIT_6", "SPLIT_9", "SPLIT_12"] = Field(
        ..., description="Producto de financiamiento (cantidad de cuotas)"
    )
    product_vertical: Literal[
        "Moda", "Supermercados", "Hogar", "Farmacias", "Tecnología", "Salud"
    ] = Field(..., description="Rubro del comercio")
    order_type: Literal["First order", "Follow up order"] = Field(
        ..., description="Si es la primera compra del cliente o no"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "gmv_usd": 120.0,
                "loan_amount_usd": 90.0,
                "down_payment_usd": 30.0,
                "first_installment_usd": 30.0,
                "second_installment_usd": 30.0,
                "third_installment_usd": 30.0,
                "credit_score": 410,
                "product": "SPLIT_3",
                "product_vertical": "Tecnología",
                "order_type": "First order",
            }
        }


class FPDResponse(BaseModel):
    fpd_probability: float = Field(..., description="Probabilidad estimada de First Payment Default")
    fpd_prediction: int = Field(..., description="1 = riesgo de FPD, 0 = sin riesgo (umbral 0.5)")
    risk_band: Literal["BAJO", "MEDIO", "ALTO"] = Field(..., description="Categorización del riesgo")
    top_drivers: list[str] = Field(..., description="Variables con mayor peso en la decisión del modelo")


class ModelInfoResponse(BaseModel):
    auc_roc_test: float
    recall_fpd: float
    precision_fpd: float
    f1_fpd: float
    note: str
