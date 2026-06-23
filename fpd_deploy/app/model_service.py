"""
Capa de lógica del modelo — separada de la capa de servicio (API).

Responsabilidades:
- Cargar el modelo entrenado y la lista de features esperadas.
- Aplicar el mismo feature engineering usado en el entrenamiento.
- Generar predicciones y explicaciones básicas de riesgo.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent.parent / "models"


class FPDModelService:
    """Encapsula el modelo de predicción de First Payment Default (FPD)."""

    def __init__(self) -> None:
        self.model = joblib.load(MODELS_DIR / "modelo_fpd.joblib")
        self.feature_columns: list[str] = joblib.load(MODELS_DIR / "features.joblib")
        with open(MODELS_DIR / "metrics.json", encoding="utf-8") as f:
            self.metrics: dict = json.load(f)

        # Categorías válidas vistas en entrenamiento (para generar dummies correctas)
        self.valid_products = ["SPLIT_1", "SPLIT_3", "SPLIT_6", "SPLIT_9", "SPLIT_12"]
        self.valid_verticals = [
            "Moda", "Supermercados", "Hogar", "Farmacias", "Tecnología", "Salud",
        ]
        self.valid_order_types = ["First order", "Follow up order"]

    # ------------------------------------------------------------------
    # Feature engineering (debe ser EXACTAMENTE el mismo que en el training)
    # ------------------------------------------------------------------
    def _build_features(self, payload: dict) -> pd.DataFrame:
        row = {
            "GMV - USD": payload["gmv_usd"],
            "Loan Amount - USD": payload["loan_amount_usd"],
            "Down Payment Amount - USD": payload["down_payment_usd"],
            "1st Installment Amount - USD": payload["first_installment_usd"],
            "2nd Installment Amount - USD": payload.get("second_installment_usd", 0.0) or 0.0,
            "3rd Installment Amount - USD": payload.get("third_installment_usd", 0.0) or 0.0,
            "User Credit Score": payload["credit_score"],
        }
        df = pd.DataFrame([row])

        n_cuotas = self._n_cuotas_from_product(payload["product"])
        df["n_cuotas"] = n_cuotas
        df["down_payment_ratio"] = (df["Down Payment Amount - USD"] / df["GMV - USD"]).clip(0, 1)
        df["avg_installment_usd"] = df["Loan Amount - USD"] / n_cuotas
        df["is_first_order"] = int(payload["order_type"] == "First order")
        df["installment_to_gmv"] = (df["avg_installment_usd"] / df["GMV - USD"]).clip(0, 5)
        df["loan_to_gmv_ratio"] = (df["Loan Amount - USD"] / df["GMV - USD"]).clip(0, 1)
        df["credit_score_bin"] = pd.cut(
            df["User Credit Score"], bins=[-100, 300, 400, 500, 2000], labels=[0, 1, 2, 3]
        ).astype(int)

        # Dummies manuales (drop_first=True como en el training)
        for p in self.valid_products[1:]:  # SPLIT_1 es la categoría base (drop_first)
            df[f"Product_{p}"] = int(payload["product"] == p)
        for v in self.valid_verticals[1:]:  # Moda es la base
            df[f"Product Type Vertical_{v}"] = int(payload["product_vertical"] == v)
        df["Order Type_Follow up order"] = int(payload["order_type"] == "Follow up order")

        # Reordenar / completar columnas exactamente como en el training
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]
        return df

    @staticmethod
    def _n_cuotas_from_product(product: str) -> int:
        return int(product.replace("SPLIT_", ""))

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------
    DECISION_THRESHOLD = 0.49  # calibrado en validación: ~recall 0.75

    def predict(self, payload: dict) -> dict:
        X = self._build_features(payload)
        proba = float(self.model.predict_proba(X)[0, 1])
        pred = int(proba >= self.DECISION_THRESHOLD)

        risk_band = self._risk_band(proba)
        top_drivers = self._top_drivers(X)

        return {
            "fpd_probability": round(proba, 4),
            "fpd_prediction": pred,
            "risk_band": risk_band,
            "top_drivers": top_drivers,
        }

    @staticmethod
    def _risk_band(proba: float) -> str:
        # Bandas calibradas sobre la distribución real de probabilidades del modelo
        # (media ~0.42, por el scale_pos_weight usado en el entrenamiento)
        if proba < 0.30:
            return "BAJO"
        if proba < 0.49:
            return "MEDIO"
        return "ALTO"

    def _top_drivers(self, X: pd.DataFrame, n: int = 3) -> list[str]:
        """Top features con mayor importancia global, presentes en este caso (informativo, no SHAP)."""
        importances = pd.Series(
            self.model.feature_importances_, index=self.feature_columns
        ).sort_values(ascending=False)
        return importances.head(n).index.tolist()


# Singleton — se carga una sola vez al iniciar la API
fpd_service = FPDModelService()
