"""
Dashboard de Mesa de Riesgo — Predicción de First Payment Default (FPD).

Usa el modelo directamente (sin pasar por HTTP) para poder desplegarse en
Streamlit Community Cloud como un único proceso.

La lógica de negocio (model_service.py) es la misma que usa la API REST
(carpeta app/), que se puede correr de forma independiente con:
    uvicorn app.main:app --reload --port 8000
Esa es la versión "servicio" pedida por la consigna; este dashboard es
la interfaz que en producción consumiría esa API.

Ejecutar localmente:
    streamlit run dashboard.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from app.model_service import fpd_service  # noqa: E402

st.set_page_config(page_title="Mesa de Riesgo — FPD", page_icon="📊", layout="wide")

REQUIRED_COLUMNS = [
    "gmv_usd", "loan_amount_usd", "down_payment_usd", "first_installment_usd",
    "second_installment_usd", "third_installment_usd", "credit_score",
    "product", "product_vertical", "order_type",
]
VALID_PRODUCTS = ["SPLIT_1", "SPLIT_3", "SPLIT_6", "SPLIT_9", "SPLIT_12"]
VALID_VERTICALS = ["Moda", "Supermercados", "Hogar", "Farmacias", "Tecnología", "Salud"]
VALID_ORDER_TYPES = ["First order", "Follow up order"]

# ──────────────────────────────────────────────────────────────────────────
# Estilos
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #0D1B2A; }
    .metric-card {
        background-color: #14253B; border-left: 4px solid #0E9AA7;
        padding: 1rem 1.2rem; border-radius: 6px; margin-bottom: 0.5rem;
    }
    h1, h2, h3, p, span, label { color: #F0F4F7 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def call_predict(payload: dict) -> dict:
    return fpd_service.predict(payload)


def df_row_to_payload(row: pd.Series) -> dict:
    return {
        "gmv_usd": float(row["gmv_usd"]),
        "loan_amount_usd": float(row["loan_amount_usd"]),
        "down_payment_usd": float(row["down_payment_usd"]),
        "first_installment_usd": float(row["first_installment_usd"]),
        "second_installment_usd": float(row.get("second_installment_usd", 0) or 0),
        "third_installment_usd": float(row.get("third_installment_usd", 0) or 0),
        "credit_score": float(row["credit_score"]),
        "product": row["product"],
        "product_vertical": row["product_vertical"],
        "order_type": row["order_type"],
    }


def validate_csv(df: pd.DataFrame) -> list[str]:
    """Devuelve una lista de errores de validación encontrados. Vacía = OK."""
    errors = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Faltan columnas: {', '.join(missing)}")
        return errors  # sin las columnas no se puede seguir validando

    bad_products = set(df["product"].dropna().unique()) - set(VALID_PRODUCTS)
    if bad_products:
        errors.append(f"Valores inválidos en 'product': {bad_products}. Válidos: {VALID_PRODUCTS}")

    bad_verticals = set(df["product_vertical"].dropna().unique()) - set(VALID_VERTICALS)
    if bad_verticals:
        errors.append(f"Valores inválidos en 'product_vertical': {bad_verticals}. Válidos: {VALID_VERTICALS}")

    bad_order_types = set(df["order_type"].dropna().unique()) - set(VALID_ORDER_TYPES)
    if bad_order_types:
        errors.append(f"Valores inválidos en 'order_type': {bad_order_types}. Válidos: {VALID_ORDER_TYPES}")

    numeric_cols = ["gmv_usd", "loan_amount_usd", "down_payment_usd", "first_installment_usd", "credit_score"]
    for col in numeric_cols:
        if df[col].isnull().any():
            errors.append(f"La columna '{col}' tiene valores vacíos.")

    return errors


def render_risk_distribution(res_df: pd.DataFrame) -> None:
    colA, colB = st.columns([1, 1])
    with colA:
        band_counts = res_df["risk_band"].value_counts().reindex(["BAJO", "MEDIO", "ALTO"]).fillna(0)
        fig = go.Figure(go.Bar(
            x=band_counts.index, y=band_counts.values,
            marker_color=["#2E7D32", "#F5C518", "#C62828"],
        ))
        fig.update_layout(
            title="Préstamos por banda de riesgo",
            paper_bgcolor="#0D1B2A", plot_bgcolor="#0D1B2A",
            font_color="#F0F4F7", height=350,
        )
        st.plotly_chart(fig, width='stretch')
    with colB:
        fig2 = px.histogram(
            res_df, x="fpd_probability", nbins=20,
            title="Distribución de probabilidad de FPD",
            color_discrete_sequence=["#0E9AA7"],
        )
        fig2.update_layout(
            paper_bgcolor="#0D1B2A", plot_bgcolor="#0D1B2A",
            font_color="#F0F4F7", height=350,
        )
        st.plotly_chart(fig2, width='stretch')


# ──────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────
st.title("📊 Mesa de Riesgo — First Payment Default")
st.caption("Sistema de scoring de riesgo crediticio")

model_info = fpd_service.metrics

tab1, tab2, tab3 = st.tabs([
    "🔍 Evaluar préstamo individual",
    "📁 Cargar base de créditos (CSV)",
    "🗂️ Demo con portfolio de ejemplo",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Evaluación individual
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Evaluar un préstamo nuevo")
    st.caption("Setea tus variables")

    col1, col2, col3 = st.columns(3)
    with col1:
        gmv = st.number_input("Valor de la compra (GMV) USD", min_value=1.0, value=100.0, step=5.0)
        down_payment = st.number_input("Pago inicial (enganche) USD", min_value=0.0, value=25.0, step=5.0)
        credit_score = st.slider("Credit Score del usuario", -30, 700, 400)
    with col2:
        loan_amount = st.number_input("Monto del préstamo USD", min_value=1.0, value=75.0, step=5.0)
        product = st.selectbox("Producto", VALID_PRODUCTS, index=1)
        vertical = st.selectbox("Rubro", VALID_VERTICALS)
    with col3:
        order_type = st.selectbox("Tipo de orden", VALID_ORDER_TYPES)
        n_cuotas = int(product.replace("SPLIT_", ""))
        cuota_estimada = loan_amount / n_cuotas
        st.metric("Cuota estimada", f"USD {cuota_estimada:.2f}")

    if st.button("🔎 Evaluar riesgo", type="primary"):
        payload = {
            "gmv_usd": gmv, "loan_amount_usd": loan_amount, "down_payment_usd": down_payment,
            "first_installment_usd": cuota_estimada,
            "second_installment_usd": cuota_estimada if n_cuotas >= 2 else 0,
            "third_installment_usd": cuota_estimada if n_cuotas >= 3 else 0,
            "credit_score": credit_score, "product": product,
            "product_vertical": vertical, "order_type": order_type,
        }
        try:
            result = call_predict(payload)
            color_map = {"BAJO": "#2E7D32", "MEDIO": "#F5C518", "ALTO": "#C62828"}
            band = result["risk_band"]

            st.divider()
            colR1, colR2 = st.columns([1, 2])
            with colR1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["fpd_probability"] * 100,
                    title={"text": "Probabilidad de FPD (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color_map[band]},
                        "steps": [
                            {"range": [0, 30], "color": "#1A3A5C"},
                            {"range": [30, 49], "color": "#2A4A6C"},
                            {"range": [49, 100], "color": "#3A1A1A"},
                        ],
                    },
                ))
                fig.update_layout(paper_bgcolor="#0D1B2A", font_color="#F0F4F7", height=300)
                st.plotly_chart(fig, width='stretch')
            with colR2:
                color_name = "green" if band == "BAJO" else "orange" if band == "MEDIO" else "red"
                st.markdown(f"### Banda de riesgo: **:{color_name}[{band}]**")
                decision = "🔴 RECHAZAR / revisar manualmente" if result["fpd_prediction"] == 1 else "🟢 APROBAR"
                st.markdown(f"**Recomendación del modelo:** {decision}")
                st.markdown("**Variables con mayor peso en esta decisión:**")
                for driver in result["top_drivers"]:
                    st.markdown(f"- `{driver}`")
        except Exception as e:
            st.error(f"Error al evaluar el préstamo: {e}")

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Carga de CSV propio
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Evaluar una base de créditos propia")
    st.caption(
        "Subí un CSV con tus propios créditos y el sistema va a evaluar el riesgo "
        "de FPD de cada uno."
    )

    with st.expander("📋 Ver formato esperado del CSV"):
        st.markdown(
            "El archivo debe tener estas columnas exactas (en este orden no es necesario, "
            "pero los nombres sí):"
        )
        example_df = pd.DataFrame({
            "gmv_usd": [120.0], "loan_amount_usd": [90.0], "down_payment_usd": [30.0],
            "first_installment_usd": [30.0], "second_installment_usd": [30.0],
            "third_installment_usd": [30.0], "credit_score": [410],
            "product": ["SPLIT_3"], "product_vertical": ["Tecnología"],
            "order_type": ["First order"],
        })
        st.dataframe(example_df, width='stretch')
        st.markdown(f"- **product** debe ser uno de: `{', '.join(VALID_PRODUCTS)}`")
        st.markdown(f"- **product_vertical** debe ser uno de: `{', '.join(VALID_VERTICALS)}`")
        st.markdown(f"- **order_type** debe ser uno de: `{', '.join(VALID_ORDER_TYPES)}`")

        try:
            with open(Path(__file__).parent / "plantilla_creditos.csv", "rb") as f:
                st.download_button(
                    "⬇️ Descargar plantilla de ejemplo",
                    data=f.read(),
                    file_name="plantilla_creditos.csv",
                    mime="text/csv",
                )
        except FileNotFoundError:
            pass

    uploaded_file = st.file_uploader("Subí tu archivo CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            user_df = None

        if user_df is not None:
            st.write(f"Archivo cargado: **{len(user_df)} créditos**")
            errors = validate_csv(user_df)

            if errors:
                st.error("Se encontraron problemas en el archivo:")
                for err in errors:
                    st.markdown(f"- {err}")
            else:
                if st.button("▶️ Evaluar todos los créditos", type="primary"):
                    progress = st.progress(0, text="Evaluando créditos...")
                    results = []
                    for i, (_, row) in enumerate(user_df.iterrows()):
                        try:
                            payload = df_row_to_payload(row)
                            pred = call_predict(payload)
                        except Exception as e:
                            pred = {
                                "fpd_probability": None, "fpd_prediction": None,
                                "risk_band": "ERROR", "top_drivers": [], "error": str(e),
                            }
                        results.append(pred)
                        progress.progress((i + 1) / len(user_df), text=f"Evaluando créditos... {i+1}/{len(user_df)}")
                    progress.empty()

                    res_df = pd.DataFrame(results)
                    final_df = pd.concat([user_df.reset_index(drop=True), res_df], axis=1)

                    n_errors = (res_df["risk_band"] == "ERROR").sum()
                    if n_errors:
                        st.warning(f"⚠️ {n_errors} créditos no pudieron evaluarse (ver columna 'error').")

                    st.success(f"✅ Evaluación completa: {len(user_df) - n_errors} créditos procesados.")

                    valid_res = res_df[res_df["risk_band"] != "ERROR"]
                    if len(valid_res) > 0:
                        render_risk_distribution(valid_res)

                    st.subheader("Resultados detallados")
                    st.dataframe(final_df, width='stretch', height=350)

                    csv_download = final_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Descargar resultados (CSV)",
                        data=csv_download,
                        file_name="creditos_evaluados.csv",
                        mime="text/csv",
                        type="primary",
                    )

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Demo con portfolio de ejemplo
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Demo con datos de ejemplo")
    st.caption(
        "Aca podras visualizar los resultados de ejemplo de la demo. "
        "Selecciona la cantidad de prestamos a evaluar."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>{model_info["auc_roc_test"]:.3f}</h3><p>AUC-ROC</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>{model_info["recall_fpd"]*100:.1f}%</h3><p>Recall FPD</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h3>{model_info["precision_fpd"]*100:.1f}%</h3><p>Precision FPD</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h3>{model_info["f1_fpd"]:.3f}</h3><p>F1 FPD</p></div>', unsafe_allow_html=True)



    st.divider()

    n_sample = st.slider("Cantidad de préstamos a evaluar", 10, 200, 50, step=10)

    if st.button("▶️ Evaluar muestra de ejemplo", type="primary"):
        portfolio = pd.read_csv(Path(__file__).parent / "sample_portfolio.csv").sample(n_sample, random_state=42)

        def legacy_row_to_payload(row):
            return {
                "gmv_usd": float(row["GMV - USD"]),
                "loan_amount_usd": float(row["Loan Amount - USD"]),
                "down_payment_usd": float(row["Down Payment Amount - USD"]),
                "first_installment_usd": float(row["1st Installment Amount - USD"]),
                "second_installment_usd": float(row.get("2nd Installment Amount - USD", 0) or 0),
                "third_installment_usd": float(row.get("3rd Installment Amount - USD", 0) or 0),
                "credit_score": float(row["User Credit Score"]) if pd.notna(row["User Credit Score"]) else 400.0,
                "product": row["Product"] if row["Product"] in VALID_PRODUCTS else "SPLIT_3",
                "product_vertical": row["Product Type Vertical"] if row["Product Type Vertical"] in VALID_VERTICALS else "Moda",
                "order_type": row["Order Type"],
            }

        progress = st.progress(0, text="Evaluando...")
        results = []
        for i, (_, row) in enumerate(portfolio.iterrows()):
            try:
                pred = call_predict(legacy_row_to_payload(row))
                pred["real_fpd"] = row["FPD"]
                results.append(pred)
            except Exception:
                pass
            progress.progress((i + 1) / len(portfolio), text=f"Evaluando... {i+1}/{len(portfolio)}")
        progress.empty()

        res_df = pd.DataFrame(results)
        render_risk_distribution(res_df)

        st.subheader("Detalle")
        display_df = res_df[["fpd_probability", "risk_band", "fpd_prediction", "real_fpd"]].copy()
        display_df.columns = ["Prob. FPD", "Banda de riesgo", "Predicción", "FPD real (histórico)"]
        st.dataframe(display_df, width='stretch', height=300)
