import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background-color: #0d1117; }
    .stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }
    h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
    .header-box { background: linear-gradient(90deg, #1f6feb 0%, #388bfd 100%); border-radius: 12px; padding: 28px 36px; margin-bottom: 32px; box-shadow: 0 4px 32px rgba(31,111,235,0.3); }
    .header-box h1 { color: white !important; font-size: 2rem !important; margin: 0 !important; }
    .header-box p { color: rgba(255,255,255,0.85); margin: 6px 0 0 0; font-size: 1rem; }
    .result-poor { background: linear-gradient(135deg, #3d0000, #7a0000); border: 1px solid #f85149; border-radius: 12px; padding: 24px; text-align: center; color: white; }
    .result-standard { background: linear-gradient(135deg, #1c2a00, #3a5200); border: 1px solid #d29922; border-radius: 12px; padding: 24px; text-align: center; color: white; }
    .result-good { background: linear-gradient(135deg, #003d1a, #006633); border: 1px solid #3fb950; border-radius: 12px; padding: 24px; text-align: center; color: white; }
    .result-title { font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight: 700; margin: 0; }
    .result-subtitle { font-size: 1rem; opacity: 0.85; margin-top: 6px; }
    .prob-bar-label { font-size: 0.85rem; color: #8b949e; margin-bottom: 2px; }
    .section-title { font-family: 'Space Mono', monospace; color: #388bfd; font-size: 0.9rem; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 12px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
    .sidebar-info { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 14px; font-size: 0.82rem; color: #8b949e; margin-top: 16px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_preprocessors():
    try:
        import keras
        if os.path.exists('ann_credit_score.keras'):
            model = keras.models.load_model('ann_credit_score.keras')
        elif os.path.exists('ann_credit_score.h5'):
            model = keras.models.load_model('ann_credit_score.h5')
        else:
            return None, None, None
        scaler = joblib.load('scaler.pkl')
        pca    = joblib.load('pca.pkl')
        return model, scaler, pca
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None, None, None

model, scaler, pca = load_model_and_preprocessors()

st.markdown("""
<div class="header-box">
    <h1>💳 Credit Score Predictor</h1>
    <p>Red Neuronal Artificial Multiclase — Clasificación de Riesgo Crediticio</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ No se encontraron los archivos del modelo (ann_credit_score.keras/.h5, scaler.pkl, pca.pkl).")
    st.stop()

col_inputs, col_result = st.columns([2, 1], gap="large")

with col_inputs:
    st.markdown('<p class="section-title">👤 Información Personal</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Edad", 18, 80, 35)
    with c2:
        occupation = st.selectbox("Ocupación", ["Scientist","Teacher","Engineer","Entrepreneur","Developer","Lawyer","Media_Manager","Doctor","Journalist","Manager","Accountant","Musician","Mechanic","Writer","Architect"])
    with c3:
        annual_income = st.slider("Ingreso Anual (USD)", 7000, 180000, 50000, step=1000)

    c4, c5 = st.columns(2)
    with c4:
        monthly_salary = st.slider("Salario Mensual Neto (USD)", 300, 15000, 4000, step=100)
    with c5:
        monthly_balance = st.slider("Balance Mensual (USD)", -500, 1500, 300, step=50)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🏦 Cuentas y Tarjetas</p>', unsafe_allow_html=True)
    c6, c7, c8 = st.columns(3)
    with c6:
        num_bank_accounts = st.slider("N° Cuentas Bancarias", 0, 10, 3)
    with c7:
        num_credit_card = st.slider("N° Tarjetas de Crédito", 0, 11, 4)
    with c8:
        num_of_loan = st.slider("N° de Préstamos", 0, 9, 2)

    c9, c10 = st.columns(2)
    with c9:
        interest_rate = st.slider("Tasa de Interés (%)", 1, 34, 14)
    with c10:
        credit_utilization = st.slider("Utilización de Crédito (%)", 0.0, 100.0, 30.0, step=0.5)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📅 Comportamiento de Pago</p>', unsafe_allow_html=True)
    c11, c12, c13 = st.columns(3)
    with c11:
        delay_from_due = st.slider("Días de Retraso", 0, 62, 10)
    with c12:
        num_delayed_payment = st.slider("N° Pagos Atrasados", 0, 28, 5)
    with c13:
        num_credit_inquiries = st.slider("N° Consultas de Crédito", 0, 17, 3)

    c14, c15 = st.columns(2)
    with c14:
        payment_min_amount = st.selectbox("¿Paga el Mínimo?", ["Yes", "No", "NM"])
    with c15:
        payment_behaviour = st.selectbox("Comportamiento de Pago", ["High_spent_Small_value_payments","Low_spent_Large_value_payments","High_spent_Medium_value_payments","Low_spent_Small_value_payments","High_spent_Large_value_payments","Low_spent_Medium_value_payments"])

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">💰 Deuda e Inversión</p>', unsafe_allow_html=True)
    c16, c17, c18 = st.columns(3)
    with c16:
        outstanding_debt = st.slider("Deuda Pendiente (USD)", 0.0, 5000.0, 800.0, step=10.0)
    with c17:
        total_emi = st.slider("EMI Mensual Total (USD)", 0.0, 2000.0, 100.0, step=5.0)
    with c18:
        amount_invested = st.slider("Inversión Mensual (USD)", 0.0, 2000.0, 200.0, step=10.0)

    c19, c20 = st.columns(2)
    with c19:
        credit_history_age = st.slider("Antigüedad Historial Crediticio (meses)", 0, 400, 180)
    with c20:
        changed_credit_limit = st.slider("Cambio en Límite de Crédito", -10.0, 30.0, 5.0, step=0.5)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🔀 Tipo de Crédito</p>', unsafe_allow_html=True)
    c21, c22 = st.columns(2)
    with c21:
        credit_mix = st.selectbox("Mezcla de Crédito", ["Good", "Standard", "Bad"])
    with c22:
        type_of_loan = st.selectbox("Tipo de Préstamo Principal", ["Personal Loan","Home Equity Loan","Mortgage Loan","Auto Loan","Student Loan","Payday Loan","Credit-Builder Loan"])

occupation_map = {"Accountant":0,"Architect":1,"Developer":2,"Doctor":3,"Engineer":4,"Entrepreneur":5,"Journalist":6,"Lawyer":7,"Manager":8,"Media_Manager":9,"Mechanic":10,"Musician":11,"Scientist":12,"Teacher":13,"Writer":14}
credit_mix_map = {"Bad":0,"Good":1,"Standard":2}
payment_min_map = {"NM":0,"No":1,"Yes":2}
payment_beh_map = {"High_spent_Large_value_payments":0,"High_spent_Medium_value_payments":1,"High_spent_Small_value_payments":2,"Low_spent_Large_value_payments":3,"Low_spent_Medium_value_payments":4,"Low_spent_Small_value_payments":5}
loan_type_map = {"Auto Loan":0,"Credit-Builder Loan":1,"Home Equity Loan":2,"Mortgage Loan":3,"Payday Loan":4,"Personal Loan":5,"Student Loan":6}

with col_result:
    st.markdown('<p class="section-title">🎯 Resultado</p>', unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predecir Credit Score", use_container_width=True, type="primary")

    if predict_btn:
        features = np.array([[
            age, occupation_map.get(occupation, 0), annual_income, monthly_salary,
            num_bank_accounts, num_credit_card, interest_rate, num_of_loan,
            loan_type_map.get(type_of_loan, 0), delay_from_due, num_delayed_payment,
            changed_credit_limit, num_credit_inquiries, credit_mix_map.get(credit_mix, 0),
            outstanding_debt, credit_utilization, credit_history_age,
            payment_min_map.get(payment_min_amount, 0), total_emi, amount_invested,
            payment_beh_map.get(payment_behaviour, 0), monthly_balance
        ]], dtype=float)

        features_scaled = scaler.transform(features)
        features_scaled = np.nan_to_num(features_scaled)
        features_scaled = features_scaled[:, :18]
        features_pca = pca.transform(features_scaled)

        # CORRECCIÓN CLAVE: model.predict() para Keras, NO predict_proba()
        probs = model.predict(features_pca)[0]
        clase_pred = int(np.argmax(probs))

        labels_map = {0: "Poor", 1: "Standard", 2: "Good"}
        emoji_map  = {0: "🔴", 1: "🟡", 2: "🟢"}
        css_map    = {0: "result-poor", 1: "result-standard", 2: "result-good"}
        desc_map   = {
            0: "Alto riesgo crediticio. Se recomienda revisar hábitos de pago y reducir deudas.",
            1: "Riesgo crediticio medio. Hay oportunidades de mejora en el perfil financiero.",
            2: "Bajo riesgo crediticio. Excelente perfil financiero."
        }

        st.markdown(f"""
        <div class="{css_map[clase_pred]}">
            <p class="result-title">{emoji_map[clase_pred]} {labels_map[clase_pred]}</p>
            <p class="result-subtitle">Clase {clase_pred}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<br><small style='color:#8b949e'>{desc_map[clase_pred]}</small>", unsafe_allow_html=True)
        st.markdown("<br>**Probabilidades:**", unsafe_allow_html=True)

        for i, (label, prob) in enumerate(zip(["Poor", "Standard", "Good"], probs)):
            st.markdown(f'<p class="prob-bar-label">{emoji_map[i]} {label}: {prob*100:.1f}%</p>', unsafe_allow_html=True)
            st.progress(float(prob))
    else:
        st.info("👈 Ajusta los parámetros del cliente y presiona **Predecir**.")
        st.markdown("""
        <div class="sidebar-info">
        <b>Clases de Credit Score:</b><br><br>
        🔴 <b>0 - Poor:</b> Alto riesgo<br>
        🟡 <b>1 - Standard:</b> Riesgo medio<br>
        🟢 <b>2 - Good:</b> Bajo riesgo
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<center><small style='color:#484f58'>ANN Multiclass · Taller de Deep Learning · Modelo: PCA + Red Neuronal Artificial</small></center>", unsafe_allow_html=True)
