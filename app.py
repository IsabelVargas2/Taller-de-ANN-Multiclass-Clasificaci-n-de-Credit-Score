import streamlit as st
import numpy as np
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
    st.error("⚠️ No se encontraron los archivos del modelo.")
    st.stop()

col_inputs, col_result = st.columns([2, 1], gap="large")

with col_inputs:
    st.markdown('<p class="section-title">👤 Información Personal</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        # Age: min=14, max=56 (datos reales)
        age = st.slider("Edad", min_value=14, max_value=56, value=35)
    with c2:
        occupation = st.selectbox("Ocupación", [
            "Accountant","Architect","Developer","Doctor","Engineer",
            "Entrepreneur","Journalist","Lawyer","Manager","Media_Manager",
            "Mechanic","Musician","Scientist","Teacher","Writer"
        ])
    with c3:
        # Annual_Income: min=7006, max=179987
        annual_income = st.slider("Ingreso Anual (USD)", min_value=7006, max_value=179987, value=50000, step=500)

    c4, c5 = st.columns(2)
    with c4:
        # Monthly_Inhand_Salary: min=304, max=15205
        monthly_salary = st.slider("Salario Mensual Neto (USD)", min_value=304, max_value=15205, value=4000, step=100)
    with c5:
        # Num_Bank_Accounts: min=0, max=10
        num_bank_accounts = st.slider("N° Cuentas Bancarias", min_value=0, max_value=10, value=3)

    c6, c7 = st.columns(2)
    with c6:
        # Num_Credit_Card: min=0, max=11
        num_credit_card = st.slider("N° Tarjetas de Crédito", min_value=0, max_value=11, value=4)
    with c7:
        # Interest_Rate: min=1, max=34
        interest_rate = st.slider("Tasa de Interés (%)", min_value=1, max_value=34, value=14)

    c8, c9 = st.columns(2)
    with c8:
        # Num_of_Loan: min=0, max=9
        num_of_loan = st.slider("N° de Préstamos", min_value=0, max_value=9, value=2)
    with c9:
        # Delay_from_due_date: min=-2, max=63
        delay_from_due = st.slider("Días de Retraso", min_value=-2, max_value=63, value=10)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📅 Comportamiento de Pago</p>', unsafe_allow_html=True)
    c10, c11, c12 = st.columns(3)
    with c10:
        # Num_of_Delayed_Payment: min=0, max=26
        num_delayed_payment = st.slider("N° Pagos Atrasados", min_value=0, max_value=26, value=5)
    with c11:
        # Changed_Credit_Limit: min=0.5, max=31.1
        changed_credit_limit = st.slider("Cambio Límite de Crédito", min_value=0.5, max_value=31.1, value=5.0, step=0.1)
    with c12:
        # Num_Credit_Inquiries: min=0, max=16
        num_credit_inquiries = st.slider("N° Consultas de Crédito", min_value=0, max_value=16, value=3)

    c13, c14 = st.columns(2)
    with c13:
        credit_mix = st.selectbox("Mezcla de Crédito", ["Good", "Standard", "Bad"])
    with c14:
        # Outstanding_Debt: min=0.2, max=4998
        outstanding_debt = st.slider("Deuda Pendiente (USD)", min_value=0.0, max_value=4998.0, value=800.0, step=10.0)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">💰 Deuda e Inversión</p>', unsafe_allow_html=True)
    c15, c16, c17 = st.columns(3)
    with c15:
        # Credit_Utilization_Ratio: min=25.5, max=42.4
        credit_utilization = st.slider("Utilización de Crédito (%)", min_value=25.5, max_value=42.4, value=32.0, step=0.1)
    with c16:
        # Credit_History_Age: min=0.4, max=33.4 años → en meses: 5 a 401
        credit_history_age = st.slider("Antigüedad Historial Crediticio (años)", min_value=0.4, max_value=33.4, value=15.0, step=0.1)
    with c17:
        payment_min_amount = st.selectbox("¿Paga el Mínimo?", ["Yes", "No", "NM"])

    c18, c19 = st.columns(2)
    with c18:
        # Total_EMI_per_month: min=0, max=1515
        total_emi = st.slider("EMI Mensual Total (USD)", min_value=0.0, max_value=1515.0, value=100.0, step=5.0)
    with c19:
        # Amount_invested_monthly: min=14.5, max=1005.8
        amount_invested = st.slider("Inversión Mensual (USD)", min_value=14.0, max_value=1006.0, value=200.0, step=10.0)

    c20, c21 = st.columns(2)
    with c20:
        payment_behaviour = st.selectbox("Comportamiento de Pago", [
            "High_spent_Large_value_payments",
            "High_spent_Medium_value_payments",
            "High_spent_Small_value_payments",
            "Low_spent_Large_value_payments",
            "Low_spent_Medium_value_payments",
            "Low_spent_Small_value_payments"
        ])
    with c21:
        # Monthly_Balance: min=93, max=1349
        monthly_balance = st.slider("Balance Mensual (USD)", min_value=93, max_value=1349, value=400)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🔀 Tipo de Préstamo</p>', unsafe_allow_html=True)
    # Type_of_Loan: 5 valores únicos tras LabelEncoder
    type_of_loan = st.selectbox("Tipo de Préstamo Principal", [
        "Auto Loan", "Credit-Builder Loan", "Home Equity Loan",
        "Mortgage Loan", "Personal Loan"
    ])

# ── Mapas de encoding (LabelEncoder alfabético) ────────────────────────────
occupation_map = {
    "Accountant":0,"Architect":1,"Developer":2,"Doctor":3,"Engineer":4,
    "Entrepreneur":5,"Journalist":6,"Lawyer":7,"Manager":8,"Media_Manager":9,
    "Mechanic":10,"Musician":11,"Scientist":12,"Teacher":13,"Writer":14
}
credit_mix_map   = {"Bad":0, "Good":1, "Standard":2}
payment_min_map  = {"NM":0, "No":1, "Yes":2}
payment_beh_map  = {
    "High_spent_Large_value_payments":0,
    "High_spent_Medium_value_payments":1,
    "High_spent_Small_value_payments":2,
    "Low_spent_Large_value_payments":3,
    "Low_spent_Medium_value_payments":4,
    "Low_spent_Small_value_payments":5
}
loan_type_map = {
    "Auto Loan":0, "Credit-Builder Loan":1, "Home Equity Loan":2,
    "Mortgage Loan":3, "Personal Loan":4
}

with col_result:
    st.markdown('<p class="section-title">🎯 Resultado</p>', unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predecir Credit Score", use_container_width=True, type="primary")

    if predict_btn:
        # Orden exacto del entrenamiento (22 columnas)
        features = np.array([[
            age,                                      # 0  Age
            occupation_map[occupation],               # 1  Occupation
            annual_income,                            # 2  Annual_Income
            monthly_salary,                           # 3  Monthly_Inhand_Salary
            num_bank_accounts,                        # 4  Num_Bank_Accounts
            num_credit_card,                          # 5  Num_Credit_Card
            interest_rate,                            # 6  Interest_Rate
            num_of_loan,                              # 7  Num_of_Loan
            delay_from_due,                           # 8  Delay_from_due_date
            num_delayed_payment,                      # 9  Num_of_Delayed_Payment
            changed_credit_limit,                     # 10 Changed_Credit_Limit
            num_credit_inquiries,                     # 11 Num_Credit_Inquiries
            credit_mix_map[credit_mix],               # 12 Credit_Mix
            outstanding_debt,                         # 13 Outstanding_Debt
            credit_utilization,                       # 14 Credit_Utilization_Ratio
            credit_history_age,                       # 15 Credit_History_Age
            payment_min_map[payment_min_amount],      # 16 Payment_of_Min_Amount
            total_emi,                                # 17 Total_EMI_per_month
            amount_invested,                          # 18 Amount_invested_monthly
            payment_beh_map[payment_behaviour],       # 19 Payment_Behaviour
            monthly_balance,                          # 20 Monthly_Balance
            loan_type_map[type_of_loan],              # 21 Type_of_Loan
        ]], dtype=float)

        features_scaled = scaler.transform(features)
        features_scaled = np.nan_to_num(features_scaled)
        features_scaled_18 = features_scaled[:, :18]
        features_pca = pca.transform(features_scaled_18)

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

        st.markdown(f"<br><small style='color:#8b949e'>{desc_map[clase_pred]}</small>",
                    unsafe_allow_html=True)
        st.markdown("<br>**Probabilidades:**", unsafe_allow_html=True)

        for i, (label, prob) in enumerate(zip(["Poor", "Standard", "Good"], probs)):
            st.markdown(
                f'<p class="prob-bar-label">{emoji_map[i]} {label}: {prob*100:.1f}%</p>',
                unsafe_allow_html=True
            )
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
st.markdown(
    "<center><small style='color:#484f58'>ANN Multiclass · Taller de Deep Learning · "
    "Modelo: PCA + Red Neuronal Artificial</small></center>",
    unsafe_allow_html=True
)
