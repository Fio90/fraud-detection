# ─────────────────────────────────────────────────────────────
# Fraud Detection System — Streamlit App
# Author: Frederick Amartey-Fio
# Institution: JUNIA ISEN — MSc Big Data
# Date: May 2026
# ─────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page configuration ────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="shield",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────
# Get the directory where app.py lives, then go up one level to models/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

@st.cache_resource
def load_model():
    model          = joblib.load(os.path.join(MODELS_DIR, "fraud_model.pkl"))
    threshold      = joblib.load(os.path.join(MODELS_DIR, "threshold.pkl"))
    feature_cols   = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    return model, threshold, feature_cols

try:
    model, threshold, feature_columns = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# ── Title ─────────────────────────────────────────────────────
st.title("Fraud Detection System")
st.markdown("""
A two-layer fraud detection system built with **XGBoost** and a **rule-based scoring engine**.
- **Layer 1** — Credit card fraud detection (ML model)
- **Layer 2** — Order fraud detection (behavioural signals)
- **Layer 3** — Combined score
""")

if not model_loaded:
    st.error(f"Could not load model: {model_error}")
    st.stop()

st.success("Model loaded successfully.")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Layer 1 — Card Fraud",
    "Layer 2 — Order Fraud",
    "Combined Score"
])

# ─────────────────────────────────────────────────────────────
# TAB 1 — LAYER 1
# ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Layer 1 — Credit Card Fraud Detection")
    st.markdown("Enter transaction details to get a fraud risk score from the XGBoost model.")

    col1, col2 = st.columns(2)

    with col1:
        amount    = st.number_input("Transaction amount ($)", min_value=0.0, value=150.0)
        time_hour = st.slider("Hour of transaction (0-23)", 0, 23, 14)
        velocity  = st.slider("Transactions in last hour", 0, 20, 1)
        country   = st.selectbox("Charge country vs billing",
                                 ["Same country", "Different country", "High-risk country"])

    with col2:
        device   = st.selectbox("Device signal",
                                ["Known device", "New device", "VPN / proxy detected"])
        merchant = st.selectbox("Merchant category",
                                ["Retail / grocery", "Electronics", "Gift cards / crypto"])
        avs_fail = st.checkbox("AVS mismatch (billing address does not match)")
        cvv_fail = st.checkbox("CVV failed or not provided")

    if st.button("Analyze transaction", key="layer1_btn"):

        # Build feature vector with all 30 features set to 0
        input_df = pd.DataFrame(
            [np.zeros(len(feature_columns))],
            columns=feature_columns
        )

        # Fill in Amount and Time using same scaling as training
        input_df["Amount_scaled"] = (amount - 88.29) / 250.11
        input_df["Time_scaled"]   = (time_hour - 12) / 6.0

        # Run prediction
        proba = model.predict_proba(input_df)[0][1]
        score = int(proba * 100)

        st.divider()

        if score < 35:
            st.success(f"Low risk — Score: {score}/100")
            st.markdown("**Decision: APPROVE**")
        elif score < 65:
            st.warning(f"Medium risk — Score: {score}/100")
            st.markdown("**Decision: REVIEW** — Send OTP challenge to cardholder")
        else:
            st.error(f"High risk — Score: {score}/100")
            st.markdown("**Decision: DECLINE** — Flag for investigation")

        st.progress(score / 100)

        st.markdown("**Signals detected:**")
        if avs_fail:
            st.markdown("- AVS mismatch — billing address does not match card records")
        if cvv_fail:
            st.markdown("- CVV failed — critical stolen card indicator")
        if country == "High-risk country":
            st.markdown("- High-risk charge country detected")
        if device == "VPN / proxy detected":
            st.markdown("- VPN / proxy detected")
        if velocity >= 5:
            st.markdown(f"- High velocity — {velocity} transactions in last hour")
        if merchant == "Gift cards / crypto":
            st.markdown("- High-risk merchant category")

# ─────────────────────────────────────────────────────────────
# TAB 2 — LAYER 2
# ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Layer 2 — Order Fraud Detection")
    st.markdown("Enter order details to get a behavioural risk score.")

    col1, col2 = st.columns(2)

    with col1:
        account_age  = st.number_input("Account age (days)", min_value=0, value=120)
        past_orders  = st.number_input("Total past orders", min_value=0, value=4)
        email_type   = st.selectbox("Email type",
                                    ["established", "disposable", "random"])
        ship_country = st.selectbox("Shipping country",
                                    ["same", "different", "high_risk"])
        ship_method  = st.selectbox("Shipping method",
                                    ["standard", "express", "freight_forwarder"])

    with col2:
        ip_orders    = st.slider("Orders from same IP today", 0, 20, 1)
        vpn          = st.checkbox("VPN / proxy detected", key="vpn2")
        product      = st.selectbox("Product category",
                                    ["normal", "elevated", "high_risk"])
        order_amt    = st.number_input("Order amount ($)", min_value=0.0, value=250.0)
        new_addr     = st.checkbox("Shipping address added in last 10 minutes")
        multi_card   = st.checkbox("Multiple cards tried on this order")
        name_mismatch = st.checkbox("Billing name differs from account name")
        prior_fraud  = st.checkbox("Email / phone linked to prior fraud")

    if st.button("Analyze order", key="layer2_btn"):

        score   = 0
        signals = []

        if account_age < 1:
            score += 22
            signals.append("HIGH — Account created today")
        elif account_age < 7:
            score += 14
            signals.append(f"HIGH — Account only {account_age} day(s) old")
        elif account_age < 30:
            score += 6
            signals.append(f"MEDIUM — Account {account_age} days old")
        else:
            signals.append(f"LOW — Established account ({account_age} days)")

        if past_orders == 0:
            score += 10
            signals.append("MEDIUM — First order ever on this account")
        elif past_orders >= 5:
            signals.append(f"LOW — {past_orders} prior orders")
        else:
            signals.append(f"LOW — {past_orders} prior order(s)")

        if email_type == "disposable":
            score += 18
            signals.append("HIGH — Disposable email address")
        elif email_type == "random":
            score += 12
            signals.append("HIGH — Suspicious email domain")
        else:
            signals.append("LOW — Established email provider")

        if ship_country == "high_risk":
            score += 22
            signals.append("HIGH — Shipping to high-risk country")
        elif ship_country == "different":
            score += 9
            signals.append("MEDIUM — Shipping country differs from billing")
        else:
            signals.append("LOW — Shipping matches billing country")

        if ship_method == "freight_forwarder":
            score += 20
            signals.append("HIGH — Freight forwarder address")
        elif ship_method == "express":
            score += 8
            signals.append("MEDIUM — Express shipping selected")
        else:
            signals.append("LOW — Standard shipping")

        if ip_orders >= 5:
            score += 20
            signals.append(f"HIGH — {ip_orders} orders from same IP today")
        elif ip_orders >= 3:
            score += 8
            signals.append(f"MEDIUM — {ip_orders} orders from same IP today")
        else:
            signals.append("LOW — Normal IP activity")

        if vpn:
            score += 15
            signals.append("HIGH — VPN / proxy detected")

        if product == "high_risk":
            score += 16
            signals.append("HIGH — High-risk product category")
        elif product == "elevated":
            score += 8
            signals.append("MEDIUM — Elevated-risk product")
        else:
            signals.append("LOW — Low-risk product category")

        if order_amt > 1000:
            score += 10
            signals.append(f"MEDIUM — High-value order: ${order_amt}")

        if new_addr:
            score += 18
            signals.append("HIGH — Shipping address added just now")
        if multi_card:
            score += 20
            signals.append("HIGH — Multiple cards attempted")
        if name_mismatch:
            score += 15
            signals.append("HIGH — Billing name differs from account name")
        if prior_fraud:
            score += 25
            signals.append("HIGH — Email / phone tied to prior fraud")

        score = min(score, 100)
        st.session_state["layer2_score"] = score

        st.divider()

        if score < 35:
            st.success(f"Low risk — Score: {score}/100")
            st.markdown("**Decision: APPROVE**")
        elif score < 65:
            st.warning(f"Medium risk — Score: {score}/100")
            st.markdown("**Decision: REVIEW**")
        else:
            st.error(f"High risk — Score: {score}/100")
            st.markdown("**Decision: DECLINE**")

        st.progress(score / 100)

        st.markdown("**Signals:**")
        for s in signals:
            st.markdown(f"- {s}")

# ─────────────────────────────────────────────────────────────
# TAB 3 — COMBINED SCORE
# ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Combined Fraud Score")
    st.markdown("""
    Combine Layer 1 and Layer 2 scores into one final decision.
    - Layer 1 (ML model) contributes **60%**
    - Layer 2 (Rule engine) contributes **40%**
    """)

    col1, col2 = st.columns(2)

    with col1:
        l1_proba = st.slider(
            "Layer 1 — Card fraud probability (0.0 to 1.0)",
            min_value=0.0, max_value=1.0, value=0.10, step=0.01
        )

    with col2:
        l2_score = st.slider(
            "Layer 2 — Order fraud score (0 to 100)",
            min_value=0, max_value=100, value=95
        )

    if st.button("Calculate combined score", key="combined_btn"):

        l1_score     = l1_proba * 100
        final_score  = int((l1_score * 0.6) + (l2_score * 0.4))
        final_score  = min(final_score, 100)

        st.divider()

        col1, col2, col3 = st.columns(3)
        col1.metric("Layer 1 score", f"{l1_score:.1f}/100")
        col2.metric("Layer 2 score", f"{l2_score}/100")
        col3.metric("Final score",   f"{final_score}/100")

        st.progress(final_score / 100)

        if final_score < 35:
            st.success(f"Final decision: APPROVE — Score {final_score}/100")
        elif final_score < 65:
            st.warning(f"Final decision: REVIEW — Score {final_score}/100")
        else:
            st.error(f"Final decision: DECLINE — Score {final_score}/100")

        if l1_proba < 0.35 and l2_score >= 65:
            st.info("""
            Key scenario: The card passed Layer 1 but Layer 2 caught
            suspicious order behaviour. This is exactly why two layers are needed.
            """)