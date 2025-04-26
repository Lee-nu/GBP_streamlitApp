# Load models and scalers
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    rf_scaler = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("xgb_scaler.pkl", "rb") as f:
    xgb_scaler = pickle.load(f)

# === Streamlit Interface ===
st.set_page_config(page_title="Britcoin Fraud Detector", layout="wide")
st.title("💷 Britcoin: Secure Digital GBP Fraud Detection")

st.sidebar.header("📌 Enter Transaction Details")
amount = st.sidebar.number_input("Amount", min_value=0.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance Origin", min_value=0.0)
newbalanceOrig = st.sidebar.number_input("New Balance Origin", min_value=0.0)
type_encoded = st.sidebar.selectbox("Transaction Type", options=[("TRANSFER", 1), ("CASH_OUT", 0)])

model_choice = st.sidebar.radio("Choose Model", ["Random Forest", "XGBoost"])

if st.sidebar.button("🚨 Submit Transaction"):
    # Input feature processing
    input_data = {
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'type_encoded': type_encoded[1],
        'step': 1,
        'oldbalanceDest': 0,
        'newbalanceDest': 0,
        'balance_diff_orig': oldbalanceOrg - newbalanceOrig,
        'balance_diff_dest': 0,
        'amount_to_balance_ratio': amount / (oldbalanceOrg + 1),
        'zero_balance_orig': int(oldbalanceOrg == 0),
        'zero_balance_dest': 1
    }
    df = pd.DataFrame([input_data])

    # Scale with correct scaler
    if model_choice == "Random Forest":
        df_scaled = rf_scaler.transform(df)
        prediction = rf_model.predict(df_scaled)[0]
    else:
        df_scaled = xgb_scaler.transform(df)
        prediction = xgb_model.predict(df_scaled)[0]

    st.subheader("🧠 Prediction Result")
    if prediction == 1:
        st.error("🚫 Fraudulent transaction detected!")
    else:
        st.success("✅ Legitimate transaction.")

    # Blockchain log
    result = blockchain.add_transaction(input_data, prediction)
    st.info(result)
