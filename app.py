# Britcoin Fraud Detection & Blockchain Streamlit App
# ===================================================

import streamlit as st
import pandas as pd
import time
import pickle
import hashlib
import json
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# === Blockchain classes ===
class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.hash_block()

    def hash_block(self):
        block_string = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, str(time.time()), "Genesis Block", "0")
        self.chain.append(genesis_block)

    def get_latest_block(self):
        return self.chain[-1]

    def add_transaction(self, data, is_fraud):
        if is_fraud:
            return "🚫 Fraudulent transaction not added to blockchain."
        last_block = self.get_latest_block()
        new_block = Block(len(self.chain), str(time.time()), data, last_block.hash)
        self.chain.append(new_block)
        return f"✅ Transaction added. Block #{new_block.index} with Hash: {new_block.hash}"

    def display_chain(self):
        return [block.__dict__ for block in self.chain]

# === Load Models & Scaler ===
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# === Streamlit UI ===
st.set_page_config(page_title="Britcoin Fraud Detection", layout="wide")
st.title("💷 Britcoin: Secure Digital GBP Fraud Detection")

st.sidebar.header("🔎 Enter Transaction Details")
amount = st.sidebar.number_input("Amount", min_value=0.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance Origin", min_value=0.0)
newbalanceOrig = st.sidebar.number_input("New Balance Origin", min_value=0.0)
type_encoded = st.sidebar.selectbox("Transaction Type", options=[("TRANSFER", 1), ("CASH_OUT", 0)])
model_choice = st.sidebar.radio("Choose Model", ["Random Forest", "XGBoost"])

if st.sidebar.button("🚀 Submit Transaction"):
    # Prepare input features
    input_data = {
        "step": 1,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": 0,
        "newbalanceDest": 0,
        "type_encoded": type_encoded[1],
        "balance_diff_orig": oldbalanceOrg - newbalanceOrig,
        "balance_diff_dest": 0,
        "amount_to_balance_ratio": amount / (oldbalanceOrg + 1),
        "zero_balance_orig": int(oldbalanceOrg == 0),
        "zero_balance_dest": 1
    }

    df_input = pd.DataFrame([input_data])
    df_scaled = scaler.transform(df_input)

    # Select and predict using chosen model
    if model_choice == "Random Forest":
        prediction = rf_model.predict(df_scaled)[0]
    else:
        prediction = xgb_model.predict(df_scaled)[0]

    # Show result
    st.subheader("🧠 Prediction Result")
    if prediction == 1:
        st.error("❌ Fraudulent transaction detected!")
    else:
        st.success("✅ Legitimate transaction.")

    # Blockchain update
    blockchain = Blockchain()
    result = blockchain.add_transaction(input_data, prediction)
    st.info(result)

    # Show ledger
    st.subheader("📒 Blockchain Ledger")
    st.dataframe(pd.DataFrame(blockchain.display_chain()), use_container_width=True)
