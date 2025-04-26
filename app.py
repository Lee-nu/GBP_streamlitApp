# ============================
# Full Corrected app.py Code
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import hashlib
import json
import time

# Blockchain classes
class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.hash_block()

    def hash_block(self):
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash
        }, sort_keys=True).encode()
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
            return "\U0001F6AB Fraudulent transaction detected! Not added to Blockchain."
        last_block = self.get_latest_block()
        new_block = Block(len(self.chain), str(time.time()), data, last_block.hash)
        self.chain.append(new_block)
        return f"\u2705 Transaction added. Block #{new_block.index} with Hash: {new_block.hash}"

    def display_chain(self):
        return [block.__dict__ for block in self.chain]

# Load models
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit app
st.set_page_config(page_title="Britcoin Fraud Detection", layout="wide")
st.title("\U0001F4B5 Britcoin: Secure Digital GBP Fraud Detection")

st.sidebar.header("\U0001F50D Enter Transaction Details")

amount = st.sidebar.number_input("Amount", min_value=0.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance Origin", min_value=0.0)
newbalanceOrig = st.sidebar.number_input("New Balance Origin", min_value=0.0)
transaction_type = st.sidebar.selectbox("Transaction Type", [("TRANSFER", 1), ("CASH_OUT", 0)])
model_choice = st.sidebar.radio("Choose Model", ("Random Forest", "XGBoost"))

if st.sidebar.button("\U0001F680 Submit Transaction"):

    # Manual feature engineering (match training)
    input_features = {
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'type_encoded': transaction_type[1],
        'balance_diff_orig': oldbalanceOrg - newbalanceOrig,
        'balance_diff_dest': 0,  # not given in input, assuming 0
        'amount_to_balance_ratio': amount / (oldbalanceOrg + 1),
        'zero_balance_orig': int(oldbalanceOrg == 0),
        'zero_balance_dest': 1,  # assume destination has 0 balance (for this app)
    }

    input_df = pd.DataFrame([input_features])

    try:
        input_scaled = scaler.transform(input_df)

        if model_choice == "Random Forest":
            prediction = rf_model.predict(input_scaled)[0]
        else:
            prediction = xgb_model.predict(input_scaled)[0]

        st.subheader("\U0001F4CA Prediction Result")

        if prediction == 1:
            st.error("\U0001F6AB Fraudulent transaction detected!")
        else:
            st.success("\u2705 Legitimate transaction.")

        # Blockchain part
        blockchain = Blockchain()
        result = blockchain.add_transaction(input_features, prediction)
        st.info(result)

        st.subheader("\U0001F4DD Blockchain Ledger")
        ledger_df = pd.DataFrame(blockchain.display_chain())
        st.dataframe(ledger_df, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

else:
    st.info("\u2139\ufe0f Please fill the transaction details and click Submit!")
