# Britcoin Fraud Detection & Blockchain Streamlit App
# ===================================================

import streamlit as st
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
import pickle

# === Blockchain classes (reuse from your notebook) ===
import hashlib
import json

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
            return "🚫 Transaction flagged as fraud and not added to the blockchain."
        last_block = self.get_latest_block()
        new_block = Block(len(self.chain), str(time.time()), data, last_block.hash)
        self.chain.append(new_block)
        return f"✅ Transaction added. Block #{new_block.index} with Hash: {new_block.hash}"

    def display_chain(self):
        return [block.__dict__ for block in self.chain]

# Initialize blockchain and (placeholder) model
blockchain = Blockchain()

# === Simulate loading a trained model and scaler (replace these with your actual model objects) ===
# For demo, we assume model is trained and saved as 'rf_model.pkl' and scaler as 'scaler.pkl'
with open('rf_model.pkl', 'rb') as f:
    rf = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# === Streamlit Interface ===
st.set_page_config(page_title="Britcoin Fraud Detector", layout="wide")
st.title("Britcoin: Secure Digital GBP Fraud Detection")

st.sidebar.header("Enter Transaction Details")
amount = st.sidebar.number_input("Amount", min_value=0.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance Origin", min_value=0.0)
newbalanceOrig = st.sidebar.number_input("New Balance Origin", min_value=0.0)
type_encoded = st.sidebar.selectbox("Transaction Type", options=[("TRANSFER", 1), ("CASH_OUT", 0)])

if st.sidebar.button("Submit Transaction"):
    # Create input row with required features
    input_data = {
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'type_encoded': type_encoded[1],
        'step': 1,  # default for simplicity
        'oldbalanceDest': 0,
        'newbalanceDest': 0,
        'balance_diff_orig': oldbalanceOrg - newbalanceOrig,
        'balance_diff_dest': 0,
        'amount_to_balance_ratio': amount / (oldbalanceOrg + 1),
        'zero_balance_orig': int(oldbalanceOrg == 0),
        'zero_balance_dest': 1
    }
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = rf.predict(input_scaled)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("🚫 Fraudulent transaction detected!")
    else:
        st.success("✅ Legitimate transaction.")

    # Blockchain logging
    result = blockchain.add_transaction(input_data, prediction)
    st.info(result)

# === Blockchain Ledger Display ===
st.subheader("Blockchain Ledger")
ledger_df = pd.DataFrame(blockchain.display_chain())
st.dataframe(ledger_df, use_container_width=True)
