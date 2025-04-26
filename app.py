import streamlit as st
import pandas as pd
import time
import pickle
import hashlib
import json

# === Blockchain classes ===
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

    def add_transaction(self, data, is_fraud):
        if is_fraud:
            return "🚫 Fraudulent transaction detected. Not added to blockchain."
        last_block = self.chain[-1]
        new_block = Block(len(self.chain), str(time.time()), data, last_block.hash)
        self.chain.append(new_block)
        return f"✅ Transaction added. Block #{new_block.index} with Hash: {new_block.hash}"

    def display_chain(self):
        return [block.__dict__ for block in self.chain]

# === Load models ===
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# === Streamlit App ===
st.set_page_config(page_title="Britcoin Fraud Detection", layout="wide")
st.title("💷 Britcoin: Secure Digital GBP Fraud Detection")

blockchain = Blockchain()

st.sidebar.header("🔵 Enter Transaction Details")
amount = st.sidebar.number_input("Amount", min_value=0.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance Origin", min_value=0.0)
newbalanceOrig = st.sidebar.number_input("New Balance Origin", min_value=0.0)
type_selected = st.sidebar.selectbox("Transaction Type", options=[("TRANSFER", 1), ("CASH_OUT", 0)])

model_choice = st.sidebar.radio("Choose Model", ("Random Forest", "XGBoost"))

if st.sidebar.button("🚀 Submit Transaction"):
    # Correct feature list used during model training
    feature_order = [
        'amount', 'amount_to_balance_ratio', 'oldbalanceOrg', 'step',
        'zero_balance_orig', 'zero_balance_dest', 'type_encoded',
        'oldbalanceDest', 'newbalanceOrig', 'balance_diff_orig',
        'balance_diff_dest', 'newbalanceDest'
    ]

    # Build full input row
    input_data = {
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'type_encoded': type_selected[1],
        'step': 1,
        'oldbalanceDest': 0.0,
        'newbalanceDest': 0.0,
        'balance_diff_orig': oldbalanceOrg - newbalanceOrig,
        'balance_diff_dest': 0.0,
        'amount_to_balance_ratio': amount / (oldbalanceOrg + 1),
        'zero_balance_orig': 1 if oldbalanceOrg == 0 else 0,
        'zero_balance_dest': 1
    }

    input_df = pd.DataFrame([input_data])

    # Arrange columns correctly
    input_df = input_df[feature_order]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    if model_choice == "Random Forest":
        prediction = rf_model.predict(input_scaled)[0]
    else:
        prediction = xgb_model.predict(input_scaled)[0]

    # Result
    st.subheader("🎯 Prediction Result")
    if prediction == 1:
        st.error("🚫 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction.")

    # Blockchain
    blockchain_result = blockchain.add_transaction(input_data, prediction)
    st.info(blockchain_result)

# === Blockchain Ledger Display ===
st.subheader("📄 Blockchain Ledger")
ledger = blockchain.display_chain()
st.dataframe(pd.DataFrame(ledger))
