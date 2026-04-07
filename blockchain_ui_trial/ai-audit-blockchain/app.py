import streamlit as st
from web3 import Web3
import pandas as pd
import json
import os
from dotenv import load_dotenv

# Force reload of .env
load_dotenv(override=True)

RPC_URL = Web3.to_checksum_address(os.getenv("RPC_URL"))
CONTRACT_ADDRESS = Web3.to_checksum_address(os.getenv("CONTRACT_ADDRESS"))

st.set_page_config(page_title="Fairness Audit Log", page_icon="🛡️", layout="wide")

st.title("🛡️ Immutable AI Fairness Registry")

# --- DEBUG SECTION (Remove later) ---
st.warning("🛠️ Debug Info")
st.write(f"**RPC URL:** `{RPC_URL}`")
st.write(f"**Contract Address:** `{CONTRACT_ADDRESS}`")
# ------------------------------------

@st.cache_resource
def get_contract():
    if not RPC_URL:
        st.error("❌ RPC_URL is missing from .env")
        return None, None
    if not CONTRACT_ADDRESS:
        st.error("❌ CONTRACT_ADDRESS is missing from .env")
        return None, None

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    
    if not w3.is_connected():
        st.error(f"❌ Failed to connect to Sepolia. RPC: {RPC_URL}")
        return None, None
    
    # ABI
    abi = json.loads('''[
        {"inputs": [], "name": "getAuditCount", "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
        {"inputs": [{"internalType": "uint256", "name": "_index", "type": "uint256"}], "name": "getAudit", "outputs": [{"components": [{"internalType": "string", "name": "modelId", "type": "string"}, {"internalType": "string", "name": "ipfsCid", "type": "string"}, {"internalType": "string", "name": "datasetHash", "type": "string"}, {"internalType": "uint256", "name": "biasScore", "type": "uint256"}, {"internalType": "uint256", "name": "timestamp", "type": "uint256"}, {"internalType": "address", "name": "auditor", "type": "address"}], "internalType": "struct AuditRegistry.AuditRecord", "name": "", "type": "tuple"}], "stateMutability": "view", "type": "function"}
    ]''')
    
    # Checksum address just in case
    try:
        c_address = Web3.to_checksum_address(CONTRACT_ADDRESS)
        contract = w3.eth.contract(address=c_address, abi=abi)
        return w3, contract
    except Exception as e:
        st.error(f"❌ Invalid Contract Address format: {e}")
        return None, None

w3, contract = get_contract()

if contract:
    try:
        # Test call
        count = contract.functions.getAuditCount().call()
        st.success(f"✅ Connected! Found {count} audits.")
        
        # ... (Rest of the logic)
        data = []
        for i in range(count - 1, -1, -1):
            record = contract.functions.getAudit(i).call()
            data.append({
                "Model Version": record[0],
                "Fairness Score": record[3] / 100.0,
                "Timestamp": pd.to_datetime(record[4], unit='s'),
                "Dataset Hash": record[2],
                "IPFS Report": record[1],
                "Auditor": record[5]
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df)

    except Exception as e:
        st.error(f"🔥 Error calling contract: {e}")
