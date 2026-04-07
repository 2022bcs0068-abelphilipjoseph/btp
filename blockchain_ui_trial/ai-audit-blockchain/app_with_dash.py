import streamlit as st
import pandas as pd
from web3 import Web3
import json
import streamlit.components.v1 as components
import graphviz
import os
from dotenv import load_dotenv

# --- 1. SETUP & CONFIG ---
load_dotenv()

# Constants from .env
RPC_URL = os.getenv("RPC_URL")
CONTRACT_ADDRESS =Web3.to_checksum_address(os.getenv("CONTRACT_ADDRESS"))
#WALLET_ADDRESS = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))
#PRIVATE_KEY = os.getenv("PRIVATE_KEY")

# Connect to Blockchain
w3 = Web3(Web3.HTTPProvider(RPC_URL))

# Smart Contract ABI
# (Assuming you are using the latest "Research Grade" contract we discussed)

CONTRACT_ABI = [
	{
		"inputs": [],
		"name": "getAuditCount",
		"outputs": [{"internalType": "uint256","name": "","type": "uint256"}],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [{"internalType": "uint256","name": "_index","type": "uint256"}],
		"name": "getAudit",
		"outputs": [
			{
				"components": [
					{"internalType": "string","name": "modelVersion","type": "string"},
					{"internalType": "string","name": "reportUrl","type": "string"},
					{"internalType": "uint256","name": "cosineBiasScore","type": "uint256"},
					{"internalType": "uint256","name": "weatScore","type": "uint256"},
					{"internalType": "uint256","name": "counterFactualDist","type": "uint256"},
					{"internalType": "uint256","name": "timestamp","type": "uint256"},
					{"internalType": "address","name": "auditor","type": "address"}
				],
				"internalType": "struct AuditRegistry.AuditRecord",
				"name": "",
				"type": "tuple"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{"internalType": "string","name": "_modelVersion","type": "string"},
			{"internalType": "string","name": "_reportUrl","type": "string"},
			{"internalType": "uint256","name": "_cosineBiasScore","type": "uint256"},
			{"internalType": "uint256","name": "_weatScore","type": "uint256"},
			{"internalType": "uint256","name": "_counterFactualDist","type": "uint256"}
		],
		"name": "logAudit",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	}
]


contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)


# --- 3. STREAMLIT UI ---

st.set_page_config(page_title="AI Fairness Registry", layout="wide")

st.title("AI Fairness Registry (Simulation Mode)")
st.markdown("---")

# === SECTION A: PUBLIC REGISTRY (Read from Chain) ===
st.subheader("Verified Audit Registry")

if st.button("🔄 Refresh Data"):
    st.rerun()

# Fetch Data
try:
    audit_count = contract.functions.getAuditCount().call()
    st.metric("Total Verified Models", audit_count)

    data = []
    # Loop backwards to show newest first
    for i in range(audit_count - 1, -1, -1):
        record = contract.functions.getAudit(i).call()
        # record order: [modelVersion, reportUrl, biasScore, maleBiasRate, isCompliant, timestamp, auditor]
        data.append({
            "Model ID": record[0],
            "Cosine Bias": f"{record[2]}%",
            "WEAT Score": f"{record[3]}%",
            "Counterfactual Distance ": f"{record[4]}%",
            "Evidence": record[1], # Showing the URL string directly
            "Auditor": f"{record[6][:6]}...{record[6][-4:]}",
        })

    if data:
        df = pd.DataFrame(data)
        
        # Display as a fancy table
        st.dataframe(
            df,
            column_config={
                "Evidence": st.column_config.LinkColumn("Evidence Link"),
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No audits found in registry yet.")

except Exception as e:
    st.error(f"Error fetching data: {e}")

st.markdown("### Architecture: ")

# Create a Graphviz flowchart
flowchart = graphviz.Digraph(engine='dot')
flowchart.attr(rankdir='LR', size='12,8', dpi='70')
flowchart.attr('node', shape='rectangle', style='filled', color='lightblue', fontname='Helvetica')

flowchart.node('A', 'Client-Side Auditor\n(MetaMask)', color='#f6851b', fontcolor='white')
flowchart.node('B', 'Smart Contract\n(RBAC Whitelist)', color='#2c3e50', fontcolor='white')
flowchart.node('C', 'Ethereum Blockchain\n(Immutable Ledger)', color='#8e44ad', fontcolor='white')
flowchart.node('D', 'Public Dashboard\n(Streamlit)', color='#27ae60', fontcolor='white')

flowchart.edge('A', 'B', label=' 1. Signs Data with\nPrivate Key')
flowchart.edge('B', 'C', label=' 2. Validates & Mines\n(Rejects if altered)')
flowchart.edge('C', 'D', label=' 3. Reads Read-Only State')

st.graphviz_chart(flowchart)
