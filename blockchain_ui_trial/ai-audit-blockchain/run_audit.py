import json
import hashlib
import os
import time
from web3 import Web3
from dotenv import load_dotenv

# 1. Load Secrets
load_dotenv()

RPC_URL = os.getenv("RPC_URL")

try:
  CONTRACT_ADDRESS = Web3.to_checksum_address(os.getenv("CONTRACT_ADDRESS"))
  WALLET_ADDRESS = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))
except Exception as e:
  print(f"❌ Address Error: Check your .env file. {e}")
  exit()

PRIVATE_KEY = os.getenv("PRIVATE_KEY")

# --- THE ABI (Same as before) ---
CONTRACT_ABI = json.loads('''[
	{
		"anonymous": false,
		"inputs": [
			{"indexed": true, "internalType": "uint256", "name": "index", "type": "uint256"},
			{"indexed": false, "internalType": "string", "name": "modelId", "type": "string"},
			{"indexed": false, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
		],
		"name": "AuditLogged",
		"type": "event"
	},
	{
		"inputs": [],
		"name": "getAuditCount",
		"outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{"internalType": "string", "name": "_modelId", "type": "string"},
			{"internalType": "string", "name": "_ipfsCid", "type": "string"},
			{"internalType": "string", "name": "_datasetHash", "type": "string"},
			{"internalType": "uint256", "name": "_biasScore", "type": "uint256"}
		],
		"name": "logAudit",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	}
]''')

def main():
    if not PRIVATE_KEY or not WALLET_ADDRESS:
        print("❌ Error: Missing credentials in .env file")
        return

    # Setup Connection
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        print("❌ Connection to Sepolia failed.")
        return

    contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
    
    print(f"🔗 Connected to Sepolia. Block: {w3.eth.block_number}")
    print(f"👤 User: {WALLET_ADDRESS}")

    # --- SIMULATE DATA ---
    print("\n1️⃣  Simulating Model Training...")
    model_version = "Simulationv1.0-Secure"
    bias_score = 79
    dataset_hash = hashlib.sha256(b"Secure Data").hexdigest()
    ipfs_cid = "QmSecureHashFromEnv" 

    # --- WRITE TO BLOCKCHAIN (ROBUST MODE) ---
    print("\n2️⃣  Submitting to Blockchain...")
    
    # 1. Get the initial nonce estimate
    nonce = w3.eth.get_transaction_count(WALLET_ADDRESS, 'pending')
    
    # 2. Retry Loop: Keeps trying until it finds the right nonce
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"   Attempt {attempt+1}: Using Nonce {nonce}")
            
            tx = contract.functions.logAudit(
                model_version, ipfs_cid, dataset_hash, bias_score
            ).build_transaction({
                'chainId': 11155111, 
                'gas': 500000,
                'gasPrice': w3.eth.gas_price,
                'nonce': nonce
            })

            signed_tx = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            # If we get here, it worked!
            print(f"⏳ Transaction sent! Hash: {w3.to_hex(tx_hash)}")
            
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            if receipt.status == 1:
                print("\n✅ Audit Successfully Logged on Chain!")
                print(f"   View on Etherscan: https://sepolia.etherscan.io/tx/{w3.to_hex(tx_hash)}")
                return # Exit successfully
            else:
                print("\n❌ Transaction Failed.")
                return

        except Exception as e:
            error_msg = str(e)
            # 3. Check if the error is specifically about the "Nonce"
            if "nonce too low" in error_msg.lower() or "nonce" in error_msg.lower():
                print(f"   ⚠️  Nonce {nonce} was too low. Auto-correcting...")
                nonce += 1 # Increment and try loop again
            else:
                # If it's a different error, crash properly
                print(f"❌ Unexpected Error: {e}")
                break

if __name__ == "__main__":
    main()
