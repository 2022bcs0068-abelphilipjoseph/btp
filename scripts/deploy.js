import { ethers } from "ethers";

async function main() {
  // Connect to local Hardhat or testnet node
  const provider = new ethers.JsonRpcProvider("http://127.0.0.1:8545");
  
  // Get list of accounts
  const accounts = await provider.listAccounts();
  console.log("Accounts:", accounts);

  // Get signer
  const signer = await provider.getSigner(accounts[0].address);
  console.log("Signer address:", await signer.getAddress());

  // Example: check balance
  const balance = await provider.getBalance(accounts[0].address);
  console.log("Balance:", ethers.formatEther(balance), "ETH");
}

main().catch(console.error);
