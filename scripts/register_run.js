// scripts/register_run.js
import { ethers } from "ethers";

async function main() {
  const registryAddress = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266";
  const RunRegistry = await ethers.getContractFactory("RunRegistry");
  const reg = await RunRegistry.attach(registryAddress);

  // example sha256 hex from python output (prefix with 0x)
  const artifactHash = "0x" + "86e3d5a36632bbddbbdab5e1a7ab9885705069adc56ead8085bf3e63ad3d0ac1";
  const tx = await reg.registerRun(artifactHash, "bias-summary-json", "local://data/artifacts/bias_summary.json");
  const receipt = await tx.wait();
  console.log("Registered run. tx:", receipt.transactionHash);
}
main().catch(console.error);
