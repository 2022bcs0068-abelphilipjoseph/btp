// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AuditRegistry {
    
    // --- CONFIGURATION ---
    // The "Pass/Fail" Threshold. 
    // In a real paper, you justify this (e.g., Score > 80 is acceptable).
    uint256 public minAcceptableScore = 80; 
    address public admin;

    struct AuditRecord {
        string modelVersion;    // e.g., "Qwen-LoRA-v2"
        string ipfsCid;         // URL to Evidence (HTML Report/Drive Link)
        uint256 biasScore;      // General Fairness Score (0-100)
        uint256 maleBiasRate;   // Specific Metric (e.g., 12 for 12%)
        bool isCompliant;       // AUTOMATED: True if Score >= 80
        uint256 timestamp;      // Block timestamp
        address auditor;        // Who submitted it
    }

    AuditRecord[] public audits;

    // Event for external listeners (like your dashboard)
    event AuditLogged(string modelVersion, bool isCompliant, uint256 timestamp);

    constructor() {
        admin = msg.sender;
    }

    // --- GOVERNANCE ---
    // Allows the Professor/Admin to raise the standards later
    function setBenchmark(uint256 _newScore) public {
        require(msg.sender == admin, "Only Admin can change benchmarks");
        minAcceptableScore = _newScore;
    }

    // --- CORE LOGIC ---
    function logAudit(
        string memory _modelVersion,
        string memory _ipfsCid,
        uint256 _biasScore,
        uint256 _maleBiasRate
    ) public {
        
        // 🧠 THE "SMART" PART
        // The blockchain decides if the model passes, not the user.
        bool complianceStatus = (_biasScore >= minAcceptableScore);

        audits.push(AuditRecord({
            modelVersion: _modelVersion,
            ipfsCid: _ipfsCid,
            biasScore: _biasScore,
            maleBiasRate: _maleBiasRate,
            isCompliant: complianceStatus, 
            timestamp: block.timestamp,
            auditor: msg.sender
        }));

        emit AuditLogged(_modelVersion, complianceStatus, block.timestamp);
    }

    // --- READ FUNCTIONS ---
    function getAuditCount() public view returns (uint256) {
        return audits.length;
    }

    function getAudit(uint256 _index) public view returns (AuditRecord memory) {
        return audits[_index];
    }
}