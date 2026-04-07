// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AuditRegistry {
    
    // Updated Struct to match your NLP UI exactly
    struct AuditRecord {
        string modelVersion;        // e.g., "Qwen-LoRA-v3"
        string reportUrl;           // e.g., GitHub/Drive link
        uint256 cosineBiasScore;    // e.g., 85
        uint256 weatScore;          // e.g., 12
        uint256 counterFactualDist; // e.g., 65
        uint256 timestamp;          // Block timestamp
        address auditor;            // Wallet address of the tester
    }

    AuditRecord[] public audits;
    
    // Event to announce a new audit on the network
    event AuditLogged(string modelVersion, uint256 timestamp);

    // Write function with updated parameters
    function logAudit(
        string memory _modelVersion,
        string memory _reportUrl,
        uint256 _cosineBiasScore,
        uint256 _weatScore,
        uint256 _counterFactualDist
    ) public {
        
        audits.push(AuditRecord({
            modelVersion: _modelVersion,
            reportUrl: _reportUrl,
            cosineBiasScore: _cosineBiasScore,
            weatScore: _weatScore,
            counterFactualDist: _counterFactualDist,
            timestamp: block.timestamp,
            auditor: msg.sender
        }));

        emit AuditLogged(_modelVersion, block.timestamp);
    }

    // Read functions
    function getAuditCount() public view returns (uint256) {
        return audits.length;
    }

    function getAudit(uint256 _index) public view returns (AuditRecord memory) {
        return audits[_index];
    }
}